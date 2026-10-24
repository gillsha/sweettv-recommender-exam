"""150 local validation runs: decay x frequency exponent x popularity weight.

Uses the same preprocessed interactions, candidates, repeats and MAP@5 as
Kaggle notebook v7. No Kaggle score is read by this search. Private caches and
per-user predictions must not be committed. CPU only, NumPy/pandas only.
"""
import os,json,time,platform,hashlib,itertools,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from core import ap_at_k,validate_submission

HALF_LIVES=[7,14,28,56,None]
POWERS=[.25,.5,.75,1.,1.5,2.]
POP_WEIGHTS=[.05,.15,.30,.50,.75]
SEED=2026

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def rank(scores,ids,pop):
    """Exact deterministic top five, with the original popularity fallback."""
    fallback=np.lexsort((ids,-pop))
    output=[]
    for row in scores:
        order=np.lexsort((ids,-pop,-row))
        chosen=order[np.isfinite(row[order]) & (row[order]>0)][:5].tolist()
        for j in fallback:
            if len(chosen)==5:break
            if int(j) not in chosen:chosen.append(int(j))
        output.append(ids[chosen].tolist())
    return output

def independent_ap(actual,pred):
    if not actual:return 0.
    hit=np.array([x in actual for x in pred],dtype=float)
    return float(np.sum(np.cumsum(hit)/np.arange(1,6)*hit)/min(5,len(actual)))

class Window:
    def __init__(self,frame,history,users,candidates,cutoff):
        self.users=users;self.uindex={u:i for i,u in enumerate(users)}
        # All logically eligible IDs remain in the ranking, including unseen ones.
        self.ids=np.array(sorted(set(candidates)-{0}),dtype=np.int64)
        index={int(x):i for i,x in enumerate(self.ids)}
        pop=frame.groupby('tv_show_id').user_id.nunique()
        self.pop=np.array([pop.get(x,0) for x in self.ids],dtype=float)
        self.pop_scores=np.broadcast_to(self.pop/self.pop.max(),(len(users),len(self.ids)))
        events=frame[frame.user_id.isin(self.uindex)&frame.tv_show_id.isin(index)].copy()
        self.ui=events.user_id.map(self.uindex).to_numpy(dtype=int)
        self.ii=events.tv_show_id.map(index).to_numpy(dtype=int)
        starts=history.start_s.to_numpy()[events.airing_key.to_numpy(dtype=int)]
        self.age_days=np.maximum(0,cutoff-starts)/86400.
        self.shape=(len(users),len(self.ids))
        self.counts=self.frequency(None)
        self.cold=self.counts.sum(axis=1)==0
    def frequency(self,half_life):
        weights=np.ones(len(self.ui)) if half_life is None else np.exp2(-self.age_days/half_life)
        out=np.zeros(self.shape,dtype=np.float64)
        np.add.at(out,(self.ui,self.ii),weights)
        return out
    def combine(self,frequency,power,pop_weight):
        normalized=np.divide(frequency,frequency.max(axis=1,keepdims=True),out=np.zeros_like(frequency),where=frequency.max(axis=1,keepdims=True)>0)
        scores=(1-pop_weight)*normalized**power+pop_weight*self.pop_scores
        scores[self.cold]=self.pop_scores[self.cold]
        return scores

def run():
    total_start=time.perf_counter()
    out=Path(os.environ.get('SWEETTV_OUTPUT','work/local150'))
    inputs=Path(os.environ.get('SWEETTV_DATA','work/data'))
    original=Path(os.environ.get('SWEETTV_REFERENCE','work/final_run'))
    template=pd.read_csv(inputs/'submission.csv',dtype=str);users=template.user_id.tolist()
    train=pd.read_pickle(out/'train.pkl');valid=pd.read_pickle(out/'validation.pkl')
    full=pd.read_pickle(out/'full.pkl');history=pd.read_pickle(out/'history.pkl')
    candidates=json.loads((out/'validation_candidates.json').read_text())
    test_candidates=json.loads((out/'test_candidates.json').read_text())
    manifest=json.loads((out/'prepare_manifest.json').read_text())
    # Dates use the same normalized relative clock as notebook v7.
    cutoff=int(pd.Timestamp(manifest['split']['cutoff']).timestamp())-3600
    final_cutoff=int(pd.Timestamp(manifest['split']['validation_end']).timestamp())-3600
    truth=valid.groupby('user_id').tv_show_id.apply(lambda x:set(map(int,x))).to_dict()
    actual={u:truth.get(u,set()) for u in users}
    reference_truth=json.loads((original/'validation_truth.json').read_text())
    assert all(actual[u]==set(reference_truth[u]) for u in users),'Local/Kaggle truth mismatch'
    t=time.perf_counter();window=Window(train,history,users,candidates,cutoff)
    baseline=rank(window.counts,window.ids,window.pop)
    reference=json.loads((original/'validation_predictions.json').read_text())['repeat']
    assert all(p==reference[u] for u,p in zip(users,baseline)),'Local/Kaggle baseline ranking mismatch'
    base_values=[independent_ap(actual[u],p) for u,p in zip(users,baseline)]
    base_score=float(np.mean(base_values));baseline_seconds=time.perf_counter()-t
    assert abs(base_score-.7608148148148147)<1e-12
    print('PARITY PASSED: exact Kaggle truth and all 1260 Repeat Top-5; MAP',base_score,flush=True)
    active=np.array([bool(actual[u]) for u in users]);results=[];best=None;best_predictions=None
    (out/'private_predictions').mkdir(exist_ok=True)
    run_id=0
    for half_life in HALF_LIVES:
        t=time.perf_counter();freq=window.frequency(half_life);frequency_seconds=time.perf_counter()-t
        for power,weight in itertools.product(POWERS,POP_WEIGHTS):
            run_id+=1;t=time.perf_counter()
            scores=window.combine(freq,power,weight)
            pred=rank(scores,window.ids,window.pop)
            ap=np.array([independent_ap(actual[u],p) for u,p in zip(users,pred)])
            # Independent existing implementation cross-check for each run.
            assert np.allclose(ap,[ap_at_k(actual[u],p) for u,p in zip(users,pred)],rtol=0,atol=1e-12)
            result={'run':run_id,'half_life_days':half_life,'frequency_power':power,'popularity_weight':weight,
                    'map5':float(ap.mean()),'active_map5':float(ap[active].mean()),
                    'score_rank_metric_seconds':time.perf_counter()-t,'shared_frequency_seconds':frequency_seconds,'seed':SEED}
            results.append(result)
            save(out/'private_predictions'/f'run_{run_id:03d}.json',dict(zip(users,pred)))
            save(out/'search_results.json',results)
            if best is None or result['map5']>best['map5']:
                best=result;best_predictions=pred
            print('RUN',run_id,'MAP',result['map5'],'PARAMS',half_life,power,weight,'BEST',best['map5'],flush=True)
    assert len(results)==150 and len({(r['half_life_days'],r['frequency_power'],r['popularity_weight']) for r in results})==150
    chosen=best if best['map5']>base_score else {'method':'original_repeat','map5':base_score}
    if chosen.get('method')=='original_repeat':best_predictions=baseline
    t=time.perf_counter();final=Window(full,history,users,test_candidates,final_cutoff)
    if chosen.get('method')=='original_repeat':scores=final.counts
    else:scores=final.combine(final.frequency(chosen['half_life_days']),chosen['frequency_power'],chosen['popularity_weight'])
    final_pred=rank(scores,final.ids,final.pop)
    submission=template.copy();submission.tv_show_id=[' '.join(map(str,p)) for p in final_pred]
    submission.to_csv(out/'submission.csv',index=False)
    check=validate_submission(inputs/'submission.csv',out/'submission.csv',test_candidates)
    elapsed=time.perf_counter()-t
    summary={'runs':150,'selected':chosen,'best_search':best,'original_repeat_map5':base_score,
        'baseline_seconds':baseline_seconds,'final_refit_predict_seconds':elapsed,
        'search_and_refit_seconds':time.perf_counter()-total_start,'cpu_machine':platform.machine(),
        'cpu_count':os.cpu_count(),'python':platform.python_version(),
        'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas']},
        'exact_Kaggle_validation_truth':True,'exact_Kaggle_repeat_rankings':True,
        'cohort':len(users),'active_users':int(active.sum()),'empty_truth_users':int((~active).sum()),
        'submission_checks':check,'submission_sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(),
        'selection_uses_Kaggle_score':False,'selection_caveat':'150 configurations reuse the same hold-out; selection optimism is possible; no independent quality guarantee',
        'scoring':'(1-w)*(decayed personal airing count / per-user max)^power + w*(unique training viewers / global max)',
        'decay':'2^(-age_days/half_life); None means no decay; age from airing start to training cutoff',
        'shared_cache':'one frequency matrix per half-life, shared by 30 configurations; per-run times include scoring/ranking/metric, not this shared construction'}
    save(out/'search_summary.json',summary)
    save(out/'selected_validation_predictions.json',dict(zip(users,best_predictions)))
    pd.DataFrame(results).to_csv(out/'search_results.csv',index=False)
    print('LOCAL SEARCH COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':run()
