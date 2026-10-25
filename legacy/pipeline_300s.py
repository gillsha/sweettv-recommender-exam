"""Full computation runs in Kaggle. Writes aggregate evidence and final CSV."""
import os,sys,json,time,platform,hashlib,importlib.metadata,resource
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
from core import ScheduleIndex,union_length,ap_at_k,map_at_k,validate_submission
from models import TrainingBundle,SEED

START=time.perf_counter()
OUT=Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
FILES={p.name:p for p in Path('/kaggle/input').rglob('*.csv')}
THRESHOLD=300
CLOCK_AFTER=int(pd.Timestamp('2020-03-29 04:00:00').timestamp())
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def stamp(s):
    raw=int(pd.Timestamp(s).timestamp())
    return raw-3600 if raw>=CLOCK_AFTER else raw
def display(s):
    value=int(s)
    if value>=CLOCK_AFTER-3600: value+=3600
    return str(pd.to_datetime(value,unit='s'))
def seconds(s,fmt):
    parsed=pd.to_datetime(s,format=fmt,errors='raise')
    assert parsed.notna().all()
    raw=parsed.astype('int64')//10**9
    return raw-(raw>=CLOCK_AFTER).astype('int64')*3600

manifest={'seed':SEED,'threshold_seconds':THRESHOLD,'python':sys.version,
          'platform':platform.platform(),'cpu_count':os.cpu_count(),
          'cpu_model':next((x.split(':',1)[-1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown'),
          'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','numba']},
          'files':{k:{'size':p.stat().st_size} for k,p in FILES.items()},'timings':{},'checks':{}}
template=pd.read_csv(FILES['submission.csv'],dtype=str)
assert list(template)==['user_id','tv_show_id'] and template.user_id.is_unique
cohort=template.user_id.tolist()
schedule=[]; schema={}; catalog=[]
for name in ['export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv']:
    f=pd.read_csv(FILES[name],low_memory=False)
    f['start_s']=seconds(f.start_time,'%d.%m.%Y %H:%M:%S')
    f['end_s']=f.start_s+f.duration
    assert f.tv_show_id.notna().all()
    labelled=f.tv_show_id.ne(0)
    schema[name]={'rows':len(f),'nulls':f.isna().sum().to_dict(),'labelled_rows':int(labelled.sum()),
                  'labelled_row_fraction':float(labelled.mean()),'labelled_duration_fraction':float(f.loc[labelled,'duration'].sum()/f.duration.sum()),
                  'nonpositive_duration_rows':int(f.duration.le(0).sum()),
                  'channels':int(f.channel_id.nunique()),'items':int(f.loc[labelled,'tv_show_id'].nunique()),
                  'start':display(f.start_s.min()),'last_start':display(f.start_s.max()),'end':display(f.end_s.max()),
                  'duration_quantiles':f.duration.quantile([0,.1,.5,.9,.99,1]).to_dict()}
    # Verify duration in seconds against adjacent start times (not assumed all contiguous).
    ordered=f.sort_values(['channel_id','start_s'])
    gap=ordered.groupby('channel_id').start_s.shift(-1)-ordered.start_s
    schema[name]['duration_equals_next_start_gap_fraction']=float((gap==ordered.duration).mean())
    f=f[f.duration.gt(0)].copy()
    labelled=f.tv_show_id.ne(0)
    f['source']=name
    catalog.append(f.loc[labelled].copy())
    schedule.append(f[['channel_id','start_s','end_s','tv_show_id','duration','source']])
    del f
all_schedule=pd.concat(schedule,ignore_index=True)
all_catalog=pd.concat(catalog,ignore_index=True)
del schedule,catalog
# ISO week boundaries are derived from the schedule dates, not from viewing outcomes.
valid_source=all_schedule[all_schedule.source.eq('export_arh_21-30-final.csv')]
test_source=all_schedule[all_schedule.source.eq('export_arh_31-42-final.csv')]
cutoff=stamp(pd.Timestamp(display(valid_source.start_s.min())).normalize())
test_start=stamp(pd.Timestamp(display(test_source.start_s.min())).normalize())
train_start=stamp(pd.Timestamp(display(all_schedule.start_s.min())).normalize())
assert train_start<cutoff<test_start
manifest['split']={'train_start':display(train_start),'cutoff':display(cutoff),'validation_end':display(test_start),
                   'cohort_size':len(cohort),'repeats_allowed':True,'empty_truth_AP':0,'candidate_policy':'labelled IDs airing in prediction window',
                   'test_schedule_features':'ID eligibility only; content metadata from historical schedule'}
history=all_schedule[(all_schedule.start_s<test_start)&all_schedule.end_s.gt(train_start)&all_schedule.tv_show_id.ne(0)].copy()
keys=['channel_id','start_s','end_s','tv_show_id']
manifest['checks']['duplicate_labelled_airings']=int(history.duplicated(keys).sum())
history=history.drop_duplicates(keys).reset_index(drop=True)
history['airing_key']=np.arange(len(history))
index=ScheduleIndex(history[['channel_id','start_s','end_s','tv_show_id','airing_key']].itertuples(index=False,name=None))
valid_candidates=set(map(int,all_schedule.loc[(all_schedule.start_s<test_start)&all_schedule.end_s.gt(cutoff),'tv_show_id']))-{0}
test_candidates=set(map(int,test_source.tv_show_id))-{0}
metadata_train=all_catalog[all_catalog.start_s<cutoff].sort_values('start_s')
metadata_full=all_catalog[all_catalog.start_s<test_start].sort_values('start_s')
manifest['split'].update(validation_candidates=len(valid_candidates),test_candidates=len(test_candidates))
save('schema_audit.json',schema)
print('Split',manifest['split'],flush=True)

segments=defaultdict(list)
coverage={'rows':0,'invalid_intervals':0,'duration_mismatch':0,'clipped_view_seconds':0,'labelled_overlap_seconds':0,
          'rows_with_labelled_overlap':0,'rows_outside_history':0,'rows_crossing_cutoff':0,'overlaps':0,'raw_duration_mismatch':0}
examples=[]; view_users=set(); duration_hist=[]; min_view=None; max_view=None
duration_deltas=Counter();mismatch_examples=[]
join_start=time.perf_counter()
for chunk in pd.read_csv(FILES['dataset11-30.csv'],dtype={'user_id':str},chunksize=250000):
    starts=seconds(chunk.start_time,'%Y-%m-%d %H:%M:%S').to_numpy()
    ends=seconds(chunk.stop_time,'%Y-%m-%d %H:%M:%S').to_numpy()
    lengths=chunk.duraton.to_numpy()
    raw_starts=starts+(starts>=CLOCK_AFTER-3600)*3600
    raw_ends=ends+(ends>=CLOCK_AFTER-3600)*3600
    raw_deltas=raw_ends-raw_starts-lengths
    coverage['raw_duration_mismatch']+=int(np.sum(raw_deltas!=0))
    for delta in raw_deltas[raw_deltas!=0]: duration_deltas[int(delta)]+=1
    for pos in np.flatnonzero(raw_deltas)[:max(0,8-len(mismatch_examples))]:
        mismatch_examples.append({'start_time':str(chunk.iloc[pos].start_time),'stop_time':str(chunk.iloc[pos].stop_time),'duraton':int(lengths[pos]),'wallclock_minus_duraton_seconds':int(raw_deltas[pos])})
    coverage['rows']+=len(chunk)
    coverage['duration_mismatch']+=int(np.sum(ends-starts!=lengths))
    coverage['invalid_intervals']+=int(np.sum(ends<=starts))
    min_view=int(starts.min()) if min_view is None else min(min_view,int(starts.min()))
    max_view=int(ends.max()) if max_view is None else max(max_view,int(ends.max()))
    duration_hist.append(lengths)
    for user,ch,start,end,reported_length in zip(chunk.user_id,chunk.vsetv_id,starts,ends,lengths):
        view_users.add(user)
        delta=int(end-start-reported_length)
        if delta or end<=start:
            continue
        a,b=max(int(start),train_start),min(int(end),test_start)
        if b<=a:
            coverage['rows_outside_history']+=1;continue
        coverage['clipped_view_seconds']+=b-a
        coverage['rows_crossing_cutoff']+=int(a<cutoff<b)
        matches=index.overlaps(int(ch),a,b)
        if matches: coverage['rows_with_labelled_overlap']+=1
        coverage['labelled_overlap_seconds']+=union_length([(x[2],x[3]) for x in matches])
        coverage['overlaps']+=len(matches)
        for item,key,left,right,program_seconds in matches:
            if left<cutoff:
                segments[(0,user,item,key,program_seconds)].append((left,min(right,cutoff)))
            if right>cutoff:
                segments[(1,user,item,key,program_seconds)].append((max(left,cutoff),right))
            if len(examples)<8 and right-left>=THRESHOLD:
                examples.append({'user_id':user,'channel_id':int(ch),'view_start':display(start),'view_end':display(end),
                    'airing_key':int(key),'tv_show_id':int(item),'overlap_start':display(left),'overlap_end':display(right),
                    'overlap_seconds':right-left,'programme_seconds':program_seconds,'view_duration':int(end-start)})
    print('Joined view rows',coverage['rows'],flush=True)
records=[];full_records=[]
for (window,user,item,key,length),ints in segments.items():
    watched=union_length(ints)
    full_records.append((user,int(item),int(key),watched,length))
    if watched>=THRESHOLD:
        records.append((window,user,int(item),int(key),watched,length,watched/length))
del segments
interactions=pd.DataFrame(records,columns=['window','user_id','tv_show_id','airing_key','watch_seconds','programme_seconds','watch_fraction'])
del records
train=interactions[interactions.window.eq(0)].copy();valid=interactions[interactions.window.eq(1)].copy()
full_interactions=pd.DataFrame(full_records,columns=['user_id','tv_show_id','airing_key','watch_seconds','programme_seconds'])
full_interactions=full_interactions.groupby(['user_id','tv_show_id','airing_key','programme_seconds'],as_index=False).watch_seconds.sum()
full_interactions=full_interactions[full_interactions.watch_seconds.ge(THRESHOLD)].copy()
del full_records
assert len(train)>0 and len(valid)>0
durations=np.concatenate(duration_hist)
coverage.update(view_users=len(view_users),first_view=display(min_view),last_view_end=display(max_view),
    duration_quantiles={str(q):float(np.quantile(durations,q)) for q in [0,.01,.1,.25,.5,.75,.9,.95,.99,1]},
    view_fraction_at_least_300=float(np.mean(durations>=THRESHOLD)),
    train_interactions=len(train),validation_interactions=len(valid),train_users=int(train.user_id.nunique()),validation_users=int(valid.user_id.nunique()),
    train_items=int(train.tv_show_id.nunique()),validation_items=int(valid.tv_show_id.nunique()),
    labelled_row_coverage=coverage['rows_with_labelled_overlap']/coverage['rows'],
    labelled_seconds_coverage=coverage['labelled_overlap_seconds']/coverage['clipped_view_seconds'],
    qualified_watch_fraction_quantiles=interactions.watch_fraction.quantile([0,.1,.5,.9,1]).to_dict())
coverage['duration_delta_counts']=dict(duration_deltas)
coverage['duration_mismatch_examples']=mismatch_examples
coverage['duration_policy']='Normalize the observed 2020-03-29 clock jump by subtracting 3600 from timestamps at/after 04:00, in BOTH schedule and views. Transition identified within training history; no absolute timezone inferred. Then require stop-start=duraton for every view.'
assert coverage['duration_mismatch']==0
assert interactions.tv_show_id.ne(0).all() and interactions.watch_fraction.le(1).all()
manifest['timings']['join_seconds']=time.perf_counter()-join_start
# Cross-check actual examples against a second, direct interval calculation.
for example in examples:
    row=history.iloc[example['airing_key']]
    a=max(stamp(example['view_start']),int(row.start_s));b=min(stamp(example['view_end']),int(row.end_s),test_start)
    assert b-a==example['overlap_seconds'] and int(row.channel_id)==example['channel_id']
    example.update(airing_start=display(row.start_s),airing_end=display(row.end_s))
manifest['checks']['actual_overlap_examples_verified']=len(examples)
save('coverage.json',coverage);save('join_examples.json',examples)
print('Coverage',json.dumps(coverage,default=str),flush=True)

truth={u:set(map(int,g.tv_show_id)) for u,g in valid.groupby('user_id') if u in set(cohort)}
assert all(items<=valid_candidates for items in truth.values())
users=sorted(set(train.user_id)|set(cohort))
bundle=TrainingBundle(train,metadata_train,users)
results=[];predictions={};score_cache={}
def evaluate(name,scores,elapsed,params):
    all_ranks=bundle.rank(scores,valid_candidates)
    ranks={u:all_ranks[u] for u in cohort}
    assert all(len(r)==len(set(r))==5 and set(r)<=valid_candidates for r in ranks.values())
    values=np.array([ap_at_k(truth.get(u,()),ranks[u]) for u in cohort])
    active=np.array([bool(truth.get(u)) for u in cohort])
    result={'method':name,'map5':float(values.mean()),'active_map5':float(values[active].mean()),'active_users':int(active.sum()),
            'empty_truth_users':int((~active).sum()),'fit_score_seconds':elapsed,'parameters':params}
    assert abs(result['map5']-map_at_k(truth,ranks,cohort))<1e-12
    results.append(result);predictions[name]=ranks;score_cache[name]=scores
    save('validation_results.json',results);print('RESULT',result,flush=True)

for name,fn,params in [
    ('popularity',lambda:bundle.popularity_scores(valid_candidates),{'statistic':'unique training viewers'}),
    ('content',lambda:bundle.content_scores(valid_candidates),{'features':'category, genres, year, director, actors','profile':'log1p airing count','similarity':'cosine'}),
    ('als32',lambda:bundle.als_scores(valid_candidates,factors=32),{'factors':32,'alpha':20,'regularization':.1,'iterations':20}),
    ('als64',lambda:bundle.als_scores(valid_candidates,factors=64),{'factors':64,'alpha':20,'regularization':.1,'iterations':20})]:
    t=time.perf_counter();scores=fn();evaluate(name,scores,time.perf_counter()-t,params)

# A repeat-frequency component tests whether persistent series habits outperform latent similarity.
c,ci,pop=bundle.candidate_map(valid_candidates)
repeat_scores=np.zeros((len(users),len(c)),np.float32); known=ci>=0
repeat_scores[:,known]=bundle.weights[:,ci[known]].toarray()
evaluate('repeat',repeat_scores,0,{'score':'log1p qualified training airing count; same popularity fallback'})
best_als=max(['als32','als64'],key=lambda n:next(r['map5'] for r in results if r['method']==n))
def rank_score(scores):
    # Reciprocal rank scores over all candidates; same deterministic tie-breaks.
    out=np.zeros_like(scores,dtype=np.float32)
    for i,row in enumerate(scores):
        ordered=np.lexsort((c,-pop,-row))
        positive=ordered[np.isfinite(row[ordered]) & (row[ordered]>0)]
        fallback=np.lexsort((c,-pop))
        positive_set=set(map(int,positive))
        order=np.concatenate([positive,np.array([j for j in fallback if int(j) not in positive_set],dtype=int)])
        out[i,order]=1/(60+np.arange(1,len(c)+1))
    return out
als_rr=rank_score(score_cache[best_als]);repeat_rr=rank_score(repeat_scores)
for w in [.25,.5,.75]:
    t=time.perf_counter();fused=w*repeat_rr+(1-w)*als_rr
    evaluate(f'hybrid_{w}',fused,time.perf_counter()-t,{'repeat_weight':w,'als':best_als,'rank_offset':60})
winner=max(results,key=lambda r:r['map5'])
manifest['selected']=winner
manifest['checks']['train_nonzero_items']=len(bundle.items)
manifest['checks']['cold_cohort_users']=sum(bool(bundle.cold[bundle.uindex[u]]) for u in cohort)
manifest['checks']['validation_new_items']=len(valid_candidates-set(bundle.items))
manifest['checks']['training_metadata_only']=True

# Predetermined activity strata; same two users for every method; no cherry-picked hits.
activity=train.groupby('user_id').size().to_dict()
eligible=sorted([u for u in cohort if truth.get(u) and activity.get(u,0)>0],key=lambda u:(activity[u],u))
chosen=[eligible[len(eligible)//4],eligible[(3*len(eligible))//4]]
lookup=all_catalog.sort_values('start_s').drop_duplicates('tv_show_id').set_index('tv_show_id')
def describe(item,user):
    row=lookup.loc[item]
    return {'id':int(item),'title':str(row.tv_show_title),'category':str(row.tv_show_category),'hit':item in truth.get(user,set()),
            'seen_in_training':bool(((train.user_id==user)&(train.tv_show_id==item)).any())}
user_examples=[]
for user in chosen:
    user_examples.append({'user_id':user,'training_airings':int(activity[user]),'actual_ids':sorted(truth[user]),
                         'training_top':train[train.user_id.eq(user)].tv_show_id.value_counts().head(5).to_dict(),
                         'methods':{n:[describe(x,user) for x in ranks[user]] for n,ranks in predictions.items()},
                         'ap5':{n:ap_at_k(truth[user],ranks[user]) for n,ranks in predictions.items()},
                         'validation_evidence':[dict(tv_show_id=int(r.tv_show_id),watch_seconds=int(r.watch_seconds),
                             programme_seconds=int(r.programme_seconds),airing_start=display(history.iloc[int(r.airing_key)].start_s),
                             channel_id=int(history.iloc[int(r.airing_key)].channel_id))
                             for r in valid[valid.user_id.eq(user)].head(5).itertuples()]})
save('user_examples.json',user_examples)
save('validation_predictions.json',predictions)
save('validation_truth.json',{u:sorted(truth.get(u,set())) for u in cohort})
# Select a nontrivial real AP exercise separately, not as the two comparison users.
exercise_users=[u for u in cohort if 0<ap_at_k(truth.get(u,()),predictions[winner['method']][u])<1]
exercise_user=next((u for u in exercise_users if len(truth[u])<=8),exercise_users[0])
save('personal_check_input.json',{'user_id':exercise_user,'method':winner['method'],
    'predictions':predictions[winner['method']][exercise_user],'actual':sorted(truth[exercise_user]),
    'formula':'AP@5=sum_{r=1..5}(hits_in_top_r/r)*hit_r / min(5, number_of_relevant_items); empty truth -> 0',
    'selected_result':winner,'iteration':[r for r in results if r['method'] in ['als32','als64']]})

# Refit only the validation-selected configuration using all permitted history.
t=time.perf_counter()
final_users=sorted(set(full_interactions.user_id)|set(cohort))
full=TrainingBundle(full_interactions,metadata_full,final_users)
name=winner['method'];par=winner['parameters']
if name=='popularity': final_scores=full.popularity_scores(test_candidates)
elif name=='content': final_scores=full.content_scores(test_candidates)
elif name.startswith('als'): final_scores=full.als_scores(test_candidates,factors=par['factors'])
else:
    c,ci,pop=full.candidate_map(test_candidates)
    final_repeat=np.zeros((len(final_users),len(c)),np.float32);known=ci>=0
    final_repeat[:,known]=full.weights[:,ci[known]].toarray()
    if name=='repeat': final_scores=final_repeat
    else:
        factors=int(par['als'].replace('als',''))
        final_als=full.als_scores(test_candidates,factors=factors)
        w=par['repeat_weight']; final_scores=w*rank_score(final_repeat)+(1-w)*rank_score(final_als)
final_ranks=full.rank(final_scores,test_candidates)
submission=template.copy();submission['tv_show_id']=[' '.join(map(str,final_ranks[u])) for u in cohort]
submission.to_csv(OUT/'submission.csv',index=False)
manifest['checks']['submission']=validate_submission(FILES['submission.csv'],OUT/'submission.csv',test_candidates)
manifest['checks']['submission_sha256']=hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest()
manifest['timings']['final_refit_predict_seconds']=time.perf_counter()-t
manifest['timings']['total_seconds']=time.perf_counter()-START
manifest['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
save('run_manifest.json',manifest)
save('test_candidates.json',sorted(test_candidates))
print('FINAL SELECTED',name,'VALIDATION MAP@5',winner['map5'],'CSV CHECK',manifest['checks']['submission'],flush=True)
