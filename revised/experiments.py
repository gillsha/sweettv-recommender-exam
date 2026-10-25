"""Select on window A, freeze, then evaluate forward window B exactly once."""
import os,sys,time,json,hashlib,platform,argparse,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from revised.data import stamp,save
from revised.models import Window,rrf,verify_target_update
from als import verify_solver
from core import ap_at_k,validate_submission

START=stamp('2020-03-09');A=stamp('2020-05-18');B=stamp('2020-06-22');END=stamp('2020-07-27')

def counted(interactions,policy='union80'):
 if policy=='union80':return interactions[5*interactions.watch_seconds>=4*interactions.programme_seconds].copy()
 if policy=='single80':return interactions[5*interactions.max_single_seconds>=4*interactions.programme_seconds].copy()
 if policy=='sessions80':return interactions.loc[interactions.index.repeat(interactions.session_count80)].copy()
 if policy=='seconds300':return interactions[interactions.watch_seconds>=300].copy()
 raise ValueError(policy)
def truth_for(events,users,ties='id_asc'):
 c=events.groupby(['user_id','tv_show_id']).size().rename('count').reset_index();out={};frequencies={};ambiguous=0
 for u,f in c.groupby('user_id'):
  if ties=='hash':
   f=f.copy();f['tie']=[int(hashlib.sha256(f'2026:{x}'.encode()).hexdigest()[:12],16) for x in f.tv_show_id]
  else:f=f.assign(tie=f.tv_show_id if ties=='id_asc' else -f.tv_show_id)
  f=f.sort_values(['count','tie'],ascending=[False,True]);out[u]=f.tv_show_id.head(5).astype(int).tolist()
  frequencies[u]=dict(zip(f.tv_show_id.astype(int),f['count'].astype(int)))
  if len(f)>5 and f.iloc[4]['count']==f.iloc[5]['count']:ambiguous+=int(u in users)
 return {u:out.get(u,[]) for u in users},frequencies,ambiguous

def evaluate(truth,pred,users):
 vals=[]
 for u in users:
  actual=set(truth[u]);hit=np.array([x in actual for x in pred[u]],float)
  ap=float(np.sum(np.cumsum(hit)/np.arange(1,6)*hit)/min(5,len(actual))) if actual else 0.
  assert abs(ap-ap_at_k(actual,pred[u]))<1e-12;vals.append(ap)
 return float(np.mean(vals))

def load_window(out,which):
 f=pd.read_pickle(out/'all_positive.pkl');h=pd.read_pickle(out/'history.pkl');s=pd.read_pickle(out/'schedule.pkl')
 f=f.merge(h[['airing_key','start_s','end_s','channel_id']],on='airing_key',validate='many_to_one')
 template=pd.read_csv(Path(os.environ.get('SWEETTV_DATA','work/data'))/'submission.csv',dtype=str);users=template.user_id.tolist()
 cutoff,end=(A,B) if which=='main' else (B,END) if which=='forward' else (END,int(s.end_s.max()))
 train_all=f[(f.start_s>=START)&(f.end_s<=cutoff)];future_all=f[(f.start_s>=cutoff)&(f.end_s<=end)]
 train=counted(train_all);future=counted(future_all)
 truth,counts,ties=truth_for(future,set(users));eligible=[u for u in users if len(truth[u])==5]
 candidates=set(s.loc[s.start_s.lt(end)&s.end_s.gt(cutoff),'tv_show_id'])
 audit={'window':which,'cutoff':cutoff,'end':end,'users':len(users),'strict5_users':len(eligible),'less5_users':sum(0<len(truth[u])<5 for u in users),'empty_users':sum(not truth[u] for u in users),'ties_across_fifth_users':ties,
        'train_airings':len(train),'validation_airings':len(future),'train_short_qualified':int(train.watch_seconds.lt(300).sum()),'validation_short_qualified':int(future.watch_seconds.lt(300).sum()),
        'excluded_boundary_user_airings':int(((f.start_s<cutoff)&(f.end_s>cutoff)).sum()),'candidates':len(candidates),'train_users':int(train.user_id.nunique())}
 window=Window(train,s,users,candidates,START,cutoff,end)
 return window,truth,eligible,future_all,audit,template,counts

def configurations():
 c=[{'id':'popularity','family':'popularity'},{'id':'repeat','family':'repeat'},{'id':'content','family':'content'},
 {'id':'implicit32','family':'implicit','params':{'factors':32,'regularization':.1}},
 {'id':'implicit32_reg1','family':'implicit','params':{'factors':32,'regularization':1.}},
 {'id':'implicit64','family':'implicit','params':{'factors':64,'regularization':.1}}]
 # Paired changes: target -> bias -> lambda; fixed dim/alpha until explicit variants.
 for name,params in [
  ('count32',{'factors':32,'regularization':.1,'alpha':20,'bias':False}),
  ('count32_bias',{'factors':32,'regularization':.1,'alpha':20,'bias':True}),
  ('count32_bias_reg1',{'factors':32,'regularization':1.,'alpha':20,'bias':True}),
  ('count32_bias_reg10',{'factors':32,'regularization':10.,'alpha':20,'bias':True}),
  ('count64_bias_reg1',{'factors':64,'regularization':1.,'alpha':20,'bias':True}),
  ('count32_bias_unweighted',{'factors':32,'regularization':1.,'alpha':0,'bias':True})]:c.append({'id':name,'family':'count_mf','params':params})
 for name,params in [('ppr85',{'damping':.85,'neighbors':50}),('ppr50',{'damping':.5,'neighbors':50}),('ppr50_k100',{'damping':.5,'neighbors':100})]:c.append({'id':name,'family':'ppr','params':params})
 return c

def run(which,out):
 out=Path(out);t0=time.perf_counter();window,truth,eligible,future_all,audit,template,counts=load_window(out,which)
 save(out/f'{which}_audit.json',audit);save(out/f'private_{which}_truth.json',truth)
 # Compilation is excluded from all timed models, not just later ones.
 checks={'weighted_target_dense_error':verify_target_update(),'implicit_solver':verify_solver()};save(out/'solver_checks.json',checks)
 results=[];cache={};configs=configurations() if which=='main' else json.loads((out/'frozen_selection.json').read_text())['families']
 def process(config):
  cid=config['id'];family=config['family'];t=time.perf_counter();component_time=0.
  if family=='hybrid':
   base=config['component'];scores=rrf(window,cache['repeat'],cache[base],config['weight']);component_time=next(r['seconds'] for r in results if r['id']=='repeat')+next(r['seconds'] for r in results if r['id']==base)
  elif family=='schedule':
   scores=cache['repeat']*window.schedule_ratio()**config['beta'];component_time=next(r['seconds'] for r in results if r['id']=='repeat')
  else:scores=window.individual(config)
  pred=window.rank(scores);elapsed=time.perf_counter()-t+component_time
  row={**config,'map5':evaluate(truth,pred,eligible),'all1260_map5':evaluate(truth,pred,window.cohort),'seconds':elapsed}
  results.append(row);cache[cid]=scores;save(out/f'private_{which}_{cid}.json',pred);save(out/f'{which}_results.json',results)
  print(which,'RESULT',json.dumps(row),flush=True)
 for config in configs:process(config)
 if which=='main':
  # Individual families are selected only by main-window MAP.
  selected=[]
  for family in ['popularity','repeat','content','implicit','count_mf','ppr']:
   selected.append(max([r for r in results if r['family']==family],key=lambda r:r['map5']))
  for component in [x['id'] for x in selected if x['family'] in ['count_mf','ppr']]+['popularity']:
   for weight in [.1,.25,.5]:process({'id':f'hybrid_{component}_{weight}','family':'hybrid','component':component,'weight':weight})
  for beta in [.25,.5,1.]:process({'id':f'schedule_{beta}','family':'schedule','beta':beta})
  selected.extend([max([r for r in results if r['family']==family],key=lambda r:r['map5']) for family in ['hybrid','schedule']])
  winner=max(results,key=lambda r:r['map5'])
  frozen={'winner':winner,'families':selected,'selected_using':'main window only; forward scores not yet computed','frozen_at_utc':pd.Timestamp.now(tz='UTC').isoformat()}
  save(out/'frozen_selection.json',frozen);print('FROZEN',json.dumps(frozen),flush=True)
 # Sensitivity uses frozen chosen-family predictions; no refit or retuning from these results.
 sensitivity=[]
 chosen=json.loads((out/'frozen_selection.json').read_text())['families']
 for policy,tie in [('union80','id_desc'),('union80','hash'),('single80','id_asc'),('sessions80','id_asc')]:
  alternative,_,_=truth_for(counted(future_all,policy),set(window.cohort),tie)
  for config in chosen:
   pred=json.loads((out/f'private_{which}_{config["id"]}.json').read_text())
   sensitivity.append({'model':config['id'],'policy':policy,'tie':tie,'map5_fixed_main_cohort':evaluate(alternative,pred,eligible),'same_users':len(eligible),'altered_truth_users':sum(set(truth[u])!=set(alternative[u]) for u in eligible)})
 save(out/f'{which}_sensitivity.json',sensitivity)
 save(out/f'{which}_runtime.json',{'wall_seconds':time.perf_counter()-t0,'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','numba','threadpoolctl']},'seed':2026,'threads':2})
 print(which,'COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['main','forward']);args=parser.parse_args()
 with threadpool_limits(limits=2):run(args.stage,os.environ.get('SWEETTV_REVISION_OUT','work/revision3'))
