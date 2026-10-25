"""Post-freeze diagnostics; never changes selected configurations."""
import json,sys,re,os
from pathlib import Path
import numpy as np
import pandas as pd
from revised.experiments import load_window,evaluate
from revised.data import save
from core import ap_at_k

def run(out):
 out=Path(out);selection=json.loads((out/'frozen_selection.json').read_text());winner=selection['winner']['id']
 bootstrap=[];rng=np.random.default_rng(2026)
 for which in ['main','forward']:
  truth=json.loads((out/f'private_{which}_truth.json').read_text());users=[u for u in truth if len(truth[u])==5]
  a=json.loads((out/f'private_{which}_{winner}.json').read_text());b=json.loads((out/f'private_{which}_repeat.json').read_text())
  diff=np.array([ap_at_k(truth[u],a[u])-ap_at_k(truth[u],b[u]) for u in users]);means=[]
  for _ in range(2000):means.append(float(rng.choice(diff,len(diff),replace=True).mean()))
  bootstrap.append({'window':which,'users':len(users),'mean_delta':float(diff.mean()),'percentile95':np.quantile(means,[.025,.975]).tolist(),'paired_resamples':2000,'seed':2026,'caution':'conditional on fixed cohort and target assumptions; main is selection-biased'})
 save(out/'robustness.json',bootstrap)
 w,t,eligible,_,audit,_,freq=load_window(out,'main')
 if (out/'example_users.json').exists():chosen=json.loads((out/'example_users.json').read_text())
 else:
  pred=json.loads((out/f'private_main_{winner}.json').read_text());ordered=sorted((ap_at_k(t[u],pred[u]),u) for u in eligible)
  chosen=[ordered[len(ordered)//4][1],ordered[3*len(ordered)//4][1]]
 metadata=w.schedule.drop_duplicates('tv_show_id').set_index('tv_show_id');examples=[]
 def item(i):
  row=metadata.loc[i];raw=str(row.tv_show_title).strip();match=re.search(r"['\"](.+)['\"]",raw)
  title=(match.group(1) if match else raw.rstrip(' .')).replace("'",'').strip()
  return {'id':int(i),'title':title,'category':str(row.tv_show_category)}
 for u in chosen:
  assert u in eligible
  train=w.train[w.train.user_id==u];n=train.groupby('tv_show_id').size().to_dict()
  truth=[dict(item(i),qualified_airings=freq[u][i]) for i in t[u]];rows=[]
  for model in selection['families']:
   pred=json.loads((out/f'private_main_{model["id"]}.json').read_text())[u]
   rows.append({'model':model['id'],'ap5':ap_at_k(t[u],pred),'items':[dict(item(i),hit=i in t[u],train_airings=int(n.get(i,0))) for i in pred]})
  examples.append({'user_id':u,'selection':'lower quartile of final AP, then ID' if u==chosen[0] else 'retained earlier illustrative user with five valid target items','train_qualified_airings':len(train),'train_unique_programmes':len(n),'truth':truth,'methods':rows})
 save(out/'private_examples.json',examples)
 short=json.loads((out/'private_short_examples.json').read_text())[0]
 # Independent sweep over segment endpoints, not the shared union implementation.
 points=sorted(set(v for x in short['intervals'] for v in x[:2]));covered=sum(b-a for a,b in zip(points,points[1:]) if any(x[0]<=a and x[1]>=b for x in short['intervals']))
 assert covered==short['union_seconds'] and 5*covered>=4*short['duration'] and covered<300
 f=pd.read_pickle(out/'all_positive.pkl')
 save(out/'reconstruction_checks.json',{'independent_short_example_union_seconds':covered,'short_example_programme_duration':short['duration'],'qualified80_below300':True,'union_le_programme_duration':bool(f.watch_seconds.le(f.programme_seconds).all()),'positive_overlap_seconds_quantiles':f.watch_seconds.quantile([0,.25,.5,.75,1]).to_dict(),'programme_duration_quantiles':f.programme_seconds.quantile([0,.25,.5,.75,1]).to_dict()})
 print('Post-freeze checks complete')

if __name__=='__main__':run(os.environ.get('SWEETTV_REVISION_OUT','work/revision3'))
