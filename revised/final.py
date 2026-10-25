"""Refit only the main-window frozen winner and validate the exact CSV."""
import os,json,time,hashlib
from pathlib import Path
from threadpoolctl import threadpool_limits
from revised.data import save
from revised.experiments import load_window
from revised.models import rrf
from core import validate_submission

def final(out):
 out=Path(out);frozen=json.loads((out/'frozen_selection.json').read_text())
 window,_,_,_,audit,template,_=load_window(out,'final')
 configs={r['id']:r for r in frozen['families']}
 def scores(c):
  if c['family']=='hybrid':return rrf(window,scores(configs['repeat']),scores(configs[c['component']]),c['weight'])
  if c['family']=='schedule':return window.repeat*window.schedule_ratio()**c['beta']
  return window.individual(c)
 start=time.perf_counter();pred=window.rank(scores(frozen['winner']));elapsed=time.perf_counter()-start
 template.iloc[:,1]=[' '.join(map(str,pred[u])) for u in template.iloc[:,0]]
 path=out/'submission_corrected.csv';template.to_csv(path,index=False)
 contract=validate_submission(Path(os.environ.get('SWEETTV_DATA','work/data'))/'submission.csv',path,window.candidates)
 save(out/'final_manifest.json',{'selected':frozen['winner'],'fit_score_rank_seconds':elapsed,'training_airings':len(window.train),
   'training_users':len(window.users),'training_items':len(window.items),'candidates':len(window.candidates),
   'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'contract':contract,'test_end_seconds':window.end})
 print('FINAL',json.dumps(json.loads((out/'final_manifest.json').read_text())),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=2):final(os.environ.get('SWEETTV_REVISION_OUT','work/revision3'))
