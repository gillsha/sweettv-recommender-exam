"""Reconstruct all positive overlaps from raw inputs, with no 300-second filter."""
import os,json,time,sys,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ScheduleIndex,union_length

CLOCK=int(pd.Timestamp('2020-03-29 04:00:00').timestamp())
def stamp(s):
 v=int(pd.Timestamp(s).timestamp());return v-3600 if v>=CLOCK else v
def seconds(s,fmt):
 a=pd.to_datetime(s,format=fmt,errors='raise').astype('int64')//10**9
 return a-(a>=CLOCK).astype('int64')*3600
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def read_schedule(folder):
 frames=[]
 for name in ['export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv']:
  p=folder/name
  if not p.exists():p=folder/(name+'.zip')
  f=pd.read_csv(p,low_memory=False);f['start_s']=seconds(f.start_time,'%d.%m.%Y %H:%M:%S');f['end_s']=f.start_s+f.duration
  f=f[f.tv_show_id.ne(0)&f.duration.gt(0)].copy();f['source']=name;frames.append(f)
 return pd.concat(frames,ignore_index=True).drop_duplicates(['channel_id','start_s','end_s','tv_show_id']).sort_values('start_s').reset_index(drop=True)
def prepare(folder,out):
 t=time.perf_counter();folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 schedule=read_schedule(folder);schedule.to_pickle(out/'schedule.pkl')
 end=stamp('2020-07-27');history=schedule[schedule.start_s.lt(end)].reset_index(drop=True);history['airing_key']=np.arange(len(history))
 index=ScheduleIndex(history[['channel_id','start_s','end_s','tv_show_id','airing_key']].itertuples(index=False,name=None))
 segments=defaultdict(list);audit={'view_rows':0,'duration_mismatch':0,'positive_overlap_rows':0,'labelled_overlap_seconds':0,'total_view_seconds':0}
 input=folder/'dataset11-30.csv'
 if not input.exists():input=folder/'dataset11-30.csv.zip'
 for f in pd.read_csv(input,dtype={'user_id':str},chunksize=250000):
  starts=seconds(f.start_time,'%Y-%m-%d %H:%M:%S').to_numpy();ends=seconds(f.stop_time,'%Y-%m-%d %H:%M:%S').to_numpy()
  audit['view_rows']+=len(f);audit['duration_mismatch']+=int(np.sum(ends-starts!=f.duraton.to_numpy()))
  for u,c,a,b in zip(f.user_id,f.vsetv_id,starts,ends):
   a=int(a);b=int(b);assert b>a
   audit['total_view_seconds']+=b-a
   matches=index.overlaps(int(c),a,min(b,end))
   audit['positive_overlap_rows']+=bool(matches)
   audit['labelled_overlap_seconds']+=union_length([(m[2],m[3]) for m in matches])
   for item,key,left,right,length in matches:segments[(u,key)].append((left,right,a,b))
  print('RAW JOIN',audit['view_rows'],flush=True)
 assert audit['duration_mismatch']==0
 records=[];duplicate_segments=0;examples=[]
 for (u,key),values in segments.items():
  unique=set(values);duplicate_segments+=len(values)-len(unique)
  row=history.iloc[key];length=int(row.duration)
  ints=[(x[0],x[1]) for x in unique];watched=union_length(ints);max_single=max(b-a for a,b in ints)
  session_count=sum(5*(b-a)>=4*length for a,b in ints)
  records.append((u,int(row.tv_show_id),key,watched,length,max_single,session_count,len(unique)))
  if len(examples)<10 and watched<300 and 5*watched>=4*length:
   examples.append({'user_id':u,'tv_show_id':int(row.tv_show_id),'channel_id':int(row.channel_id),'airing_start':str(row.start_time),'duration':length,'union_seconds':watched,'intervals':sorted(unique)})
 interactions=pd.DataFrame(records,columns=['user_id','tv_show_id','airing_key','watch_seconds','programme_seconds','max_single_seconds','session_count80','unique_sessions'])
 assert interactions.watch_seconds.le(interactions.programme_seconds).all()
 interactions.to_pickle(out/'all_positive.pkl');history.to_pickle(out/'history.pkl')
 qualified=interactions[5*interactions.watch_seconds>=4*interactions.programme_seconds]
 audit.update(positive_user_airings=len(interactions),qualified80_user_airings=len(qualified),qualified80_under300=int(qualified.watch_seconds.lt(300).sum()),deduplicated_identical_sessions=duplicate_segments,
              union_vs_single_different=int(((5*interactions.watch_seconds>=4*interactions.programme_seconds)&(5*interactions.max_single_seconds<4*interactions.programme_seconds)).sum()),
              join_seconds=time.perf_counter()-t,cache_source='fresh raw CSV join; no previous interaction cache read',threshold='5*union_seconds >= 4*programme_seconds')
 save(out/'data_audit.json',audit);save(out/'private_short_examples.json',examples)
 print('PREPARED',json.dumps(audit),flush=True)
if __name__=='__main__':prepare(os.environ.get('SWEETTV_DATA','work/data'),os.environ.get('SWEETTV_REVISION_OUT','work/revision3'))
