"""Local preprocessing equivalent to Kaggle notebook v7; creates private caches."""
import os,sys,json,time,platform,hashlib,importlib.metadata,resource
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
from core import ScheduleIndex,union_length,ap_at_k,map_at_k,validate_submission
SEED=2026

START=time.perf_counter()
OUT=Path(os.environ.get('SWEETTV_OUTPUT','work/local150')); OUT.mkdir(exist_ok=True,parents=True)
INPUT=Path(os.environ.get('SWEETTV_DATA','work/data'))
FILES={p.name.removesuffix('.zip'):p for p in INPUT.glob('*.csv*')}
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
          'cpu_model':platform.processor() or platform.machine(),
          'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas']},
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


# Private caches for the explicitly requested local search; never publish.
for name,frame in [('train',train),('validation',valid),('full',full_interactions),('history',history)]:
    frame.to_pickle(OUT/(name+'.pkl'))
save('validation_candidates.json',sorted(valid_candidates))
save('test_candidates.json',sorted(test_candidates))
save('prepare_manifest.json',manifest)
print('LOCAL PREPARATION COMPLETE',flush=True)
