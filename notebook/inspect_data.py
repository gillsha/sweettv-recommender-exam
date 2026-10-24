"""Run only inside Kaggle; audit raw competition inputs without exporting them."""
import csv, json, os, platform, time
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

started=time.perf_counter()
root=Path('/kaggle/input')
audit={'python':platform.python_version(),'cpu':platform.processor(),'cpu_count':os.cpu_count(),
       'versions':{'pandas':pd.__version__,'numpy':np.__version__},'files':{}}
for p in sorted(root.rglob('*.csv')):
    with p.open(encoding='utf-8-sig') as f:
        sample=f.read(20000)
    sep=csv.Sniffer().sniff(sample,delimiters=',;\t').delimiter
    frame=pd.read_csv(p,sep=sep,low_memory=False)
    a={'path':str(p),'separator':sep,'shape':list(frame.shape),
       'dtypes':{c:str(t) for c,t in frame.dtypes.items()},
       'nulls':{c:int(n) for c,n in frame.isna().sum().items()},
       'head':frame.head(4).fillna('').to_dict('records')}
    for c in frame:
        if c in ['start_time','stop_time','duration','duraton','tv_show_id','channel_id','vsetv_id','user_id']:
            a[c]={'min':str(frame[c].min()),'max':str(frame[c].max()),'distinct':int(frame[c].nunique())}
            if pd.api.types.is_numeric_dtype(frame[c]):
                a[c]['quantiles']=frame[c].quantile([0,.01,.1,.25,.5,.75,.9,.95,.99,1]).to_dict()
    if 'tv_show_id' in frame and 'channel_id' in frame:
        labelled=frame.tv_show_id.fillna(0).astype(int).ne(0)
        a['labelled_rows']=int(labelled.sum())
        a['labelled_sample']=frame[labelled].head(8).fillna('').to_dict('records')
        a['category_counts']=frame.loc[labelled,'tv_show_category'].value_counts(dropna=False).to_dict()
    audit['files'][p.name]=a
    Path('/kaggle/working/data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str))
    print(p.name,a['shape'],a['dtypes'],flush=True)
    del frame
audit['runtime_seconds']=time.perf_counter()-started
Path('/kaggle/working/data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str))
print(json.dumps(audit,ensure_ascii=False,default=str))
