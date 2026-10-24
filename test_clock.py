"""Regression checks using actual clock-anomaly examples; no training/data read."""
import ast
from pathlib import Path
import pandas as pd

source=ast.parse(Path(__file__).with_name('pipeline.py').read_text())
nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ['stamp','display','seconds']]
context={'pd':pd,'CLOCK_AFTER':int(pd.Timestamp('2020-03-29 04:00:00').timestamp())}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'clock-functions','exec'),context)
start=pd.Series(['2020-03-28 22:47:45','2020-03-29 01:00:23','2020-05-18 00:00:00'])
end=pd.Series(['2020-03-30 09:02:48','2020-03-29 04:02:23','2020-05-18 00:10:00'])
delta=context['seconds'](end,'%Y-%m-%d %H:%M:%S')-context['seconds'](start,'%Y-%m-%d %H:%M:%S')
assert delta.tolist()==[119703,7320,600]
for s in start.tolist()+end.tolist(): assert context['display'](context['stamp'](s))==s
print('Clock regressions passed: real durations and display round trips')
