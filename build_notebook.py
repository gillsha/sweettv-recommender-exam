"""Bundle checked source modules into a self-contained Kaggle Notebook."""
from pathlib import Path
import json,hashlib

root=Path(__file__).resolve().parent
def code(s):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':s.splitlines(True)}
def md(s):
    return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
cells=[md('# sweet.tv: temporal recommendation experiment\nElizaveta Gill. Code prepared with Codex; student verification is recorded separately.\nCPU only. No credentials, raw inputs or test viewing behaviour are embedded.'),
       code('import os\nos.environ["OPENBLAS_NUM_THREADS"]="1"\nos.environ["OMP_NUM_THREADS"]="2"\n')]
for name in ['core.py','als.py','models.py','test_core.py']:
    source=(root/name).read_text()
    cells.append(code('from pathlib import Path\nPath('+repr(name)+').write_text('+repr(source)+')\n'))
cells.append(code('import unittest\nimport test_core\nresult=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_core))\nassert result.wasSuccessful()\n'))
cells.append(code('from als import verify_solver\nprint("ALS solver check:",verify_solver())\n'))
cells.append(md('## Audit, time-aware interval join, validation, model selection, refit\nAll learned transformations use historical training data. No private Kaggle score enters model selection.'))
cells.append(code((root/'pipeline.py').read_text()))
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
for i,c in enumerate(cells): c['id']=f'cell-{i:02d}'
(root/'sweettv.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
metadata={'id':'liugrowin/sweettv-recommender-exam','title':'Sweettv Recommender Exam','code_file':'sweettv.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,'dataset_sources':[],'competition_sources':['sweettv-tv-program-recommender'],'kernel_sources':[]}
(root/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Notebook built; SHA256',hashlib.sha256((root/'sweettv.ipynb').read_bytes()).hexdigest())
