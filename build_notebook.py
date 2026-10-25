"""Bundle audited modules; execute corrected target pipeline on Kaggle CPU."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
def cell(kind,text):
 d={'cell_type':kind,'metadata':{},'source':text.splitlines(True)}
 if kind=='code':d.update(execution_count=None,outputs=[])
 return d
cells=[cell('markdown','# sweet.tv — corrected 80% / top-five target\nElizaveta Gill. Code and checks assisted by Codex. CPU only; no credentials or viewing records embedded. Main window selects configurations; forward window is opened only after freezing. Earlier 300-second/all-unique results are invalid as a Kaggle proxy and preserved in Git history.'),cell('code','import os,sys,subprocess\nfrom pathlib import Path\nos.environ["OPENBLAS_NUM_THREADS"]="2"\nos.environ["OMP_NUM_THREADS"]="2"\nos.chdir("/kaggle/working")\nPath("revised").mkdir(exist_ok=True)\n')]
for name in ['core.py','als.py','models.py','test_core.py','revised/data.py','revised/models.py','revised/experiments.py','revised/final.py','revised/test_target.py']:
 cells.append(cell('code','Path('+repr(name)+').write_text('+repr((root/name).read_text())+', encoding="utf-8")\n'))
cells.append(cell('code','inputs=list(Path("/kaggle/input").rglob("dataset11-30.csv"))\nif not inputs: inputs=list(Path("/kaggle/input").rglob("dataset11-30.csv.zip"))\nassert len(inputs)==1\nos.environ["SWEETTV_DATA"]=str(inputs[0].parent)\nos.environ["SWEETTV_REVISION_OUT"]="/kaggle/working/results"\nos.environ["PYTHONPATH"]="/kaggle/working"\nprint("Input folder:", inputs[0].parent)\nsubprocess.run([sys.executable,"revised/test_target.py"],check=True)\n'))
cells.append(cell('markdown','## Raw reconstruction and main-window selection\nFresh positive overlaps, unioned per user/airing; >=80% coverage, then five most frequent programme IDs. Unspecified tie/session conventions have sensitivity checks.'))
cells.append(cell('code','subprocess.run([sys.executable,"-m","revised.data"],check=True)\nsubprocess.run([sys.executable,"-m","revised.experiments","main"],check=True)\n'))
cells.append(cell('markdown','## Forward check and full-history refit\nForward results do not change the frozen choice. The CSV is checked against the template; submission is a separate authorized step.'))
cells.append(cell('code','subprocess.run([sys.executable,"-m","revised.experiments","forward"],check=True)\nsubprocess.run([sys.executable,"-m","revised.final"],check=True)\nimport json\nprint(json.loads(Path("results/final_manifest.json").read_text()))\n# Raw caches need not be downloaded; viewing histories remain private.\nfor p in Path("results").glob("*.pkl"):p.unlink()\n'))
for i,c in enumerate(cells):c['id']=f'corrected-{i:02d}'
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
(root/'sweettv.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
metadata={'id':'liugrowin/sweettv-recommender-exam','title':'Sweettv Recommender Exam','code_file':'sweettv.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,'dataset_sources':[],'competition_sources':['sweettv-tv-program-recommender'],'kernel_sources':[]}
(root/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Notebook SHA256',hashlib.sha256((root/'sweettv.ipynb').read_bytes()).hexdigest())
