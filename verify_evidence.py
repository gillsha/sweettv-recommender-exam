"""Independent metric/CSV verification on downloaded Kaggle outputs.

Usage: python verify_evidence.py RUN_OUTPUT_DIRECTORY ORIGINAL_TEMPLATE_CSV
No training occurs locally. Outputs an audit JSON, without copying raw inputs.
"""
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np

def verify(folder,template_path):
    folder=Path(folder)
    truth=json.loads((folder/'validation_truth.json').read_text())
    pred=json.loads((folder/'validation_predictions.json').read_text())
    reported=json.loads((folder/'validation_results.json').read_text())
    manifest=json.loads((folder/'run_manifest.json').read_text())
    with open(template_path,newline='') as f: template=list(csv.reader(f))
    users=[r[0] for r in template[1:]]
    independently_computed={}
    for result in reported:
        name=result['method'];scores=[]
        assert set(pred[name])==set(users)
        for u in users:
            rec=pred[name][u];actual=set(truth[u])
            assert len(rec)==len(set(rec))==5 and 0 not in rec
            hits=np.array([item in actual for item in rec],dtype=float)
            ap=float(np.sum(np.cumsum(hits)/np.arange(1,6)*hits)/min(5,len(actual))) if actual else 0.
            scores.append(ap)
        value=float(np.mean(scores));assert abs(value-result['map5'])<1e-12
        independently_computed[name]=value
    with (folder/'submission.csv').open(newline='') as f: submission=list(csv.reader(f))
    assert submission[0]==template[0]
    assert [r[0] for r in submission[1:]]==users
    allowed=set(json.loads((folder/'test_candidates.json').read_text()))-{0}
    for user,ranking in submission[1:]:
        ids=list(map(int,ranking.split()))
        assert len(ids)==len(set(ids))==5 and set(ids)<=allowed
    digest=hashlib.sha256((folder/'submission.csv').read_bytes()).hexdigest()
    assert digest==manifest['checks']['submission_sha256']
    examples=json.loads((folder/'user_examples.json').read_text())
    for e in examples:
        assert set(e['actual_ids'])==set(truth[e['user_id']])
        for name,recs in e['methods'].items():
            assert [r['id'] for r in recs]==pred[name][e['user_id']]
            assert all(r['hit']==(r['id'] in e['actual_ids']) for r in recs)
    checks={'independent_MAP5':independently_computed,'comparison_tolerance':1e-12,
            'exact_template_users_order_header':True,'five_unique_nonzero_candidate_IDs':True,
            'csv_sha256':digest,'user_examples_checked_against_predictions_and_truth':True,
            'local_training_performed':False}
    (folder/'independent_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__': verify(sys.argv[1],sys.argv[2])
