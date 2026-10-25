import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import union_length,ap_at_k
from revised.experiments import counted,truth_for
from revised.models import verify_target_update

class TargetTests(unittest.TestCase):
 def test_short_airing_and_exact_boundary(self):
  f=pd.DataFrame({'watch_seconds':[96,95,300],'programme_seconds':[120,120,600]})
  self.assertEqual(counted(f).index.tolist(),[0])
 def test_union_duplicate_and_split_sessions(self):
  self.assertEqual(union_length([(0,60),(40,96),(0,60)]),96)
 def test_top_five_ties_and_fewer(self):
  f=pd.DataFrame({'user_id':['u']*7+['short'],'tv_show_id':[1,1,2,3,4,5,6,9]})
  truth,_,ties=truth_for(f,{'u','short','empty'})
  self.assertEqual(truth['u'],[1,2,3,4,5]);self.assertEqual(truth['short'],[9]);self.assertEqual(truth['empty'],[]);self.assertEqual(ties,1)
  alt,_,_=truth_for(f,{'u'},'id_desc');self.assertEqual(alt['u'],[1,6,5,4,3])
 def test_target_mf_dense_reference(self):self.assertLess(verify_target_update(),1e-10)
 def test_ap_five_relevance(self):self.assertAlmostEqual(ap_at_k([1,2,3,4,5],[1,8,2,9,3]),(1+2/3+3/5)/5)
if __name__=='__main__':unittest.main()
