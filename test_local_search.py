"""Small checks for decay, mixing and deterministic fallback, no real data."""
import unittest
import numpy as np
import pandas as pd
from local_search import Window,rank,independent_ap

class LocalSearchTests(unittest.TestCase):
    def test_decay_and_cold_fallback(self):
        history=pd.DataFrame({'start_s':[0,86400,86400,86400,86400,86400]})
        frame=pd.DataFrame({'user_id':['a','a','b','b','b','b'],
            'tv_show_id':[1,2,2,3,4,5],'airing_key':[0,1,2,3,4,5]})
        w=Window(frame,history,['a','cold'],list(range(1,7)),86400)
        frequency=w.frequency(1)
        self.assertEqual(frequency[0,0],.5)
        self.assertEqual(frequency[0,1],1.)
        s=w.combine(frequency,1.,.5)
        self.assertAlmostEqual(s[0,0],.5*.5+.5*.5)
        self.assertAlmostEqual(s[0,1],1.)
        self.assertEqual(rank(s,w.ids,w.pop)[1],[2,1,3,4,5])
        self.assertEqual(w.ids[-1],6) # unseen candidates remain in the pool
    def test_metric_and_tie_order(self):
        self.assertAlmostEqual(independent_ap({1,2,8,9,10,11},[1,2,3,4,5]),.4)
        ids=np.array([1,2,3,4,5,6]);pop=np.array([1,2,2,1,1,1])
        self.assertEqual(rank(np.ones((1,6)),ids,pop)[0],[2,3,1,4,5])

if __name__=='__main__':unittest.main()
