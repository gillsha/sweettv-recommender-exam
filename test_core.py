"""Synthetic contract checks; these are AGENT checks, not student verification."""
import random
import unittest
from core import ScheduleIndex,aggregate_watches,ap_at_k,map_at_k,union_length

class CoreTests(unittest.TestCase):
    def test_ap_duplicate_empty_and_cutoff(self):
        self.assertAlmostEqual(ap_at_k([2,4],[2,2,9,4,5]),.75)
        self.assertEqual(ap_at_k([],[1,2,3,4,5]),0)
        self.assertEqual(ap_at_k([6],[1,2,3,4,5,6]),0)
        self.assertEqual(ap_at_k(range(1,11),list(range(1,6))),1)

    def test_fixed_cohort_includes_empty_truth(self):
        self.assertEqual(map_at_k({1:[2]},{1:[2],2:[3]},[1,2]),.5)

    def test_interval_boundaries_channels_unlabelled(self):
        idx=ScheduleIndex([(1,0,100,10,'a'),(1,100,200,20,'b'),
                           (1,0,300,0,'unlabelled'),(2,0,200,30,'c')])
        self.assertEqual(idx.overlaps(1,50,150),[(10,'a',50,100,100),(20,'b',100,150,100)])
        self.assertEqual(idx.overlaps(1,200,250),[])
        self.assertEqual(idx.overlaps(9,0,100),[])

    def test_union_and_temporal_clipping(self):
        idx=ScheduleIndex([(1,0,200,10,'a')])
        views=[(1,1,20,130),(1,1,40,150)]
        train=aggregate_watches(idx,views,0,100,60)
        self.assertEqual(train[0]['watch_seconds'],80)
        self.assertEqual(aggregate_watches(idx,views,100,200,60),[])
        self.assertEqual(union_length([(1,3),(2,4),(4,5),(10,12)]),6)

    def test_index_matches_brute_force_with_overlapping_schedule(self):
        rng=random.Random(2026)
        rows=[]
        for i in range(300):
            s=rng.randrange(1000)
            rows.append((rng.randrange(3),s,s+rng.randrange(1,500),i+1,str(i)))
        idx=ScheduleIndex(rows)
        for _ in range(300):
            c,s=rng.randrange(4),rng.randrange(1500)
            e=s+rng.randrange(1,400)
            expected=sorted((item,key,max(s,a),min(e,b),b-a)
                            for ch,a,b,item,key in rows if ch==c and max(s,a)<min(e,b))
            self.assertEqual(sorted(idx.overlaps(c,s,e)),expected)

if __name__=='__main__':
    unittest.main()
