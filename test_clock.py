"""Regression checks using actual clock-anomaly examples; no training/data read."""
import unittest
import pandas as pd
from revised.data import stamp,seconds
class ClockRegression(unittest.TestCase):
 def test_actual_durations_and_scalar_vector_agreement(self):
  start=pd.Series(['2020-03-28 22:47:45','2020-03-29 01:00:23','2020-05-18 00:00:00'])
  end=pd.Series(['2020-03-30 09:02:48','2020-03-29 04:02:23','2020-05-18 00:10:00'])
  delta=seconds(end,'%Y-%m-%d %H:%M:%S')-seconds(start,'%Y-%m-%d %H:%M:%S')
  self.assertEqual(delta.tolist(),[119703,7320,600])
  self.assertEqual(seconds(start,'%Y-%m-%d %H:%M:%S').tolist(),[stamp(s) for s in start])
if __name__=='__main__':unittest.main()
