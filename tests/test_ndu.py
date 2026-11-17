import unittest
import numpy as np
from adrn.ndu import score_segments,select_indices,build_geometry
class NDUTests(unittest.TestCase):
    def test_segments_and_label_preservation(self):
        y=np.array([0]*50+[1]*10);idx=np.arange(50);scores=np.linspace(.1,1,50);keep,r=select_indices(y,idx,scores,seed=1)
        self.assertEqual(len(score_segments(scores)),5);self.assertTrue(np.all(np.isin(np.arange(50,60),keep)));self.assertEqual(len(y)-len(keep),sum(i['removed'] for i in r))
        np.testing.assert_array_equal(keep,select_indices(y,idx,scores,seed=1)[0])
    def test_no_candidates(self):
        y=np.array([0,0,1]);keep,rates=select_indices(y,np.array([],dtype=int),np.array([]))
        np.testing.assert_array_equal(keep,[0,1,2]);self.assertEqual(rates,[])
    def test_small_geometry(self):
        rng=np.random.default_rng(2);X=rng.normal(size=(100,3)).astype('float32');y=np.array([0]*80+[1]*20)
        g=build_geometry(X,y);self.assertGreater(g['k'],0);self.assertEqual(len(g['indices']),len(g['scores']));self.assertTrue(np.all((g['scores']>=0)&(g['scores']<=1)))
