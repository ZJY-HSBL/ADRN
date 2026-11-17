import unittest
import numpy as np
from experiments.cisnn import run
class CISTests(unittest.TestCase):
    def test_fit_and_validation_threshold(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(150,5)).astype('float32');y=np.arange(150)%5==0;y=y.astype(int)
        result=run((x[:100],y[:100]),(x[100:125],y[100:125]),(x[125:],y[125:]),seed=2,epochs=2)
        self.assertEqual(len(result['history']),2);self.assertTrue(0<=result['test']['roc_auc']<=1)
        self.assertTrue(np.isfinite(result['threshold']))
