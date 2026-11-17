import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from adrn.data import load

class SplitTests(unittest.TestCase):
    def test_imbalance_keeps_holdout_and_checks_request(self):
        with tempfile.TemporaryDirectory() as d:
            n=500;pd.DataFrame({'a':np.arange(n),'duration':np.arange(n),'type':['a','b']*(n//2),'y':['yes' if i%5==0 else 'no' for i in range(n)]}).to_csv(Path(d)/'bank-full.csv',sep=';',index=False)
            tr,val,test,_=load('bank',root=d);tr2,val2,test2,_=load('bank',root=d,target_ir=10)
            np.testing.assert_array_equal(val[1],val2[1]);np.testing.assert_array_equal(test[1],test2[1])
            self.assertEqual(int((tr2[1]==1).sum()),int((tr2[1]==0).sum())//10)
            with self.assertRaises(ValueError):load('bank',root=d,target_ir=1)
if __name__=='__main__':unittest.main()
