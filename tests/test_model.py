import unittest
import torch
from adrn.model import ADRN
from adrn.train import threshold_f2
import numpy as np
class Tests(unittest.TestCase):
    def test_probabilities_gradients(self):
        torch.manual_seed(3);model=ADRN(5,8,4,3);x=torch.randn(9,5);out=model(x,True)
        for key in ['probability','observed','structure','gate','overlap']:
            self.assertTrue(torch.isfinite(out[key]).all());self.assertTrue(((out[key]>=0)&(out[key]<=1)).all())
        self.assertTrue((out['probability']>=out['observed']-1e-6).all())
        self.assertTrue((out['probability']<=torch.maximum(out['observed'],out['structure'])+1e-6).all())
        loss=torch.nn.functional.binary_cross_entropy(out['probability'],torch.rand(9));loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    def test_structural_probability_matches_equation(self):
        from adrn.model import structural_probability
        supports=torch.tensor([[2e-12,1e-12],[1e-8,2e-8],[1e-40,1e-40]],dtype=torch.float64)
        actual=structural_probability(supports.log())
        expected=supports[:,1]/(supports.sum(1)+1e-12)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-30)
        self.assertLess(actual[-1],1e-20)
    def test_overlap(self):
        model=ADRN(3,4,2,2);bc=model.overlap().reshape(4,4)
        self.assertTrue(torch.allclose(bc,bc.T));self.assertTrue(torch.allclose(torch.diagonal(bc),torch.ones(4)))
    def test_ablation_switches(self):
        torch.manual_seed(5)
        for variant in ('no_reliability','signed_deficit','no_gate','no_latent'):
            model=ADRN(5,8,4,3,variant=variant);x=torch.randn(10,5);v=model(x,True)
            self.assertTrue(torch.isfinite(v['probability']).all())
            if variant=='no_reliability': self.assertTrue(torch.allclose(v['reliability'],torch.ones_like(v['reliability'])))
            if variant=='no_gate': self.assertTrue(torch.allclose(v['gate'],torch.ones_like(v['gate'])))
            if variant=='no_latent': self.assertEqual(model.latent,5)
    def test_threshold(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.2,.8,.9]);self.assertEqual(threshold_f2(y,p),.8)
if __name__=='__main__':unittest.main()

class StatisticsTests(unittest.TestCase):
    def test_bh_adjustment(self):
        from adrn.statistics import bh
        np.testing.assert_allclose(bh([.01,.04,.03,.002]),[.02,.04,.04,.008])
    def test_welch_validates(self):
        from adrn.statistics import welch
        with self.assertRaises(ValueError):welch([1],[2,3])
