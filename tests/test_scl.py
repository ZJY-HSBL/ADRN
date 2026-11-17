import unittest
import torch
from adrn.scl_tpe import supervised_contrastive_loss
class SupervisedContrastiveTests(unittest.TestCase):
    def test_identical_embeddings_equal_log_candidates(self):
        z=torch.ones(4,3,requires_grad=True);y=torch.tensor([0,0,1,1]);loss=supervised_contrastive_loss(z,y,.2)
        self.assertAlmostEqual(loss.item(),torch.log(torch.tensor(3.)).item(),places=5);loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
    def test_bad_temperature(self):
        with self.assertRaises(ValueError):supervised_contrastive_loss(torch.ones(3,2),torch.tensor([0,0,1]),0)
    def test_end_to_end_small_set(self):
        import numpy as np
        from adrn.scl_tpe import scl_fit
        rng=np.random.default_rng(8);x=rng.normal(size=(120,4)).astype('float32');y=(x[:,0]>0).astype('int64')
        result=scl_fit((x[:80],y[:80]),(x[80:100],y[80:100]),(x[100:],y[100:]),4,seed=3,contrastive_epochs=1,classifier_epochs=1,contrastive_cap=48)
        self.assertTrue(0<=result['test']['pr_auc']<=1);self.assertTrue(0<=result['validation_roc_auc']<=1)
