"""Independent numpy checks of the paper's normal-range equations."""
import unittest
import numpy as np
import torch
from scipy.special import softmax,logsumexp
from adrn.model import ADRN

class PaperEquations(unittest.TestCase):
    def test_full_forward_matches_numpy(self):
        torch.manual_seed(12);net=ADRN(4,7,3,3).double();x=torch.randn(6,4,dtype=torch.float64)
        with torch.no_grad():
            result=net(x,True);z=net.encoder(x).numpy();mu=net.centers.numpy();scale=net.scale.numpy();mix=softmax(net.logits.numpy(),axis=1)
        d=((z[:,None,None,:]-mu[None,:,:,:])**2).sum(-1);log_a=-d/(2*scale[None,:,:]**2)
        log_g=np.log(mix)[None,:,:]+log_a-3*np.log(scale)[None,:,:]-1.5*np.log(2*np.pi)
        obs=softmax(logsumexp(log_g,axis=2),axis=1)[:,1]
        v=scale**2;dv=((mu[:,:,None,None,:]-mu[None,None,:,:,:])**2).sum(-1);total=v[:,:,None,None]+v[None,None,:,:]
        bc=np.exp(1.5*np.log(2*np.sqrt(v[:,:,None,None]*v[None,None,:,:])/total)-dv/(4*total))
        rel=np.zeros((2,3))
        for c in range(2):
            for k in range(3):rel[c,k]=max(1e-6,(1-bc[c,k,1-c,:].max())*(1-max(bc[c,k,c,j] for j in range(3) if j!=k)))
        support=(np.exp(log_a)*mix[None,:,:]*rel[None,:,:]).sum(axis=2)
        struct=support[:,1]/(support.sum(axis=1)+1e-12)
        responsibility=softmax(np.log(mix)[None,:,:]+log_a,axis=2)
        overlap=np.einsum('bi,ij,bj->b',responsibility[:,1,:],bc[1,:,0,:],responsibility[:,0,:])
        pred=np.clip(obs+(1-overlap)*np.maximum(struct-obs,0),1e-7,1-1e-7)
        np.testing.assert_allclose(result['observed'].numpy(),obs,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(result['structure'].numpy(),struct,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(result['overlap'].numpy(),overlap,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(result['probability'].numpy(),pred,rtol=1e-12,atol=1e-12)
