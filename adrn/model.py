"""Overlap-aware deep probabilistic reconstruction network (paper eq. 2-18)."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def structural_probability(log_support):
    """Equation (11), preserving +1e-12 in the denominator in log space."""
    log_sum=torch.logsumexp(log_support,dim=1)
    log_den=torch.logaddexp(log_sum,torch.full_like(log_sum,math.log(1e-12)))
    return torch.exp(log_support[:,1]-log_den)

class ADRN(nn.Module):
    def __init__(self,input_dim,hidden=64,latent=16,k=4,variant='full'):
        super().__init__()
        if k < 2: raise ValueError('k must be >=2 for within-class redundancy')
        self.k=k;self.latent=input_dim if variant=='no_latent' else latent;self.variant=variant
        latent=self.latent
        self.encoder=nn.Identity() if variant=='no_latent' else nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),nn.Linear(hidden,latent),nn.LayerNorm(latent))
        self.centers=nn.Parameter(torch.randn(2,k,latent)*.1)
        self.raw_scale=nn.Parameter(torch.zeros(2,k))
        self.logits=nn.Parameter(torch.zeros(2,k))
    @property
    def scale(self): return F.softplus(self.raw_scale)+1e-4
    def overlap(self):
        mu=self.centers.reshape(-1,self.latent); var=self.scale.reshape(-1).square()
        dist=(mu[:,None,:]-mu[None,:,:]).square().sum(-1)
        total=var[:,None]+var[None,:]
        log_bc=.5*self.latent*torch.log((2*torch.sqrt(var[:,None]*var[None,:])/total).clamp_min(1e-30))-dist/(4*total)
        return torch.exp(log_bc.clamp(max=0)).reshape(2,self.k,2,self.k)
    def forward(self,x,return_details=False):
        z=self.encoder(x);scale=self.scale;logpi=F.log_softmax(self.logits,dim=1)
        d2=(z[:,None,None,:]-self.centers[None,:,:,:]).square().sum(-1)
        affinity=-d2/(2*scale.square()[None,:,:])
        logq=logpi[None,:,:]+affinity-self.latent*torch.log(scale)[None,:,:]-.5*self.latent*math.log(2*math.pi)
        p_obs=torch.softmax(torch.logsumexp(logq,dim=2),dim=1)[:,1]
        bc=self.overlap(); cross=torch.stack([bc[0,:,1,:].max(-1).values,bc[1,:,0,:].max(-1).values])
        eye=torch.eye(self.k,dtype=torch.bool,device=x.device)
        within=torch.stack([bc[0,:,0,:].masked_fill(eye,0).max(-1).values,bc[1,:,1,:].masked_fill(eye,0).max(-1).values])
        reliability=((1-cross).clamp(0,1)*(1-within).clamp(0,1)).clamp_min(1e-6)
        if self.variant=='no_reliability': reliability=torch.ones_like(reliability)
        # The denominator includes the paper's 1e-12 stabilizer, in log space.
        log_support=torch.logsumexp(logpi[None,:,:]+affinity+reliability.log()[None,:,:],dim=2)
        p_struct=structural_probability(log_support)
        difference=p_struct-p_obs
        deficit=difference if self.variant=='signed_deficit' else F.relu(difference)
        responsibilities=torch.softmax(logpi[None,:,:]+affinity,dim=2)
        overlap=torch.einsum('bi,ij,bj->b',responsibilities[:,1,:],bc[1,:,0,:],responsibilities[:,0,:]).clamp(0,1)
        gate=torch.ones_like(overlap) if self.variant=='no_gate' else 1-overlap
        p=(p_obs+gate*deficit).clamp(1e-7,1-1e-7)
        if return_details:return {'probability':p,'observed':p_obs,'structure':p_struct,'overlap':overlap,'gate':gate,'deficit':deficit,'reliability':reliability}
        return p
