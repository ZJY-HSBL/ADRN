"""SCL-TPE inspired comparator. Explicit experimental approximation; see README."""
import copy, random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score,average_precision_score
from .train import metrics,threshold_f2


def supervised_contrastive_loss(z,labels,temperature):
    if z.ndim!=2 or len(z)!=len(labels) or temperature<=0:raise ValueError('invalid contrastive input')
    z=F.normalize(z,dim=1);n=len(z);scores=z@z.T/temperature
    eye=torch.eye(n,dtype=torch.bool,device=z.device);log_den=torch.logsumexp(scores.masked_fill(eye,-torch.inf),dim=1)
    positive=(labels[:,None]==labels[None,:]) & ~eye
    count=positive.sum(1);valid=count>0
    if not valid.any():raise ValueError('no positive pairs')
    # Never multiply 0 * -inf; scores are finite even on the diagonal.
    return -(((scores-log_den[:,None])*positive).sum(1)[valid]/count[valid]).mean()


def scl_fit(tr,va,te,numeric_features,seed=0,temperature=.2,contrastive_epochs=2,classifier_epochs=4,contrastive_cap=16000):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    X,y=tr;V,yv=va;T,yt=te
    if not 0<=numeric_features<=X.shape[1]:raise ValueError('invalid number of numeric features')
    encoder=nn.Sequential(nn.Linear(X.shape[1],64),nn.GELU(),nn.Linear(64,16),nn.LayerNorm(16))
    optimizer=torch.optim.Adam(encoder.parameters(),lr=.001);generator=np.random.default_rng(seed)
    for _ in range(contrastive_epochs):
        indices=generator.choice(len(y),min(contrastive_cap,len(y)),replace=False)
        dataset=TensorDataset(torch.from_numpy(X[indices]),torch.from_numpy(y[indices]))
        for xb,yb in DataLoader(dataset,batch_size=128,shuffle=True):
            a=xb.clone();b=xb.clone()
            # Noise is applied only to numeric columns; never jitter one-hot columns.
            a[:,:numeric_features]+=.05*torch.randn_like(a[:,:numeric_features]);b[:,:numeric_features]+=.05*torch.randn_like(b[:,:numeric_features])
            z=encoder(torch.cat([a,b]));loss=supervised_contrastive_loss(z,torch.cat([yb,yb]),temperature)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(encoder.parameters(),5);optimizer.step()
    encoder.eval()
    with torch.no_grad():
        ztrain=encoder(torch.from_numpy(X));zval=encoder(torch.from_numpy(V));ztest=encoder(torch.from_numpy(T))
    classifier=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,16),nn.GELU(),nn.Linear(16,1));optimizer=torch.optim.Adam(classifier.parameters(),lr=.001)
    ds=TensorDataset(ztrain,torch.from_numpy(y.astype('float32')));best=-1;state=None;history=[]
    for epoch in range(classifier_epochs):
        classifier.train()
        for xb,yb in DataLoader(ds,batch_size=512,shuffle=True):
            optimizer.zero_grad();loss=F.binary_cross_entropy_with_logits(classifier(xb).squeeze(1),yb);loss.backward();optimizer.step()
        classifier.eval()
        with torch.no_grad():p=torch.sigmoid(classifier(zval).squeeze(1)).numpy()
        score=average_precision_score(yv,p);history.append(score)
        if score>best:best=score;state=copy.deepcopy(classifier.state_dict())
    classifier.load_state_dict(state);classifier.eval()
    with torch.no_grad():pv=torch.sigmoid(classifier(zval).squeeze(1)).numpy();pt=torch.sigmoid(classifier(ztest).squeeze(1)).numpy()
    threshold=threshold_f2(yv,pv)
    return {'seed':seed,'temperature':temperature,'validation_ap':best,'validation_roc_auc':float(roc_auc_score(yv,pv)),'threshold':threshold,'test':metrics(yt,pt,threshold),'validation_history':history}
