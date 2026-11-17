"""Training flow: supervised pretraining, KMeans bases, joint fitting."""
import copy,random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_auc_score,precision_score,recall_score,f1_score,fbeta_score
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from .model import ADRN

def metrics(y,p,threshold):
    pred=p>=threshold
    return {'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'f2':fbeta_score(y,pred,beta=2,zero_division=0),'roc_auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p)}

def threshold_f2(y,p):
    precision,recall,thresholds=precision_recall_curve(y,p)
    f2=5*precision[:-1]*recall[:-1]/np.maximum(4*precision[:-1]+recall[:-1],1e-15)
    return float(thresholds[np.argmax(f2)]) if len(thresholds) else .5

def init_bases(model,x,y,seed):
    model.eval()
    with torch.no_grad(): z=model.encoder(torch.from_numpy(x)).numpy()
    for c in (0,1):
        zc=z[y==c];km=KMeans(n_clusters=model.k,random_state=seed,n_init=5).fit(zc)
        for j in range(model.k):
            group=zc[km.labels_==j];mean=km.cluster_centers_[j]; mse=np.mean(np.sum((group-mean)**2,axis=1)/z.shape[1]);scale=max(np.sqrt(max(mse,1e-6))-1e-4,1e-6)
            with torch.no_grad():
                model.centers[c,j].copy_(torch.from_numpy(mean));model.raw_scale[c,j]=float(np.log(np.expm1(scale)) if scale<20 else scale);model.logits[c,j]=float(np.log(len(group)/len(zc)))

def train(train_set,val_set,test_set,seed=0,epochs=12,pretrain_epochs=4,k=4,hidden=64,latent=16,batch_size=512,lr=.001,beta=.5,variant='full'):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    X,y=train_set;V,yv=val_set;T,yt=test_set
    model=ADRN(X.shape[1],hidden,latent,k,variant);head=nn.Linear(model.latent,1)
    counts=np.bincount(y,minlength=2);weights=torch.from_numpy((counts[y]**(-beta)).astype('float64'))
    ds=TensorDataset(torch.from_numpy(X),torch.from_numpy(y.astype('float32')))
    def loader():return DataLoader(ds,batch_size=batch_size,sampler=WeightedRandomSampler(weights,len(y),replacement=True))
    opt=torch.optim.Adam(list(model.encoder.parameters())+list(head.parameters()),lr=lr)
    best=-1;state=None
    history=[]
    for epoch in range(pretrain_epochs if variant!='no_latent' else 0):
        model.train();head.train()
        for xb,yb in loader():
            opt.zero_grad();loss=F.binary_cross_entropy_with_logits(head(model.encoder(xb)).squeeze(1),yb);loss.backward();nn.utils.clip_grad_norm_(list(model.encoder.parameters())+list(head.parameters()),5);opt.step()
        model.eval();head.eval()
        with torch.no_grad(): p=torch.sigmoid(head(model.encoder(torch.from_numpy(V))).squeeze()).numpy()
        score=average_precision_score(yv,p)
        if score>best:best=score;state=copy.deepcopy(model.encoder.state_dict())
    if variant!='no_latent': model.encoder.load_state_dict(state)
    init_bases(model,X,y,seed)
    opt=torch.optim.Adam(model.parameters(),lr=lr);best=-1;state=None
    for epoch in range(epochs):
        model.train();total=0
        for xb,yb in loader():
            opt.zero_grad();out=model(xb);loss=F.binary_cross_entropy(out,yb);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(yb)
        model.eval()
        with torch.no_grad():p=model(torch.from_numpy(V)).numpy()
        ap=average_precision_score(yv,p);history.append({'epoch':epoch+1,'train_loss':total/len(y),'validation_ap':ap})
        if ap>best:best=ap;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    with torch.no_grad():
        pv=model(torch.from_numpy(V)).numpy(); detail=model(torch.from_numpy(T),return_details=True)
    pt=detail['probability'].numpy();threshold=threshold_f2(yv,pv)
    extra={key:value.detach().numpy() for key,value in detail.items() if key!='reliability'}
    return model,{'seed':seed,'threshold':threshold,'validation_ap':best,'test':metrics(yt,pt,threshold),'history':history,'mechanism':{'positive_deficit_fraction':float((extra['deficit']>0).mean()),'mean_gate':float(extra['gate'].mean()),'mean_overlap':float(extra['overlap'].mean())}},(yt,pt)
