"""Explicit implementations of four comparison baselines; not author code."""
import argparse,copy,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from adrn.data import load
from adrn.train import metrics,threshold_f2

def mlp(tr,va,te,seed,focal=False,epochs=6):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    X,y=tr;V,yv=va;T,yt=te
    model=nn.Sequential(nn.Linear(X.shape[1],64),nn.GELU(),nn.Linear(64,16),nn.LayerNorm(16),nn.Linear(16,1));opt=torch.optim.Adam(model.parameters(),lr=.001)
    weights=torch.from_numpy((np.bincount(y,minlength=2)[y]**(-.5)).astype('float64'));ds=TensorDataset(torch.from_numpy(X),torch.from_numpy(y.astype('float32')))
    best=-1;state=None
    for _ in range(epochs):
        model.train()
        for xb,yb in DataLoader(ds,batch_size=512,sampler=WeightedRandomSampler(weights,len(y),replacement=True)):
            opt.zero_grad();logit=model(xb).squeeze(1);bce=F.binary_cross_entropy_with_logits(logit,yb,reduction='none')
            if focal:
                prob=torch.sigmoid(logit);pt=torch.where(yb>0,prob,1-prob);alpha=torch.where(yb>0,.25,.75);loss=(alpha*(1-pt).square()*bce).mean()
            else:loss=bce.mean()
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        model.eval()
        with torch.no_grad():pv=torch.sigmoid(model(torch.from_numpy(V)).squeeze(1)).numpy()
        ap=average_precision_score(yv,pv)
        if ap>best:best=ap;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    with torch.no_grad():pv=torch.sigmoid(model(torch.from_numpy(V)).squeeze(1)).numpy();pt=torch.sigmoid(model(torch.from_numpy(T)).squeeze(1)).numpy()
    threshold=threshold_f2(yv,pv)
    return {'seed':seed,'threshold':threshold,'validation_ap':best,'test':metrics(yt,pt,threshold)}

def tree(tr,va,te,seed,smote=False):
    X,y=tr;V,yv=va;T,yt=te
    if smote:X,y=SMOTE(random_state=seed).fit_resample(X,y)
    model=XGBClassifier(n_estimators=150,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='aucpr',n_jobs=2,random_state=seed,tree_method='hist')
    model.fit(X,y,eval_set=[(V,yv)],verbose=False)
    pv=model.predict_proba(V)[:,1];pt=model.predict_proba(T)[:,1];threshold=threshold_f2(yv,pv)
    return {'seed':seed,'threshold':threshold,'validation_ap':float(average_precision_score(yv,pv)),'test':metrics(yt,pt,threshold)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['gmsc','bank','default']);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]);p.add_argument('--methods',nargs='+',default=['mlp','focal_mlp','xgboost','smote_xgboost']);args=p.parse_args()
    for name in args.datasets:
        tr,va,te,meta=load(name);path=Path('results',name+'_baselines.json');res=json.loads(path.read_text()) if path.exists() else {'data':meta,'methods':{}}
        for method in args.methods:
            runs=[]
            for seed in args.seeds:
                print(name,method,seed,flush=True)
                r=mlp(tr,va,te,seed,method=='focal_mlp') if method in ('mlp','focal_mlp') else tree(tr,va,te,seed,method=='smote_xgboost')
                runs.append(r);print(json.dumps(r['test']),flush=True)
            summary={m:{'mean':float(np.mean([r['test'][m] for r in runs])),'sd':float(np.std([r['test'][m] for r in runs],ddof=1)) if len(runs)>1 else None} for m in runs[0]['test']}
            res['methods'][method]={'runs':runs,'summary':summary};path.write_text(json.dumps(res,indent=2))
if __name__=='__main__':main()
