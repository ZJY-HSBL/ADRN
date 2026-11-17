"""Independent adapter for CISNN; source algorithm Huangza123/CIS, not a code copy.

Architecture and sampling choices follow the public reference; evaluation uses
validation AP instead of the reference script's repeated test monitoring.
"""
import argparse,copy,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from adrn.data import load
from adrn.train import metrics,threshold_f2

def run(tr,va,te,seed,epochs):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    X,y=tr;V,yv=va;T,yt=te;majority=X[y==0];minority=X[y==1]
    km=KMeans(n_clusters=15,n_init=10,random_state=seed).fit(majority)
    groups=[np.flatnonzero(km.labels_==i) for i in range(15)];centers=km.cluster_centers_
    distance=NearestNeighbors(n_neighbors=1,n_jobs=2).fit(minority).kneighbors(centers,return_distance=True)[0].ravel()
    distribution=torch.from_numpy(np.exp(-distance).astype('float32'));cent=torch.from_numpy(centers);pos=torch.from_numpy(minority);Xt=torch.from_numpy(majority)
    # Upstream uses a 10-unit ReLU layer followed by a two-unit ReLU output.
    net=nn.Sequential(nn.Linear(X.shape[1],10),nn.ReLU(),nn.Linear(10,2),nn.ReLU())
    opt=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=.001);rng=np.random.default_rng(seed);best=-1;state=None;history=[]
    for epoch in range(epochs):
        net.train()
        with torch.no_grad():
            difficulty=1-torch.softmax(net(cent),dim=1)[:,0];selected=torch.topk(distribution+.9*difficulty,k=4).indices.numpy()
        chosen=np.concatenate([groups[j] for j in selected]);others=np.concatenate([groups[j] for j in range(15) if j not in selected]);sample=rng.choice(others,size=len(chosen)//2,replace=True);negidx=np.concatenate([chosen,sample])
        posloss=F.cross_entropy(net(pos),torch.ones(len(pos),dtype=torch.long));negloss=F.cross_entropy(net(Xt[negidx]),torch.zeros(len(negidx),dtype=torch.long))
        loss=posloss+negloss;opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():pv=torch.softmax(torch.tanh(net(torch.from_numpy(V))),dim=1)[:,1].numpy()
        ap=average_precision_score(yv,pv);history.append({'epoch':epoch+1,'validation_ap':float(ap)})
        if ap>best:best=ap;state=copy.deepcopy(net.state_dict())
    net.load_state_dict(state);net.eval()
    with torch.no_grad():pv=torch.softmax(torch.tanh(net(torch.from_numpy(V))),dim=1)[:,1].numpy();pt=torch.softmax(torch.tanh(net(torch.from_numpy(T))),dim=1)[:,1].numpy()
    threshold=threshold_f2(yv,pv)
    return {'seed':seed,'validation_ap':float(best),'threshold':threshold,'test':metrics(yt,pt,threshold),'history':history}

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['gmsc','bank','default']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]);p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
    for name in a.datasets:
        tr,va,te,meta=load(name);runs=[]
        for seed in a.seeds:
            print(name,'CISNN adapter seed',seed,flush=True);r=run(tr,va,te,seed,a.epochs);runs.append(r);print(json.dumps(r['test']),flush=True)
        summary={m:{'mean':float(np.mean([r['test'][m] for r in runs])),'sd':float(np.std([r['test'][m] for r in runs],ddof=1)) if len(runs)>1 else None} for m in runs[0]['test']}
        res={'data':meta,'config':{'seeds':a.seeds,'epochs':a.epochs,'clusters':15,'top_clusters':4,'eta':.9,'optimizer':'Adam','lr':.01,'weight_decay':.001,'selection':'validation average precision'},'reference':'https://github.com/Huangza123/CIS/blob/0d1313db2cdeaa0deb713b7d27665d45abc6f7e8/train_CIS.py','summary':summary,'runs':runs}
        Path('results',name+'_cisnn.json').write_text(json.dumps(res,indent=2))
        path=Path('results',name+'_baselines.json');b=json.loads(path.read_text());b['methods']['cisnn_adapter']={'runs':runs,'summary':summary};path.write_text(json.dumps(b,indent=2))
if __name__=='__main__':main()
