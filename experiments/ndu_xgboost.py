"""Paper-guided NDU approximation + XGBoost on the original training split."""
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from adrn.data import load
from adrn.ndu import build_geometry,select_indices
from adrn.train import metrics,threshold_f2

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['gmsc','bank','default']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]);args=p.parse_args()
    for name in args.datasets:
        start=time.time();tr,va,te,meta=load(name);X,y=tr;V,yv=va;T,yt=te
        print(name,'building approximate neighbours and GMM',flush=True)
        g=build_geometry(X,y);runs=[]
        print(name,'geometry',g['k'],g['minority_overlap'],g['expanded_overlap'],flush=True)
        for seed in args.seeds:
            keep,rates=select_indices(y,g['indices'],g['scores'],seed=seed)
            print(name,'seed',seed,'train rows',len(keep),flush=True)
            model=XGBClassifier(n_estimators=150,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='aucpr',n_jobs=2,random_state=seed,tree_method='hist')
            model.fit(X[keep],y[keep],eval_set=[(V,yv)],verbose=False)
            pv=model.predict_proba(V)[:,1];pt=model.predict_proba(T)[:,1];threshold=threshold_f2(yv,pv)
            runs.append({'seed':seed,'training_rows':len(keep),'removed_majority':len(y)-len(keep),'segments':rates,'threshold':threshold,'validation_ap':float(average_precision_score(yv,pv)),'test':metrics(yt,pt,threshold)})
            print(json.dumps(runs[-1]['test']),flush=True)
        summary={m:{'mean':float(np.mean([r['test'][m] for r in runs])),'sd':float(np.std([r['test'][m] for r in runs],ddof=1)) if len(runs)>1 else None} for m in runs[0]['test']}
        result={'data':meta,'geometry':{k:v for k,v in g.items() if k not in ['indices','scores']},'config':{'seeds':args.seeds,'neighbor_backend':'FAISS HNSW approximate','density_normalization':'exp(logpdf - maximum logpdf)','gmm_covariance':'full','beta':.8,'segments':'five equal-count F-score groups','deletion':'uniform without replacement','geometry_random_state':0,'smote':False},'runtime_seconds':time.time()-start,'runs':runs,'summary':summary}
        path=Path('results',name+'_ndu_approx.json');path.write_text(json.dumps(result,indent=2))
        baseline=Path('results',name+'_baselines.json');b=json.loads(baseline.read_text());b['methods']['ndu_xgboost_approx']={'runs':runs,'summary':summary};baseline.write_text(json.dumps(b,indent=2))
if __name__=='__main__':main()
