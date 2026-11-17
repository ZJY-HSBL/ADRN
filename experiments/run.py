"""Reproducible ADRN experiment. Example: python -m experiments.run --seeds 0 1 2 3 4"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve,roc_curve,confusion_matrix
from adrn.data import load
from adrn.train import train


def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['gmsc','bank','default']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]);p.add_argument('--epochs',type=int,default=6);p.add_argument('--pretrain-epochs',type=int,default=3);p.add_argument('--raw-dir',default='data/raw');args=p.parse_args()
    for d in ['results','figures','checkpoints']:Path(d).mkdir(exist_ok=True)
    all_results={}
    for name in args.datasets:
        print('Loading',name,flush=True);train_set,val,test,meta=load(name,args.raw_dir)
        runs=[];first=None
        for seed in args.seeds:
            print('Training',name,'seed',seed,flush=True)
            model,result,preds=train(train_set,val,test,seed=seed,epochs=args.epochs,pretrain_epochs=args.pretrain_epochs)
            runs.append(result);print(json.dumps(result['test']),flush=True)
            if first is None:
                first=(result,preds);torch.save({'state_dict':model.state_dict(),'input_dim':train_set[0].shape[1],'hidden':64,'latent':16,'k':4,'threshold':result['threshold']},f'checkpoints/{name}_seed{seed}.pt')
        names=runs[0]['test'].keys();summary={key:{'mean':float(np.mean([r['test'][key] for r in runs])),'sd':float(np.std([r['test'][key] for r in runs],ddof=1)) if len(runs)>1 else None} for key in names}
        all_results[name]={'data':meta,'config':{'epochs':args.epochs,'pretrain_epochs':args.pretrain_epochs,'seeds':args.seeds,'hidden':64,'latent':16,'k':4,'batch_size':512,'lr':.001,'sampling_beta':.5,'optimizer':'Adam','hyperparameter_search':'none'},'summary':summary,'runs':runs}
        Path('results',name+'.json').write_text(json.dumps(all_results[name],indent=2),encoding='utf8')
        r,(y,score)=first;precision,recall,_=precision_recall_curve(y,score);fpr,tpr,_=roc_curve(y,score);cm=confusion_matrix(y,score>=r['threshold']);hist=r['history']
        fig,ax=plt.subplots(2,2,figsize=(10,7.5),constrained_layout=True)
        ax[0,0].plot(recall,precision,color='#635bcb');ax[0,0].axhline(y.mean(),color='#dc9550',ls='--',label='Prevalence');ax[0,0].set(xlabel='Recall',ylabel='Precision',title=f"PR curve · AP {r['test']['pr_auc']:.3f}");ax[0,0].legend(frameon=False)
        ax[0,1].plot(fpr,tpr,color='#635bcb');ax[0,1].plot([0,1],[0,1],ls='--',color='#94a3b8');ax[0,1].set(xlabel='False positive rate',ylabel='True positive rate',title=f"ROC curve · AUC {r['test']['roc_auc']:.3f}")
        ax[1,0].plot([x['epoch'] for x in hist],[x['validation_ap'] for x in hist],marker='o',color='#635bcb');ax[1,0].set(xlabel='Epoch',ylabel='Validation AP',title='Validation during joint training')
        im=ax[1,1].imshow(cm,cmap='Purples');ax[1,1].set(xlabel='Predicted',ylabel='Actual',title=f"Confusion matrix · threshold {r['threshold']:.3f}",xticks=[0,1],yticks=[0,1]);
        for i in range(2):
            for j in range(2):ax[1,1].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        fig.suptitle(f'{name.upper()} · independent test set · seed {r["seed"]}',fontsize=14,weight='bold');fig.savefig('figures/'+name+'_evaluation.png',dpi=150);plt.close(fig)
    Path('results/summary.json').write_text(json.dumps(all_results,indent=2),encoding='utf8')
if __name__=='__main__':main()
