"""TPE selection solely by validation AP; fixed split, independent test evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import optuna
from adrn.data import load
from adrn.train import train
p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['gmsc','bank','default']);p.add_argument('--trials',type=int,default=12);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]);p.add_argument('--epochs',type=int,default=6);p.add_argument('--pretrain-epochs',type=int,default=3);args=p.parse_args()
Path('results').mkdir(exist_ok=True)
for name in args.datasets:
    tr,va,te,meta=load(name);print('TPE',name,flush=True)
    def objective(trial):
        config={'hidden':trial.suggest_categorical('hidden',[32,64]),'latent':trial.suggest_categorical('latent',[8,16]),'k':trial.suggest_categorical('k',[4,8]),'lr':trial.suggest_float('lr',.0005,.002,log=True),'beta':trial.suggest_float('beta',.25,.75)}
        # During search the real held-out test set is neither passed nor evaluated.
        _,r,_=train(tr,va,va,seed=100+trial.number,epochs=args.epochs,pretrain_epochs=args.pretrain_epochs,**config)
        return r['validation_ap']
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=2))
    study.optimize(objective,n_trials=args.trials)
    cfg=study.best_params;print(name,'best',cfg,flush=True);runs=[]
    for seed in args.seeds:
        print(name,'test',seed,flush=True)
        _,r,_=train(tr,va,te,seed=seed,epochs=args.epochs,pretrain_epochs=args.pretrain_epochs,**cfg);runs.append(r)
    metrics=runs[0]['test'].keys();summary={m:{'mean':float(np.mean([r['test'][m] for r in runs])),'sd':float(np.std([r['test'][m] for r in runs],ddof=1)) if len(runs)>1 else None} for m in metrics}
    res={'data':meta,'search':{'method':'TPESampler','seed':42,'startup_trials':2,'trials':args.trials,'objective':'validation_average_precision','space':{'hidden':[32,64],'latent':[8,16],'k':[4,8],'lr':[.0005,.002],'beta':[.25,.75]},'history':[{'number':t.number,'value':t.value,'params':t.params} for t in study.trials],'best_config':cfg,'best_validation_ap':study.best_value},'config':{'epochs':args.epochs,'pretrain_epochs':args.pretrain_epochs,'seeds':args.seeds,'batch_size':512},'summary':summary,'runs':runs}
    Path('results',name+'_tpe.json').write_text(json.dumps(res,indent=2))
