"""Small-budget TPE temperature search using validation ROC-AUC only."""
import argparse,json
from pathlib import Path
import numpy as np
import optuna
from adrn.data import load
from adrn.scl_tpe import scl_fit

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['gmsc','bank','default']);p.add_argument('--trials',type=int,default=5);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]);args=p.parse_args()
    for name in args.datasets:
        tr,va,te,meta=load(name);print('SCL-TPE',name,flush=True)
        def objective(trial):
            temp=trial.suggest_float('temperature',.05,1.,log=True)
            r=scl_fit(tr,va,va,meta['numeric_features'],seed=100+trial.number,temperature=temp)
            return r['validation_roc_auc']
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=2));study.optimize(objective,n_trials=args.trials)
        temp=study.best_params['temperature'];runs=[]
        for seed in args.seeds:
            print(name,'final seed',seed,'temperature',temp,flush=True);runs.append(scl_fit(tr,va,te,meta['numeric_features'],seed=seed,temperature=temp))
        summary={m:{'mean':float(np.mean([r['test'][m] for r in runs])),'sd':float(np.std([r['test'][m] for r in runs],ddof=1)) if len(runs)>1 else None} for m in runs[0]['test']}
        result={'data':meta,'search':{'objective':'validation_roc_auc','trials':args.trials,'temperature_range':[.05,1.0],'startup_trials':2,'best_temperature':temp,'history':[{'number':t.number,'value':t.value,'params':t.params} for t in study.trials]},'config':{'seeds':args.seeds,'contrastive_epochs':2,'contrastive_cap':16000,'classifier_epochs':4,'batch_size_contrastive':128,'numeric_noise_sd':.05},'summary':summary,'runs':runs}
        Path('results',name+'_scl_tpe.json').write_text(json.dumps(result,indent=2))
        baseline_path=Path('results',name+'_baselines.json')
        if baseline_path.exists():
            baseline=json.loads(baseline_path.read_text());baseline['methods']['scl_tpe_approx']={'runs':runs,'summary':summary}
            baseline_path.write_text(json.dumps(baseline,indent=2))
if __name__=='__main__':main()
