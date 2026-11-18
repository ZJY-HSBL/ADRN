"""Reconstruct and save preprocessing fitted on the fixed training split."""
from pathlib import Path
import hashlib
import joblib,torch
from adrn.data import load

def main():
    for name in ['gmsc','bank','default']:
        path=Path('checkpoints',name+'_seed0.pt')
        if not path.exists():raise FileNotFoundError(f'Run experiments.run first: {path}')
        tr,_,_,metadata,transformer=load(name,return_transformer=True)
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['input_dim']!=tr[0].shape[1]:raise ValueError('model and preprocessing dimensions disagree')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        joblib.dump({'dataset':name,'default_encoding':'raw','metadata':metadata,'checkpoint_sha256':digest,'transformer':transformer},Path('checkpoints',name+'_preprocess.joblib'))
        print(name,'saved preprocessing for',checkpoint['input_dim'],'features')
if __name__=='__main__':main()
