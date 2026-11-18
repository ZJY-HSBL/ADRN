"""Predict from original feature columns and an explicitly trusted local artifact."""
import argparse
import hashlib
from pathlib import Path
import joblib
import pandas as pd
import torch
from .model import ADRN

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['gmsc','bank','default'],required=True);parser.add_argument('--input',required=True,help='CSV with original feature names');parser.add_argument('--output',default='predictions.csv');parser.add_argument('--checkpoint-dir',default='checkpoints');args=parser.parse_args()
    root=Path(args.checkpoint_dir);checkpoint_path=root/(args.dataset+'_seed0.pt');ck=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    bundle=joblib.load(root/(args.dataset+'_preprocess.joblib'))
    if bundle['dataset']!=args.dataset:raise ValueError('preprocessor dataset mismatch')
    if bundle.get('checkpoint_sha256')!=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest():raise ValueError('preprocessor does not match checkpoint; rerun experiments.package_checkpoints')
    df=pd.read_csv(args.input,sep=';' if args.dataset=='bank' else ',')
    transformer=bundle['transformer'];columns=transformer.feature_names_in_;missing=sorted(set(columns)-set(df.columns))
    if missing:raise ValueError('missing feature columns: '+', '.join(missing))
    if args.dataset=='default':
        if bundle['default_encoding']=='grouped':
            df['EDUCATION']=df['EDUCATION'].replace({0:4,5:4,6:4});df['MARRIAGE']=df['MARRIAGE'].replace({0:3})
        for col in ['SEX','EDUCATION','MARRIAGE']:df[col]=df[col].astype(str)
    X=transformer.transform(df.loc[:,columns]).astype('float32');model=ADRN(ck['input_dim'],ck['hidden'],ck['latent'],ck['k']);model.load_state_dict(ck['state_dict']);model.eval()
    with torch.no_grad():prob=model(torch.from_numpy(X)).numpy()
    pd.DataFrame({'positive_probability':prob,'predicted_label':(prob>=ck['threshold']).astype(int)}).to_csv(args.output,index=False)
    print(f'Saved {len(df)} predictions to {args.output}; threshold={ck["threshold"]:.6f}')
if __name__=='__main__':main()
