"""Training-only preprocessing and fixed stratified train/validation/test split."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load(name,root='data/raw',split_seed=42,target_ir=None,ir_seed=0,default_encoding='raw',return_transformer=False):
    root=Path(root)
    if name=='gmsc':
        df=pd.read_csv(root/'cs-training.csv'); y=df.pop('SeriousDlqin2yrs').astype(int).to_numpy();df=df.loc[:,~df.columns.str.startswith('Unnamed')]
    elif name=='bank':
        df=pd.read_csv(root/'bank-full.csv',sep=';');y=(df.pop('y')=='yes').astype(int).to_numpy();df=df.drop(columns=['duration'])
    elif name=='default':
        df=pd.read_excel(root/'default.xls',header=1);y=df.pop('default payment next month').astype(int).to_numpy();df=df.drop(columns=['ID'])
        if default_encoding not in ('raw','grouped'): raise ValueError('unknown default encoding')
        if default_encoding=='grouped':
            # Explicit research assumption: combine education 0/5/6 with 4,
            # and undocumented marriage 0 with 3; not confirmed by the paper.
            df['EDUCATION']=df['EDUCATION'].replace({0:4,5:4,6:4})
            df['MARRIAGE']=df['MARRIAGE'].replace({0:3})
        for c in ('SEX','EDUCATION','MARRIAGE'):df[c]=df[c].astype(str)
    else: raise ValueError('unknown dataset')
    indices=np.arange(len(y)); tr, rest=train_test_split(indices,test_size=.2,stratify=y,random_state=split_seed);va,te=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=split_seed)
    if target_ir is not None:
        if target_ir <= 0: raise ValueError('target_ir must be positive')
        negative=tr[y[tr]==0];positive=tr[y[tr]==1];target=int(len(negative)//target_ir)
        if target<1 or target>len(positive): raise ValueError('target ratio cannot be achieved by minority undersampling')
        selected=np.random.default_rng(ir_seed).choice(positive,size=target,replace=False)
        tr=np.concatenate([negative,selected])
    cat=df.select_dtypes(include=['object','category']).columns.tolist();num=[c for c in df.columns if c not in cat]
    numeric=Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    categorical=Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    transformer=ColumnTransformer([('num',numeric,num),('cat',categorical,cat)])
    a=transformer.fit_transform(df.iloc[tr]).astype('float32');b=transformer.transform(df.iloc[va]).astype('float32');c=transformer.transform(df.iloc[te]).astype('float32')
    metadata={'rows':len(y),'train':len(tr),'validation':len(va),'test':len(te),'input_features':df.shape[1],'numeric_features':len(num),'encoded_features':a.shape[1],'split_seed':split_seed,'default_encoding':default_encoding if name=='default' else None,'positive':int(y.sum())}
    result=((a,y[tr]),(b,y[va]),(c,y[te]),metadata)
    return (*result,transformer) if return_transformer else result
