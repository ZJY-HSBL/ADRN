"""NDU paper-guided approximation. Undersampling only; no SMOTE is applied.

The paper does not define ComputeThreshold / Segment / RS in detail; here
five equal-count score segments and uniform deletion are explicit choices.
A FAISS HNSW index is an approximate neighbour-search implementation.
"""
import numpy as np
from sklearn.mixture import GaussianMixture


def score_segments(score,n_segments=5):
    score=np.asarray(score,dtype=float)
    if score.ndim!=1 or not np.isfinite(score).all():raise ValueError('scores must be a finite vector')
    if n_segments<1:raise ValueError('segments must be positive')
    if not len(score):return []
    return [g for g in np.array_split(np.argsort(score,kind='stable'),min(n_segments,len(score))) if len(g)]


def select_indices(y,overlap_indices,scores,seed=0,n_segments=5):
    y=np.asarray(y);overlap_indices=np.asarray(overlap_indices,dtype=int);scores=np.asarray(scores,float)
    if len(scores)!=len(overlap_indices) or np.any(y[overlap_indices]!=0):raise ValueError('candidate/score mismatch')
    rng=np.random.default_rng(seed);removed=[];rates=[]
    for ids in score_segments(scores,n_segments):
        values=scores[ids];maxval=float(values.max());rate=0. if maxval<=0 else float(np.clip(1-values.mean()/maxval,0,1))
        count=int(np.floor(rate*len(ids)));removed.extend(overlap_indices[rng.choice(ids,count,replace=False)].tolist());rates.append({'n':len(ids),'rate':rate,'removed':count})
    keep=np.ones(len(y),dtype=bool);keep[removed]=False
    return np.flatnonzero(keep),rates


def build_geometry(X,y,beta=.8,gmm_components=5,n_segments=5,seed=0):
    import faiss
    X=np.ascontiguousarray(X,dtype='float32');y=np.asarray(y,dtype=int)
    if X.ndim!=2 or len(X)!=len(y) or not np.isfinite(X).all() or set(np.unique(y))!={0,1}:raise ValueError('expected finite binary data')
    n=len(y);n0=int((y==0).sum());n1=int((y==1).sum())
    if n0<=n1:raise ValueError('this implementation assumes label 0 is majority')
    k=max(1,min(n-1,int(np.sqrt(n/(n0/n1)))))
    faiss.omp_set_num_threads(2);index=faiss.IndexHNSWFlat(X.shape[1],16);index.hnsw.efConstruction=80;index.hnsw.efSearch=max(100,2*k);index.add(X)
    # Chunked queries avoid retaining an N x K distance matrix.
    cross=np.zeros(n,bool);local=np.zeros(n,float)
    for start in range(0,n,4096):
        stop=min(n,start+4096);_,neighbors=index.search(X[start:stop],k+1)
        for j,row in enumerate(neighbors):
            ix=start+j;row=row[(row>=0)&(row!=ix)][:k]
            if len(row)!=k:raise RuntimeError('neighbour search returned too few neighbours')
            labels=y[row];cross[ix]=bool(np.any(labels!=y[ix]));local[ix]=float(np.mean(labels==0))
    minority_overlap=np.flatnonzero(cross & (y==1));initial_majority=np.flatnonzero(cross & (y==0))
    if len(minority_overlap)==0:
        return {'indices':initial_majority,'scores':np.array([]),'k':k,'minority_overlap':0,'expanded_overlap':0,'gmm_count':0}
    majority=np.flatnonzero(y==0);index_major=faiss.IndexHNSWFlat(X.shape[1],16);index_major.hnsw.efConstruction=80;index_major.hnsw.efSearch=max(100,2*k);index_major.add(X[majority]);expanded=np.zeros(n,bool);expanded[initial_majority]=True
    for start in range(0,len(minority_overlap),4096):
        _,neighbors=index_major.search(X[minority_overlap[start:start+4096]],min(k,len(majority)))
        if np.any(neighbors<0): raise RuntimeError('majority-neighbour search returned too few neighbours')
        expanded[majority[neighbors.ravel()]]=True
    candidates=np.flatnonzero(expanded);gmm_ids=np.concatenate([candidates,minority_overlap]);components=min(gmm_components,len(gmm_ids))
    model=GaussianMixture(n_components=components,covariance_type='full',reg_covar=1e-5,random_state=seed,max_iter=100).fit(X[gmm_ids])
    logpdf=model.score_samples(X[candidates]);density=np.exp(np.maximum(logpdf-logpdf.max(),-700))
    scores=beta*local[candidates]+(1-beta)*density
    return {'indices':candidates,'scores':scores,'k':k,'minority_overlap':len(minority_overlap),'expanded_overlap':len(candidates),'gmm_count':len(gmm_ids),'gmm_converged':bool(model.converged_),'gmm_iterations':int(model.n_iter_)}
