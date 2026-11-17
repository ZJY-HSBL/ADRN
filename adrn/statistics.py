"""Welch t tests and Benjamini-Hochberg adjusted p-values."""
import numpy as np
from scipy.stats import ttest_ind

def bh(p_values):
    p=np.asarray(p_values,dtype=float)
    if p.ndim!=1 or not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):raise ValueError('p must be a finite vector of probabilities')
    order=np.argsort(p);n=len(p);adjusted=np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1].clip(0,1)
    output=np.empty(n);output[order]=adjusted;return output

def welch(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<2 or len(b)<2 or not (np.isfinite(a).all() and np.isfinite(b).all()):raise ValueError('two finite samples of length >=2 required')
    value=float(ttest_ind(a,b,equal_var=False).pvalue)
    if not np.isfinite(value):raise ValueError('test undefined for degenerate data')
    return value
