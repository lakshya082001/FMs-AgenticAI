"""Exploratory check: do lane-level late rates have temporal dependence in raw DataCo? (result: no, see README)"""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
d=pd.read_csv('data/raw/dataco.csv',encoding='latin-1');d.columns=d.columns.str.strip()
d=d[d['Delivery Status']!='Shipping canceled'].copy()
d['t']=pd.to_datetime(d['order date (DateOrders)'])
d['done']=pd.to_datetime(d['shipping date (DateOrders)'])+pd.to_timedelta(d['Days for shipping (real)'],unit='D')
y='Late_delivery_risk'
for name,keys in [('mode x market',['Shipping Mode','Market']),('mode x region',['Shipping Mode','Order Region']),('category',['Category Name'])]:
    d['lane']=d[keys].astype(str).agg('|'.join,axis=1)
    # weekly lane late rate, lag-1 autocorr of lane-demeaned series
    w=d.groupby(['lane',d.t.dt.to_period('W')])[y].agg(['mean','size']).reset_index()
    w=w[w['size']>=20].sort_values(['lane','t']); w['dm']=w['mean']-w.groupby('lane')['mean'].transform('mean')
    w['lag']=w.groupby('lane')['dm'].shift(1); w=w.dropna()
    print(f'{name}: lanes={d.lane.nunique()} weekly lag1 autocorr={w.dm.corr(w.lag):.3f} (n={len(w)})')
    for W in [7,30]:
        rate=np.full(len(d),np.nan); cnt=np.zeros(len(d))
        for _,g in d.groupby('lane'):
            c=g.sort_values('done'); comp=c.done.values; cs=np.concatenate([[0],np.cumsum(c[y].values)])
            hi=np.searchsorted(comp,g.t.values,'left'); lo=np.searchsorted(comp,(g.t-pd.Timedelta(days=W)).values,'left')
            n=hi-lo; idx=d.index.get_indexer(g.index)
            cnt[idx]=n; rate[idx]=np.where(n>0,(cs[hi]-cs[lo])/np.maximum(n,1),np.nan)
        m=(cnt>=20)&(d.t>='2015-03-01').values
        lm=d.groupby('lane')[y].transform('mean').values
        print(f'  trailing {W}d: usable={m.mean():.2f} corr(rate-lanemean, y-lanemean)={np.corrcoef(rate[m]-lm[m], d[y].values[m]-lm[m])[0,1]:.3f} AUC lane-mean={roc_auc_score(d[y][m],lm[m]):.3f} AUC trailing={roc_auc_score(d[y][m],rate[m]):.3f}')
