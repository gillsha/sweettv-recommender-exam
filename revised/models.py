"""Training-only count models, biased MF and personalized PageRank."""
import sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from numba import njit,prange,set_num_threads
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from als import ImplicitALS
from models import TrainingBundle

@njit(parallel=True)
def target_update(indptr,indices,confidence,targets,fixed,bias,reg):
 gram=fixed.T@fixed;offset=-(fixed.T@bias);out=np.zeros((len(indptr)-1,fixed.shape[1]))
 for row in prange(len(indptr)-1):
  a=gram.copy();b=offset.copy()
  for j in range(fixed.shape[1]):a[j,j]+=reg[j]
  for q in range(indptr[row],indptr[row+1]):
   i=indices[q];c=confidence[q];y=fixed[i]
   b+=(c*targets[q]-(c-1)*bias[i])*y
   for j in range(fixed.shape[1]):
    for k in range(fixed.shape[1]):a[j,k]+=(c-1)*y[j]*y[k]
  out[row]=np.linalg.solve(a,b)
 return out

class CountMF:
 def __init__(self,factors=32,regularization=.1,alpha=20,bias=False,iterations=20,bias_regularization=1.):
  self.factors=factors;self.reg=regularization;self.alpha=alpha;self.use_bias=bias;self.iterations=iterations;self.bias_reg=bias_regularization
 def fit(self,counts):
  set_num_threads(2);c=counts.astype(np.float64).tocsr();ct=c.T.tocsr()
  target=np.log1p(c.data);target_t=np.log1p(ct.data)
  conf=1+self.alpha*target;conf_t=1+self.alpha*target_t
  y=np.random.default_rng(2026).normal(0,.01,(c.shape[1],self.factors));b=np.zeros(c.shape[1])
  with threadpool_limits(limits=1):
   for iteration in range(self.iterations):
    x=target_update(c.indptr,c.indices,conf,target,y,b,np.full(self.factors,self.reg))
    if self.use_bias:
     aug=np.column_stack([x,np.ones(c.shape[0])]);reg=np.r_[np.full(self.factors,self.reg),self.bias_reg]
     result=target_update(ct.indptr,ct.indices,conf_t,target_t,aug,np.zeros(c.shape[0]),reg)
     y=result[:,:-1].copy();b=result[:,-1].copy()
    else:y=target_update(ct.indptr,ct.indices,conf_t,target_t,x,np.zeros(c.shape[0]),np.full(self.factors,self.reg))
  self.user_factors=x;self.item_factors=y;self.item_bias=b;return self
 def predict(self,rows):return self.user_factors[rows]@self.item_factors.T+self.item_bias

def verify_target_update():
 fixed=np.array([[.1,.2],[.3,-.2],[.5,.4]]);bias=np.array([.2,-.1,.3]);reg=np.array([.7,.8])
 c=csr_matrix([[3.,0.,5.],[0.,0.,0.]]);targets=np.log1p(np.array([2.,4.]));actual=target_update(c.indptr,c.indices,c.data,targets,fixed,bias,reg)
 W=np.diag([3.,1.,5.]);t=np.array([targets[0],0,targets[1]])
 expected=np.linalg.solve(fixed.T@W@fixed+np.diag(reg),fixed.T@W@(t-bias))
 cold=np.linalg.solve(fixed.T@fixed+np.diag(reg),-fixed.T@bias)
 assert np.allclose(actual[0],expected,atol=1e-10) and np.allclose(actual[1],cold,atol=1e-10)
 return float(np.max(np.abs(actual[0]-expected)))

class Window:
 def __init__(self,train,schedule,cohort,candidates,start,cutoff,end):
  self.train=train;self.schedule=schedule;self.cohort=cohort;self.start=start;self.cutoff=cutoff;self.end=end
  self.users=sorted(set(train.user_id)|set(cohort));self.uindex={u:i for i,u in enumerate(self.users)}
  self.items=np.array(sorted(train.tv_show_id.unique()),dtype=int);self.iindex={i:j for j,i in enumerate(self.items)}
  self.candidates=np.array(sorted(candidates),dtype=int);self.ci=np.array([self.iindex.get(i,-1) for i in self.candidates]);self.known=self.ci>=0
  pairs=train.groupby(['user_id','tv_show_id']).size().rename('n').reset_index()
  self.counts=csr_matrix((pairs.n.to_numpy(float),(pairs.user_id.map(self.uindex),pairs.tv_show_id.map(self.iindex))),shape=(len(self.users),len(self.items)))
  self.rows=np.array([self.uindex[u] for u in cohort]);self.cold=np.asarray(self.counts[self.rows].sum(axis=1)).ravel()==0
  self.pop=np.asarray(self.counts.sum(axis=0)).ravel();self.cpop=np.zeros(len(self.candidates));self.cpop[self.known]=self.pop[self.ci[self.known]]
  self.repeat=self.project(self.counts[self.rows].toarray())
 def project(self,s):
  out=np.zeros((len(self.cohort),len(self.candidates)));out[:,self.known]=s[:,self.ci[self.known]];return out
 def rank(self,scores):
  fallback=np.lexsort((self.candidates,-self.cpop));out={}
  for u,row,cold in zip(self.cohort,scores,self.cold):
   if cold:chosen=fallback[:5]
   else:
    order=np.lexsort((self.candidates,-self.cpop,-np.nan_to_num(row,nan=-np.inf)))
    chosen=order[np.isfinite(row[order])&(row[order]>0)][:5].tolist()
    for j in fallback:
     if len(chosen)==5:break
     if int(j) not in chosen:chosen.append(int(j))
   out[u]=self.candidates[chosen].tolist()
  return out
 def individual(self,config):
  family=config['family'];p=config.get('params',{})
  if family=='popularity':return np.broadcast_to(self.cpop,(len(self.cohort),len(self.candidates))).copy()
  if family=='repeat':return self.repeat.copy()
  if family=='content':
   metadata=self.schedule[(self.schedule.start_s>=self.start)&(self.schedule.end_s<=self.cutoff)]
   bundle=TrainingBundle(self.train,metadata,self.users)
   return bundle.content_scores(self.candidates)[self.rows]
  if family=='implicit':
   c=self.counts.copy();c.data=1+20*np.log1p(c.data)
   model=ImplicitALS(factors=p.get('factors',32),regularization=p.get('regularization',.1),iterations=20,random_state=2026,num_threads=2).fit(c)
   return self.project(model.user_factors[self.rows]@model.item_factors.T)
  if family=='count_mf':return self.project(CountMF(**p).fit(self.counts).predict(self.rows))
  if family=='ppr':
   damping=p.get('damping',.85);neighbors=p.get('neighbors',50)
   binary=(self.counts>0).astype(float);co=(binary.T@binary).toarray();norm=np.sqrt(np.diag(co));co/=np.maximum(norm[:,None]*norm[None,:],1e-12);np.fill_diagonal(co,0)
   rows=[];cols=[];vals=[]
   for i,row in enumerate(co):
    ids=np.lexsort((self.items,-row))[:neighbors];ids=ids[row[ids]>0]
    rows.extend([i]*len(ids));cols.extend(ids);vals.extend(row[ids])
   P=normalize(csr_matrix((vals,(rows,cols)),shape=co.shape),norm='l1',axis=1)
   restart=normalize(self.counts[self.rows],norm='l1',axis=1).toarray();score=restart.copy()
   for iteration in range(100):
    propagated=np.asarray(score@P);lost=score.sum(axis=1)-propagated.sum(axis=1)
    new=(1-damping)*restart+damping*(propagated+lost[:,None]*restart)
    delta=np.max(np.abs(new-score));score=new
    if delta<1e-10:break
   self.ppr_iterations=iteration+1
   return self.project(score)
  raise ValueError(family)
 def schedule_ratio(self):
  # Channel preferences use past qualified airings only; no future viewer signal.
  channels=np.array(sorted(self.schedule.channel_id.unique()));chidx={c:i for i,c in enumerate(channels)}
  x=self.train.groupby(['user_id','channel_id']).size().rename('n').reset_index()
  weights=csr_matrix((x.n.to_numpy(float),(x.user_id.map(self.uindex),x.channel_id.map(chidx))),shape=(len(self.users),len(channels)))
  weights=normalize(weights[self.rows],norm='l1',axis=1)
  ci={x:i for i,x in enumerate(self.candidates)}
  def availability(a,b):
   f=self.schedule[(self.schedule.start_s>=a)&(self.schedule.end_s<=b)&self.schedule.tv_show_id.isin(ci)]
   z=f.groupby(['channel_id','tv_show_id']).size().rename('n').reset_index()
   return csr_matrix((z.n.to_numpy(float)/((b-a)/86400),(z.channel_id.map(chidx),z.tv_show_id.map(ci))),shape=(len(channels),len(ci)))
  past=(weights@availability(self.start,self.cutoff)).toarray();future=(weights@availability(self.cutoff,self.end)).toarray()
  return np.clip((future+.02)/(past+.02),.25,4.)

def rrf(window,a,b,weight):
 def reciprocal(s):
  out=np.empty_like(s)
  for i,row in enumerate(s):
   # Same score/fallback ordering before rank fusion.
   order=np.lexsort((window.candidates,-window.cpop,-np.maximum(row,0)))
   if window.cold[i]:order=np.lexsort((window.candidates,-window.cpop))
   out[i,order]=1/(20+np.arange(1,len(order)+1))
  return out
 return (1-weight)*reciprocal(a)+weight*reciprocal(b)
