"""Batched full-branch PLR search with the production multistart safeguards.

Vectorizes independent nuisance and bracket searches; does not reduce the grid,
skip local maxima, use truth-centered windows or substitute a fixed alternative.
"""
from methods import *

def batch_profile(p,c):
    p=np.atleast_2d(p);c=np.broadcast_to(c,p.shape);m=p.shape[1]
    lo=np.zeros(len(p));hi=np.ones(len(p))
    for _ in range(42):
        eta=(lo+hi)/2;q=(1-eta[:,None])*p+eta[:,None]/m
        grad=np.sum(c*(1/m-p)/q,axis=1)
        lo=np.where(grad>0,eta,lo);hi=np.where(grad<=0,eta,hi)
    eta=(lo+hi)/2;q=(1-eta[:,None])*p+eta[:,None]/m
    value=np.sum(xlogy(c,q),axis=1)
    left=np.sum(xlogy(c,p),axis=1);right=-c.sum(axis=1)*np.log(m)
    eta=np.where(left>value,0.,eta);value=np.maximum(value,left)
    eta=np.where(right>value,1.,eta);value=np.maximum(value,right)
    return value,eta

def batch_maximum(ex,counts):
    c=np.atleast_2d(counts);m=2**ex.n;grid=np.linspace(*ex.bounds,32*m+1)
    p=ex.probabilities(grid);ng=len(grid);nc=len(c)
    values,_=batch_profile(np.tile(p,(nc,1)),np.repeat(c,ng,axis=0));values=values.reshape(nc,ng)
    best=values.max(axis=1);owner=[];left=[];right=[]
    for row in range(nc):
        v=values[row];peaks=np.flatnonzero((v[1:-1]>=v[:-2])&(v[1:-1]>=v[2:]))+1
        peaks=[i for i in peaks if v[i]>-sum(c[row])*np.log(m)+1e-10]
        brackets=[(grid[0],grid[1]),(grid[-2],grid[-1])]+[(grid[i-1],grid[i+1]) for i in peaks]
        for lo,hi in brackets:owner.append(row);left.append(lo);right.append(hi)
    owner=np.array(owner);left=np.array(left);right=np.array(right);cc=c[owner]
    golden=(np.sqrt(5)-1)/2
    x1=right-golden*(right-left);x2=left+golden*(right-left)
    v1=batch_profile(ex.probabilities(x1),cc)[0];v2=batch_profile(ex.probabilities(x2),cc)[0]
    # Initial brackets <=2 bin_width/32; 60 contractions exceed 1e-12 resolution.
    for _ in range(60):
        choose=v1>v2
        right=np.where(choose,x2,right);left=np.where(choose,left,x1)
        new1=right-golden*(right-left);new2=left+golden*(right-left)
        # A surviving interior point retains its evaluated likelihood. Only
        # the new point requires another independent nuisance optimization.
        value=batch_profile(ex.probabilities(np.where(choose,new1,new2)),cc)[0]
        old1,old2=x1,x2;oldv1,oldv2=v1,v2
        x1=np.where(choose,new1,old2);x2=np.where(choose,old1,new2)
        v1=np.where(choose,value,oldv2);v2=np.where(choose,oldv1,value)
    np.maximum.at(best,owner,np.maximum(v1,v2))
    return best
