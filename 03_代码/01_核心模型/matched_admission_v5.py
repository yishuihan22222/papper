"""One-step receiver-location intervention with matched origin admissions.

This is a diagnostic experiment, not a dispatch algorithm novelty claim.
After the first synchronous failure update all variants use exactly the same
legacy LP continuation. No inventory or cumulative throughput is introduced.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack


PLAN_NAMES = ('nearest_cost', 'balanced_util', 'low_id_probe', 'high_id_probe')


@dataclass
class AdmissionPlan:
    name: str
    src: np.ndarray
    dst: np.ndarray
    flow: np.ndarray
    own: np.ndarray
    q: np.ndarray
    load: np.ndarray
    completed_source: np.ndarray
    completed_receiver: np.ndarray
    cost: float
    excess: float
    assigned: float
    served: float
    max_error: float
    excess_optimum: float | None = None
    peak_util_optimum: float | None = None


def evaluate_flow(a, failed, demand, src, dst, flow, name='manual'):
    """Physical accounting for a supplied first-step flow, without optimization."""
    D=np.asarray(demand,float);n=len(a.ids);K=a.nominal_capacity
    src=np.asarray(src,int);dst=np.asarray(dst,int);flow=np.asarray(flow,float)
    if D.shape!=(n,) or not np.isfinite(D).all() or (D<0).any() or D.sum()<=0:
        raise ValueError('Invalid demand')
    if not set(failed)<=set(a.ids) or src.shape!=dst.shape or src.shape!=flow.shape:
        raise ValueError('Invalid supplied plan')
    if len(flow) and (min(src.min(),dst.min())<0 or max(src.max(),dst.max())>=n):
        raise ValueError('Invalid endpoint')
    active=np.array([node not in failed for node in a.ids])
    own=np.where(active,np.minimum(D,K),0.)
    path=a.paths(frozenset(failed)).copy();ix=np.flatnonzero(active);path[ix,ix]=0.
    if not np.isfinite(flow).all() or (len(flow) and flow.min()<-1e-7):raise ValueError('Invalid flow')
    if len(flow) and not np.isfinite(path[src[flow>1e-9],dst[flow>1e-9]]).all():
        raise ValueError('Flow on unavailable path')
    # Only numerical negative values tolerated by the underlying LP are clipped.
    correction=float(-np.minimum(flow,0.).sum())
    if correction>1e-7:raise ValueError('Excessive negative-flow correction')
    flow=np.maximum(flow,0.)
    outgoing=np.bincount(src,weights=flow,minlength=n)
    incoming=np.bincount(dst,weights=flow,minlength=n)
    load=own+incoming;frac=np.divide(np.minimum(load,K),load,out=np.zeros(n),where=load>0)
    receiver=load*frac;completed=own*frac+np.bincount(src,weights=flow*frac[dst],minlength=n)
    errors=[max(0.,float(np.max(outgoing-(D-own)))),
            max(0.,float(np.max(load-a.dispatch_limit))),
            max(0.,float(np.max(completed-D))),
            abs(float(receiver.sum()-completed.sum())),correction]
    if max(errors)>1e-5:raise AssertionError(('Supplied plan accounting',errors))
    valid=flow>0
    return AdmissionPlan(name,src,dst,flow,own,outgoing,load,completed,receiver,
        float(np.dot(flow[valid],path[src[valid],dst[valid]])),float(np.maximum(load-K,0.).sum()),
        float((own+outgoing).sum()),float(completed.sum()),max(errors))


def matched_plans(a, failed, demand):
    """Fix the baseline LP's maximum admission VECTOR, then minimum excess.

    All four variants share each source's q, not just the total. Minimum excess
    implies identical initial aggregate completion. Two rank objectives are
    arbitrary stress probes; neither is a recommended operational policy.
    """
    baseline=a.evaluate(failed,demand,details=True)
    arcs=baseline[3]['arcs'];D=np.asarray(demand,float);n=len(a.ids)
    src=np.array([a.pos[v] for v in arcs.source_id],int)
    dst=np.array([a.pos[v] for v in arcs.target_id],int)
    original=evaluate_flow(a,failed,D,src,dst,arcs.flow.to_numpy(float),'baseline')
    q=original.q.copy();m=len(src)
    if not m:
        plans={name:evaluate_flow(a,failed,D,src,dst,[],name) for name in PLAN_NAMES}
        for p in plans.values():p.excess_optimum=0.
        return plans,baseline
    columns=np.arange(m)
    S=coo_matrix((np.ones(m),(src,columns)),shape=(n,m)).tocsr()
    R=coo_matrix((np.ones(m),(dst,columns)),shape=(n,m)).tocsr()
    own=original.own;K=a.nominal_capacity
    A=vstack([hstack([R,csr_matrix((n,n))]),hstack([R,-eye(n)])],format='csr')
    b=np.r_[a.dispatch_limit-own,K-own]
    E=hstack([S,csr_matrix((n,n))],format='csr');eq=q.copy()

    def solve(c,au=A,bu=b,ae=E,be=eq):
        sol=linprog(c,A_ub=au,b_ub=bu,A_eq=ae,b_eq=be,bounds=(0,None),method='highs')
        if not sol.success:raise RuntimeError('Matched admission LP: '+sol.message)
        return sol

    first=solve(np.r_[np.zeros(m),np.ones(n)])
    optimum=max(0.,float(first.x[m:].sum()))
    E=vstack([E,csr_matrix(np.r_[np.zeros(m),np.ones(n)][None,:])],format='csr')
    eq=np.r_[eq,optimum]
    cost=arcs.cost.to_numpy(float);plans={}
    for name in PLAN_NAMES:
        peak=None
        if name=='balanced_util':
            aa=vstack([hstack([A,csr_matrix((len(b),1))]),
                       hstack([R,csr_matrix((n,n)),-csr_matrix(K[:,None])])],format='csr')
            bb=np.r_[b,-own];ee=hstack([E,csr_matrix((E.shape[0],1))],format='csr')
            sol=solve(np.r_[np.zeros(m+n),1.],aa,bb,ee,eq)
            peak=float(sol.x[-1])
            aa=vstack([A,hstack([R,csr_matrix((n,n))])],format='csr')
            bb=np.r_[b,peak*K-own]
            sol=solve(np.r_[cost,np.zeros(n)],aa,bb,E,eq)
        elif name=='nearest_cost':
            sol=solve(np.r_[cost,np.zeros(n)],A,b,E,eq)
        else:
            rank=a.canonical_rank[dst].astype(float)/max(1,n-1)
            if name=='high_id_probe':rank=-rank
            c=np.r_[rank,np.zeros(n)]
            sol=solve(c,A,b,E,eq);objective=float(c@sol.x)
            ee=vstack([E,csr_matrix(c[None,:])],format='csr')
            sol=solve(np.r_[cost,np.zeros(n)],A,b,ee,np.r_[eq,objective])
        p=evaluate_flow(a,failed,D,src,dst,sol.x[:m],name)
        p.excess_optimum=optimum;p.peak_util_optimum=peak
        if np.max(abs(p.q-q))>1e-5 or abs(p.excess-optimum)>1e-5:
            raise AssertionError('Matched constraints violated')
        plans[name]=p
    return plans,baseline


def intervene(engine, plan, continuation_cache=None):
    """Only the first allocation differs; use the common LP after its update."""
    initial=engine.initial
    cause=engine.classify(initial,plan.load/engine.a.nominal_capacity)
    start=initial|cause['new']
    key=(engine.cache_namespace,start)
    if continuation_cache is None:continuation_cache={}
    if key not in continuation_cache:
        failed,metrics,iterations=engine.closure(start)
        continuation_cache[key]=(failed,dict(metrics),iterations,list(engine.cascade_logs[start]))
    failed,metrics,iterations,logs=continuation_cache[key]
    return dict(first_new=sorted(cause['new']),first_overload_only=sorted(cause['overload_only']),
        first_risk_only=sorted(cause['risk_only']),first_both=sorted(cause['both']),
        continuation_start=sorted(start),final_failed=sorted(failed),
        final_service=float(metrics['Service_Rate']),final_served=float(metrics['Served_Load']),
        final_assigned=float(metrics['Attempted_Assigned_Load']),
        continuation_iterations=iterations,continuation_logs=logs)
