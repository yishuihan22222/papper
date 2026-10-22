"""Small independent checks of fixed-state flow claims, not cascade guarantees."""
from pathlib import Path
import hashlib,json,itertools,platform
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'项目结果/dispatch_theory_feasibility_20261018'

def maxflow(d,s,arcs):
    ni=len(d);nj=len(s);sink=ni+nj+1;rows=[];cols=[];values=[]
    for i,v in enumerate(d):rows.append(0);cols.append(1+i);values.append(int(v))
    for j,v in enumerate(s):rows.append(1+ni+j);cols.append(sink);values.append(int(v))
    for i,j in arcs:rows.append(1+i);cols.append(1+ni+j);values.append(int(sum(d)+1))
    g=csr_matrix((np.asarray(values,dtype=np.int64),(rows,cols)),shape=(sink+1,sink+1))
    return int(maximum_flow(g,0,sink).flow_value)

def excess_lp(d,s,u,arcs,q,cost=None):
    m=len(arcs);n=len(s);a=np.zeros((len(d)+2*n,m+n))
    for col,(i,j) in enumerate(arcs):a[i,col]=1;a[len(d)+j,col]=1;a[len(d)+n+j,col]=1
    a[len(d)+n:,m:]=-np.eye(n)
    b=np.r_[d,u,s];eq=np.r_[np.ones(m),np.zeros(n)][None,:]
    obj=np.r_[np.zeros(m),np.ones(n)]
    r=linprog(obj,A_ub=a,b_ub=b,A_eq=eq,b_eq=[q],bounds=(0,None),method='highs')
    if not r.success:raise RuntimeError(r.message)
    opt=float(r.fun)
    if cost is not None:
        r=linprog(np.r_[cost,np.zeros(n)],A_ub=a,b_ub=b,A_eq=np.vstack([eq,obj]),b_eq=[q,opt],bounds=(0,None),method='highs')
        if not r.success:raise RuntimeError(r.message)
    incoming=np.zeros(n)
    for col,(_,j) in enumerate(arcs):incoming[j]+=r.x[col]
    return opt,incoming

def main():
    OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(2026101801);rows=[]
    for case in range(200):
        ni=int(rng.integers(1,6));nj=int(rng.integers(1,6))
        d=rng.integers(0,7,ni);s=rng.integers(0,7,nj);u=s+rng.integers(0,4,nj)
        arcs=list(zip(*np.nonzero(rng.random((ni,nj))<.55)))
        qs=maxflow(d,s,arcs);qe=maxflow(d,u,arcs)
        cuts=[];deficits=[]
        for bits in itertools.product([False,True],repeat=ni):
            subset={i for i,v in enumerate(bits) if v};neighbors={j for i,j in arcs if i in subset}
            dd=sum(int(d[i]) for i in subset);ss=sum(int(s[j]) for j in neighbors)
            cuts.append(int(sum(d))-dd+ss);deficits.append(dd-ss)
        assert qs==min(cuts) and int(sum(d))-qs==max(deficits)
        for q in sorted(set([0.,float(qs),float(qe),float(qe)/2])):
            observed,incoming=excess_lp(d,s,u,arcs,q)
            expected=max(0.,q-qs)
            ok=abs(observed-expected)<1e-8 and abs(np.maximum(incoming-s,0).sum()-expected)<1e-8
            assert ok
            if abs(q-qe)<1e-12:assert abs(np.minimum(incoming,s).sum()-qs)<1e-8
            rows.append(dict(case=case,origins=ni,receivers=nj,arcs=len(arcs),safe_maximum=qs,emergency_maximum=qe,
                             admission=q,excess_lp=observed,excess_certificate=expected,passed=ok))
    arcs=[(0,0),(0,1),(1,1)]
    opt,incoming=excess_lp(np.array([6,6]),np.array([5,5]),np.array([8,8]),arcs,12,cost=np.array([0,1,0]))
    alternative=np.array([5.,7.])
    counterexample=dict(supply=[6,6],safe_receiver_capacity=[5,5],emergency_receiver_capacity=[8,8],
                        allowed_arcs=arcs,arc_cost=[0,1,0],minimum_excess=opt,
                        lexicographic_receiver_load=incoming.tolist(),lexicographic_overloaded=int((incoming>5+1e-8).sum()),
                        feasible_alternative_arc_flow=[5,1,6],alternative_receiver_load=alternative.tolist(),
                        alternative_excess=float(np.maximum(alternative-5,0).sum()),alternative_overloaded=1,
                        conclusion='Minimum total excess, including the cost tie-break, need not minimize overloaded node count.')
    assert np.allclose(incoming,[6,6]) and opt==2 and counterexample['lexicographic_overloaded']==2
    pd.DataFrame(rows).to_csv(OUT/'79a_flow_certificate_checks.csv',index=False,encoding='utf-8-sig')
    (OUT/'79b_overload_count_counterexample.json').write_text(json.dumps(counterexample,ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(completed=True,random_bipartite_instances=200,frontier_checks=len(rows),all_passed=True,
                counterexample_verified=True,python=platform.python_version(),seed=2026101801,
                proof_is_in_separate_note=True,numerical_checks_are_not_proofs=True,production_model_unchanged=True,
                sources={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    result['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.name in ['79a_flow_certificate_checks.csv','79b_overload_count_counterexample.json']}
    (OUT/'79_completion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
