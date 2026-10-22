"""Second bounded pilot: receiver calendar revealed progressively.

Forecast horizon is an information assumption, not an empirically estimated model.
All recovery/calendar combinations are retained (no assumed correlation).
"""
from functools import lru_cache
from itertools import combinations, product
import json
import time
from pilot import Config, step, solver, INF, dump, write_csv, OUT
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint


def beliefs(scenarios, t, lookahead, H):
    groups={}
    for i,(r,cap) in enumerate(scenarios):
        key=(r if r<=t else None,cap[:min(H,t+lookahead+1)])
        groups.setdefault(key,[]).append(i)
    return tuple(tuple(v) for v in groups.values())


def solve_uncertain(x,K,calendars,rs,lag=1,lookahead=0,mode="sum"):
    H=len(calendars[0])
    scenarios=tuple((r,tuple(c)) for c in calendars for r in rs)
    cfgs=tuple(Config(H,x,K,c,(r,),lag) for r,c in scenarios)
    groups_by_t=[beliefs(scenarios,t,lookahead,H) for t in range(H+1)]

    @lru_cache(None)
    def value(t,left,queue,pipe,support):
        if t==H:
            return (0,0,0) if left+queue+sum(pipe)==0 else (INF,INF,-1)
        representative=support[0]
        cfg=cfgs[representative]
        recovered=scenarios[representative][0]<=t
        assert len({scenarios[i][0]<=t for i in support})==1
        assert len({scenarios[i][1][t] for i in support})==1
        groups=[tuple(i for i in g if i in support) for g in groups_by_t[t+1]]
        groups=[g for g in groups if g]
        candidates=[]
        for y in range(left+1):
            nxt,_,_=step(cfg,t,left,queue,pipe,recovered,y)
            children=[value(t+1,*nxt,g) for g in groups]
            if any(v[0]>=INF for v in children):
                continue
            worst=y+max(v[0] for v in children)
            total=len(support)*y+sum(v[1] for v in children)
            key=(total,worst,y) if mode=="sum" else (y,total,worst)
            candidates.append((key,worst,total,y))
        if not candidates:
            return INF,INF,-1
        _,w,s,y=min(candidates)
        return w,s,y

    roots=groups_by_t[0]
    vals=[value(0,x,0,(0,)*lag,g) for g in roots]
    result=(max(v[0] for v in vals),sum(v[1] for v in vals),
            [v[2] for v in vals])
    return value,scenarios,groups_by_t,result


def independent_scenario_milp(x,K,calendars,rs,lag,lookahead):
    scenarios=tuple((r,tuple(c)) for c in calendars for r in rs)
    H=len(calendars[0]);n=len(scenarios)*5*H
    def at(s,k,t):
        return s*5*H+k*H+t
    eq=[];rhs=[];cost=np.zeros(n);lo=np.zeros(n);hi=np.full(n,np.inf)
    for s,(r,cap) in enumerate(scenarios):
        for t in range(H):
            cost[at(s,0,t)]=1
            hi[at(s,0,t)]=x if t+lag<H else 0
            hi[at(s,1,t)]=K if r<=t else 0
            hi[at(s,2,t)]=cap[t]
            if t==H-1:
                hi[at(s,3,t)]=hi[at(s,4,t)]=0
            a=np.zeros(n)
            a[at(s,0,t)]=a[at(s,1,t)]=a[at(s,3,t)]=1
            if t:
                a[at(s,3,t-1)]=-1
            eq.append(a);rhs.append(x if t==0 else 0)
            a=np.zeros(n)
            a[at(s,2,t)]=a[at(s,4,t)]=1
            if t:
                a[at(s,4,t-1)]=-1
            if t>=lag:
                a[at(s,0,t-lag)]=-1
            eq.append(a);rhs.append(0)
    for t in range(H):
        independent_groups={}
        for s,(r,cap) in enumerate(scenarios):
            history=(tuple(r<=u for u in range(t+1)),
                     tuple(cap[u] for u in range(min(H,t+lookahead+1))))
            independent_groups.setdefault(history,[]).append(s)
        for group in independent_groups.values():
            for s in group[1:]:
                a=np.zeros(n);a[at(s,0,t)]=1;a[at(group[0],0,t)]=-1
                eq.append(a);rhs.append(0)
    A=np.array(eq);b=np.array(rhs)
    result=milp(cost,integrality=np.ones(n),bounds=Bounds(lo,hi),
                constraints=LinearConstraint(A,b,b),options={"time_limit":30})
    if result.status==2:
        return None
    assert result.success,result.message
    assert np.max(np.abs(A@result.x-b))<1e-6
    assert np.max(np.abs(result.x-np.round(result.x)))<1e-6
    return int(round(result.fun))


def trace_uncertain(x,K,calendars,rs,lag,lookahead):
    value,scenarios,groups,result=solve_uncertain(x,K,calendars,rs,lag,lookahead)
    traces=[]
    for i,(r,cap) in enumerate(scenarios):
        cfg=Config(len(cap),x,K,cap,(r,),lag)
        left,queue,pipe=x,0,(0,)*lag
        completed=sent=0;rows=[]
        for t in range(len(cap)):
            support=next(g for g in groups[t] if i in g)
            y=value(t,left,queue,pipe,support)[2]
            assert y>=0
            (left,queue,pipe),local,remote=step(cfg,t,left,queue,pipe,r<=t,y)
            sent+=y;completed+=local+remote
            assert completed+left+queue+sum(pipe)==x
            rows.append(dict(t=t,R_revealed=r if r<=t else None,
                             calendar_prefix=list(cap[:min(len(cap),t+lookahead+1)]),
                             dispatch=y,local=local,remote=remote,
                             local_left=left,remote_queue=queue,pipe=list(pipe)))
        assert completed==x
        traces.append(dict(R=r,calendar=cap,sent=sent,rows=rows))
    return dict(result=result,traces=traces)


def main():
    started=time.time()
    profiles=[(0,)+c for c in product(range(3),repeat=3)]
    design=dict(horizon=4,calendar_pairs=list(combinations(profiles,2)),
                batches=[3,4],local_rate=1,recovery_supports=[[1,2,3],[1,2,3,4]],
                lags=[1,2],forecast_horizons=[0,1,2,3],
                no_calibrated_probabilities=True,cartesian_recovery_capacity_support=True)
    dump(design,"uncertain_capacity_design.json")
    rows=[];checks=[]
    for j,(pair,x,lag,rs) in enumerate(product(design["calendar_pairs"],
                                          design["batches"],design["lags"],
                                          design["recovery_supports"])):
        rs=tuple(rs)
        results=[]
        for f in design["forecast_horizons"]:
            v,sc,groups,result=solve_uncertain(x,1,pair,rs,lag,f)
            results.append(result)
        known=sum(solver(Config(4,x,1,c,rs,lag))(0,x,0,(0,)*lag,rs)[1]
                  for c in pair)
        online=results[0]
        feasible=online[0]<INF
        safe=solve_uncertain(x,1,pair,rs,lag,0,"safe_min")[3]
        if feasible:
            assert results[-1][1]==known
            assert all(results[f][1]>=results[f+1][1] for f in range(3))
            assert safe[0]<INF
            # Removing capacity uncertainty cannot invalidate a feasible policy.
            assert online[1]>=known
        envelope=tuple(min(a,b) for a,b in zip(*pair))
        env=solver(Config(4,x,1,envelope,rs,lag))(0,x,0,(0,)*lag,rs)
        rows.append(dict(id=j,calendars=json.dumps(pair),batch=x,lag=lag,recoveries=json.dumps(rs),
                         feasible=feasible,online_sum=online[1] if feasible else "",
                         known_calendar_sum=known if feasible else "",
                         uncertainty_cost=online[1]-known if feasible else "",
                         safe_gap=safe[1]-online[1] if feasible else "",
                         envelope_feasible=env[0]<INF,
                         forecast_sums=json.dumps([z[1] if z[0]<INF else None for z in results]),
                         forecast_to_match_known=next((f for f,z in enumerate(results)
                             if feasible and z[1]==known),None)))
        if j%109==0:
            ip=independent_scenario_milp(x,1,pair,rs,lag,0)
            assert ip==(online[1] if feasible else None)
            checks.append(dict(id=j,forecast=0,value=ip,passed=True))
        if j and j%300==0:
            print(json.dumps({"uncertain_progress":j}),flush=True)
    write_csv(rows,"uncertain_capacity_grid.csv")
    write_csv(checks,"uncertain_capacity_independent_checks.csv")
    good=[r for r in rows if r["feasible"]]
    example_pair=((0,2,1,0),(0,2,1,2))
    example={str(f):trace_uncertain(3,1,example_pair,(1,2,3,4),1,f) for f in (0,1,2,3)}
    dump(example,"forecast_information_example.json")
    correlated_pair=((0,2,1,0),(0,0,1,2))
    correlation_example=trace_uncertain(3,1,correlated_pair,(1,2,3,4),1,0)
    dump(correlation_example,"nonrectangular_calendar_example.json")
    summary=dict(configurations=len(rows),feasible=len(good),
                 independent_checks=len(checks),
                 positive_uncertainty_cost=sum(r["uncertainty_cost"]>0 for r in good),
                 safe_min_mismatches=sum(r["safe_gap"]!=0 for r in good),
                 conservative_envelope_false_infeasible=sum(not r["envelope_feasible"] for r in good),
                 need_forecast_longer_than_transport_lag=sum(
                     r["forecast_to_match_known"]>r["lag"] for r in good),
                 example_sums={f:v["result"][1] for f,v in example.items()},
                 elapsed_seconds=round(time.time()-started,3),
                 empirical_validation=False,novelty_established=False)
    dump(summary,"uncertain_capacity_summary.json")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
