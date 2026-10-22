"""Exact finite-scenario pilot; all quantities are uncalibrated illustrative units."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / ".runtime_packages"))
import numpy as np
import scipy
from scipy.optimize import linprog, milp, Bounds, LinearConstraint

OUT = Path(__file__).resolve().parents[1]
INF = 10**9


@dataclass(frozen=True)
class Config:
    horizon: int
    batch: int
    local_rate: int
    capacities: tuple[int, ...]
    recoveries: tuple[int, ...]
    lag: int = 1
    dispatch_limit: int = 100

    def __post_init__(self):
        assert len(self.capacities) == self.horizon
        assert self.recoveries and min(self.recoveries) >= 1
        assert self.lag >= 0 and self.dispatch_limit > 0
        assert min(self.capacities) >= 0 and self.local_rate > 0


def split_observations(support, t):
    """Binary signal histories: first recovery time, or not yet recovered."""
    groups = {}
    for r in support:
        key = r if r <= t else None
        groups.setdefault(key, []).append(r)
    return tuple(tuple(v) for v in groups.values())


def step(cfg, t, x, queue, pipe, recovered, y):
    assert 0 <= y <= min(x, cfg.dispatch_limit)
    assert len(pipe) == cfg.lag
    arriving = pipe[0] if cfg.lag else y
    next_pipe = pipe[1:] + (y,) if cfg.lag else ()
    local_done = min(cfg.local_rate, x-y) if recovered else 0
    remote_done = min(cfg.capacities[t], queue+arriving)
    state = x-y-local_done, queue+arriving-remote_done, next_pipe
    assert min(state[0], state[1], *state[2], 0) >= 0
    assert x+queue+sum(pipe) == sum(state[:2])+sum(next_pipe)+local_done+remote_done
    return state, local_done, remote_done


def solver(cfg, mode="sum"):
    """Exact nonanticipative DP. sum=equal-scenario sum, worst=minimax."""
    @lru_cache(None)
    def value(t, x, queue, pipe, support):
        if t == cfg.horizon:
            return (0, 0, 0) if x+queue+sum(pipe) == 0 else (INF, INF, -1)
        flags = {r <= t for r in support}
        assert len(flags) == 1
        recovered = next(iter(flags))
        options = []
        for y in range(min(x, cfg.dispatch_limit)+1):
            nxt, _, _ = step(cfg,t,x,queue,pipe,recovered,y)
            children = [value(t+1,*nxt,g) for g in split_observations(support,t+1)]
            if any(c[0] >= INF for c in children):
                continue
            worst = y + max(c[0] for c in children)
            total = len(support)*y + sum(c[1] for c in children)
            # Minimum y is a reproducible latest-dispatch tie-break.
            key = (total,worst,y) if mode=="sum" else (worst,total,y)
            if mode=="safe_min":
                key = (y,total,worst)
            options.append((key,worst,total,y))
        if not options:
            return INF, INF, -1
        _, worst, total, y = min(options)
        return worst, total, y
    return value


def trajectory(cfg, r, value=None, kind="dp", dispatch_time=0, fixed_y=0):
    x,queue,pipe=cfg.batch,0,(0,)*cfg.lag
    support=cfg.recoveries
    sent=done=0
    rows=[]
    for t in range(cfg.horizon):
        support=next(g for g in split_observations(support,t) if r in g)
        if kind=="dp":
            _,_,y=value(t,x,queue,pipe,support)
            if y<0:
                return {"feasible":False,"sent":sent,"rows":rows}
        elif kind=="all":
            y=min(x,cfg.dispatch_limit) if t==0 else 0
        elif kind=="single":
            y=min(x,fixed_y,cfg.dispatch_limit) if t==dispatch_time else 0
        elif kind=="no_info":
            y=min(x, value[t], cfg.dispatch_limit)
        else:
            raise ValueError(kind)
        prev=(x,queue,pipe)
        (x,queue,pipe),local,remote=step(cfg,t,x,queue,pipe,r<=t,y)
        sent+=y
        done+=local+remote
        assert done+x+queue+sum(pipe)==cfg.batch
        rows.append(dict(t=t,R=r,recovered=r<=t,dispatch=y,local=local,
                         remote=remote,source_left=x,receiver_queue=queue,
                         pipeline=list(pipe),completed=done,
                         x_before=prev[0],queue_before=prev[1]))
    return {"feasible":done==cfg.batch,"sent":sent,"rows":rows}


def single_action(cfg):
    """Strong single-epoch baseline: choose epoch, adapt amount to observed history."""
    candidates=[]
    for t in range(cfg.horizon):
        total=0
        actions={}
        feasible=True
        for group in split_observations(cfg.recoveries,t):
            group_best=None
            for q in range(min(cfg.batch,cfg.dispatch_limit)+1):
                runs=[trajectory(cfg,r,kind="single",dispatch_time=t,fixed_y=q) for r in group]
                if all(z["feasible"] for z in runs):
                    item=(sum(z["sent"] for z in runs),q)
                    if group_best is None or item<group_best:
                        group_best=item
            if group_best is None:
                feasible=False
                break
            total+=group_best[0]
            actions[str(group)]=group_best[1]
        if feasible:
            candidates.append((total,t,actions))
    return min(candidates,key=lambda z:(z[0],-z[1])) if candidates else None


def independent_program(cfg, recoveries=None, no_information=False):
    """Independent flow balances: source stock, receiver stock, dispatch and service.

    No call to DP transition. Scenario coupling equates dispatch on common
    recovery histories. Single scenario uses continuous LP; multi uses MILP
    to match DP's discrete quantity units, without assuming integrality of NA LP.
    """
    rs=tuple(recoveries or cfg.recoveries)
    H=cfg.horizon
    n=len(rs)*5*H
    def at(s,kind,t):
        return s*5*H+kind*H+t
    # kinds: dispatch, local_service, remote_service, source_stock, remote_stock
    cost=np.zeros(n)
    low=np.zeros(n)
    high=np.full(n,np.inf)
    eq=[]; rhs=[]
    for s,r in enumerate(rs):
        for t in range(H):
            cost[at(s,0,t)]=1
            high[at(s,0,t)]=min(cfg.batch,cfg.dispatch_limit) if t+cfg.lag<H else 0
            high[at(s,1,t)]=cfg.local_rate if t>=r else 0
            high[at(s,2,t)]=cfg.capacities[t]
            if t==H-1:
                high[at(s,3,t)]=high[at(s,4,t)]=0
            a=np.zeros(n)
            for kind in (0,1,3):
                a[at(s,kind,t)]=1
            if t:
                a[at(s,3,t-1)]=-1
            eq.append(a); rhs.append(cfg.batch if t==0 else 0)
            a=np.zeros(n)
            a[at(s,2,t)]=a[at(s,4,t)]=1
            if t:
                a[at(s,4,t-1)]=-1
            if t>=cfg.lag:
                a[at(s,0,t-cfg.lag)]=-1
            eq.append(a); rhs.append(0)
    for t in range(H):
        groups={}
        for s,r in enumerate(rs):
            key=None if no_information else (r if r<=t else None)
            groups.setdefault(key,[]).append(s)
        for group in groups.values():
            for s in group[1:]:
                a=np.zeros(n)
                a[at(s,0,t)]=1
                a[at(group[0],0,t)]=-1
                eq.append(a);rhs.append(0)
    A=np.asarray(eq); b=np.asarray(rhs)
    if len(rs)==1:
        result=linprog(cost,A_eq=A,b_eq=b,bounds=list(zip(low,high)),method="highs")
    else:
        result=milp(cost,integrality=np.ones(n),bounds=Bounds(low,high),
                    constraints=LinearConstraint(A,b,b), options={"time_limit":30})
    if result.status==2:
        return None
    assert result.success, result.message
    assert np.max(np.abs(A@result.x-b))<1e-6
    assert np.min(result.x-low)>-1e-6
    assert np.max(result.x-high)<1e-6
    assert np.max(np.abs(result.x-np.round(result.x)))<1e-6
    return int(round(result.fun))


def write_csv(rows,name):
    if not rows:
        return
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def dump(obj,name):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    started=time.time()
    # This rectangular design is fixed before examining the outputs.
    design=dict(horizon=4,batches=[2,3,4,5],local_rates=[1,2],
                capacities=list(product(range(3),repeat=4)),
                recoveries=[[1,2,3],[1,2,3,4]],lags=[0,1,2],
                criteria=["worst","sum","safe_min"],
                no_operational_calibration=True,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(design,"design.json")
    grid=[]; checked=[]; mismatches=[]
    for j,(cap,x,k,rs,lag) in enumerate(product(design["capacities"],design["batches"],
                      design["local_rates"],design["recoveries"],design["lags"])):
        cfg=Config(4,x,k,tuple(cap),tuple(rs),lag)
        initial=(0,x,0,(0,)*lag,tuple(rs))
        opt=solver(cfg,"sum"); v=opt(*initial)
        worst=solver(cfg,"worst")(*initial)
        safe=solver(cfg,"safe_min"); sv=safe(*initial)
        feasible=v[0]<INF
        single=single_action(cfg) if feasible else None
        if feasible:
            for r in rs:
                run=trajectory(cfg,r,opt)
                assert run["feasible"]
                sr=trajectory(cfg,r,safe)
                assert sr["feasible"]
                if sr["sent"] != run["sent"]:
                    mismatches.append(dict(config=asdict(cfg),R=r,optimal=run["sent"],safe=sr["sent"]))
        grid.append(dict(id=j,capacities=json.dumps(cap),batch=x,local_rate=k,
                         recoveries=json.dumps(rs),lag=lag,feasible=feasible,
                         dp_sum=v[1] if feasible else "",dp_worst=worst[0] if feasible else "",
                         first_dispatch=v[2] if feasible else "",
                         safe_sum=sv[1] if feasible else "",
                         single_sum=single[0] if single else "",
                         single_time=single[1] if single else "",
                         sum_safe_gap=sv[1]-v[1] if feasible else "",
                         single_gap=single[0]-v[1] if single else ""))
        # Uniformly spaced deterministic audit subset, including infeasible cases.
        if j%79==0:
            for r in (rs[0],rs[-1]):
                deterministic=Config(4,x,k,tuple(cap),(r,),lag)
                dv=solver(deterministic)(0,x,0,(0,)*lag,(r,))
                lp=independent_program(deterministic)
                expected=None if dv[0]>=INF else dv[1]
                assert lp==expected,(cfg,r,lp,expected)
                checked.append(dict(grid_id=j,kind="deterministic_lp",R=r,value=lp,passed=True))
        if j%157==0:
            ip=independent_program(cfg)
            expected=None if not feasible else v[1]
            assert ip==expected,(cfg,ip,expected)
            checked.append(dict(grid_id=j,kind="scenario_milp",R="all",value=ip,passed=True))
        if j and j%500==0:
            print(json.dumps({"progress_cases":j}),flush=True)
    write_csv(grid,"grid.csv")
    write_csv(checked,"independent_checks.csv")
    dump(mismatches,"safe_min_mismatches.json")
    cases={}
    for label,cap in (("front_loaded",(0,2,2,0,0)),("flat",(0,1,1,1,1)),
                      ("late",(0,0,0,2,2))):
        cfg=Config(5,4,1,cap,(1,2,3,4),1)
        opt=solver(cfg)
        value=opt(0,4,0,(0,),cfg.recoveries)
        cases[label]=dict(config=asdict(cfg),value=value,
                         scenario_milp=independent_program(cfg),
                         no_information_milp=independent_program(cfg,no_information=True),
                         best_single=single_action(cfg),
                         runs={str(r):trajectory(cfg,r,opt) for r in cfg.recoveries},
                         immediate={str(r):trajectory(cfg,r,kind="all") for r in cfg.recoveries})
        assert cases[label]["scenario_milp"]==value[1]
    dump(cases,"case_studies.json")
    feasible_rows=[z for z in grid if z["feasible"]]
    summary=dict(configurations=len(grid),feasible=len(feasible_rows),
                 infeasible=len(grid)-len(feasible_rows),independent_checks=len(checked),
                 safe_min_objective_mismatches=sum(z["sum_safe_gap"]!=0 for z in feasible_rows),
                 safe_min_trajectory_mismatches=len(mismatches),
                 partial_initial_actions=sum(0<z["first_dispatch"]<z["batch"] for z in feasible_rows),
                 single_infeasible_dynamic_feasible=sum(z["single_sum"]=="" for z in feasible_rows),
                 strict_improvement_over_best_single=sum(z["single_gap"]!="" and z["single_gap"]>0 for z in feasible_rows),
                 python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                 elapsed_seconds=round(time.time()-started,3),
                 real_data_experiment=False,
                 novelty_established=False)
    dump(summary,"summary.json")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
