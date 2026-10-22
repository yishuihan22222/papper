"""Replace the feasibility oracle with a direct residual-service calculation."""
from functools import lru_cache
import csv
import json
from pilot import Config, OUT, solver, trajectory, step, dump


def residual_feasible(cfg, t, x, queue, pipe, latest_r):
    # Already committed work has priority. All current in-transit work arrives
    # strictly before new dispatch at t would arrive (t+lag).
    earliest_new_arrival=t+cfg.lag
    spare=0
    for u in range(t,cfg.horizon):
        incoming=pipe[0] if cfg.lag else 0
        pipe=pipe[1:]+(0,) if cfg.lag else ()
        queue+=incoming
        used=min(queue,cfg.capacities[u])
        queue-=used
        if u>=earliest_new_arrival:
            spare+=cfg.capacities[u]-used
    if queue+sum(pipe):
        return False
    local_remaining=cfg.local_rate*max(0,cfg.horizon-max(t,latest_r))
    external_needed=max(0,x-local_remaining)
    return external_needed<=spare


def analytic_rule(cfg):
    assert cfg.dispatch_limit>=cfg.batch
    @lru_cache(None)
    def rule(t,x,queue,pipe,support):
        if t==cfg.horizon:
            return 0,0,0 if x+queue+sum(pipe)==0 else -1
        latest=max(support)
        for y in range(x+1):
            next_state,_,_=step(cfg,t,x,queue,pipe,latest<=t,y)
            if residual_feasible(cfg,t+1,*next_state,latest):
                return 0,0,y
        return 0,0,-1
    return rule


def main():
    rows=list(csv.DictReader((OUT/"grid.csv").open(encoding="utf-8-sig")))
    path_checks=0; mismatches=[]
    for row in rows:
        cfg=Config(4,int(row["batch"]),int(row["local_rate"]),
                   tuple(json.loads(row["capacities"])),
                   tuple(json.loads(row["recoveries"])),int(row["lag"]))
        ar=analytic_rule(cfg)
        init=(0,cfg.batch,0,(0,)*cfg.lag,cfg.recoveries)
        assert (ar(*init)[2]>=0)==(row["feasible"]=="True")
        if row["feasible"]!="True":
            continue
        dp=solver(cfg)
        for r in cfg.recoveries:
            actual=trajectory(cfg,r,ar)
            expected=trajectory(cfg,r,dp)
            path_checks+=1
            if not actual["feasible"] or actual["sent"]!=expected["sent"]:
                mismatches.append(dict(id=row["id"],R=r,rule=actual,dp=expected))
    result=dict(configurations=len(rows),scenario_paths=path_checks,
                mismatches=len(mismatches),
                full_dp_used_in_rule=False,
                applicability="Known receiver service calendar, one destination, common deadline, homogeneous units, no binding dispatch limit",
                mismatching_cases=mismatches)
    dump(result,"analytic_rule_checks.json")
    print(json.dumps({k:v for k,v in result.items() if k!="mismatching_cases"}))


if __name__=="__main__":
    main()
