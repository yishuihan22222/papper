"""Additional checks motivated by a specific concern, not a new outcome search."""
from dataclasses import asdict
from itertools import product
import csv
import json
from pathlib import Path
from pilot import Config, solver, independent_program, OUT, INF, dump, write_csv


def main():
    grid=list(csv.DictReader((OUT/"grid.csv").open(encoding="utf-8-sig")))
    initial_errors=[]; worst_errors=[]
    for row in grid:
        if row["feasible"]!="True":
            continue
        x=int(row["batch"]);K=int(row["local_rate"])
        rs=json.loads(row["recoveries"]);cap=json.loads(row["capacities"])
        lag=int(row["lag"]);H=4
        q=max(0,x-K*max(0,H-max(rs)))
        now=max(0,q-sum(cap[lag+1:]))
        if q!=int(row["dp_worst"]):
            worst_errors.append({"id":row["id"],"formula":q,"dp":row["dp_worst"]})
        if now!=int(row["first_dispatch"]):
            initial_errors.append({"id":row["id"],"formula":now,"dp":row["first_dispatch"]})
    # Test per-slot dispatch limits, absent from the first grid.
    limited=[]; matched=0
    for cap,x,limit in product(product(range(3),repeat=4),(3,5),(1,2)):
        cfg=Config(4,x,1,cap,(1,2,3,4),1,limit)
        initial=(0,x,0,(0,),cfg.recoveries)
        v=solver(cfg)(*initial);s=solver(cfg,"safe_min")(*initial)
        if v[0]<INF:
            matched+=s[1]==v[1]
        limited.append(dict(capacities=json.dumps(cap),batch=x,limit=limit,
                            feasible=v[0]<INF,optimal_sum=v[1] if v[0]<INF else "",
                            safe_sum=s[1] if s[0]<INF else ""))
    write_csv(limited,"dispatch_limit_ablation.csv")
    # Independent end-to-end checks of pipeline timing including infeasibility.
    timing=[]
    for lag,cap,x in ((0,(2,0,0),2),(1,(2,0,0),2),(1,(0,2,0),2),
                      (2,(0,2,0),2),(2,(0,0,2),2),(3,(0,0,2),2)):
        cfg=Config(3,x,1,cap,(3,),lag)
        v=solver(cfg)(0,x,0,(0,)*lag,(3,))
        lp=independent_program(cfg)
        assert lp==(None if v[0]>=INF else v[1])
        timing.append(dict(config=asdict(cfg),expected_feasible=lag in (0,1,2) and cap[lag]>0,
                           solved_feasible=lp is not None))
        assert timing[-1]["expected_feasible"]==timing[-1]["solved_feasible"]
    # Hand-check the algebra in the two-clock local note: derivative -1/K.
    note_counterexamples=dict(
        final_batch_time={"t":1,"tau":1,"x":4,"K":1,"y0_finish":6,"y1_finish":5,
                          "derivative_wrt_y":-1},
        boundary={"H":10,"tau":2,"x":4,"beta":1,
                  "zero_advance_latest_wait":4,"all_advance_boundary":8,
                  "point":"H-tau-x/beta makes y_min zero, not x."},
        streaming_rate={"m":2,"K":2,"harmonic":1,"pipeline_steady_rate":2,
                        "point":"Harmonic rate requires no overlap across the entire batch."})
    result=dict(initial_formula_checks=sum(r["feasible"]=="True" for r in grid),
                initial_formula_mismatches=initial_errors,
                minimax_formula_mismatches=worst_errors,
                finite_dispatch_cases=len(limited),
                finite_dispatch_feasible=sum(r["feasible"] for r in limited),
                finite_dispatch_safe_matches=matched,
                timing_checks=timing,note_counterexamples=note_counterexamples)
    dump(result,"structural_checks.json")
    print(json.dumps({k:v for k,v in result.items() if k!="timing_checks"}))


if __name__=="__main__":
    main()
