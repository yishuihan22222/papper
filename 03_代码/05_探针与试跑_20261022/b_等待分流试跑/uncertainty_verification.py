"""Stratified feasible-case verification after uniform checks were mostly infeasible."""
import csv
import json
from uncertain_capacity import solve_uncertain, independent_scenario_milp, trace_uncertain
from pilot import OUT, INF, dump, write_csv


def verify_information(trace, horizon):
    seen={}
    for scenario in trace["traces"]:
        r=scenario["R"]; cap=scenario["calendar"]
        for row in scenario["rows"]:
            t=row["t"]
            key=(t,tuple(r<=u for u in range(t+1)),tuple(row["calendar_prefix"]))
            if key in seen:
                assert seen[key]==row["dispatch"],"Nonanticipativity violated"
            seen[key]=row["dispatch"]
    return len(seen)


def main():
    grid=list(csv.DictReader((OUT/"uncertain_capacity_grid.csv").open(encoding="utf-8-sig")))
    good=[r for r in grid if r["feasible"]=="True"]
    targets=[]
    # Repeat 13 originally uniformly spaced checks using independent history keys.
    for row in grid[::109]:
        targets.append(("uniform",row,0))
    # Twelve evenly spaced feasible cases; validate every forecast horizon.
    positions=[round(i*(len(good)-1)/11) for i in range(12)]
    for i in positions:
        targets.extend(("feasible",good[i],f) for f in range(4))
    checked=[];histories=0
    for kind,row,f in targets:
        pair=tuple(tuple(c) for c in json.loads(row["calendars"]))
        x=int(row["batch"]);lag=int(row["lag"])
        rs=tuple(json.loads(row["recoveries"]))
        dp=solve_uncertain(x,1,pair,rs,lag,f)[3]
        ip=independent_scenario_milp(x,1,pair,rs,lag,f)
        assert ip==(None if dp[0]>=INF else dp[1])
        if ip is not None:
            tr=trace_uncertain(x,1,pair,rs,lag,f)
            histories+=verify_information(tr,4)
        checked.append(dict(kind=kind,id=row["id"],lag=lag,forecast=f,
                            feasible=ip is not None,value=ip,passed=True))
    for pair in (((0,2,1,0),(0,2,1,2)),((0,2,1,0),(0,0,1,2))):
        for f in range(4):
            tr=trace_uncertain(3,1,pair,(1,2,3,4),1,f)
            ip=independent_scenario_milp(3,1,pair,(1,2,3,4),1,f)
            assert ip==tr["result"][1]
            histories+=verify_information(tr,4)
            checked.append(dict(kind="named_example",id=json.dumps(pair),lag=1,
                                forecast=f,feasible=True,value=ip,passed=True))
    write_csv(checked,"uncertain_capacity_extended_checks.csv")
    summary=dict(independent_program_checks=len(checked),
                 feasible_checks=sum(c["feasible"] for c in checked),
                 nonanticipative_history_checks=histories,
                 all_passed=True,shared_dp_transition_in_milp=False,
                 shared_observation_group_function_in_milp=False)
    dump(summary,"uncertainty_verification_summary.json")
    print(json.dumps(summary))


if __name__=="__main__":
    main()
