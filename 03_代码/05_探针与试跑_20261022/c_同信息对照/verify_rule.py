"""Verify the closed-form action and individual-scenario optimality."""
from itertools import product
from pathlib import Path
import csv
import json
import numpy as np
from compare import PILOT, OUT, policy_run, history, write_rows, dump
from scipy.optimize import milp, Bounds, LinearConstraint


def formula(t, state, K, calendars, rmax, lag):
    H = len(calendars[0]); left, queue, pipe = state
    needed = max(0, left - K * max(0, H - max(t, rmax)))
    suffix = min(sum(c[t + lag + 1:]) for c in calendars)
    feasible = all(
        (queue if b == t else 0) + sum(v for k, v in enumerate(pipe) if t + k >= b)
        + (needed if t + lag >= b else 0) <= sum(c[b:])
        for c in calendars for b in range(t, H + 1)
    )
    return dict(feasible=feasible, action=max(0, needed - suffix),
                required_external=needed, guaranteed_tail=suffix)


def target_scenario_milp(x, K, pair, rs, lag, f, target):
    """Minimize only one scenario's diversion, while guaranteeing every scenario."""
    scenarios = tuple(product(rs, pair)); H = len(pair[0]); n = len(scenarios)*H*5
    def at(s, k, t):
        return (s*5+k)*H+t
    cost = np.zeros(n); high = np.full(n, np.inf); eq = []; rhs = []
    for s, (r, cap) in enumerate(scenarios):
        for t in range(H):
            cost[at(s, 0, t)] = int(s == target)
            high[at(s, 0, t)] = x if t + lag < H else 0
            high[at(s, 1, t)] = K if t >= r else 0
            high[at(s, 2, t)] = cap[t]
            if t == H-1:
                high[at(s, 3, t)] = high[at(s, 4, t)] = 0
            a = np.zeros(n)
            a[at(s, 0, t)] = a[at(s, 1, t)] = a[at(s, 3, t)] = 1
            if t:
                a[at(s, 3, t-1)] = -1
            eq.append(a); rhs.append(x if t == 0 else 0)
            a = np.zeros(n); a[at(s, 2, t)] = a[at(s, 4, t)] = 1
            if t:
                a[at(s, 4, t-1)] = -1
            if t >= lag:
                a[at(s, 0, t-lag)] = -1
            eq.append(a); rhs.append(0)
    for t in range(H):
        # Built independently from the policy's posterior helper.
        groups = {}
        for s, (r, cap) in enumerate(scenarios):
            key = (r if r <= t else None, tuple(cap[u] for u in range(min(H, t+f+1))))
            groups.setdefault(key, []).append(s)
        for group in groups.values():
            for s in group[1:]:
                a = np.zeros(n); a[at(s, 0, t)] = 1; a[at(group[0], 0, t)] = -1
                eq.append(a); rhs.append(0)
    A = np.asarray(eq); b = np.asarray(rhs)
    result = milp(cost, integrality=np.ones(n), bounds=Bounds(np.zeros(n), high),
                  constraints=LinearConstraint(A, b, b), options={"time_limit": 30})
    assert result.success, result.message
    assert np.max(np.abs(A @ result.x-b)) < 1e-6
    return int(round(result.fun))


def main():
    grid = list(csv.DictReader((PILOT/"uncertain_capacity_grid.csv").open(encoding="utf-8-sig")))
    initial_checks = path_state_checks = 0
    for row in grid:
        pair = tuple(tuple(c) for c in json.loads(row["calendars"]))
        rs = tuple(json.loads(row["recoveries"])); x = int(row["batch"]); lag = int(row["lag"])
        for f in range(4):
            roots = {}
            for cap in pair:
                roots.setdefault(cap[:f+1], []).append(cap)
            feasible = all(formula(0, (x,0,(0,)*lag), 1, tuple(caps), max(rs), lag)["feasible"]
                           for caps in roots.values())
            assert feasible == (row["feasible"] == "True")
            initial_checks += 1
            if not feasible:
                continue
            run = policy_run(x, 1, pair, rs, lag, f, traces=True)
            for path in run["paths"]:
                for rec in path["records"]:
                    result = formula(rec["t"], rec["state"], 1,
                                     rec["consistent_calendars"], max(rec["recovery_support"]), lag)
                    assert result["feasible"]
                    assert result["action"] == rec["dispatch"]
                    path_state_checks += 1
    good = [r for r in grid if r["feasible"] == "True"]
    cases = []
    for idx in (0, len(good)//2, len(good)-1):
        row = good[idx]
        cases.append(dict(label="grid_"+row["id"], x=int(row["batch"]), K=1,
                          pair=tuple(tuple(c) for c in json.loads(row["calendars"])),
                          rs=tuple(json.loads(row["recoveries"])), lag=int(row["lag"])))
    cases.append(dict(label="positive_capacity_K2", x=6, K=2,
                      pair=((0,3,1,1),(0,3,1,3)), rs=(1,3), lag=1))
    checks = []
    for case in cases:
        for f in range(4):
            run = policy_run(case["x"], case["K"], case["pair"], case["rs"], case["lag"], f, traces=True)
            for target, path in enumerate(run["paths"]):
                value = target_scenario_milp(case["x"], case["K"], case["pair"], case["rs"],
                                             case["lag"], f, target)
                assert value == path["sent"], (case, f, target, value, path["sent"])
                checks.append(dict(case=case["label"], f=f, target=target, R=path["R"],
                                   calendar=json.dumps(path["calendar"]), rule_sent=path["sent"],
                                   target_optimum=value, passed=True))
    write_rows(checks, "individual_scenario_checks.csv")
    summary = dict(initial_feasibility_checks=initial_checks,
                   formula_trajectory_state_checks=path_state_checks,
                   individual_scenario_milp_checks=len(checks), all_passed=True,
                   counts_are_not_distinct_real_world_cases=True)
    dump(summary, "formula_verification_summary.json")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
