"""Same-information policy comparison; see 00_design.md for the fixed design."""
from collections import Counter
from functools import lru_cache
from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import time

OUT = Path(__file__).resolve().parent
PILOT = OUT.parent / "wait_divert_pilot_20261022"
sys.path.insert(0, str(PILOT / "code"))
from pilot import Config, step, INF
import numpy as np
import scipy
from scipy.optimize import milp, Bounds, LinearConstraint


def dump(value, name):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(rows, name):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def history(r, cap, t, f):
    return tuple(r <= u for u in range(t + 1)), cap[:min(len(cap), t + f + 1)]


@lru_cache(None)
def minimum_action(t, state, K, calendars, plan_r, lag):
    """Least current dispatch with a common feasible continuation.

    After the current slot, all still-necessary external units are sent at t+1.
    No scenario's actual future calendar is selected as if it were observed.
    """
    H = len(calendars[0]); left = state[0]
    for y in range(left + 1):
        valid = True; future_y = None
        for cap in calendars:
            cfg = Config(H, left, K, cap, (plan_r,), lag)
            s, _, _ = step(cfg, t, *state, plan_r <= t, y)
            q = max(0, s[0] - K * max(0, H - max(t + 1, plan_r)))
            if future_y is None:
                future_y = q
            assert future_y == q
            for u in range(t + 1, H):
                s, _, _ = step(cfg, u, *s, plan_r <= u, q if u == t + 1 else 0)
            if sum(s[:2]) + sum(s[2]) != 0:
                valid = False
                break
        if valid:
            return y, future_y or 0
    return None


def policy_run(x, K, calendars, rs, lag, f, threshold=None, traces=False):
    H = len(calendars[0]); total = worst = 0
    seen = {}; paths = []
    for r, cap in product(rs, calendars):
        cfg = Config(H, x, K, cap, (r,), lag)
        state = (x, 0, (0,) * lag); sent = completed = 0; records = []
        for t in range(H):
            observed = history(r, cap, t, f)
            r_support = tuple(z for z in rs if tuple(z <= u for u in range(t + 1)) == observed[0])
            c_support = tuple(c for c in calendars if c[:min(H, t + f + 1)] == observed[1])
            gated = threshold is not None and r > t and max(r_support) > threshold
            plan_r = H if gated else max(r_support)
            action = minimum_action(t, state, K, c_support, plan_r, lag)
            key = (t, observed)
            if key in seen:
                assert seen[key] == (state, action), "Future information leaked into actions"
            seen[key] = state, action
            if action is None:
                return dict(feasible=False, total=None, worst=None, paths=paths,
                            failure=dict(R=r, calendar=cap, t=t, state=state, gated=gated))
            y = action[0]
            if traces:
                records.append(dict(t=t, state=state, dispatch=y, gated=gated,
                                    plan_recovery=plan_r, consistent_calendars=c_support,
                                    recovery_support=r_support, visible_calendar=observed[1]))
            state, local, remote = step(cfg, t, *state, r <= t, y)
            sent += y; completed += local + remote
            assert completed + state[0] + state[1] + sum(state[2]) == x
        assert completed == x
        total += sent; worst = max(worst, sent)
        if traces:
            paths.append(dict(R=r, calendar=cap, sent=sent, records=records))
    return dict(feasible=True, total=total, worst=worst, paths=paths,
                observed_histories=len(seen))


def one_epoch(x, K, calendars, rs, lag, f):
    """Optimal fixed epoch per INITIAL information set; quantity adapts at epoch."""
    H = len(calendars[0])
    scenarios = tuple(product(rs, calendars)); roots = {}
    for s in scenarios:
        roots.setdefault(history(*s, 0, f), []).append(s)
    roots_result = []; total = 0
    for initial_history, group in roots.items():
        candidates = []
        for epoch in range(H):
            partitions = {}
            for r, cap in group:
                partitions.setdefault(history(r, cap, epoch, f), []).append((r, cap))
            actions = []; cost = 0; possible = True
            for observed, subgroup in partitions.items():
                best_q = None
                for q in range(x + 1):
                    valid = True
                    for r, cap in subgroup:
                        state = (x, 0, (0,) * lag)
                        cfg = Config(H, x, K, cap, (r,), lag)
                        for t in range(H):
                            y = q if t == epoch else 0
                            if y > state[0]:
                                valid = False; break
                            state, _, _ = step(cfg, t, *state, r <= t, y)
                        if not valid or state[0] + state[1] + sum(state[2]):
                            valid = False; break
                    if valid:
                        best_q = q; break
                if best_q is None:
                    possible = False; break
                cost += len(subgroup) * best_q
                actions.append(dict(history=observed, quantity=best_q, scenarios=subgroup))
            if possible:
                candidates.append(dict(total=cost, epoch=epoch, actions=actions))
        if not candidates:
            return None
        best = min(candidates, key=lambda z: (z["total"], -z["epoch"]))
        total += best["total"]
        roots_result.append(dict(initial_history=initial_history, scenarios=group, **best))
    return dict(total=total, roots=roots_result)


def flow_program(scenarios, K, lag, t, state, f=0, open_loop=False, epoch=None):
    """Independent inventory balances, no simulation step or feasibility rule used.

    open_loop=True: all future dispatches common across capacity calendars,
    with lexicographic total-dispatch/current-dispatch objective.
    Otherwise: usual observation-based coupling, optionally one allowed epoch.
    """
    H = len(scenarios[0][1]); ntime = H - t; n = len(scenarios) * 5 * ntime
    def ix(s, kind, u):
        return s * 5 * ntime + kind * ntime + u - t
    eq = []; rhs = []; obj = np.zeros(n); upper = np.full(n, np.inf)
    left, queue, pipe = state
    for s, (r, cap) in enumerate(scenarios):
        for u in range(t, H):
            obj[ix(s, 0, u)] = (left + 1) if open_loop else 1
            if open_loop and u == t:
                obj[ix(s, 0, u)] += 1
            upper[ix(s, 0, u)] = left if u + lag < H and (epoch is None or u == epoch) else 0
            upper[ix(s, 1, u)] = K if r <= u else 0
            upper[ix(s, 2, u)] = cap[u]
            if u == H - 1:
                upper[ix(s, 3, u)] = upper[ix(s, 4, u)] = 0
            a = np.zeros(n)
            a[ix(s, 0, u)] = a[ix(s, 1, u)] = a[ix(s, 3, u)] = 1
            if u > t:
                a[ix(s, 3, u - 1)] = -1
            eq.append(a); rhs.append(left if u == t else 0)
            a = np.zeros(n)
            a[ix(s, 2, u)] = a[ix(s, 4, u)] = 1
            if u > t:
                a[ix(s, 4, u - 1)] = -1
            if u - lag >= t:
                a[ix(s, 0, u - lag)] = -1
            arriving = pipe[u - t] if u - t < lag else 0
            eq.append(a); rhs.append(arriving + (queue if u == t else 0))
    for u in range(t, H):
        groups = {}
        for s, (r, cap) in enumerate(scenarios):
            key = "common" if open_loop else history(r, cap, u, f)
            groups.setdefault(key, []).append(s)
        for group in groups.values():
            for s in group[1:]:
                a = np.zeros(n); a[ix(s, 0, u)] = 1; a[ix(group[0], 0, u)] = -1
                eq.append(a); rhs.append(0)
    A = np.asarray(eq); b = np.asarray(rhs)
    result = milp(obj, integrality=np.ones(n), bounds=Bounds(np.zeros(n), upper),
                  constraints=LinearConstraint(A, b, b), options={"time_limit": 30})
    if result.status == 2:
        return None
    assert result.success, result.message
    assert np.max(np.abs(A @ result.x - b)) < 1e-6
    assert np.max(np.abs(result.x - np.round(result.x))) < 1e-6
    return dict(total=int(round(sum(result.x[ix(s, 0, u)]
                                   for s in range(len(scenarios)) for u in range(t, H)))),
                first=int(round(result.x[ix(0, 0, t)])))


def main():
    started = time.time()
    source = PILOT / "uncertain_capacity_grid.csv"
    grid = list(csv.DictReader(source.open(encoding="utf-8-sig")))
    good = [r for r in grid if r["feasible"] == "True"]
    # Select audits before inspecting this comparison's outputs.
    audit_ids = {good[round(i * (len(good) - 1) / 7)]["id"] for i in range(8)}
    dump(dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              configurations=len(grid), forecast_horizons=[0, 1, 2, 3],
              audit_grid_ids=sorted(audit_ids, key=int), python=platform.python_version(),
              numpy=np.__version__, scipy=scipy.__version__,
              design_sha256=hashlib.sha256((OUT / "00_design.md").read_bytes()).hexdigest(),
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
         "run_metadata.json")
    rows = []; thresholds = []; checks = []; representatives = {}
    checked_states = set()
    for j, row in enumerate(grid):
        pair = tuple(tuple(c) for c in json.loads(row["calendars"]))
        rs = tuple(json.loads(row["recoveries"]))
        x = int(row["batch"]); lag = int(row["lag"]); H = len(pair[0])
        expected = json.loads(row["forecast_sums"])
        for f in range(H):
            item = dict(id=int(row["id"]), f=f, H=H, lag=lag, batch=x,
                        recoveries=json.dumps(rs), calendars=json.dumps(pair),
                        feasible=row["feasible"] == "True", exact_sum=expected[f],
                        rule_sum=None, rule_worst=None, single_sum=None, single_gap=None,
                        best_gated_sum=None, optimal_deltas=None)
            if not item["feasible"]:
                rows.append(item); continue
            rule = policy_run(x, 1, pair, rs, lag, f, traces=row["id"] in audit_ids)
            assert rule["feasible"]
            # Record discrepancies instead of redefining the candidate if one occurs.
            single = one_epoch(x, 1, pair, rs, lag, f)
            assert single is not None and single["total"] >= expected[f]
            item.update(rule_sum=rule["total"], rule_worst=rule["worst"],
                        single_sum=single["total"], single_gap=single["total"] - expected[f])
            gated_results = []
            for delta in range(H + 1):
                run = policy_run(x, 1, pair, rs, lag, f, threshold=delta)
                if run["feasible"]:
                    assert run["total"] >= expected[f]
                    gated_results.append((run["total"], delta))
                thresholds.append(dict(id=item["id"], f=f, delta=delta,
                                       delta_over_H=delta / H, feasible=run["feasible"],
                                       total=run["total"], worst=run["worst"],
                                       exact_sum=expected[f]))
            best = min(z[0] for z in gated_results)
            assert best <= rule["total"]  # delta=H includes the ungated rule.
            item.update(best_gated_sum=best,
                        optimal_deltas=json.dumps([d for v, d in gated_results if v == best]))
            rows.append(item)
            if row["id"] in audit_ids:
                for root in single["roots"]:
                    values = [flow_program(root["scenarios"], 1, lag, 0, (x, 0, (0,) * lag),
                                           f=f, epoch=e) for e in range(H)]
                    opt = min(v["total"] for v in values if v is not None)
                    assert opt == root["total"]
                    checks.append(dict(kind="single_epoch_optimum", id=item["id"], f=f,
                                       t=0, expected=opt, actual=root["total"], passed=True))
                for path in rule["paths"]:
                    for rec in path["records"]:
                        for gated in (False, True):
                            # Gating is only meaningful while source is unrecovered.
                            if gated and path["R"] <= rec["t"]:
                                continue
                            plan_r = H if gated else rec["plan_recovery"]
                            state = (rec["state"][0], rec["state"][1], tuple(rec["state"][2]))
                            caps = tuple(rec["consistent_calendars"])
                            key = (rec["t"], state, plan_r, caps, lag)
                            if key in checked_states:
                                continue
                            checked_states.add(key)
                            action = minimum_action(rec["t"], state, 1, caps, plan_r, lag)
                            ip = flow_program(tuple((plan_r, c) for c in caps), 1, lag,
                                              rec["t"], state, open_loop=True)
                            assert (action is None) == (ip is None)
                            if action is not None:
                                assert action[0] == ip["first"]
                                assert (action[0] + action[1]) * len(caps) == ip["total"]
                            checks.append(dict(kind="rolling_flow_gate" if gated else "rolling_flow",
                                               id=item["id"], f=f, t=rec["t"],
                                               expected=None if ip is None else ip["first"],
                                               actual=None if action is None else action[0], passed=True))
            if item["single_gap"] > 0 and f not in representatives:
                representatives[f] = dict(config=item, single=single,
                                           rule=policy_run(x, 1, pair, rs, lag, f, traces=True))
        if j and j % 400 == 0:
            print(json.dumps({"processed_configurations": j}), flush=True)
    write_rows(rows, "comparison_grid.csv")
    write_rows(thresholds, "threshold_grid.csv")
    write_rows(checks, "independent_flow_checks.csv")
    dump(representatives, "representative_paths.json")
    usable = [r for r in rows if r["feasible"]]
    by_f = {}
    for f in range(4):
        subset = [r for r in usable if r["f"] == f]
        positive = [r for r in subset if r["single_gap"] > 0]
        by_f[f] = dict(feasible=len(subset), single_strict_gap=len(positive),
                       sum_exact=sum(r["exact_sum"] for r in subset),
                       sum_single=sum(r["single_sum"] for r in subset),
                       max_single_gap=max(r["single_gap"] for r in subset),
                       rule_dp_mismatches=sum(r["rule_sum"] != r["exact_sum"] for r in subset),
                       best_gated_dp_mismatches=sum(r["best_gated_sum"] != r["exact_sum"] for r in subset))
    threshold_summary = []
    for f, delta in product(range(4), range(5)):
        subset = [r for r in thresholds if r["f"] == f and r["delta"] == delta]
        feasible = [r for r in subset if r["feasible"]]
        threshold_summary.append(dict(f=f, delta=delta, delta_over_H=delta / 4,
                                      feasible=len(feasible), no_guaranteed_plan=len(subset)-len(feasible),
                                      worse_than_dp=sum(r["total"] > r["exact_sum"] for r in feasible)))
    summary = dict(configurations=len(grid), information_configurations=len(rows),
                   feasible_information_configurations=len(usable),
                   rule_dp_mismatches=sum(r["rule_sum"] != r["exact_sum"] for r in usable),
                   best_gated_dp_mismatches=sum(r["best_gated_sum"] != r["exact_sum"] for r in usable),
                   independent_checks=len(checks), check_kinds=dict(Counter(r["kind"] for r in checks)),
                   all_independent_checks_passed=True, by_forecast=by_f,
                   thresholds=threshold_summary, elapsed_seconds=round(time.time()-started, 3),
                   empirical_validation=False, literature_algorithm_reproduced=False,
                   novel_algorithm_established=False)
    dump(summary, "summary.json")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
