"""Verify a counterexample family using the existing DP and independent MILP.

Illustrative structural verification only; no estimated operational probabilities.
Run with the same Python/runtime dependencies as wait_divert_pilot_20261022.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
CODE = WORK / "outputs" / "wait_divert_pilot_20261022" / "code"
sys.path.insert(0, str(CODE))
from uncertain_capacity import solve_uncertain, independent_scenario_milp


def main():
    rows = []
    for H in range(3, 7):
        x = H - 1
        for lag in range(1, H - 1):
            a = tuple(x if t == lag else 0 for t in range(H))
            b = tuple(x if t in (lag, H - 1) else 0 for t in range(H))
            for f in range(H):
                expected = (3 if f == H - 1 else 4) * x
                dp = solve_uncertain(x, 1, (a, b), (1, H), lag, f)[3]
                ip = independent_scenario_milp(x, 1, (a, b), (1, H), lag, f)
                assert dp[1] == expected == ip, (H, lag, f, dp, ip, expected)
                assert dp[0] == x
                rows.append(dict(H=H, x=x, K=1, lag=lag, forecast=f,
                                 calendar_a=json.dumps(a), calendar_b=json.dumps(b),
                                 recovery_support=json.dumps([1, H]),
                                 expected_sum=expected, dp_sum=dp[1], milp_sum=ip,
                                 worst_diversion=dp[0], passed=True))
    with (OUT / "lookahead_boundary_checks.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        horizon_lag_configurations=len({(r["H"], r["lag"]) for r in rows}),
        independent_milp_checks=len(rows), all_passed=True,
        checked_horizons=[3, 4, 5, 6],
        proven_family="H >= 3; 1 <= L <= H-2; x=H-1; K=1; R in {1,H}",
        critical_lookahead="H-1", empirical_validation=False,
        novelty_relative_to_literature_established=False,
        objective="sum of diversion over four equally weighted scenarios",
        source_sha256={name: hashlib.sha256((CODE / name).read_bytes()).hexdigest()
                       for name in ("pilot.py", "uncertain_capacity.py")},
    )
    (OUT / "lookahead_boundary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
