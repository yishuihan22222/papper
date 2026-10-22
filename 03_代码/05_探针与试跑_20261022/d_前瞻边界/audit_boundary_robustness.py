"""Audit whether the lookahead counterexample needs extreme load/recovery assumptions."""
import csv
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "wait_divert_pilot_20261022" / "code"))
from uncertain_capacity import solve_uncertain, independent_scenario_milp


def main():
    rows = []
    for H in range(3, 7):
        for lag in range(1, H - 1):
            # Both families hold quantities and capacity fixed as H grows.
            a = tuple(1 if t == lag else 0 for t in range(H))
            b = tuple(1 if t in (lag, H - 1) else 0 for t in range(H))
            for kind, x, rs in (("fixed_unit_load", 1, (1, H)),
                                ("both_recover_before_deadline", 2, (1, H - 1))):
                for f in range(H):
                    dp = solve_uncertain(x, 1, (a, b), rs, lag, f)[3]
                    ip = independent_scenario_milp(x, 1, (a, b), rs, lag, f)
                    expected = 3 if f == H - 1 else 4
                    assert dp[1] == ip == expected, (kind, H, lag, f, dp, ip)
                    assert dp[0] == 1
                    rows.append(dict(kind=kind, H=H, lag=lag, x=x, K=1, forecast=f,
                                     recovery_support=json.dumps(rs),
                                     calendars=json.dumps([a, b]),
                                     expected=expected, dp_sum=dp[1], milp_sum=ip,
                                     worst_diversion=dp[0], passed=True))
    # A separate diagnostic: all receiving slots reachable from t=0 have
    # strictly positive capacity; both possible restorations precede H.
    pair = ((0, 3, 1, 1), (0, 3, 1, 3))
    for f in range(4):
        dp = solve_uncertain(6, 2, pair, (1, 3), 1, f)[3]
        ip = independent_scenario_milp(6, 2, pair, (1, 3), 1, f)
        expected = 10 if f == 3 else 12
        assert dp[1] == ip == expected, (f, dp, ip, expected)
        rows.append(dict(kind="positive_capacity_every_reachable_slot", H=4,
                         lag=1, x=6, K=2, forecast=f, recovery_support="[1, 3]",
                         calendars=json.dumps(pair), expected=expected,
                         dp_sum=dp[1], milp_sum=ip, worst_diversion=dp[0], passed=True))
    with (OUT / "boundary_robustness_checks.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(additional_independent_checks=len(rows), all_passed=True,
                   fixed_unit_load_checks=50, before_deadline_recovery_checks=50,
                   positive_receiving_capacity_checks=4,
                   targeted_structural_diagnostics=True,
                   empirical_frequency_estimated=False)
    (OUT / "boundary_robustness_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
