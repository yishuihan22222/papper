"""Independent artifact reload and ledger verification for prototypes94/95."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '项目结果/reversible_blocking_validation_20261018'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    with p.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def dump(d, p):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = [ROOT / '项目结果/dynamic_coupling_decision_20261018/93_final_manifest.json',
                 ROOT / '项目结果/reversible_blocking_prototype_20261018/94_completion.json',
                 ROOT / '项目结果/reversible_blocking_peak_horizon_20261018/95_completion.json']
    preserved = {}
    for p in manifests:
        m = json.loads(p.read_text(encoding='utf-8'))
        assert m['completed']
        for name, h in m['files'].items():
            assert sha(ROOT / name) == h, name
            preserved[name] = h
        preserved[str(p.relative_to(ROOT))] = sha(p)
    folder94 = ROOT / '项目结果/reversible_blocking_prototype_20261018'
    folder95 = ROOT / '项目结果/reversible_blocking_peak_horizon_20261018'
    count_rows = count_cases = 0
    for folder, number, full_policy, bypass_policy in [
            (folder94, '94', 'offline_fifo', 'offline_bypass'),
            (folder95, '95', 'offline_fifo', 'offline_bypass')]:
        rows = read_csv(folder / f'{number}a_policy_results.csv')
        cases = {}
        for r in rows:
            if number == '94':
                total = sum(map(len, json.loads(r['arrivals_json'])))
                assert total == int(r['completed']) + int(r['rejected']) + int(r['terminal_inventory'])
                assert int(r['accepted']) == int(r['completed']) + int(r['terminal_inventory'])
            else:
                peak = r['peak']
                total = 4 if peak.startswith('repeated') else 3 if peak.startswith('mixed') else 2
                assert total == int(r['completed']) + int(r['rejected']) + int(r['terminal_inventory'])
            assert int(r['cost_ticks']) == 10 * (total - int(r['completed'])) + int(r['holding_ticks']) * int(r['waiting_units'])
            if int(r['drain_slots']):
                assert int(r['terminal_inventory']) == 0
            cases.setdefault(r['case_key'], {})[r['policy']] = r
        for c in cases.values():
            exact = int(c[full_policy]['cost_ticks'])
            assert int(c[bypass_policy]['cost_ticks']) <= exact
            assert all(int(r['cost_ticks']) >= exact for p, r in c.items() if p != bypass_policy)
        assert len(rows) == (25000 if number == '94' else 1920)
        assert len(cases) == (5000 if number == '94' else 240)
        count_rows += len(rows)
        count_cases += len(cases)
    # Recompute the reversible recovery witness without importing the simulator.
    tr = read_csv(folder94 / '94f_recovery_witness.csv')
    lane, buffers = [0, 1], [1, 0]
    served = 0
    blocked = []
    for t, r in enumerate(tr):
        assert json.loads(r['lane_before_json']) == lane
        assert [int(r['buffer_A_before']), int(r['buffer_B_before'])] == buffers
        k = [0 if t in (0, 1) else 1, 1]
        complete = [min(buffers[v], k[v]) for v in range(2)]
        buffers = [buffers[v] - complete[v] for v in range(2)]
        while lane and buffers[lane[0]] < 1:
            buffers[lane.pop(0)] += 1
        served += sum(complete)
        assert served + len(lane) + sum(buffers) == 3
        assert json.loads(r['lane_after_json']) == lane
        assert [int(r['buffer_A_after']), int(r['buffer_B_after'])] == buffers
        assert sum(complete) == int(r['completed'])
        blocked.append(bool(lane))
    assert served == 3 and not lane and sum(buffers) == 0
    assert blocked[:2] == [True, True] and not blocked[2]
    clear = read_csv(folder94 / '94g_all_state_clearance.csv')
    assert len(clear) == 88
    for r in clear:
        assert int(r['completed']) == len(json.loads(r['lane_json'])) + int(r['buffer_A']) + int(r['buffer_B'])
    brute = read_csv(folder94 / '94e_independent_action_enumeration.csv')
    assert len(brute) == 500
    assert all(int(r['exact_ticks']) == int(r['independently_ticks']) for r in brute)
    symmetry = read_csv(folder94 / '94d_label_symmetry.csv')
    assert len(symmetry) == 5000
    assert all(int(r['original_ticks']) == int(r['renamed_ticks']) for r in symmetry)
    flows = read_csv(folder94 / '94c_fluid_lp_bounds.csv')
    assert all(float(r['fluid_lp_ticks']) <= int(r['bypass_ticks']) + 1e-7 <= int(r['fifo_ticks']) + 1e-7 for r in flows)
    assert sum(int(r['fifo_ticks']) > int(r['bypass_ticks']) for r in flows) == 120
    windows = read_csv(folder95 / '95d_horizon_summary.csv')
    cleared = {int(r['horizon']): r for r in windows if int(r['drain_slots']) == 6}
    assert [int(cleared[h]['equal_to_offline']) for h in (2, 3, 4, 6)] == [60, 90, 106, 120]
    findings = dict(completed=True, reloaded_policy_rows=count_rows, configurations=count_cases,
                    independent_recovery_replay=True, all_state_clearing_checks=len(clear),
                    action_tree_checks=len(brute), label_symmetry_checks=len(symmetry),
                    prototype_fifo_order_gap_cases=120,
                    longer_horizon_equals_offline={h: int(cleared[h]['equal_to_offline']) for h in (2, 3, 4, 6)},
                    longer_horizon_conditions=120, forecast_assumption='Perfect within-window information',
                    no_endogenous_permanent_failure=True, no_inventory_loss=True,
                    physical_scope='Shared FIFO lane and two processing stations; not calibrated city-wide facilities',
                    manuscript_changed=False, original_models_preserved=True, new_algorithm_claim=False)
    dump(findings, OUT / '96_verification.json')
    extra = [Path(__file__), OUT / '96_verification.json']
    preserved.update({str(p.relative_to(ROOT)): sha(p) for p in extra})
    dump(dict(completed=True, files=preserved, verified_file_count=len(preserved),
              manuscript_changed=False), OUT / '96_final_manifest.json')
    print(json.dumps(findings, ensure_ascii=False))
    print('Verified file references:', len(preserved))


if __name__ == '__main__':
    main()
