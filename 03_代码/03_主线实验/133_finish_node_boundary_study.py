"""Execute frozen boundary cells with reusable allocators and three processes.

Only execution/caching changes relative to 132. The 00_design.json grid, fixed
inputs and every cell's scientific specification are retained unchanged.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from capacity_precision_v5 import formula_precision_nodes
from configured_dual_mechanism_v5 import ConfiguredDispatchAllocator, ConfiguredDualMechanismRecovery
from model_config import ModelConfig
from project_paths import RESULTS_DIR
from public_data_inputs_v1 import sha

core = importlib.import_module('83_dispatch_mainline_core')
study = importlib.import_module('132_node_boundary_and_scope_study')
OUT = study.OUT


def run_cell(job):
    d = json.loads((OUT / '00_design.json').read_text(encoding='utf-8'))
    n, e, _, _ = core.load_study()
    demands = json.loads((OUT / 'frozen_inputs/demands.json').read_text(encoding='utf-8'))
    keep = ~n.node_id.isin(d['excluded']).to_numpy()
    nn, _ = formula_precision_nodes(core.capacity_nodes(n, job['capacity_id']), n)
    scope = job['scope']
    if scope == 'reduced242':
        nn = nn.loc[keep].reset_index(drop=True)
        e = e[e.Source_id.isin(nn.node_id) & e.Target_id.isin(nn.node_id)].reset_index(drop=True)
    rows = []
    for theta in d['thresholds']:
        configs = {arm: ModelConfig.for_arm(arm, theta_risk=theta,
                  theta_source='Frozen historical threshold, not observed') for arm in d['arms']}
        allocators = {arm: ConfiguredDispatchAllocator(nn, e, configs[arm]) for arm in d['arms']}
        states, pressures = {}, {}
        for case in d['cases']:
            for demand in d['demand_names']:
                parts = np.array(demands[demand]['parts'], dtype=float)
                groups = demands[demand]['groups']
                if scope == 'matched_tasks246': parts[:, ~keep] = 0.
                if scope == 'reduced242': parts = parts[:, keep]
                for arm in d['arms']:
                    en = ConfiguredDualMechanismRecovery(allocators[arm], e, parts, groups,
                        case['initial_ids'], case['kind'], configs[arm])
                    en.cache = states.setdefault((arm, demand), {})
                    en.pressure_cache = pressures.setdefault((arm, case['kind']), {})
                    failed, metric, rounds = en.closure(en.initial)
                    if metric['Max_LP_Violation'] > 1e-5 or metric['Service_Conservation_Error'] > 1e-7:
                        raise AssertionError('Invalid allocation')
                    rows.append(dict(scope=scope, capacity_id=job['capacity_id'], theta=theta,
                        case_id=case['case_id'], kind=case['kind'], demand=demand, arm=arm,
                        initial_count=len(en.initial), nodes=len(nn), edges=len(e), final_failed=len(failed),
                        failure_fraction=len(failed)/len(nn), rounds=rounds,
                        final_ids_json=json.dumps(sorted(failed), ensure_ascii=False), **metric))
    path = OUT / 'shards' / (scope + '_' + job['capacity_id'] + '.csv')
    study.save(rows, path)
    return str(path)


def main():
    started = time.perf_counter()
    d = json.loads((OUT / '00_design.json').read_text(encoding='utf-8'))
    lock = json.loads((OUT / 'design_lock.json').read_text(encoding='utf-8'))
    assert sha(OUT / '00_design.json') == lock['design']
    assert sha(OUT / '00_design.md') == lock['note']
    for p, h in d['protected'].items(): assert sha(p) == h, p
    study.dump(dict(scientific_design_changed=False, allocator_reuse=True, workers=3,
                    source_sha256=sha(__file__), reason='Same 5184 frozen trajectories; execution caching only'), OUT / 'execution_v2.json')
    pending, complete = [], []
    for job in d['jobs']:
        path = OUT / 'shards' / (job['scope'] + '_' + job['capacity_id'] + '.csv')
        if path.exists():
            r = pd.read_csv(path)
            assert len(r) == 576 and r[['scope','capacity_id','theta','case_id','demand','arm']].drop_duplicates().shape[0] == 576
            complete.append(str(path))
        else: pending.append(job)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_cell, job): job for job in pending}
        for future in as_completed(futures):
            complete.append(future.result())
            print('complete cell', futures[future], 'cells', len(complete), '/9', flush=True)
    r = pd.concat([pd.read_csv(path) for path in sorted(complete)], ignore_index=True)
    checks = []
    def check(label, value):
        checks.append(dict(check=label, passed=bool(value)))
        if not value:
            study.save(checks, OUT / 'failed_checks.csv')
            raise AssertionError(label)
    check('5184 predeclared trajectories', len(r) == d['expected_runs'] == 5184)
    check('all allocations feasible and conserving tasks', r.Max_LP_Violation.max() <= 1e-5 and
          r.Service_Conservation_Error.max() <= 1e-7 and r.Assignment_Conservation_Error.max() <= 1e-7)
    keys = ['scope', 'capacity_id', 'theta', 'case_id', 'kind', 'demand']
    p = r.pivot(index=keys, columns='arm', values='Service_Rate').reset_index()
    p['C_B_pp'] = 100*(p.C-p.B); p['C_A_pp'] = 100*(p.C-p.A)
    check('safe-policy upper bound in all conditions', p.C_B_pp.min() >= -1e-7 and p.C_A_pp.min() >= -1e-7)
    summaries = []
    for (scope, cap), g in p.groupby(['scope','capacity_id'], sort=False):
        low = g[~g.demand.eq('external_plus150')]
        summaries.append(dict(scope=scope, capacity_id=cap, conditions=len(g), C_B_mean_pp=g.C_B_pp.mean(),
            C_B_median_pp=g.C_B_pp.median(), C_B_better=int((g.C_B_pp>1e-7).sum()),
            C_B_equal=int((g.C_B_pp.abs()<=1e-7).sum()), C_B_worse=int((g.C_B_pp< -1e-7).sum()),
            excluding_highest_conditions=len(low), excluding_highest_mean_pp=low.C_B_pp.mean()))
    task_keys = ['capacity_id','theta','case_id','kind','demand','arm']
    a = r[r.scope.eq('matched_tasks246')].set_index(task_keys).sort_index()
    b = r[r.scope.eq('reduced242')].set_index(task_keys).sort_index()
    denominator_error = float((a.Demand_Load-b.Demand_Load).abs().max())
    check('same task denominators within numerical summation tolerance', denominator_error <= 1e-7)
    effect = b[[]].copy()
    effect['service_change_pp'] = 100*(b.Service_Rate-a.Service_Rate)
    effect['completed_load_change'] = b.Served_Load-a.Served_Load
    effect['denominator_error'] = (a.Demand_Load-b.Demand_Load).abs()
    n, e, _, _ = core.load_study()
    keep = ~n.node_id.isin(d['excluded']).to_numpy()
    resources = []
    for scope in d['scopes']:
        for cap in d['capacity_ids']:
            nn, _ = formula_precision_nodes(core.capacity_nodes(n, cap), n)
            ee = e
            if scope == 'reduced242':
                nn = nn.loc[keep].reset_index(drop=True)
                ee = e[e.Source_id.isin(nn.node_id) & e.Target_id.isin(nn.node_id)]
            resources.append(dict(scope=scope, capacity_id=cap, nodes=len(nn), edges=len(ee),
                total_capacity=nn.overload_threshold.sum(), retained_base_load=nn.proxy_daily_load.sum()))
    demands = json.loads((OUT / 'frozen_inputs/demands.json').read_text(encoding='utf-8'))
    cold = []
    for scope, arm in [('full246','B'),('matched_tasks246','B'),('reduced242','B'),('reduced242','C')]:
        for cap in d['capacity_ids']:
            nn, _ = formula_precision_nodes(core.capacity_nodes(n, cap), n)
            ee = e; case = d['cases'][-1]; demand = d['demand_names'][0]
            parts = np.array(demands[demand]['parts'])
            if scope == 'matched_tasks246': parts[:,~keep] = 0.
            if scope == 'reduced242':
                nn = nn.loc[keep].reset_index(drop=True); parts = parts[:,keep]
                ee = e[e.Source_id.isin(nn.node_id)&e.Target_id.isin(nn.node_id)].reset_index(drop=True)
            en = study.engine_for(nn,ee,parts,demands[demand]['groups'],case,arm,.42)
            f,m,_ = en.closure(en.initial)
            old = r[r.scope.eq(scope)&r.capacity_id.eq(cap)&r.case_id.eq(case['case_id'])&
                    r.demand.eq(demand)&r.arm.eq(arm)&r.theta.eq(.42)].iloc[0]
            equal = set(json.loads(old.final_ids_json)) == set(f) and abs(old.Service_Rate-m['Service_Rate']) <= 1e-12
            check('cold restart '+scope+cap+arm,equal)
            cold.append(dict(scope=scope,capacity_id=cap,arm=arm,identical=equal))
    for key in ['C_B_pp','C_A_pp']:
        check('policy differences finite '+key,np.isfinite(p[key]).all())
    for path,h in d['protected'].items(): check('protected '+path,sha(path)==h)
    for rows,name in [(r,'04_model_runs.csv'),(resources,'05_scope_resources.csv'),(p,'06_policy_pairs.csv'),
                      (summaries,'07_policy_summary.csv'),(effect.reset_index(),'08_matched_boundary_effect.csv'),
                      (cold,'09_cold_checks.csv'),(checks,'10_checks.csv')]: study.save(rows,OUT/name)
    findings = dict(completed=True,model_runs=len(r),policy_conditions=len(p),checks=len(checks),
        passed=sum(v['passed'] for v in checks),cold_restarts=len(cold),denominator_roundoff_max=denominator_error,
        actual_real_capacity_replacements=0,elapsed_seconds=time.perf_counter()-started)
    study.dump(findings,OUT/'completion.json')
    study.dump(dict(files={str(path.relative_to(OUT)):sha(path) for path in OUT.rglob('*') if path.is_file()}),OUT/'delivery_manifest.json')
    print(json.dumps(findings,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
