"""Predeclared administrative-role boundary comparison; no observed K claimed."""
import importlib
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy

from capacity_precision_v5 import formula_precision_nodes
from configured_dual_mechanism_v5 import ConfiguredDispatchAllocator, ConfiguredDualMechanismRecovery
from model_config import ModelConfig
from project_paths import DATA_DIR, RESULTS_DIR, SOURCES_DIR
from public_data_inputs_v1 import sha

core = importlib.import_module('83_dispatch_mainline_core')
OUT = RESULTS_DIR / 'node_boundary_scope_20261021'


def dump(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def build_design():
    if OUT.exists():
        raise FileExistsError('New frozen version required: ' + str(OUT))
    OUT.mkdir(parents=True)
    (OUT / 'shards').mkdir()
    (OUT / 'frozen_inputs').mkdir()
    n, e, demands, _ = core.load_study()
    prior = json.loads((core.OUT / '84_design.json').read_text(encoding='utf-8'))
    role_path = SOURCES_DIR / '宏观与设施补强_20261014/facility_review_records.json'
    roles = json.loads(role_path.read_text(encoding='utf-8'))
    excluded = sorted(v['node_id'] for v in roles if v['排除对照'])
    if len(excluded) != 4 or not set(excluded) <= set(n.node_id):
        raise ValueError('Expected the four previously documented role-boundary candidates')
    keep = ~n.node_id.isin(excluded).to_numpy()
    payload = {}
    for name in prior['demand_names']:
        part, groups = demands[name]
        payload[name] = dict(parts=np.asarray(part).tolist(), groups=list(groups))
    dump(payload, OUT / 'frozen_inputs/demands.json')
    roles_by_id = {v['node_id']: v for v in roles}
    audit = []
    for row in n.to_dict('records'):
        evidence = roles_by_id.get(row['node_id'], {})
        audit.append(dict(node_id=row['node_id'], name=row['名称'], family=row['family'],
                          conservative_exclusion=row['node_id'] in excluded,
                          decision='exclude_in_sensitivity_only' if row['node_id'] in excluded else 'retain_not_newly_verified',
                          prior_review_available=bool(evidence),
                          identity_note=evidence.get('身份结论', '本轮未新增点位事实核验；保留代理身份'),
                          source_url=evidence.get('核验来源', ''), empirically_calibrated=False))
    save(audit, OUT / '01_node_role_audit.csv')
    save([v for v in audit if v['conservative_exclusion']], OUT / '02_four_role_boundary_candidates.csv')
    n.to_csv(OUT / 'frozen_inputs/original_nodes.csv', index=False, encoding='utf-8-sig')
    e.to_csv(OUT / 'frozen_inputs/original_edges.csv', index=False, encoding='utf-8-sig')
    cases, changes = [], []
    for c in prior['cases']:
        common = sorted(set(c['initial_ids']) - set(excluded))
        changes.append(dict(case_id=c['case_id'], original_initial_count=len(c['initial_ids']),
                            common_initial_count=len(common),
                            removed_initial_ids=json.dumps(sorted(set(c['initial_ids']) & set(excluded)), ensure_ascii=False)))
        cases.append(dict(c, initial_ids=common))
    save(changes, OUT / '03_shock_projection.csv')
    capacities = {}
    for key in prior['capacity_ids']:
        corrected, _ = formula_precision_nodes(core.capacity_nodes(n, key), n)
        capacities[key] = corrected
        corrected.to_csv(OUT / f'frozen_inputs/{key}_formula_nodes.csv', index=False, encoding='utf-8-sig')
    scopes = ('full246', 'matched_tasks246', 'reduced242')
    jobs = [dict(scope=s, capacity_id=k) for s, k in itertools.product(scopes, prior['capacity_ids'])]
    protected = list(DATA_DIR.glob('*.csv'))
    protected += list(Path(__file__).parent.glob('*.py'))
    protected += [role_path, core.OUT / '84_design.json', core.OUT / '84a_capacity_policy_results.csv',
                  RESULTS_DIR / 'capacity_precision_audit_20261020/08_paper_policy_pairs.csv',
                  core.ROOT.parent / '中文论文初稿_全图美化与正文优化V5_20261019.docx',
                  core.ROOT.parent / '论文初稿配套图表_V5_20261019/初稿源文件/manuscript.md']
    design = dict(date='2026-10-21', schema='node-boundary-v1', excluded=excluded,
                  excluded_basis='Previously documented administrative/regulatory role candidates; not a declaration of nonexistence or zero business influence',
                  cases=cases, thresholds=prior['thresholds'], demand_names=prior['demand_names'],
                  capacity_ids=prior['capacity_ids'], scopes=scopes, jobs=jobs, arms=['A', 'B', 'C'],
                  expected_runs=len(jobs)*len(cases)*len(prior['thresholds'])*len(prior['demand_names'])*3,
                  comparison_rules={
                      'full246': 'Original demand components, with the common projected shocks used in all scopes',
                      'matched_tasks246': 'Same original graph and K; demand at the four excluded sources zeroed, no refitting',
                      'reduced242': 'Induced graph after removing the four candidates; retained tasks and K unchanged, no capacity rescaling'},
                  inference='matched_tasks246 vs reduced242 holds task vector and denominator fixed, but removes capacity and paths/risk relations jointly; not an isolated topology effect or empirical causal estimate',
                  fixed=['Original L and risk attributes', 'K second-rounding removed using the established precision formula',
                         'Demand components and retained-source values', 'No type reassignment, coordinate merging, new links, new shocks or threshold search'],
                  stops=['No extra exclusions chosen from outcomes', 'No capacity or task rescaling to preserve policy ranking',
                         'Do not call remaining 242 nodes verified physical processing facilities'],
                  protected={str(p): sha(p) for p in protected if p.exists()},
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
    dump(design, OUT / '00_design.json')
    note = '''# 节点边界对照 运行前设计

仅采用既有核验记录的4个行政/监管角色候选作保守排除敏感性，不因关键词或缺少容量资料继续删点，不宣称其不存在或对物流无影响。
保留原246图；增加原图共同任务版本及242节点诱导子图。共同任务版本将4个候选的来源任务置零，242图保留相同其余来源任务、能力及边属性，不重做IPF，不补边，不归一化剩余容量。
16组冲击均取与剩余节点的交集，三种边界使用同一集合；原始8点损伤及本次变化逐组另存。不将修改后的损伤冒充原论文相同实验。
容量采用前轮已验证的公式精度版本；三套既有容量、四档既有需求、三风险阈值、三政策全部运行，共5184条轨迹。
共同任务246与242对照可以比较同分母服务差，但同时改变了承接资源和网络/风险关系，不能归因为纯拓扑因果。原246与242的需求总量不同，另列。
不新增实测K，不把条件相关仿真视为独立城市或灾害，不预设清洗后的排序或效应幅度。
'''
    (OUT / '00_design.md').write_text(note, encoding='utf-8')
    dump(dict(design=sha(OUT / '00_design.json'), note=sha(OUT / '00_design.md')),
         OUT / 'design_lock.json')
    return design, n, e, payload, capacities, keep


def engine_for(nn, ee, parts, groups, case, arm, theta):
    cfg = ModelConfig.for_arm(arm, theta_risk=theta, theta_source='Frozen historical threshold, not observed')
    a = ConfiguredDispatchAllocator(nn, ee, cfg)
    return ConfiguredDualMechanismRecovery(a, ee, parts, groups, case['initial_ids'], case['kind'], cfg)


def main():
    started = time.perf_counter()
    d, n, e, demands, capacity_versions, keep = build_design()
    rows, counts, checks = [], [], []
    def check(label, value):
        checks.append(dict(check=label, passed=bool(value)))
        if not value:
            save(checks, OUT / 'failed_checks.csv')
            raise AssertionError(label)
    check('design precedes all new outcomes', sha(OUT / '00_design.json') == json.loads((OUT / 'design_lock.json').read_text())['design'])
    for job in d['jobs']:
        scope, cap = job['scope'], job['capacity_id']
        nn = capacity_versions[cap].copy()
        ee = e.copy()
        if scope == 'reduced242':
            nn = nn.loc[keep].reset_index(drop=True)
            ee = ee[ee.Source_id.isin(nn.node_id) & ee.Target_id.isin(nn.node_id)].reset_index(drop=True)
        counts.append(dict(scope=scope, capacity_id=cap, nodes=len(nn), edges=len(ee),
                           total_capacity=float(nn.overload_threshold.sum()), retained_base_load=float(nn.proxy_daily_load.sum())))
        allocators = {arm: ConfiguredDispatchAllocator(nn, ee, ModelConfig.for_arm(
            arm, theta_risk=d['thresholds'][0], theta_source='Frozen historical threshold, not observed')) for arm in d['arms']}
        shard = []
        for theta, case, demand in itertools.product(d['thresholds'], d['cases'], d['demand_names']):
            parts = np.array(demands[demand]['parts'], dtype=float)
            groups = demands[demand]['groups']
            if scope == 'matched_tasks246':
                parts[:, ~keep] = 0.
            elif scope == 'reduced242':
                parts = parts[:, keep]
            outcomes = {}
            for arm in d['arms']:
                en = engine_for(nn, ee, parts, groups, case, arm, theta)
                failed, metric, iterations = en.closure(en.initial)
                row = dict(scope=scope, capacity_id=cap, theta=theta, case_id=case['case_id'],
                           kind=case['kind'], demand=demand, arm=arm, initial_count=len(en.initial),
                           nodes=len(nn), edges=len(ee), final_failed=len(failed), failure_fraction=len(failed)/len(nn),
                           rounds=iterations, final_ids_json=json.dumps(sorted(failed), ensure_ascii=False), **metric)
                shard.append(row)
                outcomes[arm] = (failed, metric)
                check('allocation/closure accounting ' + str((scope, cap, theta, case['case_id'], demand, arm)),
                      metric['Max_LP_Violation'] <= 1e-5 and metric['Service_Conservation_Error'] <= 1e-7
                      and metric['Assignment_Conservation_Error'] <= 1e-7 and set(en.initial) <= failed)
            for arm in ['A', 'B']:
                check('conditional safe-policy bound ' + str((scope, cap, theta, case['case_id'], demand, arm)),
                      outcomes['C'][0] <= outcomes[arm][0] and
                      outcomes['C'][1]['Service_Rate'] >= outcomes[arm][1]['Service_Rate'] - 1e-9)
        rows.extend(shard)
        save(shard, OUT / f'shards/{scope}_{cap}.csv')
        dump(dict(scope=scope, capacity_id=cap, finished_runs=len(rows), elapsed_seconds=time.perf_counter()-started), OUT / 'progress.json')
        print(scope, cap, 'runs', len(rows), 'elapsed', round(time.perf_counter()-started, 1), flush=True)
    r = pd.DataFrame(rows)
    check('all predeclared runs completed', len(r) == d['expected_runs'])
    save(r, OUT / '04_model_runs.csv')
    save(counts, OUT / '05_scope_resources.csv')
    keys = ['scope', 'capacity_id', 'theta', 'case_id', 'kind', 'demand']
    p = r.pivot(index=keys, columns='arm', values='Service_Rate').reset_index()
    p['C_B_pp'] = 100*(p.C-p.B)
    p['C_A_pp'] = 100*(p.C-p.A)
    save(p, OUT / '06_policy_pairs.csv')
    summary = []
    for (scope, cap), g in p.groupby(['scope', 'capacity_id'], sort=False):
        low = g[~g.demand.eq('external_plus150')]
        summary.append(dict(scope=scope, capacity_id=cap, conditions=len(g),
                            C_B_mean_pp=g.C_B_pp.mean(), C_B_median_pp=g.C_B_pp.median(),
                            C_B_better=int((g.C_B_pp > 1e-7).sum()), C_B_equal=int((g.C_B_pp.abs() <= 1e-7).sum()),
                            C_B_worse=int((g.C_B_pp < -1e-7).sum()),
                            excluding_highest_conditions=len(low), excluding_highest_mean_pp=low.C_B_pp.mean()))
    save(summary, OUT / '07_policy_summary.csv')
    task_keys = ['capacity_id', 'theta', 'case_id', 'kind', 'demand', 'arm']
    a = r[r.scope.eq('matched_tasks246')].set_index(task_keys)
    b = r[r.scope.eq('reduced242')].set_index(task_keys)
    check('same retained task demand denominator in matched comparison', np.array_equal(a.Demand_Load.to_numpy(), b.Demand_Load.to_numpy()))
    effect = b[[]].copy()
    effect['service_change_pp'] = 100*(b.Service_Rate-a.Service_Rate)
    effect['completed_load_change'] = b.Served_Load-a.Served_Load
    effect['same_demand'] = (a.Demand_Load-b.Demand_Load).abs() < 1e-9
    save(effect.reset_index(), OUT / '08_matched_boundary_effect.csv')
    cold_rows = []
    for cap in d['capacity_ids']:
        nn = capacity_versions[cap].loc[keep].reset_index(drop=True)
        ee = e[e.Source_id.isin(nn.node_id) & e.Target_id.isin(nn.node_id)].reset_index(drop=True)
        for arm in ['B', 'C']:
            case = d['cases'][-1]
            demand = d['demand_names'][0]
            parts = np.array(demands[demand]['parts'])[:, keep]
            en = engine_for(nn, ee, parts, demands[demand]['groups'], case, arm, .42)
            f, m, _ = en.closure(en.initial)
            old = r[r.scope.eq('reduced242') & r.capacity_id.eq(cap) & r.case_id.eq(case['case_id']) &
                    r.demand.eq(demand) & r.arm.eq(arm) & r.theta.eq(.42)].iloc[0]
            passed = set(json.loads(old.final_ids_json)) == set(f) and abs(old.Service_Rate-m['Service_Rate']) <= 1e-12
            check('cold restart ' + cap + arm, passed)
            cold_rows.append(dict(capacity_id=cap, arm=arm, identical=passed))
    save(cold_rows, OUT / '09_cold_checks.csv')
    for path, h in d['protected'].items():
        check('protected source ' + path, sha(path) == h)
    save(checks, OUT / '10_checks.csv')
    manifest = dict(completed=True, model_runs=len(r), policy_conditions=len(p), checks=len(checks),
                    passed=sum(v['passed'] for v in checks), excluded_node_count=4,
                    conservative_nodes=242, actual_real_capacity_replacements=0,
                    elapsed_seconds=time.perf_counter()-started,
                    files={str(f.relative_to(OUT)): sha(f) for f in OUT.rglob('*') if f.is_file()})
    dump(manifest, OUT / 'completion.json')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
