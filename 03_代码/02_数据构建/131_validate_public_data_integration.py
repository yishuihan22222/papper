"""Independent ingestion checks and a bounded V5 before/after execution check."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy

from closed_loop_recovery_v3 import load_inputs, DemandModel, DemandScenario
from configured_dual_mechanism_v5 import ConfiguredDispatchAllocator, ConfiguredDualMechanismRecovery
from model_config import ModelConfig
from project_paths import RESULTS_DIR
from public_data_inputs_v1 import (DEFAULT_DIR, LOAD, CAPACITY, sha, apply_replacements,
                                   load_public_inputs)


def main():
    started = time.perf_counter()
    out = DEFAULT_DIR
    destination = out / 'validation'
    if destination.exists():
        raise FileExistsError('Validation version already exists: ' + str(destination))
    destination.mkdir()
    checks, results, comparisons = [], [], []

    def check(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))
        if not passed:
            pd.DataFrame(checks).to_csv(destination / 'failed_checks.csv', index=False, encoding='utf-8-sig')
            raise AssertionError(name)

    def rejects(name, call):
        try:
            call()
        except ValueError as error:
            check(name, True, rejection=str(error))
        else:
            check(name, False)

    old_n, old_e, old_ci, old_cross = load_inputs()
    bundle = load_public_inputs()
    for path, expected in bundle.manifest['protected_inputs'].items():
        check('protected source unchanged: ' + path, sha(path) == expected)
    n, e, ci, cross = bundle.legacy_inputs()
    check('246 unique existing nodes', len(n) == 246 and n.node_id.is_unique and old_n.node_id.tolist() == n.node_id.tolist())
    numeric_old = old_n.select_dtypes(include='number').columns
    unchanged = np.array_equal(old_n[numeric_old].to_numpy(), n[numeric_old].to_numpy())
    max_difference = float(np.max(np.abs(old_n[numeric_old].to_numpy() - n[numeric_old].to_numpy())))
    check('all original numeric node fields unchanged after input round trip', unchanged,
          max_absolute_difference=max_difference)
    for label, old, new in [('edges', old_e, e), ('city_data', old_ci, ci), ('cross_layer', old_cross, cross)]:
        pd.testing.assert_frame_equal(old, new, check_exact=True)
        check(label + ' unchanged', True)
    check('30 observations, 19 anchors, no false measurements', len(bundle.observations) == 30
          and len({o['anchor_id'] for o in bundle.observations}) == 19
          and not any(o['independently_reverified_this_run'] for o in bundle.observations))
    check('all node links are context only', not bundle.links.confirmed_parameter_entity_match.any())
    check('12 nodes contain loaded numeric public context', int(n.public_context_count.gt(0).sum()) == 12
          and sum(len(json.loads(v)) for v in n.public_observations_json) == len(bundle.links))
    for obs in bundle.observations:
        check('finite numeric observation or explicitly missing: ' + obs['observation_id'],
              obs['value'] is None or (np.isfinite(obs['value']) and obs['value'] >= 0))
    record = {o['observation_id']: o for o in bundle.observations}
    check('only unit prefixes converted, no time annualization',
          record['W01_cargo']['value'] == 102.5 * 10000 and
          record['W03_yangluo']['value'] == 238.1 * 10000 and
          record['W06_trade']['value'] == 800 * 100000000 and
          record['W11_built']['unit'] == 'tonne_inventory' and
          record['W10_duration']['value'] == 2 * 60)
    check('ambiguous million-level volume remains missing', record['W12_sorting']['value'] is None)
    check('no L/K values falsely reported as replaced', bundle.manifest['actual_parameter_replacements'] == 0
          and np.array_equal(n[LOAD], old_n[LOAD]) and np.array_equal(n[CAPACITY], old_n[CAPACITY]))
    rejects('strict real-change request fails explicitly', lambda: load_public_inputs(require_real_parameter_change=True))
    raw = dict(node_id=n.node_id.iloc[0], parameter=CAPACITY, model_value=300000., model_unit='piece/day')
    rejects('physical pieces/day cannot overwrite proxy capacity', lambda: apply_replacements(n, [raw]))
    proxy = dict(raw, model_unit='proxy_service_unit_per_slot')
    rejects('changing unit label alone cannot authorize an overwrite', lambda: apply_replacements(n, [proxy]))
    complete = dict(proxy, **{field: True for field in ('entity_match_resolved', 'period_aligned',
                     'service_compatible', 'quantity_definition_aligned', 'unit_conversion_calibrated')})
    rejects('source and unit-conversion evidence are required', lambda: apply_replacements(n, [complete]))
    missing = dict(complete, node_id='NO_SUCH_FACILITY', source_observation_id='example_only', conversion_evidence='example_only')
    rejects('new facility cannot be inserted through a parameter overwrite', lambda: apply_replacements(n, [missing]))
    check('source files readable without replacing the legacy entry', old_n.shape[1] < n.shape[1])

    design = json.loads((RESULTS_DIR / 'protect_repair_v6_20261017/71_design.json').read_text(encoding='utf-8'))
    case_ids = ('test_random_01', 'test_channel_00')
    cases = [c for c in design['evaluation_cases'] if c['case_id'] in case_ids]
    check('two existing shocks, no invented observed event', len(cases) == 2)
    demands = {}
    for version, nn, ee, cc, cr in [('legacy', old_n, old_e, old_ci, old_cross), ('public_context', n, e, ci, cross)]:
        dm = DemandModel(nn, ee, cc, cr)
        demands[version] = {
            'nominal': dm.components(DemandScenario('nominal')),
            'external_plus20': dm.components(DemandScenario('external_plus20', tuple((c, 1.2) for c in dm.cities)))
        }
    for name in demands['legacy']:
        check('same generated demand components: ' + name,
              np.array_equal(demands['legacy'][name][0], demands['public_context'][name][0]))
    for case in cases:
        for demand in demands['legacy']:
            for arm in ('A', 'B', 'C'):
                pair = []
                for version, nn, ee in [('legacy', old_n, old_e), ('public_context', n, e)]:
                    cfg = ModelConfig.for_arm(arm, theta_risk=.42, theta_source='existing bounded ingestion parity check, not observed')
                    allocator = ConfiguredDispatchAllocator(nn, ee, cfg)
                    parts, groups = demands[version][demand]
                    engine = ConfiguredDualMechanismRecovery(allocator, ee, parts, groups,
                        case['initial_ids'], case['kind'], cfg)
                    failed, metric, rounds = engine.closure(engine.initial)
                    row = dict(version=version, case_id=case['case_id'], demand=demand, arm=arm,
                               theta=.42, final_failed=len(failed), final_ids=json.dumps(sorted(failed), ensure_ascii=False),
                               rounds=rounds, **metric)
                    results.append(row)
                    pair.append((failed, metric, rounds))
                    check('conservation and LP feasibility ' + str((version, case['case_id'], demand, arm)),
                          metric['Service_Conservation_Error'] <= 1e-7 and
                          metric['Assignment_Conservation_Error'] <= 1e-7 and metric['Max_LP_Violation'] <= 1e-5)
                keys = pair[0][1].keys()
                equal = pair[0][0] == pair[1][0] and pair[0][2] == pair[1][2] and all(pair[0][1][key] == pair[1][1][key] for key in keys)
                check('closure identical after context import ' + str((case['case_id'], demand, arm)), equal)
                comparisons.append(dict(case_id=case['case_id'], demand=demand, arm=arm,
                                        identical=equal, service_rate_delta=pair[1][1]['Service_Rate'] - pair[0][1]['Service_Rate'],
                                        failed_count_delta=len(pair[1][0])-len(pair[0][0])))
            print('parity checked', case['case_id'], demand, flush=True)
    for path, expected in bundle.manifest['protected_inputs'].items():
        check('protected source unchanged after execution: ' + path, sha(path) == expected)
    for frame, name in [(checks, 'checks.csv'), (results, 'model_runs.csv'), (comparisons, 'paired_results.csv')]:
        pd.DataFrame(frame).to_csv(destination / name, index=False, encoding='utf-8-sig')
    summary = dict(completed=True, checks=len(checks), passed=sum(v['passed'] for v in checks),
                   model_runs=len(results), compared_conditions=len(comparisons),
                   identical_conditions=sum(v['identical'] for v in comparisons),
                   actual_real_parameter_replacements=0, real_data_effect_estimated=False,
                   purpose='Input execution/regression check only; not a new calibrated experiment',
                   scipy=scipy.__version__, elapsed_seconds=time.perf_counter()-started,
                   source_sha256=sha(__file__), files={name: sha(destination / name) for name in
                         ('checks.csv', 'model_runs.csv', 'paired_results.csv')})
    (destination / 'completion.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (destination / '运行验证说明.md').write_text(
        f"# 新输入版本运行验证\n\n{len(checks)}项检查通过；执行{len(results)}条轨迹，"
        f"比较{len(comparisons)}组相同条件，终态、完成服务和全部输出指标均一致。\n\n"
        "这是检查背景资料接入没有改变计算的回归验证，不是新增真实数据实验。"
        "真实需求/容量替换均为0；不据此声称实测容量稳健性或外部预测验证。\n",
        encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
