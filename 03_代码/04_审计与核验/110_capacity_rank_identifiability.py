"""Direction 1 evidence gate and bounded conditional capacity response audit."""
from pathlib import Path
import hashlib
import importlib
import itertools
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '项目结果/capacity_rank_identifiability_20261019'
DATA = ROOT / '数据及代码/data'
CODE = Path(__file__).parent
PRIOR = ROOT / '项目结果/dispatch_capacity_null_mainline_20261018'
FAC = ROOT / '数据来源/重点设施核验与校准_20261014'
CAP = ROOT / '项目结果/capacity_heterogeneity_design_20261018'
CANDIDATES = [
    '中国邮政武汉市黄金口邮件处理中心@@武汉市@@汉阳区',
    '京东亚一分拣中心@@武汉市@@新洲区',
    '中通快递(武汉分拨中心)@@武汉市@@东西湖区',
]
CAP_IDS = ['baseline', 'node_a0.20_s00', 'family_a0.20_s00']
CHECKS = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(obj, name):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def save(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def check(label, passed):
    CHECKS.append(dict(check=label, passed=bool(passed)))
    if not passed:
        save(CHECKS, 'validation_failed.csv')
        raise AssertionError(label)


def begin():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'source_manifest.json'
    if path.exists():
        m = json.loads(path.read_text(encoding='utf-8'))
        for p, h in m['protected'].items():
            check('unchanged source ' + p, sha(p) == h)
        check('frozen source', sha(__file__) == m['new_source_sha256'])
        check('frozen design', sha(OUT / '00_design.md') == m['design_sha256'])
        return m
    paths = list(DATA.glob('*.csv')) + list(CODE.glob('*.py'))
    paths += list(FAC.glob('*.csv')) + list(FAC.glob('*.json'))
    paths += [PRIOR / '84a_capacity_policy_results.csv', PRIOR / '86_design_v3.json',
              CAP / '82a_assumed_capacity_scenarios.csv', CAP / '82_completion.json',
              ROOT / '项目文档/重点设施身份与容量校准核验_20261014.md',
              ROOT.parent / '中文论文初稿_全图美化与正文优化V5_20261019.docx']
    m = dict(new_source_sha256=sha(__file__), design_sha256=sha(OUT / '00_design.md'),
             runtime=dict(python=platform.python_version(), numpy=np.__version__,
                          pandas=pd.__version__, scipy=scipy.__version__),
             protected={str(p): sha(p) for p in paths if p.is_file() and p.resolve() != Path(__file__).resolve()})
    dump(m, 'source_manifest.json')
    return m


def evidence_gate(n):
    # Facts paraphrased from primary pages opened on 2026-10-19. No source is
    # converted to a capacity bound or an investment price.
    public = [dict(source_id='NEW_POST_2023', node_id=CANDIDATES[0],
                   title='2023年武汉盛辉、武汉黄金口、十堰邮件处理中心内部处理业务外包项目招标公告',
                   url='https://www.chinapost.com.cn/cn/report/2308/3704-1.htm',
                   published='2023-08-07', retrieved='2026-10-19',
                   fact='黄金口年预测业务量4311万袋件；月日均计费分段5万/12万袋件；服务期到2025-03-31。',
                   metric_class='forecast_workload_and_processing_tariff',
                   limitation='预测不是已观测负荷；阶梯不是设备能力；单价不是增容价格；袋件与代理单位未映射。'),
              dict(source_id='NEW_JD_2023', node_id=CANDIDATES[1],
                   title='武汉市政府2023年1月报道中的武汉亚一二期分拣中心',
                   url='https://www.wuhan.gov.cn/ztzl/23zt/xczjc/202301/t20230119_2136808.shtml',
                   published='2023-01-19', retrieved='2026-10-19',
                   fact='报道日约75万件完成分拣出港；二期中心自2019年底使用，日订单处理能力超过100万单。',
                   metric_class='reported_daily_workload_and_capability_statement',
                   limitation='2023年而非2024同期间；件与订单非必然同口径；报道二期与原POI边界未闭合；超过不是有限上界。'),
              dict(source_id='NEW_ZTO_2024', node_id=CANDIDATES[2],
                   title='再加码，扩产能，中通快递华中总部基地在东西湖连连加速',
                   url='https://www.dxh.gov.cn/ZSYZ/XMTJ_16715/202406/t20240617_2416528.shtml',
                   published='2024-06-06', retrieved='2026-10-19',
                   fact='武汉转运中心一期进港、二期出港；报道园区进出港合计日均320万票，2024仍有升级扩建。',
                   metric_class='combined_operating_volume_across_two_phases',
                   limitation='两期合计不等于单个POI；进出港计量需去重；业务量不是能力上界；扩建前后不是同一固定能力。')]
    dump(dict(status='primary_page_reading_notes_not_full_page_archives', sources=public), '01_public_source_reading_notes.json')
    samples = pd.read_csv(FAC / '设施级运营量与能力样本.csv')
    oldrows = []
    for r in samples.to_dict('records'):
        oldrows.append(dict(sample_id=r['样本ID'], entity=r['设施实体'], metric_class=r['指标性质'],
                            unit=r['统一单位'], period=r['统计或适用期'], match_status=r['设施对应状态'],
                            source_url=r['来源URL'], direct_model_capacity_bound=False,
                            failure_reason=str(r['限制及处理']) + '；现有代理单位未物理标定'))
    save(oldrows, '02_prior_sample_bound_audit.csv')
    diag = json.loads((FAC / 'calibration_diagnostics.json').read_text(encoding='utf-8'))
    check('prior calibration still unclosed', not diag['can_fit_absolute_proxy_to_ton_mapping'])
    c = n[n.node_id.isin(CANDIDATES)].copy()
    check('three exact candidate IDs', len(c) == 3)
    rows = []
    for node in CANDIDATES:
        r = c[c.node_id == node].iloc[0]
        p = next(v for v in public if v['node_id'] == node)
        rows.append(dict(node_id=node, model_family=r.family, proxy_load=float(r.proxy_daily_load),
                         proxy_capacity=float(r.overload_threshold), source_id=p['source_id'],
                         physical_boundary_confirmed=False, period_and_metric_aligned=False,
                         proxy_unit_calibrated=False, empirical_capacity_interval_available=False,
                         incremental_investment_cost_available=False,
                         missing_evidence=p['limitation']))
    save(rows, '03_candidate_evidence_gate.csv')
    cap = pd.read_csv(CAP / '82a_assumed_capacity_scenarios.csv')
    check('existing capacity table explicitly uncalibrated', (~cap.empirically_calibrated).all())
    gate = dict(empirical_ranking_gate_passed=False, empirical_interval_set_defined=False,
                verified_interval_candidates=0, candidates=3, reviewed_prior_samples=len(samples),
                reviewed_prior_node_records=diag['reviewed_nodes'], new_primary_pages=len(public),
                model_nodes=len(n), full_new_246_node_source_audit=False,
                existing_capacity_scenarios=cap.scenario.nunique(),
                capacity_scenarios_are_observation_supported=False,
                equal_money_budget_comparison_defined=False,
                numerical_information_value_defined=False,
                can_claim_real_world_ranking_reversal=False,
                can_run_conditional_point_audit=True,
                reason='没有同设施同期间同单位的容量区间与增容成本关系；既有合成情景不能替代这些证据。')
    dump(gate, '04_evidence_gate.json')
    return gate


def independent_safe_lp(allocator, failed, demand):
    """Independent single-stage LP with fixed local-priority accounting."""
    alive = np.array([k not in failed for k in allocator.ids])
    K = allocator.nominal_capacity
    own = np.where(alive, np.minimum(K, demand), 0.)
    remaining = demand - own
    spare = np.where(alive, K - own, 0.)
    paths = allocator.paths(failed).copy()
    paths[np.diag_indices_from(paths)] = np.where(alive, 0., np.inf)
    i, j = np.nonzero((remaining[:, None] > 1e-9) & (spare[None, :] > 1e-9) & np.isfinite(paths))
    if len(i) == 0:
        return float(own.sum())
    m, size = len(i), len(K)
    A = coo_matrix((np.ones(2*m), (np.r_[i, size+j], np.r_[np.arange(m), np.arange(m)])),
                  shape=(2*size, m)).tocsr()
    z = linprog(-np.ones(m), A_ub=A, b_ub=np.r_[remaining, spare], bounds=(0, None), method='highs')
    check('independent safe LP solved', z.success)
    check('independent safe LP conservation', np.max(A @ z.x - np.r_[remaining, spare]) < 1e-6)
    return float(own.sum() - z.fun)


def point_audit():
    core = importlib.import_module('83_dispatch_mainline_core')
    n, edges, demands, _ = core.load_study()
    design = json.loads((PRIOR / '86_design_v3.json').read_text(encoding='utf-8'))
    old = pd.read_csv(PRIOR / '84a_capacity_policy_results.csv')
    rows = []
    for cap_id in CAP_IDS:
        current = core.capacity_nodes(n, cap_id)
        base = core.DispatchAllocatorV5(current, edges, 1., 'min_overload', 2)
        upgraded = {}
        for node in CANDIDATES:
            nn = current.copy()
            nn.loc[nn.node_id == node, 'overload_threshold'] += 1.
            check('one unit capacity increase', abs(nn.overload_threshold.sum() - current.overload_threshold.sum() - 1) < 1e-7)
            upgraded[node] = core.DispatchAllocatorV5(nn, edges, 1., 'min_overload', 2)
        configurations = 0
        for case, theta, demand_name in itertools.product(design['cases'], design['thresholds'], design['demands']):
            parts, groups = demands[demand_name]
            engine = core.DualMechanismRecoveryV5(base, edges, parts, groups, case['initial_ids'], case['kind'], theta, gamma=1.)
            failed, metric, _ = engine.closure(engine.initial)
            total_demand = engine.demand.sum()
            tag = dict(capacity_id=cap_id, case_id=case['case_id'], theta=theta, demand=demand_name)
            ref = old[(old.capacity_id == cap_id) & (old.arm == 'C') & (old.case_id == case['case_id'])
                      & (old.demand == demand_name) & np.isclose(old.theta, theta, atol=1e-12, rtol=0)]
            check('exact old C reference', len(ref) == 1)
            rr = ref.iloc[0]
            check('old C result reproduced', abs(metric['Service_Rate'] - rr.Service_Rate) < 1e-9
                  and {base.pos[k] for k in failed} == set(json.loads(rr.final_failed_indices_json)))
            target = independent_safe_lp(base, failed, engine.demand)
            check('baseline independent safe LP match', abs(target - metric['Served_Load']) < 1e-6)
            rows.append(dict(**tag, intervention='baseline', node_id='', capacity_increment=0.,
                             node_failed=False, served_load=metric['Served_Load'], service=metric['Service_Rate'],
                             gain_load=0., gain_service_pp=0., final_failed=len(failed),
                             final_ids_json=json.dumps(sorted(failed), ensure_ascii=False), oracle_served=target))
            for node, alloc in upgraded.items():
                engine2 = core.DualMechanismRecoveryV5(alloc, edges, parts, groups, case['initial_ids'], case['kind'], theta, gamma=1.)
                mm, util, _, _ = alloc.evaluate(failed, engine.demand)
                check('upgraded endpoint stable with C', util.max() <= 1 + 1e-7
                      and not engine2.classify(failed, util)['new'])
                if configurations == 0:
                    ff, full, _ = engine2.closure(engine2.initial)
                    check('capacity-independent risk closure full replay', ff == failed
                          and abs(full['Served_Load'] - mm['Served_Load']) < 1e-7)
                oracle = independent_safe_lp(alloc, failed, engine.demand)
                check('upgraded independent safe LP match', abs(oracle - mm['Served_Load']) < 1e-6)
                gain = mm['Served_Load'] - metric['Served_Load']
                check('one-unit gain nonnegative and bounded', -1e-6 <= gain <= 1 + 1e-6)
                check('C accounting', mm['Nominal_Excess_Load'] < 1e-6 and mm['Max_LP_Violation'] < 1e-5
                      and mm['Service_Conservation_Error'] < 1e-6)
                if node in failed:
                    check('capacity is not repair', abs(gain) < 1e-6)
                rows.append(dict(**tag, intervention='add_one_proxy_unit', node_id=node, capacity_increment=1.,
                                 node_failed=node in failed, served_load=mm['Served_Load'], service=mm['Service_Rate'],
                                 gain_load=gain, gain_service_pp=100 * gain / total_demand, final_failed=len(failed),
                                 final_ids_json=json.dumps(sorted(failed), ensure_ascii=False), oracle_served=oracle))
            configurations += 1
            base.paths.cache_clear()
            for a in upgraded.values():
                a.paths.cache_clear()
        check('24 point configurations ' + cap_id, configurations == 24)
        print('completed capacity point', cap_id, flush=True)
    frame = pd.DataFrame(rows)
    check('288 point configurations', len(frame) == 288)
    save(frame, '05_point_configurations.csv')
    effects = frame[frame.intervention == 'add_one_proxy_unit'].copy()
    summary = []
    for (cap_id, demand, node), g in effects.groupby(['capacity_id', 'demand', 'node_id']):
        summary.append(dict(capacity_id=cap_id, demand=demand, node_id=node, conditions=len(g),
                            mean_gain_load=float(g.gain_load.mean()), min_gain_load=float(g.gain_load.min()),
                            max_gain_load=float(g.gain_load.max()), positive=int((g.gain_load > 1e-7).sum()),
                            zero=int((g.gain_load.abs() <= 1e-7).sum()), negative=int((g.gain_load < -1e-7).sum()),
                            failed_conditions=int(g.node_failed.sum())))
    sums = pd.DataFrame(summary)
    save(sums, '06_point_response_summary.csv')
    pairwise = []
    for demand in design['demands']:
        for a, b in itertools.combinations(CANDIDATES, 2):
            diffs = []
            for cap_id in CAP_IDS:
                sub = sums[(sums.capacity_id == cap_id) & (sums.demand == demand)].set_index('node_id')
                difference = float(sub.loc[a, 'mean_gain_load'] - sub.loc[b, 'mean_gain_load'])
                diffs.append(difference)
                pairwise.append(dict(demand=demand, capacity_id=cap_id, candidate_A=a, candidate_B=b,
                                     A_minus_B_mean_gain=difference, order='A>B' if difference > 1e-7
                                     else 'B>A' if difference < -1e-7 else 'tie'))
            for r in pairwise[-3:]:
                r['strict_reversal_across_three_codes'] = bool(min(diffs) < -1e-7 and max(diffs) > 1e-7)
    save(pairwise, '07_point_pairwise_orders.csv')
    return frame, sums, pd.DataFrame(pairwise)


def report(gate, frame, summary, pairs, manifest):
    short = {node: node.split('@@')[0] for node in CANDIDATES}
    lines = ['|容量编码|需求|候选|平均完成量增益|有正效应格数/12|节点失效格数/12|',
             '|---|---|---|---|---|---|']
    for r in summary.itertuples():
        lines.append(f'|{r.capacity_id}|{r.demand}|{short[r.node_id]}|{r.mean_gain_load:.6f}|{r.positive}|{r.failed_conditions}|')
    real_reversals = pairs[pairs.strict_reversal_across_three_codes].drop_duplicates(['demand', 'candidate_A', 'candidate_B'])
    claims = dict(first_round_complete=True, empirical_ranking_ready=False,
                  real_world_nonidentifiability_proved=False, optimal_measurement_identified=False,
                  model_configurations=len(frame), conditional_interventions=int((frame.intervention != 'baseline').sum()),
                  strict_code_reversal_pairs=len(real_reversals), all_checks_passed=all(r['passed'] for r in CHECKS),
                  production_files_changed=False, next_stage='stop_empirical_ranking_until_measurement_interface_defined')
    dump(claims, '09_findings.json')
    note = f'''# 方向1首轮：容量干预排序的证据门槛与条件性响应

日期：2026-10-19。首轮完成；现实稳健排序、现实排名反转证书及最优补数选择均未完成。原因是容量不确定集合与投入到能力增量的映射尚无同口径证据。这里没有把缺数据写成已经证明不可识别，也没有声称算法创新。

## 一、实质发现与停止判断

已有246节点的处理能力仍是代理编码。旧资料中20个重点节点身份审查、22条数值记录，本轮对三个候选的三份公开页核读，未提供可直接写入当前模型的同设施、同期间、同单位有限容量区间。当前核查范围不足以断言全网任何地方都不存在可用资料。

81套既有容量情景明确标为非实测标定；固定总容量和扰动幅度是设计假设，不是观测给出的物理约束。本轮只使用原主线的三个既定点检查模型响应，不把它们组成现实不确定集合。另无三设施单位增容成本，无法比较同货币预算。黄金口业务处理单价不是扩建价格。

因此，方向1的实证门槛当前未过。停止增加参数点和优化器：没有有依据的集合U，就不报告min/max_U收益界、排名保证、现实非识别反例或数值信息价值。这个限制比求解难度更早；我此前建议应收紧为“先通过数据接口门槛，再研究排序”。

## 二、本轮新增公开证据

- 黄金口：[中国邮政2023招标公告](https://www.chinapost.com.cn/cn/report/2308/3704-1.htm)有预测业务量、按月日均量计费阶梯及服务期。可以支持处理业务存在，不能把计费分段当容量上下限，也不能把外包处理费当增容价格。
- 京东亚一：[武汉市政府2023年报道](https://www.wuhan.gov.cn/ztzl/23zt/xczjc/202301/t20230119_2136808.shtml)给出二期当日约75万件与日订单处理能力超过100万单的表述。年份、二期范围、件/订单口径及原POI对应仍须闭合；没有可直接使用的有限能力上界。
- 中通武汉：[东西湖区2024年报道](https://www.dxh.gov.cn/ZSYZ/XMTJ_16715/202406/t20240617_2416528.shtml)给出一期进港、二期出港及合计日均320万票。两期合计不能自动分给一个POI，进出港计数不能自动当不重复订单，实际量也不是最大能力。

01是带URL/日期/事实与限制的核读记录，不是原始HTML快照；02逐条保留旧22条数值为何不能成为本模型容量界；03是三个候选的缺口表。公开原页中的采购操作说明不构成本任务的行动指令，本轮没有注册、采购或联系运营方。

## 三、已经完成的条件性计算

固定原图、风险参数、需求与损伤，采用安全派入C。三个候选由会话事先指定；3套既有容量编码×4损伤×3阈值×2需求=72基线，每个分别为三个候选新增1代理容量单位，共216次干预、288个终态配置。全部基线匹配旧84的C服务和失效集合；所有终态用独立单阶段运输LP核对。每个容量点首条件的三个增容另完整重放闭包，核对容量不改变纯风险失效集合。

新增1单位是计算中的单位归一化，既不是现实误差半径，也不是1吨/日、1件/日或同预算。负荷保持原代理值；这不表示容量和负荷已被独立观测。以下是等权网格描述均值，不是真实场景期望，不将格子当独立事件样本。失效节点增容不等于修复。

{chr(10).join(lines)}

三套编码间有严格正负反转的“候选对×需求”数量为{len(real_reversals)}。详细正、零、负响应和并列见05—07表。无论有无反转，都不能用有限点的极值充当连续集合的严格收益界；未见反转也不证明稳健排序。

## 四、为什么现有计算本身不是新方法

C下不存在过载退出，风险压力不含容量或利用率，故固定其他输入时风险闭包F不依赖K。在该F上，固定本地优先承接b_j=min(D_j,K_j)，失效节点b=0；剩余来源量D-b和存活接收余量K-b组成标准运输最大流。终态总完成H=sum(b)+Q_safe。

本轮独立LP对每个配置核对该式。增容响应因此首先是既有局部优先规则下的标准流量敏感性，没有据此建立新结构定理、稳健优化算法或最优测量策略。若只停留在这个层面，应按原停损线停止方法包装。

## 五、下一步所需的是测量接口，不是更多仿真

08表提供三个候选的补数清单，顺序只是逻辑依赖，并非已计算出的信息价值排序：

1. 确认同一个物理场地/期次/作业环节，排除公司、园区与车间重复计量；当前京东二期、中通一期/二期尤其需明确。
2. 选择同一研究期间、同一计量单位；需求、完成、能力必须相容。需说明袋/件/票/订单的统计关系，不能直接互换。
3. 获得该口径下可持续处理能力的测量或有依据的上下界，同时区分设备名义速度、实测业务量和能力。仅一日吞吐通常不足以给出能力上界。
4. 若目标是现实投资优先级，补充每个候选的增容方案、可增加能力、价格、实施条件。若只研究相同物理能力增量，则明确放弃“同预算”解释。

只核实一个节点的K通常也不足以确定全网排序：其他节点容量、负荷、边的实际转运含义仍可能不确定。本轮没有指定“最值得测的一个节点”，因为没有有依据的U、观测误差模型或先验；不虚构这些要素来算信息价值。不把上述公开能力数字硬套到当前246节点代理网络。

原一周探索投入上限不重置。当前可交付的是可复核的证据缺口、条件性响应与标准流等价核对；当前不能交付的是现实稳健排名或已经确认的新颖性。

## 六、复现

脚本110_capacity_rank_identifiability.py。Python {manifest['runtime']['python']}，NumPy {manifest['runtime']['numpy']}，pandas {manifest['runtime']['pandas']}，SciPy {manifest['runtime']['scipy']}，HiGHS。00在运行前冻结；source_manifest记录旧源码、原数据、相关历史实验、资料及指定稿件哈希。10_validation保存检查；completion含结果与新增源码哈希。原文件未改。
'''
    (OUT / '方向1_容量排序可识别性首轮结果_20261019.md').write_text(note, encoding='utf-8')
    return claims


def main():
    start = time.perf_counter()
    manifest = begin()
    n = pd.read_csv(DATA / '04b_武汉节点代理容量.csv')
    gate = evidence_gate(n)
    tasks = []
    for node in CANDIDATES:
        for order, field, reason in [
            (1, '物理场地、期次、作业环节与node_id对应', '先于数值标定；避免合并或重复'),
            (2, '同期间同口径需求与完成量', '用于分离实际业务量与能力'),
            (3, '可持续处理能力及其测量误差或依据明确的界', '定义容量可行集合；不是用±比例代替'),
            (4, '与模型单位的可解释映射', '原代理负荷无物理单位，不能直接混算'),
            (5, '增容方案、成本与能力增量', '用于同货币预算，不能用服务计费阶梯代替')]:
            tasks.append(dict(node_id=node, dependency_order=order, required_field=field,
                              purpose=reason, value_of_information_rank_not_established=True))
    save(tasks, '08_measurement_dependency_checklist.csv')
    frame, summary, pairs = point_audit()
    for p, h in manifest['protected'].items():
        check('final preserved ' + p, sha(p) == h)
    check('design preserved', sha(OUT / '00_design.md') == manifest['design_sha256'])
    check('computation source preserved', sha(__file__) == manifest['new_source_sha256'])
    claims = report(gate, frame, summary, pairs, manifest)
    save(CHECKS, '10_validation.csv')
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'completion.json']
    dump(dict(completed=True, scope='evidence gate and conditional point audit only',
              checks=len(CHECKS), passed=sum(r['passed'] for r in CHECKS),
              new_source_sha256=sha(__file__), elapsed_seconds=time.perf_counter() - start,
              files={str(p.relative_to(OUT)): sha(p) for p in files}, findings=claims), 'completion.json')
    print(json.dumps(claims, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
