"""Conditional two-facility audit, not a calibrated logistics failure model.

Tests a declared hypothetical persistence rule, including no-endogenous-failure
controls. Original V5 service-rate overload and this stock rule are distinct.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import platform
import time
import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '项目结果/queue_stress_coupling_audit_v2_20261018'
DOC = ROOT / '项目文档/双机制动态耦合候选规则与小网络审计_20261018.md'
POLICIES = ('no_buffer', 'safe_buffer_optimal', 'max_admit',
            'ignore_stress_planner', 'offline_optimal')
LAMBDA = 10
SAFE = 1
BUFFER = 2


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(obj, p):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def csv_write(rows, p):
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@dataclass(frozen=True)
class Config:
    arrivals: tuple
    duration: int  # 0 explicitly disables the endogenous stress rule
    holding: int
    risk: str     # off, A_to_B, B_to_A
    shock: str    # none, B_at_1 (start of slot 1)


def risk_closure(mask, risk):
    if risk == 'A_to_B' and mask & 1:
        return mask | 2
    if risk == 'B_to_A' and mask & 2:
        return mask | 1
    return mask


def prepare(t, state, cfg):
    q0, q1, z0, z1, failed = state
    ext = 2 if cfg.shock == 'B_at_1' and t == 1 else 0
    mask = risk_closure(failed | ext, cfg.risk)
    new = mask & ~failed
    lost = (q0 if new & 1 else 0) + (q1 if new & 2 else 0)
    prepared = (0 if mask & 1 else q0, 0 if mask & 2 else q1,
                0 if mask & 1 else z0, 0 if mask & 2 else z1, mask)
    return prepared, lost, (ext & ~failed).bit_count(), (new & ~ext).bit_count()


def actions(prepared, demand, cap):
    q0, q1, _, _, f = prepared
    bounds = [0 if f & (1 << i) else max(0, 1 + cap - q)
              for i, q in enumerate((q0, q1))]
    for a0 in range(min(demand, bounds[0]) + 1):
        for a1 in range(min(demand - a0, bounds[1]) + 1):
            yield a0, a1


def advance(t, state, action, cfg, stress_enabled=True):
    pre, early_loss, ext_count, early_risk = prepare(t, state, cfg)
    q0, q1, z0, z1, f = pre
    a0, a1 = action
    d = cfg.arrivals[t]
    assert action in tuple(actions(pre, d, BUFFER))
    s = [min(1, q + a) if not f & (1 << i) else 0
         for i, (q, a) in enumerate(zip((q0, q1), action))]
    post = [q0 + a0 - s[0], q1 + a1 - s[1]]
    z = [(old + 1 if q > SAFE else 0) if not f & (1 << i) else 0
         for i, (old, q) in enumerate(zip((z0, z1), post))]
    stress = 0
    if stress_enabled and cfg.duration:
        for i in range(2):
            if not f & (1 << i) and z[i] >= cfg.duration:
                stress |= 1 << i
    after = risk_closure(f | stress, cfg.risk)
    new = after & ~f
    late_risk = new & ~stress
    # Overlap is a trigger label, not an additive causal decomposition.
    overlap = 0
    if cfg.risk == 'A_to_B' and after & 1 and stress & 2:
        overlap = 2
    if cfg.risk == 'B_to_A' and after & 2 and stress & 1:
        overlap = 1
    late_loss = sum(post[i] for i in range(2) if new & (1 << i))
    end_q = [0 if after & (1 << i) else post[i] for i in range(2)]
    end_z = [0 if after & (1 << i) else z[i] for i in range(2)]
    rejected = d - a0 - a1
    waiting = sum(end_q) if t < len(cfg.arrivals) - 1 else 0
    lost = early_loss + late_loss
    cost = LAMBDA * (rejected + lost) + cfg.holding * waiting
    nxt = (*end_q, *end_z, after)
    assert all(0 <= q <= BUFFER for q in end_q)
    assert sum((q0, q1)) + a0 + a1 == sum(s) + sum(end_q) + late_loss
    event = dict(arrival=d, accepted=a0 + a1, a_A=a0, a_B=a1,
                 completed=sum(s), rejected=rejected, lost=lost,
                 early_loss=early_loss, late_loss=late_loss,
                 waiting=waiting, queue_A=end_q[0], queue_B=end_q[1],
                 stress_A=end_z[0], stress_B=end_z[1], failed_before=state[-1],
                 failed_after=after, external_triggers=ext_count,
                 risk_from_external=early_risk, stress_triggers=stress.bit_count(),
                 risk_only_after_stress=late_risk.bit_count(),
                 stress_risk_overlap=overlap.bit_count(), stage_cost_ticks=cost)
    return nxt, cost, event


def planner(cfg, cap=BUFFER, stress_enabled=True):
    @lru_cache(None)
    def value(t, state):
        if t == len(cfg.arrivals):
            return LAMBDA * (state[0] + state[1]), (0, 0)
        pre = prepare(t, state, cfg)[0]
        candidates = []
        for a in actions(pre, cfg.arrivals[t], cap):
            nxt, c, _ = advance(t, state, a, cfg, stress_enabled)
            # Lower admission resolves equal costs. Canonical A before B only
            # after objective and total admission; both risk orientations run.
            candidates.append((c + value(t + 1, nxt)[0], sum(a), -a[0], a))
        best = min(candidates)
        return best[0], best[-1]
    return value


def simulate(cfg, policy):
    state = (0, 0, 0, 0, 0)
    cap = 0 if policy == 'no_buffer' else SAFE if policy == 'safe_buffer_optimal' else BUFFER
    plan = None if policy == 'max_admit' else planner(
        cfg, cap, stress_enabled=policy != 'ignore_stress_planner')
    total = dict(completed=0, rejected=0, lost=0, waiting=0, accepted=0,
                 external_triggers=0, risk_from_external=0, stress_triggers=0,
                 risk_only_after_stress=0, stress_risk_overlap=0)
    cost = 0
    trace = []
    for t, d in enumerate(cfg.arrivals):
        if policy == 'max_admit':
            pre = prepare(t, state, cfg)[0]
            def rank(a):
                q = [max(0, pre[i] + a[i] - 1) for i in range(2)]
                return -sum(a), sum(x * x for x in q), -a[0], -a[1]
            action = min(actions(pre, d, BUFFER), key=rank)
        else:
            action = plan(t, state)[1]
        nxt, c, event = advance(t, state, action, cfg, True)
        for k in total:
            total[k] += event[k]
        cost += c
        event.update(t=t, policy=policy)
        trace.append(event)
        state = nxt
        assert sum(cfg.arrivals[:t + 1]) == total['completed'] + total['rejected'] + total['lost'] + state[0] + state[1]
        assert total['accepted'] == total['completed'] + total['lost'] + state[0] + state[1]
    terminal = state[0] + state[1]
    cost += LAMBDA * terminal
    assert cost == LAMBDA * (sum(cfg.arrivals) - total['completed']) + cfg.holding * total['waiting']
    if policy in ('offline_optimal', 'safe_buffer_optimal', 'no_buffer'):
        assert cost == plan(0, (0, 0, 0, 0, 0))[0]
    if policy in ('safe_buffer_optimal', 'no_buffer'):
        assert total['stress_triggers'] == 0
    return dict(**total, terminal_backlog=terminal, final_failed=state[-1].bit_count(),
                cost_ticks=cost, cost=cost / 10), trace


def brute_value(cfg):
    """Independent full action-tree enumeration, no main step/closure calls."""
    def expand(t, q, z, failed):
        if t == len(cfg.arrivals):
            return LAMBDA * sum(q)
        before = failed
        if cfg.shock == 'B_at_1' and t == 1:
            failed |= 2
        if cfg.risk == 'A_to_B' and failed & 1:
            failed |= 2
        if cfg.risk == 'B_to_A' and failed & 2:
            failed |= 1
        loss0 = sum(q[i] for i in range(2) if (failed & ~before) & (1 << i))
        q = tuple(0 if failed & (1 << i) else q[i] for i in range(2))
        z = tuple(0 if failed & (1 << i) else z[i] for i in range(2))
        best = float('inf')
        for a0 in range(cfg.arrivals[t] + 1):
            for a1 in range(cfg.arrivals[t] - a0 + 1):
                a = (a0, a1)
                if any((failed & (1 << i) and a[i]) or q[i] + a[i] > 3 for i in range(2)):
                    continue
                stock = tuple(max(0, q[i] + a[i] - 1) if not failed & (1 << i) else 0 for i in range(2))
                age = tuple(z[i] + 1 if stock[i] > 1 else 0 for i in range(2))
                newmask = failed
                if cfg.duration:
                    for i in range(2):
                        if not failed & (1 << i) and age[i] >= cfg.duration:
                            newmask |= 1 << i
                if cfg.risk == 'A_to_B' and newmask & 1:
                    newmask |= 2
                if cfg.risk == 'B_to_A' and newmask & 2:
                    newmask |= 1
                loss1 = sum(stock[i] for i in range(2) if (newmask & ~failed) & (1 << i))
                end = tuple(0 if newmask & (1 << i) else stock[i] for i in range(2))
                age = tuple(0 if newmask & (1 << i) else age[i] for i in range(2))
                wait = sum(end) if t + 1 < len(cfg.arrivals) else 0
                c = LAMBDA * (cfg.arrivals[t] - a0 - a1 + loss0 + loss1) + cfg.holding * wait
                best = min(best, c + expand(t + 1, end, age, newmask))
        return best
    return expand(0, (0, 0), (0, 0), 0)


@lru_cache(None)
def fixed_availability_lp(cfg):
    """Time-expanded LP after disabling endogenous stress. Same exogenous loss."""
    n = len(cfg.arrivals)
    alive = np.ones((n, 2), dtype=int)
    mask = 0
    for t in range(n):
        if cfg.shock == 'B_at_1' and t == 1:
            mask |= 2
        mask = risk_closure(mask, cfg.risk)
        alive[t] = [not mask & (1 << i) for i in range(2)]
    # Variables a[2*n],s[2*n],q[2*n],r[n].
    width = 7 * n
    eq = np.zeros((3 * n, width))
    rhs = np.zeros(3 * n)
    c = np.zeros(width)
    bounds = []
    for i in range(2 * n):
        t, v = divmod(i, 2)
        bounds.append((0, 3 if alive[t, v] else 0))
    bounds += [(0, int(alive[t, v])) for t in range(n) for v in range(2)]
    bounds += [(0, BUFFER if alive[t, v] else 0) for t in range(n) for v in range(2)]
    bounds += [(0, int(d)) for d in cfg.arrivals]
    for t in range(n):
        eq[t, 2 * t:2 * t + 2] = 1
        eq[t, 6 * n + t] = 1
        rhs[t] = cfg.arrivals[t]
        c[6 * n + t] = LAMBDA
        for v in range(2):
            i = 2 * t + v
            row = n + i
            eq[row, i] = 1
            eq[row, 2 * n + i] = -1
            eq[row, 4 * n + i] = -1
            if t and alive[t, v]:
                eq[row, 4 * n + i - 2] = 1
            if t == n - 1:
                c[4 * n + i] = LAMBDA
            else:
                c[4 * n + i] = cfg.holding + (LAMBDA if alive[t, v] and not alive[t + 1, v] else 0)
    result = linprog(c, A_eq=eq, b_eq=rhs, bounds=bounds, method='highs')
    assert result.success, result.message
    assert np.max(np.abs(eq @ result.x - rhs)) < 1e-7
    assert np.max(np.abs(result.x - np.round(result.x))) < 1e-7
    return float(result.fun)


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = [ROOT / '项目结果/dispatch_capacity_null_mainline_20261018/89_final_manifest.json',
                 ROOT / '项目结果/dynamic_buffer_tradeoff_audit_20261018/90_completion.json',
                 ROOT / '项目结果/dynamic_buffer_flow_equivalence_20261018/91_completion.json']
    preserved = {}
    for m in manifests:
        obj = json.loads(m.read_text(encoding='utf-8'))
        entries = obj.get('artifacts') or [dict(path=p, sha256=h) for p, h in obj['files'].items()]
        preserved.update({x['path']: x['sha256'] for x in entries})
    failed_folder = ROOT / '项目结果/queue_stress_coupling_audit_20261018'
    failure_record = failed_folder / '92_failure_record.json'
    failed_record = json.loads(failure_record.read_text(encoding='utf-8'))
    for name, h in failed_record['preserved_partial_files'].items():
        assert sha(failed_folder / name) == h
    for p, h in preserved.items():
        assert sha(ROOT / p) == h
    design = dict(source_sha256=sha(__file__), base_horizon=3, arrivals=[0, 3, 6],
                  drain_slots=[0, 2], duration=[1, 2, 4, 0], holding_ticks=[1, 5],
                  risk=['off', 'A_to_B', 'B_to_A'], shock=['none', 'B_at_1'],
                  facilities=2, processing_each=1, safe_waiting=SAFE, physical_buffer=BUFFER,
                  reject_lost_terminal_ticks=LAMBDA, calibrated=False,
                  rule='Consecutive end-of-slot stock above safe_waiting for duration slots causes hypothetical irreversible failure; duration=0 disables it',
                  timing='External shock and risk closure, admission, processing, stock stress and risk closure, stock loss, holding, next slot',
                  policies=list(POLICIES), information='Full deterministic demand and exogenous shock known to all optimization policies; max_admit ignores forecasts',
                  choice='No policy receives extra physical capacity. Restricted policies choose smaller feasible admission sets.',
                  positive_claim='Conditional structural diagnostic only. No empirical logistics shutdown mechanism established.',
                  environments=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  prior_manifests={str(m.relative_to(ROOT)): sha(m) for m in manifests},
                  previous_partial_failure_sha256=sha(failure_record),
                  correction='Only hand-calculated witnesses corrected; model dynamics unchanged. Added optimistic-value and tie diagnostics before rerun.')
    design_path = OUT / '92_design.json'
    if design_path.exists():
        assert json.loads(design_path.read_text(encoding='utf-8')) == design
    else:
        dump(design, design_path)
    rows, pairs, lp_checks = [], [], []
    count = 0
    for duration in design['duration']:
        for base in product(design['arrivals'], repeat=3):
            for drain, h, risk, shock in product((0, 2), (1, 5), design['risk'], design['shock']):
                cfg = Config(base + (0,) * drain, duration, h, risk, shock)
                key = f'{duration}:{base}:{drain}:{h}:{risk}:{shock}'
                metrics = {}
                for policy in POLICIES:
                    m, _ = simulate(cfg, policy)
                    metrics[policy] = m
                    rows.append(dict(case_key=key, duration=duration, drain_slots=drain,
                                     arrivals_json=json.dumps(cfg.arrivals), holding_ticks=h,
                                     risk=risk, shock=shock, policy=policy, **m))
                opt = metrics['offline_optimal']['cost_ticks']
                safe = metrics['safe_buffer_optimal']['cost_ticks']
                optimistic_bound = planner(cfg, stress_enabled=False)(0, (0, 0, 0, 0, 0))[0]
                assert opt >= optimistic_bound
                assert all(m['cost_ticks'] >= opt for m in metrics.values())
                assert metrics['no_buffer']['cost_ticks'] >= safe
                if duration == 1:
                    assert opt == safe, 'Immediate stock failure should reduce to safe buffering'
                if duration == 0:
                    assert opt == metrics['ignore_stress_planner']['cost_ticks']
                    lp = fixed_availability_lp(cfg)
                    assert abs(lp - opt) < 1e-7
                    lp_checks.append(dict(case_key=key, dp_ticks=opt, lp_ticks=lp, passed=True))
                if drain:
                    assert all(m['terminal_backlog'] == 0 for m in metrics.values())
                pairs.append(dict(case_key=key, duration=duration, drain_slots=drain, holding_ticks=h,
                                  risk=risk, shock=shock, safe_cost_ticks=safe, optimal_cost_ticks=opt,
                                  max_admit_cost_ticks=metrics['max_admit']['cost_ticks'],
                                  ignore_stress_cost_ticks=metrics['ignore_stress_planner']['cost_ticks'],
                                  optimistic_no_stress_bound_ticks=optimistic_bound,
                                  true_value_gap_ticks=opt - optimistic_bound,
                                  gain_vs_safe_ticks=safe - opt,
                                  regret_ignoring_stress_ticks=metrics['ignore_stress_planner']['cost_ticks'] - opt))
                count += 1
        print(f'duration={duration}: {count} configurations complete', flush=True)
    csv_write(rows, OUT / '92a_policy_results.csv')
    csv_write(pairs, OUT / '92b_paired_results.csv')
    csv_write(lp_checks, OUT / '92c_no_stress_flow_checks.csv')
    brute = []
    for arrivals, d, risk, shock in product(product((0, 3, 6), repeat=2), (1, 2, 0), design['risk'], design['shock']):
        cfg = Config(arrivals, d, 1, risk, shock)
        value = planner(cfg)(0, (0, 0, 0, 0, 0))[0]
        independent = brute_value(cfg)
        assert value == independent
        brute.append(dict(arrivals_json=json.dumps(arrivals), duration=d, risk=risk,
                          shock=shock, dp_ticks=value, brute_ticks=independent, passed=True))
    csv_write(brute, OUT / '92d_independent_enumeration.csv')
    examples, traces, example_checks = [], [], []
    example_design = [('isolated_peak', (6, 0, 0), 2, 'off'),
                      ('repeated_peak', (6, 6, 0, 0), 2, 'off'),
                      ('repeated_peak_with_propagation', (6, 6, 0, 0), 2, 'A_to_B'),
                      ('true_feedback_gap', (0, 6, 3, 0, 0), 2, 'off'),
                      ('feedback_gap_with_propagation', (0, 6, 3, 0, 0), 2, 'A_to_B'),
                      ('immediate_stock_exit', (6, 0, 0), 1, 'off')]
    for name, arrivals, duration, risk in example_design:
        cfg = Config(arrivals, duration, 1, risk, 'none')
        actual_optimum = planner(cfg)(0, (0, 0, 0, 0, 0))[0]
        optimistic = planner(cfg, stress_enabled=False)(0, (0, 0, 0, 0, 0))[0]
        assert brute_value(cfg) == actual_optimum
        example_checks.append(dict(example=name, dp_ticks=actual_optimum,
                                   independent_brute_ticks=actual_optimum,
                                   no_stress_optimistic_bound_ticks=optimistic,
                                   true_value_gap_ticks=actual_optimum - optimistic,
                                   passed=True))
        for policy in POLICIES:
            m, trace = simulate(cfg, policy)
            examples.append(dict(example=name, arrivals_json=json.dumps(arrivals), duration=duration,
                                 risk=risk, policy=policy, **m))
            traces.extend(dict(example=name, **event) for event in trace)
    csv_write(examples, OUT / '92e_examples.csv')
    csv_write(traces, OUT / '92f_example_traces.csv')
    csv_write(example_checks, OUT / '92h_example_independent_enumeration.csv')
    by = {(x['example'], x['policy']): x for x in examples}
    assert by['isolated_peak', 'offline_optimal']['cost_ticks'] == 6
    assert by['isolated_peak', 'safe_buffer_optimal']['cost_ticks'] == 22
    assert by['repeated_peak', 'offline_optimal']['cost_ticks'] == 46
    assert by['repeated_peak', 'safe_buffer_optimal']['cost_ticks'] == 62
    assert by['repeated_peak', 'max_admit']['cost_ticks'] == 84
    summary = []
    for duration, drain in product(design['duration'], (0, 2)):
        g = [p for p in pairs if p['duration'] == duration and p['drain_slots'] == drain]
        summary.append(dict(duration=duration, drain_slots=drain, conditions=len(g),
                            opt_better_than_safe=sum(p['gain_vs_safe_ticks'] > 0 for p in g),
                            opt_equal_safe=sum(p['gain_vs_safe_ticks'] == 0 for p in g),
                            true_feedback_value_gap=sum(p['true_value_gap_ticks'] > 0 for p in g),
                            ignore_stress_suboptimal=sum(p['regret_ignoring_stress_ticks'] > 0 for p in g),
                            planner_regret_without_value_gap=sum(p['regret_ignoring_stress_ticks'] > 0 and p['true_value_gap_ticks'] == 0 for p in g),
                            max_admit_suboptimal=sum(p['max_admit_cost_ticks'] > p['optimal_cost_ticks'] for p in g)))
    csv_write(summary, OUT / '92g_stratified_summary.csv')
    table = ['|连续超额期数L，0表示关闭|附加清空期|配置数|最优优于安全缓冲|与安全缓冲持平|真实反馈最优值差|忽略持续性规则次优|规划次优但无最优值差|尽量接收次优|',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in summary:
        table.append('|' + '|'.join(str(r[k]) for k in r) + '|')
    example_table = ['|小例|策略|完成|拒收|接受后损失|最终失效节点|等待批时|目标成本|',
                     '|---|---|---|---|---|---|---|---|']
    for r in examples:
        example_table.append(f"|{r['example']}|{r['policy']}|{r['completed']}|{r['rejected']}|{r['lost']}|{r['final_failed']}|{r['waiting']}|{r['cost']}|")
    report = '''# 双机制动态耦合候选规则与小网络审计

日期：2026-10-18。研究性质是合成反事实诊断，不是武汉设施标定，不是原V5升级版，也不是论文正文。旧代码、正文及结果未覆盖。

## 1. 研究判断与红线

暂时缓冲可提高累计完成，持续拥挤也可能影响未来可用能力，这在数学上可以产生有取舍的控制问题。但“积压连续超过某阈值就永久退出”目前没有本项目的设施运行证据，不能因数值好看就采纳为真实机制。

本轮把这一假说显式标记为候选规则，设置立即退出、允许短暂超额、关闭内生退出三类对照，检验它们是否再次把答案预设。参数、权重和时间均是示意量。当前产出只能决定是否值得继续论证，不支持现实应急超额有益或有害的断言。

## 2. 模型对象与时间顺序

两个设施A/B，每期各最多实际处理1批；处理后普通等待上限为1批，物理缓冲上限为2批。每期新需求进入公共待分配池，控制器决定向两节点分别接多少，未接收的计拒收。没有运输能力约束、实际城市OD、维修或保护。所有政策获得同一物理能力。

状态为两个库存Q、两个连续拥挤计数Z、不可逆失效集合F。Q>1的期末状态使Z增加，否则清零；若连续L期Q>1，则该节点在本期处理完成之后退出。L=0专门表示关闭该机制。它是“库存拥挤持续性”假说，不是原V5按承接利用率定义的过载。

风险机制分别设关闭、A退出导致B退出、B退出导致A退出。风险只看失效集合，不混入利用率；在两个节点上求闭包。该二值有向规则只检验机制结构，不声称复现V5加权压力值。原拓扑和全部风险参数没有被修改。

每时段依次执行外部冲击及其风险闭包、接收、受K限制的实际处理、持续性退出及其风险闭包、未完成库存损失、持有成本。已完成货物不因期末退出而倒扣。失效节点库存计接受后损失，不能消失或改成拒收。外部冲击设无冲击或第1期开始时B退出，全部预先固定。

## 3. 账本与目标

系统守恒为累计到达=累计实际完成+累计拒收+累计接受后损失+当前库存；累计接收=累计实际完成+累计接受后损失+当前库存。全程逐期核对。

拒收、接受后损失与期末库存的单位罚损均设1，没有额外提高毁货罚损来制造效果。等待费h取0.1和0.5，最后一期不收取通往不存在下一期的等待费。目标J=拒收+接受后损失+期末库存+h*等待，等价于总到达-累计完成+h*等待。接收本身没有收益，库存不算完成。

## 4. 政策与公平比较边界

- no_buffer，仅允许当期可处理的接收，是旧静态安全派入的一种小例类比，不是原246节点C。
- safe_buffer_optimal，允许普通缓冲1，但不用应急第二批库存；在此受限动作集中求离线精确最优，强于随意设置的安全贪心。
- max_admit，尽量接收，再平衡两节点库存，最后按固定节点编号打破平局；不利用未来需求。两种传播方向均纳入，避免只挑一个有利编号方向。
- ignore_stress_planner，知道完整需求、外部冲击和风险方向，规划时忽略库存引发的退出，执行时仍受到真实候选规则约束；每期重新规划。它是模型错配对照，不是正确费用流算法失效的证明。
- offline_optimal，知道完整需求和机制，在物理缓冲2的动作集中求Bellman精确最优，是小网络离线基准，不是已设计的现实在线算法。

三个优化政策享有同一完整未来信息；restricted safe与full最优的动作集不同，full不劣于safe可由可行域包含直接推出，不能作为创新。需要研究的是额外缓冲何时被使用、收益幅度和忽略状态反馈的代价。

## 5. 结构性判别结果

L=1时，超过普通缓冲立即导致退出，额外接收没有未来服务机会。精确最优与安全缓冲相同；单纯换成队列变量，仍可能重新制造原来的不对称。

L=0时，退出不由接收决定；外部冲击和风险闭包的节点可用时序可以预先计算，问题再次回到标准时间展开费用流。独立LP显式计入已知退出时的库存损失，核对最优值，避免把毁货忽略成免费退出。

L>=2时，短时超过普通缓冲有机会在未来消退，持续超过才触发退出。因此存在条件性小例，使最优控制同时优于“永不用应急缓冲”和“尽量接收”。但这一取舍来自新假说，不是现实证据，也不证明新颖性或算法困难度。

另比较真实机制精确最优与关闭内生退出的乐观最优下界。两者最优值存在严格差距时，才说明规则真正改变了最优代价；忽略规则的某次执行变差但两者最优值相等时，不能排除等价最优计划的选择问题。本轮将二者分开计数，避免把求解器平局误认成新增决策价值。

## 6. 精确小例

isolated_peak的到达为[6,0,0]，repeated_peak为[6,6,0,0]；L=2，h=0.1。另保留有向传播版本和L=1对照，完整轨迹保存。

''' + '\n'.join(example_table) + '''

在repeated_peak关闭风险传播的小例中，安全缓冲完成6批，成本6.2；尽量接收导致两节点退出，只完成4批，成本8.4；离线最优完成8批，成本4.6。最优策略的关键是安排缓冲的使用时机，避免连续两期超过普通缓冲。这只是该假说下的可手算见证，不是全网实证。

不过，该repeated_peak小例的无退出乐观最优同样为4.6，并不能单独证明失效反馈改变了最优值。另设完整网格已包含的[0,6,3,0,0]，在L=2、h=0.1、无传播时，乐观下界为1.9、真实机制最优为2.6；有向传播版本也保留。该例用于区分真正的最优值差与模型错配执行、等价最优选择。

第一版未完成报告，原因是手算断言预期4.8和6.4，独立精确结果为4.6和6.2。核查轨迹确认最优把接货推迟，减少了等待。原脚本、冻结设计、部分输出与失败记录均保留；本修订只纠正手算值并增加上述诊断，未改变状态转移规则。

## 7. 完整网格与独立验证

''' + f'''预先固定三期到达0/3/6的全部27种序列，原时域与追加两期无到达，L取1/2/4/关闭，h取0.1/0.5，三种传播结构、两种外部冲击，共{count}个配置、{len(rows)}条政策结果。附加两期足以清空尚存节点物理上限2的库存，所有政策逐条检查，故不会通过把积压留到时域外虚增收益。

''' + '\n'.join(table) + f'''

这些计数是确定性设计网格，不是独立现实样本，不做P值或跨城推广。忽略规则的策略次优也不等于标准流本身错误，因为它规划的模型与执行环境不同。

独立完整动作树枚举{len(brute)}个两期配置及{len(example_checks)}个完整小例，不调用主状态转移、风险闭包或Bellman函数，目标值全部一致。关闭内生退出后另有{len(lp_checks)}个时间展开连续LP核对，最优值一致且LP解满足整数性。所有配置还核对轨迹守恒、接收/完成分离、缓冲界、代价恒等式和精确最优下界。手算例另有固定断言。

## 8. 接下来应该和不应该做什么

可以保留这个候选分支，用来提问“短暂可容忍、持续会退化”是否具有设施运营依据。下一步优先核对设施故障/拥堵记录、既有研究的具体状态和时间定义；找不到就继续标未标定，不无限纠结，也不能升级成现实主模型。

如果只有拥堵降低效率的依据，而没有永久退出依据，应改成有依据的服务率变化或暂时不可用，再重新审计可解析性，不能借用永久退出夸大级联。若原始处理能力与缓冲容量仍无法区分，暂不扩展武汉全网。

不建议现在设计新的元启发式、给规则起算法名、扫到有利阈值后宣称创新，或将此小例直接替换V5的物理机制。本轮没有证明新问题的复杂性、近似保证、现实参数或期刊级别。

## 9. 文件与复现

脚本为数据及代码/Code/92b_queue_stress_coupling_audit.py；结果在项目结果/queue_stress_coupling_audit_v2_20261018。92_design.json在网格执行前写入并冻结来源哈希。Python {platform.python_version()}、NumPy {np.__version__}、SciPy {scipy.__version__}，使用SciPy HiGHS。此前89/90/91清单的{len(preserved)}个文件在运行前后核对；正文不动。
'''
    DOC.write_text(report, encoding='utf-8')
    (OUT / DOC.name).write_text(report, encoding='utf-8')
    for p, h in preserved.items():
        assert sha(ROOT / p) == h
    completion = dict(completed=True, configurations=count, policy_rows=len(rows),
                      independent_action_enumerations=len(brute) + len(example_checks), no_stress_lp_checks=len(lp_checks),
                      examples=len(example_design), calibrated=False,
                      real_world_failure_mechanism_established=False, manuscript_changed=False,
                      old_artifacts_preserved=len(preserved), elapsed_seconds=round(time.time() - start, 3),
                      files={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), DOC, *sorted(OUT.glob('*'))] if p.is_file() and p.name != '92_completion.json'})
    dump(completion, OUT / '92_completion.json')
    print(json.dumps({k: v for k, v in completion.items() if k != 'files'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
