"""Reversible finite-buffer blocking on a small shared FIFO input lane.

Synthetic operating module only. No endogenous permanent failure, cargo loss,
Wuhan calibration, new algorithm claim, or modification of the V5 model.
"""
from dataclasses import dataclass
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
OUT = ROOT / '项目结果/reversible_blocking_prototype_20261018'
DOC = ROOT / '项目文档/可恢复拥堵最小原型与精确基线_20261018.md'
ALPHABET = ((), (0,), (1,), (0, 1), (1, 0))
POLICIES = ('direct_only_optimal', 'max_accept_fifo', 'rolling_2_perfect', 'offline_fifo', 'offline_bypass')
REJECT = 10
BUFFER = 1


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(obj, p):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def save_csv(rows, p):
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@dataclass(frozen=True)
class Config:
    arrivals: tuple
    lane_capacity: int
    holding: int
    outage: str


def capacities(t, outage):
    a = 0 if outage == 'A_down_01' and t in (0, 1) or outage == 'A_down_1' and t == 1 else 1
    b = 0 if outage == 'B_down_01' and t in (0, 1) or outage == 'B_down_1' and t == 1 else 1
    return a, b


def admissions(word, room):
    # Unique accepted subsequences, preserving all relative arrival order.
    return tuple(sorted({tuple(word[i] for i in range(len(word)) if mask & (1 << i))
                         for mask in range(1 << len(word))
                         if mask.bit_count() <= room}, key=lambda x: (len(x), x)))


def step(t, state, accept, cfg, bypass=False):
    lane, p0, p1 = state
    d = cfg.arrivals[t]
    assert accept in admissions(d, cfg.lane_capacity - len(lane))
    k = capacities(t, cfg.outage)
    served = (min(p0, k[0]), min(p1, k[1]))
    buffers = [p0 - served[0], p1 - served[1]]
    packed = lane + accept
    assert len(packed) <= cfg.lane_capacity
    transferred = []
    if bypass:
        waiting = []
        for v in packed:
            if buffers[v] < BUFFER:
                buffers[v] += 1
                transferred.append(v)
            else:
                waiting.append(v)
        final_lane = tuple(waiting)
    else:
        index = 0
        while index < len(packed) and buffers[packed[index]] < BUFFER:
            v = packed[index]
            buffers[v] += 1
            transferred.append(v)
            index += 1
        final_lane = packed[index:]
    blocked = bool(final_lane) and buffers[final_lane[0]] == BUFFER
    hol_opportunity = blocked and any(buffers[v] < BUFFER for v in final_lane[1:])
    stock = len(final_lane) + sum(buffers)
    waiting_cost_units = stock if t < len(cfg.arrivals) - 1 else 0
    rejected = len(d) - len(accept)
    cost = REJECT * rejected + cfg.holding * waiting_cost_units
    event = dict(t=t, arrivals_json=json.dumps(d), accepted_json=json.dumps(accept),
                 lane_before_json=json.dumps(lane), buffer_A_before=p0, buffer_B_before=p1,
                 processing_A=k[0], processing_B=k[1], completed_A=served[0], completed_B=served[1],
                 accepted=len(accept), completed=sum(served), rejected=rejected,
                 transferred_json=json.dumps(transferred), lane_after_json=json.dumps(final_lane),
                 buffer_A_after=buffers[0], buffer_B_after=buffers[1], inventory=stock,
                 lane_full_at_admission=int(len(lane) == cfg.lane_capacity),
                 input_blocked_at_end=int(blocked), hol_opportunity_missed=int(hol_opportunity),
                 waiting_units=waiting_cost_units, stage_cost_ticks=cost)
    assert len(lane) + p0 + p1 + len(accept) == sum(served) + stock
    assert len(final_lane) <= cfg.lane_capacity and all(0 <= x <= BUFFER for x in buffers)
    return (final_lane, *buffers), cost, event


def planner(cfg, *, bypass=False, direct=False, stop=None):
    end = len(cfg.arrivals) if stop is None else stop

    @lru_cache(None)
    def value(t, state):
        if t == end:
            return REJECT * (len(state[0]) + state[1] + state[2]), ()
        candidates = []
        for a in admissions(cfg.arrivals[t], cfg.lane_capacity - len(state[0])):
            nxt, c, _ = step(t, state, a, cfg, bypass)
            if direct and nxt[0]:
                continue
            cost = c + value(t + 1, nxt)[0]
            candidates.append((cost, len(a), a))
        if not candidates:
            raise AssertionError('No feasible action in reachable state')
        best = min(candidates)
        return best[0], best[-1]
    return value


def simulate(cfg, policy, initial=((), 0, 0)):
    state = initial
    if policy in ('offline_fifo', 'offline_bypass', 'direct_only_optimal'):
        plan = planner(cfg, bypass=policy == 'offline_bypass', direct=policy == 'direct_only_optimal')
    else:
        plan = None
    totals = dict(completed=0, accepted=0, rejected=0, waiting_units=0,
                  blocked_slots=0, hol_missed_slots=0, entry_full_slots=0, blockage_clearances=0)
    cost = 0
    trace = []
    previously_blocked = False
    for t, word in enumerate(cfg.arrivals):
        if policy == 'max_accept_fifo':
            candidates = []
            for a in admissions(word, cfg.lane_capacity - len(state[0])):
                _, _, event = step(t, state, a, cfg)
                # Maximum admission, then least immediate lane inventory, then
                # least head-of-line obstruction, then fixed class numbering.
                candidates.append((-len(a), len(json.loads(event['lane_after_json'])),
                                   event['hol_opportunity_missed'], a))
            accept = min(candidates)[-1]
        elif policy == 'rolling_2_perfect':
            accept = planner(cfg, stop=min(t + 2, len(cfg.arrivals)))(t, state)[1]
        else:
            accept = plan(t, state)[1]
        nxt, c, event = step(t, state, accept, cfg, policy == 'offline_bypass')
        totals['completed'] += event['completed']
        totals['accepted'] += event['accepted']
        totals['rejected'] += event['rejected']
        totals['waiting_units'] += event['waiting_units']
        totals['blocked_slots'] += event['input_blocked_at_end']
        totals['hol_missed_slots'] += event['hol_opportunity_missed']
        totals['entry_full_slots'] += event['lane_full_at_admission']
        totals['blockage_clearances'] += int(previously_blocked and not event['input_blocked_at_end'])
        previously_blocked = bool(event['input_blocked_at_end'])
        cost += c
        event.update(policy=policy)
        trace.append(event)
        state = nxt
        arrived = sum(map(len, cfg.arrivals[:t + 1])) + len(initial[0]) + initial[1] + initial[2]
        inventory = len(state[0]) + state[1] + state[2]
        assert arrived == totals['completed'] + totals['rejected'] + inventory
    terminal = len(state[0]) + state[1] + state[2]
    cost += REJECT * terminal
    demand = sum(map(len, cfg.arrivals)) + len(initial[0]) + initial[1] + initial[2]
    assert cost == REJECT * (demand - totals['completed']) + cfg.holding * totals['waiting_units']
    if plan is not None:
        assert cost == plan(0, initial)[0]
    return dict(**totals, terminal_inventory=terminal, cost_ticks=cost, cost=cost / 10), trace


def brute(cfg, initial=((), 0, 0), bypass=False):
    """Independent complete action tree, with no step/admissions/planner calls."""
    def visit(t, lane, p):
        if t == len(cfg.arrivals):
            return REJECT * (len(lane) + sum(p))
        k = capacities(t, cfg.outage)
        after = [max(0, p[i] - k[i]) for i in range(2)]
        best = float('inf')
        d = cfg.arrivals[t]
        for mask in range(1 << len(d)):
            a = tuple(d[i] for i in range(len(d)) if mask & (1 << i))
            if len(lane) + len(a) > cfg.lane_capacity:
                continue
            buf = after.copy()
            remaining = []
            stopped = False
            for v in lane + a:
                if buf[v] < 1 and (bypass or not stopped):
                    buf[v] += 1
                else:
                    remaining.append(v)
                    if not bypass:
                        stopped = True
            inventory = len(remaining) + sum(buf)
            wait = inventory if t + 1 < len(cfg.arrivals) else 0
            c = REJECT * (len(d) - len(a)) + cfg.holding * wait
            best = min(best, c + visit(t + 1, tuple(remaining), tuple(buf)))
        return best
    return visit(0, initial[0], (initial[1], initial[2]))


def fluid_lp(cfg):
    """Order-free two-destination LP relaxation, same slot/space convention.

    a,s,x,p,q indexed by (time,destination). Shared pre-transfer lane capacity
    retained. FIFO is deliberately relaxed. No unconditional integrality claim.
    """
    n = len(cfg.arrivals)
    width = 10 * n
    eq = np.zeros((4 * n, width))
    ub = np.zeros((3 * n, width))
    rhs = np.zeros(4 * n)
    ub_rhs = np.r_[np.full(n, cfg.lane_capacity), np.zeros(2 * n)]
    c = np.zeros(width)
    bounds = []
    demands = [[word.count(v) for v in range(2)] for word in cfg.arrivals]
    # Each block has 2*n variables: a, service, transfer, station p, lane q.
    bounds += [(0, demands[t][v]) for t in range(n) for v in range(2)]
    bounds += [(0, capacities(t, cfg.outage)[v]) for t in range(n) for v in range(2)]
    bounds += [(0, None)] * (2 * n)
    bounds += [(0, BUFFER)] * (2 * n)
    bounds += [(0, cfg.lane_capacity)] * (2 * n)
    for t in range(n):
        for v in range(2):
            i = 2 * t + v
            # p_t - p_prev + service_t - transfer_t = 0.
            eq[i, 6 * n + i] = 1
            eq[i, 2 * n + i] = 1
            eq[i, 4 * n + i] = -1
            # q_t - q_prev - admission_t + transfer_t = 0.
            eq[2 * n + i, 8 * n + i] = 1
            eq[2 * n + i, i] = -1
            eq[2 * n + i, 4 * n + i] = 1
            if t:
                eq[i, 6 * n + i - 2] = -1
                eq[2 * n + i, 8 * n + i - 2] = -1
                ub[t, 8 * n + i - 2] = 1
                ub[n + i, 6 * n + i - 2] = -1
            ub[t, i] = 1
            # Service occurs before admission and transfer.
            ub[n + i, 2 * n + i] = 1
            c[i] = -REJECT
            hold = cfg.holding if t < n - 1 else REJECT
            c[6 * n + i] = c[8 * n + i] = hold
    result = linprog(c, A_eq=eq, b_eq=rhs, A_ub=ub, b_ub=ub_rhs,
                     bounds=bounds, method='highs')
    assert result.success, result.message
    residual = max(float(np.abs(eq @ result.x - rhs).max()),
                   float(np.maximum(ub @ result.x - ub_rhs, 0).max()))
    assert residual < 1e-7
    return (float(result.fun + REJECT * sum(map(len, cfg.arrivals))),
            float(np.abs(result.x - np.round(result.x)).max()))


def swap_config(cfg):
    names = dict(normal='normal', A_down_01='B_down_01', B_down_01='A_down_01',
                 A_down_1='B_down_1', B_down_1='A_down_1')
    return Config(tuple(tuple(1 - v for v in w) for w in cfg.arrivals), cfg.lane_capacity,
                  cfg.holding, names[cfg.outage])


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    previous = ROOT / '项目结果/dynamic_coupling_decision_20261018/93_final_manifest.json'
    preserved = json.loads(previous.read_text(encoding='utf-8'))['files']
    for p, h in preserved.items():
        assert sha(ROOT / p) == h
    design = dict(source_sha256=sha(__file__), base_horizon=3, alphabet=ALPHABET,
                  lane_capacity=[2, 3], station_waiting_capacity=BUFFER,
                  holding_ticks=[1, 5], rejection_terminal_ticks=REJECT,
                  outages=['normal', 'A_down_01', 'B_down_01', 'A_down_1', 'B_down_1'],
                  drain_slots=[0, 6], policies=POLICIES,
                  event_order='Process old station stock; select arrival subsequence subject to current lane free space; FIFO transfer; hold; next slot',
                  restriction='New arrivals cannot be processed until a later slot. Lane peak capacity is checked before transfer, uniformly for all policies and LP.',
                  accepted_stock_loss=False, endogenous_failure=False, empirical_calibration=False,
                  FIFO='Shared arrival lane; destination fixed per batch; no resequencing or free cancellation after acceptance',
                  controls='Bypass changes mechanics and is an ablation, not a fair same-physics algorithm competitor; LP further relaxes order/integrality',
                  information='Offline exact controllers know future arrivals/capacity; rolling_2 has perfect two-slot forecast. These are not validated real forecasts.',
                  manuscript_changed=False, prior_manifest_sha256=sha(previous),
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    path = OUT / '94_design.json'
    if path.exists():
        # JSON roundtrip normalizes tuples in the frozen design.
        assert json.loads(path.read_text(encoding='utf-8')) == json.loads(json.dumps(design))
    else:
        dump(design, path)
    rows, pairs, lp_rows, symmetry = [], [], [], []
    for outage in design['outages']:
        for seq, lane_cap, hold, drain in product(product(ALPHABET, repeat=3), (2, 3), (1, 5), (0, 6)):
            cfg = Config(seq + ((),) * drain, lane_cap, hold, outage)
            key = json.dumps((seq, lane_cap, hold, drain, outage), separators=(',', ':'))
            metrics = {}
            for policy in POLICIES:
                m, _ = simulate(cfg, policy)
                metrics[policy] = m
                rows.append(dict(case_key=key, outage=outage, lane_capacity=lane_cap,
                                 holding_ticks=hold, drain_slots=drain,
                                 arrivals_json=json.dumps(cfg.arrivals), policy=policy, **m))
                if drain:
                    assert m['terminal_inventory'] == 0
            opt = metrics['offline_fifo']['cost_ticks']
            bypass = metrics['offline_bypass']['cost_ticks']
            assert all(metrics[p]['cost_ticks'] >= opt for p in POLICIES if p != 'offline_bypass')
            assert bypass <= opt
            lp, fraction = fluid_lp(cfg)
            assert lp <= bypass + 1e-6
            lp_rows.append(dict(case_key=key, fluid_lp_ticks=lp, fifo_ticks=opt,
                                bypass_ticks=bypass, fifo_order_gap_ticks=opt - bypass,
                                lp_relaxation_gap_ticks=bypass - lp,
                                maximum_fractionality=fraction, passed=True))
            other = planner(swap_config(cfg))(0, ((), 0, 0))[0]
            assert other == opt
            symmetry.append(dict(case_key=key, original_ticks=opt, renamed_ticks=other, passed=True))
            pairs.append(dict(case_key=key, outage=outage, lane_capacity=lane_cap,
                              holding_ticks=hold, drain_slots=drain,
                              direct_ticks=metrics['direct_only_optimal']['cost_ticks'],
                              fifo_optimal_ticks=opt, bypass_optimal_ticks=bypass,
                              max_accept_ticks=metrics['max_accept_fifo']['cost_ticks'],
                              rolling_2_ticks=metrics['rolling_2_perfect']['cost_ticks'],
                              buffer_gain_ticks=metrics['direct_only_optimal']['cost_ticks'] - opt,
                              max_accept_regret_ticks=metrics['max_accept_fifo']['cost_ticks'] - opt,
                              rolling_regret_ticks=metrics['rolling_2_perfect']['cost_ticks'] - opt,
                              order_cost_ticks=opt - bypass))
        print(f'{outage}: {len(pairs)} configurations complete', flush=True)
    save_csv(rows, OUT / '94a_policy_results.csv')
    save_csv(pairs, OUT / '94b_paired_results.csv')
    save_csv(lp_rows, OUT / '94c_fluid_lp_bounds.csv')
    save_csv(symmetry, OUT / '94d_label_symmetry.csv')
    brute_rows = []
    for seq, lane, outage in product(product(ALPHABET, repeat=2), (2, 3), design['outages']):
        cfg = Config(seq, lane, 1, outage)
        for bypass in (False, True):
            exact = planner(cfg, bypass=bypass)(0, ((), 0, 0))[0]
            independently = brute(cfg, bypass=bypass)
            assert exact == independently
            brute_rows.append(dict(arrivals_json=json.dumps(seq), lane_capacity=lane, outage=outage,
                                   bypass=bypass, exact_ticks=exact, independently_ticks=independently, passed=True))
    save_csv(brute_rows, OUT / '94e_independent_action_enumeration.csv')
    # Mechanism recovery witness starts with existing inventory and no arrivals.
    recovery = Config(((),) * 6, 3, 1, 'A_down_01')
    recovered, recovery_trace = simulate(recovery, 'max_accept_fifo', ((0, 1), 1, 0))
    assert recovered['completed'] == 3 and recovered['terminal_inventory'] == 0
    assert recovery_trace[0]['input_blocked_at_end'] and recovery_trace[1]['input_blocked_at_end']
    assert recovery_trace[0]['completed'] == 0 and recovery_trace[2]['completed'] == 1
    assert recovery_trace[3]['completed'] == 2
    assert recovered['blockage_clearances'] >= 1
    save_csv(recovery_trace, OUT / '94f_recovery_witness.csv')
    # Clearing bound for every valid lane/station stock state after capacity
    # recovers, with zero new arrivals. No free deletion or forced reset.
    clear_checks = []
    for lane_cap in (2, 3):
        for length in range(lane_cap + 1):
            for word in product((0, 1), repeat=length):
                for p0, p1 in product((0, 1), repeat=2):
                    cfg = Config(((),) * 6, lane_cap, 1, 'normal')
                    m, _ = simulate(cfg, 'max_accept_fifo', (word, p0, p1))
                    assert m['terminal_inventory'] == 0 and m['completed'] == length + p0 + p1
                    clear_checks.append(dict(lane_capacity=lane_cap, lane_json=json.dumps(word),
                                             buffer_A=p0, buffer_B=p1, completed=m['completed'], passed=True))
    save_csv(clear_checks, OUT / '94g_all_state_clearance.csv')
    examples, traces = [], []
    specifications = [('buffer_retains_peak', ((0, 0), (), (), (), ()), 1, 'normal'),
                      ('waiting_can_cost_more', ((0, 0), (0, 0), (), ()), 5, 'normal'),
                      ('head_of_line_recovery', ((0, 0), (1,), (), (), (), ()), 1, 'A_down_01')]
    for name, seq, hold, outage in specifications:
        cfg = Config(seq, 2, hold, outage)
        for policy in POLICIES:
            m, trace = simulate(cfg, policy)
            examples.append(dict(example=name, policy=policy, **m))
            traces.extend(dict(example=name, **t) for t in trace)
    save_csv(examples, OUT / '94h_examples.csv')
    save_csv(traces, OUT / '94i_example_traces.csv')
    summary = []
    for outage, drain in product(design['outages'], (0, 6)):
        g = [r for r in pairs if r['outage'] == outage and r['drain_slots'] == drain]
        summary.append(dict(outage=outage, drain_slots=drain, configurations=len(g),
                            buffering_strict_gain=sum(r['buffer_gain_ticks'] > 0 for r in g),
                            max_accept_suboptimal=sum(r['max_accept_regret_ticks'] > 0 for r in g),
                            rolling_2_suboptimal=sum(r['rolling_regret_ticks'] > 0 for r in g),
                            strict_fifo_order_cost=sum(r['order_cost_ticks'] > 0 for r in g)))
    save_csv(summary, OUT / '94j_stratified_summary.csv')
    order_witness = next((r for r in pairs if r['order_cost_ticks'] > 0), None)
    if order_witness:
        seq, lane, hold, drain, outage = json.loads(order_witness['case_key'])
        cfg = Config(tuple(tuple(w) for w in seq) + ((),) * drain, lane, hold, outage)
        witness_traces = []
        for policy in ('offline_fifo', 'offline_bypass'):
            _, tr = simulate(cfg, policy)
            witness_traces.extend(tr)
        save_csv(witness_traces, OUT / '94k_first_order_gap_trace.csv')
    dump(dict(first_order_gap=order_witness,
              selection='First lexicographic strictly positive gap in the complete predeclared grid; diagnostic only'),
         OUT / '94_order_gap_diagnostic.json')
    summary_table = ['|处理能力情景|附加清空期|配置数|缓冲严格有益|尽量接收次优|两步滚动次优|FIFO本身增加最优成本|',
                     '|---|---|---|---|---|---|---|']
    for r in summary:
        summary_table.append('|' + '|'.join(str(v) for v in r.values()) + '|')
    example_table = ['|小例|政策|实际完成|拒收|期末库存|等待批时|成本|', '|---|---|---|---|---|---|---|']
    for r in examples:
        example_table.append('|' + '|'.join(str(r[k]) for k in ('example', 'policy', 'completed', 'rejected', 'terminal_inventory', 'waiting_units', 'cost')) + '|')
    report = f'''# 可恢复拥堵最小原型与精确基线

日期：2026-10-18。新增独立运行模块，不改正文、V5/V6或既有结果。所有到达、容量、时段和成本是未标定合成量；不把仓内输送机机制直接推广到246个混合POI。

## 1. 完成的工作

实现两个处理站、一个有限容量共享FIFO入口通道和两个有限下游缓冲。下游库存持续按当期处理能力出清；队首目的缓冲无空间时，该货及后续货等待，库存不毁损、不永久失效。空间释放后按同一转运规则恢复，没有人工清零或额外“恢复”按钮。

本轮只实现运营拥堵模块。给定处理能力时序可以表示短时外部停机，不含原V5的加权风险闭包，也没有声称已形成全网双机制动态模型。

## 2. 物理时序与账本

批次目的站A/B固定。每期先处理站内旧库存，再从当期到达序列选择一个保序子序列接入，接着FIFO转运到目的缓冲，最后计库存等待。已接收货不能免费取消或重排。接入前检查共享通道当前空位，属于同步批次接入约定；所有政策和LP都保留相同时序，不将时间约定当作现实标定。新货最早下一期完成处理。

共享通道容量取2/3，每个目的缓冲容量1，正常每站每期处理1。外部给定短暂停机为无停机、A/B前两期停机、A/B仅第1期停机；之后能力均恢复。队首阻塞只限制转运与接入，不关闭下游已有库存的处理。

累计到达加初始库存=累计完成+累计拒收+当前库存。没有接受后损失、虚增服务或库存失踪。目标J=拒收+期末库存+h*等待批时，h取0.1/0.5；等价于总到达加初始库存减累计完成再加等待费。最后一刻不收通往不存在下一时段的等待费。

## 3. 政策、信息与机制对照

- direct_only_optimal，不保留共享入口队列，但可用目的站缓冲；在此受限动作集中求离线精确最优。这是强受限基线，不是旧网络C的原样复算。
- max_accept_fifo，最大接收后减少即时入口库存及队首阻碍，按固定目的站编号打破剩余平局，不利用未来。
- rolling_2_perfect，每期求两步精确规划，只执行首步。只享有两期完美预测；窗口末库存按单位拒收费估值，不代表实际预测已验证。
- offline_fifo，知道全时域到达和给定能力的精确Bellman最优，是离线基准。
- offline_bypass，同样精确求解但允许后续货绕过堵住的队首。这改变通道机制，单独作为FIFO消融，不能把优势归为同一物理系统上的算法改进。

另建无顺序的两目的地连续流量LP，保留逐类守恒、下游容量、旧库存处理顺序、共享通道接入峰值约束。它是放松下界，不能不加证明就称为所有FIFO问题的等价费用流。单独记录连续解的分数性和LP到绕行精确解的差距。

## 4. 固定设计与完整结果

三期到达从空、A、B、AB、BA完整枚举125个序列；通道2/3、等待费0.1/0.5、五种能力情景，原时域和额外六期无到达两种版本，共{len(pairs)}个配置、{len(rows)}条政策结果。到达顺序不是按胜负挑选的。六期清空使存活库存全部被处理，逐条断言，原时域未完成仍足额记罚损。

{chr(10).join(summary_table)}

上述计数是确定性网格结果，不是独立现实样本或事件概率，不进行显著性解释。精确最优不劣于受限策略可由可行域包含直接推出，不能作为创新。FIFO与绕行的差值描述机械顺序代价，不是算法最优差距。

## 5. 可恢复性见证及小例

可恢复性见证初始有A站库存1、入口队列AB，没有新增货。A前两期处理能力为0，B正常。前两期A堵住入口，B后面的货也不能入站；第2期A恢复后完成1批并释放空间，A/B均得到转入货，第3期完成剩余2批。全部3批正常完成，拥堵自行解除，无毁货、永久退出或人工重置。完整逐期记录见94f。

独立预定的收益/代价小例另列如下。目的为AA的单峰可以用入口保留随后能处理的货；重复AA且等待费较高时，尽量接收可能不值得。具体成本以精确账本为准，不用手算预设赢家。

{chr(10).join(example_table)}

全网格中第一个严格FIFO代价配置单独保存为94_order_gap_diagnostic.json和逐期轨迹。这是探索性诊断选择，不能当成独立确认样本。

## 6. 独立核验

- 每条轨迹检查货量守恒、真实完成与接收分离、缓冲与通道上界、代价恒等式。
- {len(brute_rows)}个两期配置用不调用主转移/动作生成/规划器的完整动作树核对FIFO和绕行最优值。
- {len(symmetry)}个配置交换A/B编号及对应停机时序，FIFO精确最优值相同。启发式平局动作不保证逐条镜像，未将这种保证扩大到启发式。
- {len(lp_rows)}个独立流量LP核对线性可行性与下界关系，记录整数性而不预设所有实例整数。
- 穷举{len(clear_checks)}个合法通道/站内库存状态，在无新增货、正常处理能力下均在六期内清空，排除因为入口堵塞而永久停工的人造死锁。

## 7. 目前能说与不能说的结论

可恢复拥堵已经有可守恒、可清空、可比较精确最优的最小实现。它比“超过阈值就永久退出”更贴近此次核对到的特定仓内阻塞语义，但还不是城市物流节点的实证模型。

如果仅保留可自由绕行的线性流量规则，很多部分仍是标准优化；若保留FIFO，必须检查实际机械或业务顺序是否存在，不能为增加难度人为禁止绕行。出现LP差距或FIFO顺序代价也不自动构成新方法或复杂性定理。

下一步是否扩展应取决于同口径对象能否定义处理、暂存及连接关系，以及是否有足够研究价值；不应只因为小例有正结果就覆盖旧模型或直接重跑全网。

## 8. 证据与复现

机制参考为93号采用边界记录所核对的仓储案例和有限缓冲研究，例如[4OR仓储阻塞案例](https://link.springer.com/article/10.1007/s10288-026-00620-7)。该案例支持的是有具体输送结构的暂时阻塞和解除，不提供本项目参数。

脚本：数据及代码/Code/94_reversible_blocking_prototype.py。结果：项目结果/reversible_blocking_prototype_20261018。Python {platform.python_version()}、NumPy {np.__version__}、SciPy {scipy.__version__}，LP为HiGHS。设计在运行前写入并冻结来源哈希，旧93清单引用的{len(preserved)}个文件在前后核对。
'''
    DOC.write_text(report, encoding='utf-8')
    (OUT / DOC.name).write_text(report, encoding='utf-8')
    for p, h in preserved.items():
        assert sha(ROOT / p) == h
    completion = dict(completed=True, configurations=len(pairs), policy_rows=len(rows),
                      independent_enumerations=len(brute_rows), lp_checks=len(lp_rows),
                      label_symmetry_checks=len(symmetry), all_initial_state_clearance_checks=len(clear_checks),
                      max_lp_fractionality=max(r['maximum_fractionality'] for r in lp_rows),
                      positive_fifo_gaps=sum(r['fifo_order_gap_ticks'] > 0 for r in lp_rows),
                      positive_lp_relaxation_gaps=sum(r['lp_relaxation_gap_ticks'] > 1e-6 for r in lp_rows),
                      calibrated=False, manuscript_changed=False, original_model_changed=False,
                      old_artifacts_preserved=len(preserved), elapsed_seconds=round(time.time() - start, 3),
                      files={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), DOC, *sorted(OUT.glob('*'))] if p.is_file() and p.name != '94_completion.json'})
    dump(completion, OUT / '94_completion.json')
    print(json.dumps({k: v for k, v in completion.items() if k != 'files'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
