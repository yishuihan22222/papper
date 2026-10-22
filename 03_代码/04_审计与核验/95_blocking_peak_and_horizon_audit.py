"""Targeted peak and forecast-horizon diagnostics following prototype 94.

Exploratory model development, not a new held-out empirical confirmation set.
"""
from pathlib import Path
from itertools import product
import importlib.util
import sys
import json
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '项目结果/reversible_blocking_peak_horizon_20261018'
DOC = ROOT / '项目文档/可恢复拥堵高峰与滚动窗口补充核验_20261018.md'
SOURCE = Path(__file__).with_name('94_reversible_blocking_prototype.py')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('reversible94', SOURCE)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def dump(d, p):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def csv_write(rows, p):
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rolling(cfg, horizon):
    state = ((), 0, 0)
    cost = served = rejected = waiting = 0
    for t, d in enumerate(cfg.arrivals):
        plan = core.planner(cfg, stop=min(t + horizon, len(cfg.arrivals)))
        action = plan(t, state)[1]
        state, c, e = core.step(t, state, action, cfg)
        cost += c
        served += e['completed']
        rejected += e['rejected']
        waiting += e['waiting_units']
        assert sum(map(len, cfg.arrivals[:t + 1])) == served + rejected + len(state[0]) + state[1] + state[2]
    terminal = len(state[0]) + state[1] + state[2]
    cost += 10 * terminal
    assert cost == 10 * (sum(map(len, cfg.arrivals)) - served) + cfg.holding * waiting
    return dict(cost_ticks=cost, completed=served, rejected=rejected,
                waiting_units=waiting, terminal_inventory=terminal)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = ROOT / '项目结果/reversible_blocking_prototype_20261018/94_completion.json'
    prev = json.loads(previous.read_text(encoding='utf-8'))
    for p, h in prev['files'].items():
        assert sha(ROOT / p) == h
    sequences = dict(single_A_peak=((0, 0), (), ()), repeated_A_peak=((0, 0), (0, 0), ()),
                     mixed_A_then_B=((0, 0), (1,), ()), single_B_peak=((1, 1), (), ()),
                     repeated_B_peak=((1, 1), (1, 1), ()), mixed_B_then_A=((1, 1), (0,), ()))
    design = dict(source_sha256=sha(__file__), prototype_sha256=sha(SOURCE),
                  previous_completion_sha256=sha(previous), sequences=sequences,
                  outages=['normal', 'A_down_01', 'B_down_01', 'A_down_1', 'B_down_1'],
                  holding_ticks=[1, 5], lane_capacity=[2, 3], drain_slots=[0, 6],
                  rolling_horizons=[2, 3, 4, 6],
                  reason='Prototype94 normal arrival alphabet had at most one batch per station per slot; add explicit repeated peaks and stronger horizons',
                  inference='Targeted synthetic model-development cases, no hold-out or empirical validation claim',
                  forecast='Perfect capacity and arrival information within each window, not a learned forecast',
                  calibrated=False, manuscript_changed=False)
    p = OUT / '95_design.json'
    if p.exists():
        assert json.loads(p.read_text(encoding='utf-8')) == json.loads(json.dumps(design))
    else:
        dump(design, p)
    rows, pairs, lp_checks = [], [], []
    for (name, sequence), outage, hold, cap, drain in product(sequences.items(), design['outages'], (1, 5), (2, 3), (0, 6)):
        cfg = core.Config(sequence + ((),) * drain, cap, hold, outage)
        key = f'{name}:{outage}:{hold}:{cap}:{drain}'
        metrics = {}
        for policy in ('direct_only_optimal', 'max_accept_fifo', 'offline_fifo', 'offline_bypass'):
            m, _ = core.simulate(cfg, policy)
            metrics[policy] = {k: m[k] for k in ('cost_ticks', 'completed', 'rejected', 'waiting_units', 'terminal_inventory')}
        for horizon in (2, 3, 4, 6):
            metrics[f'rolling_{horizon}'] = rolling(cfg, horizon)
        # Independent earlier implementation must agree for the same window.
        old, _ = core.simulate(cfg, 'rolling_2_perfect')
        assert old['cost_ticks'] == metrics['rolling_2']['cost_ticks']
        opt = metrics['offline_fifo']['cost_ticks']
        for policy, m in metrics.items():
            if policy != 'offline_bypass':
                assert m['cost_ticks'] >= opt
            if drain:
                assert m['terminal_inventory'] == 0
            rows.append(dict(case_key=key, peak=name, outage=outage, holding_ticks=hold,
                             lane_capacity=cap, drain_slots=drain, policy=policy, **m))
        lp, fraction = core.fluid_lp(cfg)
        bypass = metrics['offline_bypass']['cost_ticks']
        assert lp <= bypass + 1e-7
        lp_checks.append(dict(case_key=key, lp_ticks=lp, bypass_ticks=bypass,
                              fifo_ticks=opt, fractionality=fraction, passed=True))
        pairs.append(dict(case_key=key, peak=name, outage=outage, holding_ticks=hold,
                          lane_capacity=cap, drain_slots=drain, optimum=opt,
                          direct_gain=metrics['direct_only_optimal']['cost_ticks'] - opt,
                          max_accept_regret=metrics['max_accept_fifo']['cost_ticks'] - opt,
                          **{f'regret_{h}': metrics[f'rolling_{h}']['cost_ticks'] - opt for h in (2, 3, 4, 6)},
                          fifo_order_cost=opt - bypass))
    csv_write(rows, OUT / '95a_policy_results.csv')
    csv_write(pairs, OUT / '95b_paired_results.csv')
    csv_write(lp_checks, OUT / '95c_flow_relaxation.csv')
    summary = []
    for drain in (0, 6):
        g = [r for r in pairs if r['drain_slots'] == drain]
        for horizon in (2, 3, 4, 6):
            values = [r[f'regret_{horizon}'] / 10 for r in g]
            summary.append(dict(drain_slots=drain, horizon=horizon, configurations=len(g),
                                equal_to_offline=sum(x == 0 for x in values),
                                strictly_suboptimal=sum(x > 0 for x in values),
                                mean_regret=sum(values) / len(values), max_regret=max(values)))
    csv_write(summary, OUT / '95d_horizon_summary.csv')
    nonmonotone = [r for r in pairs if any(r[f'regret_{a}'] < r[f'regret_{b}'] for a, b in ((2, 3), (3, 4), (4, 6)))]
    dump(dict(count=len(nonmonotone), cases=nonmonotone,
              interpretation='A longer receding window is not assumed monotonically better; terminal approximation and replanning may matter'),
         OUT / '95_horizon_nonmonotonicity.json')
    table = ['|清空期|窗口|配置数|达到离线最优|次优|平均代价差|最大代价差|', '|---|---|---|---|---|---|---|']
    for r in summary:
        table.append('|' + '|'.join(str(v) for v in r.values()) + '|')
    report = f'''# 可恢复拥堵的高峰与滚动窗口补充核验

日期：2026-10-18。基于94号原型，正文与原模型不动。本轮是发现覆盖缺口后新增的合成开发诊断，不是未见实测验证集。

## 1. 为什么补这一步

94号全网格的当期到达只含空、A、B、AB、BA。在正常每站每期处理1的设定下，每目的站最多到达1批，未覆盖同站重复高峰，因此正常场景中所有政策一致不能解释成“正常时缓冲总无益”。94号另有AA单峰小例，本轮进一步固定单峰、重复峰、混合队首三种序列及A/B对称版本。

此外，两步窗口在当前先处理旧库存、后接收和转运的时序下可能看不到一个入口批次被实际处理的时刻。只拿两步策略比较离线最优不足以评价滚动决策方法，需要更长窗口对照。

## 2. 固定设计

六种三期序列分别是AA-空-空、AA-AA-空、AA-B-空及A/B交换版本；通道2/3、等待费0.1/0.5、五种给定短停机时序、原时域和附加六期无到达，共{len(pairs)}个配置、{len(rows)}条策略结果。

比较不留入口积压的强离线基线、尽量接收、FIFO离线精确最优、可绕行机制消融，以及2/3/4/6步滚动精确规划。窗口内需求与能力均为完美信息，窗口末库存仍以单位1计价；没有真实预测训练或误差校准。代价全部无量纲。

## 3. 窗口结果

{chr(10).join(table)}

所有差距均针对相同FIFO物理系统的离线最优。表中均值只是等权合成配置的描述，不是现实期望损失。A/B编号与给定停机情景成对保留。

相邻窗口增大后出现更高执行代价的配置为{len(nonmonotone)}个，完整列表保留；不预设窗口越长必然越好。达到本批小例离线最优也不能推成一般网络的最优性保证。

## 4. 机制与基线边界

共有{sum(r['direct_gain'] > 0 for r in pairs)}个配置中使用入口缓冲可严格改善最优代价，{sum(r['max_accept_regret'] > 0 for r in pairs)}个配置中尽量接收次优。它们是当前合成网格的计数，不表示现实发生比例。

FIFO相对可绕行精确最优有严格额外成本的配置为{sum(r['fifo_order_cost'] > 0 for r in pairs)}个。放松LP到可绕行精确解出现严格目标差的配置为{sum(r['bypass_ticks'] - r['lp_ticks'] > 1e-6 for r in lp_checks)}个，最大分数性为{max(r['fractionality'] for r in lp_checks)}。保留这些数值用于区分队列顺序、控制策略与线性放松的作用，不将机械差异宣传成算法优势。

## 5. 对下一步的实际建议

如果继续开发滚动派入，应使用足以覆盖处理/转运和可预见短停机的窗口，并与强离线基线比较；两步不是自然合理的默认值。但在实际信息有限时不能直接借用完美窗口的表现，后续需要区分预测误差与优化误差。

当前最小原型已完成收益、代价、自恢复、标准LP界、精确最优和窗口诊断。尚未证明新理论，尚未识别新的算法贡献，尚无理由把仓内FIFO规则统一施加到城市混合POI或立刻重跑246节点网络。

## 6. 复现与保留

脚本为数据及代码/Code/95_blocking_peak_and_horizon_audit.py，结果在项目结果/reversible_blocking_peak_horizon_20261018。物理转移直接复用冻结94代码，未改动该模块；两步滚动与原94实现逐配置核对一致。守恒、代价恒等式、离线下界、LP下界以及附加清空时段的零库存均通过。原94清单前后核对。
'''
    DOC.write_text(report, encoding='utf-8')
    (OUT / DOC.name).write_text(report, encoding='utf-8')
    for p, h in prev['files'].items():
        assert sha(ROOT / p) == h
    completion = dict(completed=True, configurations=len(pairs), policy_rows=len(rows),
                      horizon_values=[2, 3, 4, 6], nonmonotone_horizon_cases=len(nonmonotone),
                      manuscript_changed=False, original_prototype_preserved=True, empirical_calibration=False,
                      files={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), DOC, *sorted(OUT.glob('*'))] if p.is_file() and p.name != '95_completion.json'})
    dump(completion, OUT / '95_completion.json')
    print(json.dumps({k: v for k, v in completion.items() if k != 'files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
