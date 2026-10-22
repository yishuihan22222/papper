"""Exact, integer-valued diagnostic of buffering benefit, not a V5 replacement.

One homogeneous facility; no failure/repair/route model is claimed here.
All penalties, time slots and batch sizes are illustrative, not calibrated.
"""
from functools import lru_cache
from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '项目结果/dynamic_buffer_tradeoff_audit_20261018'
DOC = ROOT / '项目文档/动态缓冲收益与取舍小例审计_20261018.md'
REJECT_TICKS = 10
HOLD_TICKS = (0, 1, 5, 10, 20)
POLICIES = ('no_buffer', 'max_admit', 'lookahead_2', 'offline_optimal')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(rows, path):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def transition(q, arrival, accepted, k, b):
    assert 0 <= accepted <= min(arrival, k + b - q)
    completed = min(k, q + accepted)
    q_next = q + accepted - completed
    assert 0 <= q_next <= b
    return q_next, completed, arrival - accepted


def solve(arrivals, b, hold, capacities=None, global_terminal=None):
    """Finite horizon Bellman solution. Integer ticks avoid near-optimal ties.

    At a tied cost, prefer less admission (less waiting inventory).
    An artificial lookahead endpoint has the same unit backlog penalty as
    rejection; its holding charge still follows the true outer horizon.
    """
    arrivals = tuple(arrivals)
    capacities = tuple(capacities or (1,) * len(arrivals))
    n = len(arrivals)
    charge_last_holding = global_terminal is False

    @lru_cache(None)
    def value(t, q):
        if t == n:
            return REJECT_TICKS * q, -1
        d, k = arrivals[t], capacities[t]
        candidates = []
        for a in range(min(d, k + b - q) + 1):
            nxt, _, rejected = transition(q, d, a, k, b)
            waiting = hold * nxt if t < n - 1 or charge_last_holding else 0
            cost = REJECT_TICKS * rejected + waiting + value(t + 1, nxt)[0]
            candidates.append((cost, a))
        return min(candidates)

    return value


def simulate(arrivals, b, hold, policy):
    arrivals = tuple(arrivals)
    n = len(arrivals)
    optimal = solve(arrivals, b, hold) if policy == 'offline_optimal' else None
    q = completed = rejected = waiting_units = total_cost = accepted_total = 0
    trace = []
    for t, d in enumerate(arrivals):
        if policy == 'no_buffer':
            a = min(d, max(1 - q, 0))
        elif policy == 'max_admit':
            # Do not deliberately add new terminal leftovers. Existing stock
            # cannot be cancelled free of charge or silently discarded.
            limit = max(1 - q, 0) if t == n - 1 else 1 + b - q
            a = min(d, limit)
        elif policy == 'lookahead_2':
            end = min(t + 2, n)
            local = solve(arrivals[t:end], b, hold, global_terminal=end == n)
            a = local(0, q)[1]
        elif policy == 'offline_optimal':
            a = optimal(t, q)[1]
        else:
            raise ValueError(policy)
        nxt, s, r = transition(q, d, a, 1, b)
        wait = nxt if t < n - 1 else 0
        step_cost = REJECT_TICKS * r + hold * wait
        trace.append(dict(t=t, arrival=d, queue_before=q, accepted=a,
                          completed=s, rejected=r, queue_after=nxt,
                          waiting_units=wait, stage_cost=step_cost / 10))
        q = nxt
        completed += s
        rejected += r
        waiting_units += wait
        total_cost += step_cost
        accepted_total += a
        assert completed + rejected + q == sum(arrivals[:t + 1])
        assert accepted_total == completed + q
    total_cost += REJECT_TICKS * q
    assert sum(arrivals) == completed + rejected + q
    assert total_cost == REJECT_TICKS * (sum(arrivals) - completed) + hold * waiting_units
    if optimal is not None:
        assert total_cost == optimal(0, 0)[0]
    return dict(cost_ticks=total_cost, cost=total_cost / 10,
                completed=completed, rejected=rejected, terminal_backlog=q,
                waiting_units=waiting_units, accepted=accepted_total), trace


def brute_force(arrivals, b, hold):
    """Independent path enumeration, with no Bellman recursion or transition()."""
    states = [(0, 0)]
    for t, d in enumerate(arrivals):
        new = []
        for q, cost in states:
            for a in range(d + 1):
                served = min(1, q + a)
                nxt = q + a - served
                if not 0 <= nxt <= b:
                    continue
                c = cost + REJECT_TICKS * (d - a)
                if t < len(arrivals) - 1:
                    c += hold * nxt
                new.append((nxt, c))
        states = new
    return min(cost + REJECT_TICKS * q for q, cost in states)


def count_compare(rows, p, q):
    pairs = {}
    for r in rows:
        pairs.setdefault(r['case_key'], {})[r['policy']] = r
    outcomes = [g[p]['cost_ticks'] - g[q]['cost_ticks'] for g in pairs.values()]
    return dict(comparison=f'{p} vs {q}', conditions=len(outcomes),
                lower_cost=sum(x < 0 for x in outcomes),
                equal_cost=sum(x == 0 for x in outcomes),
                higher_cost=sum(x > 0 for x in outcomes),
                interpretation='Deterministic grid counts, not independent statistical samples')


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    prior = ROOT / '项目结果/dispatch_capacity_null_mainline_20261018/89_final_manifest.json'
    preserved = json.loads(prior.read_text(encoding='utf-8'))['artifacts']
    for item in preserved:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    source_hash = sha(__file__)
    design = dict(
        status='Frozen before grid execution; illustrative diagnostic only',
        source_sha256=source_hash, buffers=[1, 2], holding_ticks=list(HOLD_TICKS),
        reject_and_terminal_ticks=REJECT_TICKS, horizons=[4, 6],
        arrivals_per_slot=[0, 1, 2], drain_slots=[0, 2],
        processing_per_slot=1, exact_integer_units=True,
        queue_capacity='End-of-slot waiting inventory; processing is separate',
        objective='Reject cost + waiting cost + terminal backlog cost',
        waiting='End-of-slot backlog charged only before a further true slot',
        policies=list(POLICIES),
        information='lookahead_2 sees exact next slot; offline_optimal sees full sequence. Neither is a forecast-validated controller.',
        no_failures=True, no_empirical_calibration=True,
        source_data='Synthetic complete enumeration, no facility measurements',
        units='One arbitrary cargo batch; one uncalibrated time slot; dimensionless penalty',
        comparison='Identical physical buffer and processing capability available to all policies; no_buffer elects not to retain stock',
        terminal='All leftover charged once at rejection unit cost; no free unfinished work',
        prior_manifest_sha256=sha(prior),
        assumption_map_sha256=sha(ROOT / '项目文档/结论假设与应急承接收益边界审计_20261018.md'))
    design_path = OUT / '90_design.json'
    if design_path.exists():
        assert json.loads(design_path.read_text(encoding='utf-8')) == design, 'Frozen design changed'
    else:
        dump(design, design_path)

    rows = []
    brute_checks = configurations = 0
    for base_t in design['horizons']:
        for sequence in product(range(3), repeat=base_t):
            for drain in (0, 2):
                arrivals = sequence + (0,) * drain
                for b in (1, 2):
                    for hold in HOLD_TICKS:
                        configurations += 1
                        key = f'{base_t}:{"".join(map(str, sequence))}:{drain}:{b}:{hold}'
                        case = []
                        for policy in POLICIES:
                            metrics, _ = simulate(arrivals, b, hold, policy)
                            r = dict(case_key=key, base_horizon=base_t, drain_slots=drain,
                                     arrivals_json=json.dumps(arrivals), buffer=b,
                                     holding_penalty=hold / 10, policy=policy, **metrics)
                            rows.append(r)
                            case.append(r)
                        optimal = case[-1]['cost_ticks']
                        assert all(r['cost_ticks'] >= optimal for r in case)
                        if b == 1:
                            assert all(r['terminal_backlog'] == 0 for r in case)
                        if drain == 2:
                            assert all(r['terminal_backlog'] == 0 for r in case)
                        if base_t == 4 and drain == 0:
                            assert brute_force(arrivals, b, hold) == optimal
                            brute_checks += 1
        print(f'Finished base horizon {base_t}: {configurations} configurations', flush=True)
    write_csv(rows, OUT / '90a_grid_results.csv')
    comparisons = []
    for drain in (0, 2):
        subset = [r for r in rows if r['drain_slots'] == drain]
        for p, q in [('max_admit', 'no_buffer'), ('lookahead_2', 'no_buffer'),
                     ('lookahead_2', 'max_admit'), ('offline_optimal', 'no_buffer'),
                     ('offline_optimal', 'lookahead_2')]:
            comparisons.append(dict(drain_slots=drain, **count_compare(subset, p, q)))
    write_csv(comparisons, OUT / '90b_grid_comparisons.csv')

    examples = []
    traces = []
    example_design = [('peak_then_idle', (2, 0), .1),
                      ('sustained_load', (2, 2), .1),
                      ('waiting_too_costly', (2, 0), 2.),
                      ('delayed_spare_capacity', (2, 1, 0), .1)]
    for name, arrivals, h in example_design:
        for policy in POLICIES:
            metrics, trace = simulate(arrivals, 1, round(h * 10), policy)
            examples.append(dict(example=name, arrivals_json=json.dumps(arrivals),
                                 buffer=1, holding_penalty=h, policy=policy, **metrics))
            traces.extend(dict(example=name, policy=policy, **t) for t in trace)
    write_csv(examples, OUT / '90c_exact_examples.csv')
    write_csv(traces, OUT / '90d_example_traces.csv')
    # Hand-derived witnesses checked separately from grid summaries.
    by = {(x['example'], x['policy']): x for x in examples}
    assert by['peak_then_idle', 'no_buffer']['cost_ticks'] == 10
    assert by['peak_then_idle', 'max_admit']['cost_ticks'] == 1
    assert by['sustained_load', 'no_buffer']['cost_ticks'] == 20
    assert by['sustained_load', 'max_admit']['cost_ticks'] == 21
    assert by['waiting_too_costly', 'offline_optimal']['cost_ticks'] == 10
    assert by['waiting_too_costly', 'max_admit']['cost_ticks'] == 20
    assert by['delayed_spare_capacity', 'lookahead_2']['cost_ticks'] > by['delayed_spare_capacity', 'offline_optimal']['cost_ticks']

    table = ['|小例|到达序列|等待权重|政策|完成|拒收|期末积压|目标成本|',
             '|---|---|---|---|---|---|---|---|']
    names = dict(peak_then_idle='高峰后空闲', sustained_load='连续高负荷',
                 waiting_too_costly='等待成本高', delayed_spare_capacity='空闲在两步以后')
    for r in examples:
        table.append(f"|{names[r['example']]}|{r['arrivals_json']}|{r['holding_penalty']}|{r['policy']}|{r['completed']}|{r['rejected']}|{r['terminal_backlog']}|{r['cost']}|")
    cmp_table = ['|额外清空时段|比较，前者相对后者|条件数|成本更低|持平|成本更高|',
                 '|---|---|---|---|---|---|']
    for c in comparisons:
        cmp_table.append(f"|{c['drain_slots']}|{c['comparison']}|{c['conditions']}|{c['lower_cost']}|{c['equal_cost']}|{c['higher_cost']}|")
    report = '''# 动态缓冲收益与决策取舍的小例审计

日期：2026-10-18。正文、V5/V6模型及既有结果均未改动。本报告是合成机制诊断，不是武汉设施动态运行验证，也不是新算法结果。

## 1. 直接结论

有限缓冲可以在后续有闲置处理能力时增加累计实际完成量；但“尽量多接”不是普遍最优，因为它可能只让本来就能完成的货提前排队、增加等待，或产生不值得承担的等待成本。无需人为加入新的级联参数，就能构造这两类情形。

这证明“额外接收可能有跨期价值”可以被一致地表示，不证明现实中B应胜C。旧V5的B没有队列，不等于本试验的max_admit；no_buffer只是静态安全派入的一设施类比，也不是把旧246节点C原样重跑。

## 2. 状态、动作与账本

一个设施，处理能力每时段K=1，所有货物同质、可按整数批次计量。B为处理完成后的等待缓冲容量，分别取1或2。它不是处理能力倍数，也不是失效阈值。

每期先观察新到达D_t，选择接收a_t，满足0<=a_t<=min(D_t,K+B-Q_t)。实际完成S_t=min(K,Q_t+a_t)，拒收R_t=D_t-a_t，下一期队列Q_(t+1)=Q_t+a_t-S_t。接收库存不能免费取消。没有运输、风险、过载退出、修复或预测误差；库存全程保守计量。

目标J=ΣR_t+h*Σ(t<T-1)Q_(t+1)+Q_T。拒收与期末积压的单位罚损均为1，h取0/0.1/0.5/1/2。所有量都是未标定的示意单位。期末积压记一次未完成损失，不算服务收益；最后一刻不再收取通往不存在下一时段的等待费。

全程有ΣD=ΣS+ΣR+Q_T，故J=(ΣD-ΣS)+h*累计等待。该恒等式防止算法通过多接却不处理虚增成绩。加入排队不是提高K，更没有把入库当成完成。

## 3. 对照政策与信息条件

- no_buffer，不保留跨期库存，仅接收当期可完成量。所有政策使用同一物理B，no_buffer选择不使用。
- max_admit，在允许缓冲内尽可能接收，末期不故意新增无用库存。不是原V5算法B。
- lookahead_2，准确看到当前及下一时段到达，求两步精确优化后只执行首步。窗口末库存按单位1计价。它具有额外未来信息，不是已经验证的真实预测控制器。
- offline_optimal，知道完整到达序列的Bellman精确解，是离线下界，不可作为实际可实现的性能承诺。

精确解与其他政策拥有相同物理动作集；它能选择完全不排队，因此“不劣于no_buffer”本身也是可行域包含关系，不能作为新方法发现。两步策略与离线策略的信息差必须显式报告。

## 4. 手算小例与逐步记录

下表全部B=1，拒收单位成本1。

''' + '\n'.join(table) + '''

高峰后空闲[2,0]时，接收第二批使其等待一次后完成，多完成1批，付出0.1等待成本；J由1降至0.1。连续高负荷[2,2]时，两种政策都只能完成2批，多接只是用排队货替换随后到达的货，未增加累计完成，J由2变2.1。这两个例子使用完全相同的处理能力、缓冲与罚损参数，只改变需求时序。

将h提高到2，不值得为保住价值1的一批货承担2的等待成本，离线最优选择拒收。h=1有等价最优，程序以较少接收打破平局，不把同成本动作差异误报成算法优势。

[2,1,0]进一步说明两步窗口可能看不到稍后空闲，因而错过有价值的提前缓冲。它是短视界失败例，不是为滚动策略预设成功。完整轨迹保存在90d_example_traces.csv。

## 5. 完整小网格与验证

''' + f'''枚举基础长度4与6的全部0/1/2到达序列，各自保留原时域并另做追加2个零到达处理时段的对照；B为1/2，h为五个固定值，共{configurations}个配置、{len(rows)}条政策结果。不按政策胜负选择序列。每种清空时域各8100个配置。

''' + '\n'.join(cmp_table) + f'''

这些是确定性完整网格的计数，不是独立样本、现实频率或显著性检验。追加清空时段的对照保留所有政策，可检查结论对终止时刻的依赖；在K=1、B<=2且没有新增货的本试验中，两个时段足以清空所有库存，已逐条断言。原时域中未清空的剩余仍有明确罚损。

所有轨迹核对每期队列界、处理上限、累计货量守恒和代价恒等式；所有配置核对离线精确成本不高于各政策。基础4期无附加时段的{brute_checks}个配置，另用不调用Bellman递推或主转移函数的完整动作路径枚举核对最优值，全部一致。手算例另外固定断言。成本以整数十分之一单位精确计算，不依赖浮点近似排名。

## 6. 对主线的含义与不成立的推广

1. 当前静态模型遗漏的跨期完成收益可以明确建模。但这一标准有限缓冲问题本身不是硬创新。
2. 需要区分接收、处理和容纳，不能再把接收超过K直接等同于处理过载。本阶段没有设施失效，所以尚未建立“缓冲收益与级联风险”的完整取舍。
3. 引入等待后，动态目标下哪些货值得保留可以依赖未来需求；但额外动作集和未来信息本身会带来结构性优势，不能再次包装成算法发现。
4. 下一步若接回双机制，必须先明确设施退出的物理对象及依据。不能为让缓冲策略变差而任意设队列触发失效；也不能直接把原util与队列相加。
5. 若无可辩护的退出机制，可将后续问题诚实定位为风险情景下的有限缓冲派入，而不是宣称已验证过载失效的现实机制。需要统一所有政策的时段、状态、服务/拒收账本和信息条件。
6. 此处没有拟合真实需求、吞吐、仓储容量、货损或等待成本，也没有重新评价保护、维修、全网路由、城市公平或SCI分区可发表性。

## 7. 复现与保留

新增脚本为数据及代码/Code/90_dynamic_buffer_tradeoff_audit.py；输入设计及全部网格、比较、轨迹、验证清单在项目结果/dynamic_buffer_tradeoff_audit_20261018。现有550个冻结产出在运行前后逐一核对哈希。此报告与结论假设表是独立审计材料，不改论文正文。
'''
    DOC.write_text(report, encoding='utf-8')
    (OUT / DOC.name).write_text(report, encoding='utf-8')
    for item in preserved:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    completion = dict(completed=True, configurations=configurations, policy_rows=len(rows),
                      exhaustive_independent_checks=brute_checks,
                      hand_examples=len(example_design), old_artifacts_preserved=len(preserved),
                      calibrated=False, full_network_run=False, failure_mechanism_included=False,
                      manuscript_changed=False, source_sha256=source_hash,
                      elapsed_seconds=round(time.time() - started, 3),
                      files={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), DOC, *sorted(OUT.glob('*'))] if p.is_file() and p.name != '90_completion.json'})
    dump(completion, OUT / '90_completion.json')
    print(json.dumps({k: v for k, v in completion.items() if k != 'files'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
