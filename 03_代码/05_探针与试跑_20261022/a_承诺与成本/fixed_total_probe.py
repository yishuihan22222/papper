# -*- coding: utf-8 -*-
"""
承诺探针·第二组：给日历加"跨时段总量固定"约束后，非平凡区域是否变大？

动机：第二轮报告 §4 已指出，退化来自"恢复支持 × 容量日历 = 笛卡尔积"，
最坏恢复与最忙日历总能同时出现。承诺把问题从排班变成覆盖，但若日历仍按
逐槽独立取值，覆盖的约束就退化为"逐槽最小包络"，于是 65% 的情形溢价为零。

本组把日历限制为"可用槽总量固定 T，只在槽间重新分配"，打破笛卡尔积。
"""
import csv, json, itertools, os

H = 4
K = 1.0

def q_rob(batch, rmax):
    return max(0.0, batch - K * max(0, H - rmax))

def cals_fixed_total(L, T):
    """可用槽 s>=L，各取 0..2，总和固定为 T"""
    slots = H - L
    out = []
    for v in itertools.product([0, 1, 2], repeat=slots):
        if sum(v) == T:
            out.append((0,) * L + v)
    return out

def cals_free(L):
    return [(0,) * L + v for v in itertools.product([0, 1, 2], repeat=H - L)]

def commit_cost(cals, q, L):
    slots = list(range(L, H))
    hi = int(max(max(c[s] for c in cals) for s in slots))
    best = None
    for z in itertools.product(range(hi + 1), repeat=len(slots)):
        if all(sum(min(z[i], c[s]) for i, s in enumerate(slots)) >= q - 1e-9 for c in cals):
            t = sum(z)
            if best is None or t < best:
                best = t
    return best

def run(calset_fn, tag):
    rows = []
    for L in (1, 2):
        for batch in (3, 4):
            for rmax in (3, 4):
                q = q_rob(batch, rmax)
                for T in (2, 3, 4):
                    C = calset_fn(L, T)
                    if len(C) < 2:
                        continue
                    for a, b in itertools.combinations(C, 2):
                        if min(sum(c[s] for s in range(L, H)) for c in (a, b)) < q - 1e-9:
                            continue
                        cost = commit_cost([a, b], q, L)
                        rows.append(dict(tag=tag, L=L, batch=batch, rmax=rmax, T=T, q=q,
                                         commit=cost, premium=cost - q,
                                         env=sum(min(a[s], b[s]) for s in range(L, H))))
    return rows

if __name__ == '__main__':
    out = os.path.dirname(os.path.abspath(__file__))
    rows = run(cals_fixed_total, 'fixed_total')
    # 对照：同规模自由笛卡尔集（每个 T 下取"总量恰好为 T"的补集做对照不合适，
    # 因此直接复用前一组结果做全局对照，这里只报本组）
    with open(os.path.join(out, 'fixed_total_grid.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    prem = [r['premium'] for r in rows]
    n = len(rows)
    print('固定总量组：配置 %d' % n)
    print('溢价 mean %.3f  max %d' % (sum(prem) / n, max(prem)))
    print('溢价>0 占比 %.1f%%' % (100 * sum(1 for p in prem if p > 0) / n))
    from collections import Counter
    print('分布', dict(sorted(Counter(prem).items())))
    for T in (2, 3, 4):
        s = [r['premium'] for r in rows if r['T'] == T]
        if s:
            print('  T=%d: n=%d mean %.2f  >0 占比 %.1f%%' % (T, len(s), sum(s) / len(s), 100 * sum(1 for p in s if p > 0) / len(s)))
    # 与自由组对照
    fr = [r for r in csv.DictReader(open(os.path.join(out, 'commitment_grid.csv'), encoding='utf-8-sig')) if r['feasible'] == 'True']
    fp = [float(r['premium']) for r in fr]
    print('自由笛卡尔组：n=%d mean %.3f  >0 占比 %.1f%%' % (len(fp), sum(fp) / len(fp), 100 * sum(1 for p in fp if p > 0) / len(fp)))
