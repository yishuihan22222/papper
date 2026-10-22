# -*- coding: utf-8 -*-
"""
承诺探针：把"什么时候转、转多少"换成"提前承诺哪些时段、多少量"后，
问题是否仍然退化？

设定（与第二轮试跑一致）
  H=4 槽，本地处理率 K=1，转运时滞 L∈{1,2}，批量 x∈{3,4}
  恢复支持 {1,2,3} 或 {1,2,3,4}；q_rob = [x - K*(H - R_max)]_+
  接收日历：首槽为 0，其余 3 槽各取 0/1/2，共 27 条；取全部 351 个无序日历对

承诺模型
  在 t=0（还不知道是哪条日历）承诺 z_s，s = L..H-1
  承诺是预留：实际可用 = min(z_s, c_s)，用不上就浪费
  要求：对集合内每条日历，Σ_s min(z_s, c_s) >= q_rob
  目标：min Σ_s z_s

  承诺免费 -> z 取无穷大 -> 退化。所以本探针的全部内容押在"承诺有成本"上。
"""
import csv, json, itertools, os

H = 4
K = 1.0

def q_rob(batch, rmax):
    return max(0.0, batch - K * max(0, H - rmax))

def calendars():
    out = []
    for rest in itertools.product([0, 1, 2], repeat=H - 1):
        out.append((0,) + rest)
    return out

CALS = calendars()

def commit_cost(cals, q, L):
    """最小承诺成本（整数规划，穷举；规模小）"""
    slots = list(range(L, H))
    hi = int(max(max(c[s] for c in cals) for s in slots))
    best = None
    best_z = None
    for z in itertools.product(range(hi + 1), repeat=len(slots)):
        ok = True
        for c in cals:
            got = sum(min(z[i], c[s]) for i, s in enumerate(slots))
            if got < q - 1e-9:
                ok = False
                break
        if ok:
            tot = sum(z)
            if best is None or tot < best:
                best, best_z = tot, z
    return best, best_z

def dominates(a, b, L):
    """只在可用槽（s>=L）上判断支配：时滞之前的槽根本用不上"""
    ua, ub = a[L:], b[L:]
    return all(x >= y for x, y in zip(ua, ub)) or all(y >= x for x, y in zip(ua, ub))

def crossing(a, b, L):
    """交叉槽数：只在可用槽上，一处高一处在低的槽数"""
    ua, ub = a[L:], b[L:]
    return sum(1 for x, y in zip(ua, ub) if x != y)

def main():
    rows = []
    for L in (1, 2):
        for batch in (3, 4):
            for rmax in (3, 4):
                q = q_rob(batch, rmax)
                for a, b in itertools.combinations(CALS, 2):
                    cals = [a, b]
                    A = [sum(c[s] for s in range(L, H)) for c in cals]
                    feasible = min(A) >= q - 1e-9
                    cost, z = commit_cost(cals, q, L) if feasible else (None, None)
                    # 上界：按逐槽最大值承诺（必然可行，代价最高）
                    zmax = [max(c[s] for c in cals) for s in range(L, H)]
                    ub = sum(zmax) if feasible else None
                    # 逐槽最小值承诺（可能不可行）
                    zmin = [min(c[s] for c in cals) for s in range(L, H)]
                    lb_feasible = all(
                        sum(min(zmin[i], c[s]) for i, s in enumerate(range(L, H))) >= q - 1e-9
                        for c in cals) if feasible else False
                    rows.append(dict(
                        L=L, batch=batch, rmax=rmax, q=q,
                        A=A[0], B=A[1], feasible=feasible,
                        commit=cost, premium=(cost - q) if cost is not None else None,
                        upper=ub, upper_gap=(ub - cost) if cost is not None else None,
                        env_min_cost=sum(zmin), env_min_feasible=lb_feasible,
                        dominates=dominates(a, b, L), crossing=crossing(a, b, L),
                        cal_a=list(a), cal_b=list(b),
                    ))
    return rows

if __name__ == '__main__':
    rows = main()
    out = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out, 'commitment_grid.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    feas = [r for r in rows if r['feasible']]
    prem = [r['premium'] for r in feas]
    print('配置 %d，可行 %d' % (len(rows), len(feas)))
    print('承诺溢价 premium = commit - q_rob')
    print('  min %d  max %d  mean %.3f' % (min(prem), max(prem), sum(prem)/len(prem)))
    print('  溢价=0 的组数: %d / %d' % (sum(1 for p in prem if p == 0), len(prem)))
    # 猜想：溢价=0 <=> 存在支配关系
    d0 = [r for r in feas if r['premium'] == 0]
    nd = [r for r in feas if not r['dominates']]
    print('  溢价=0 且无支配: %d （应为 0）' % sum(1 for r in d0 if not r['dominates']))
    print('  无支配且溢价=0: %d （应为 0）' % sum(1 for r in nd if r['premium'] == 0))
    # 上界差距：聪明承诺 vs 逐槽最大承诺
    ug = [r['upper_gap'] for r in feas]
    print('逐槽最大承诺(可行但贵) 相对最优的超出: max %d  mean %.3f' % (max(ug), sum(ug)/len(ug)))
    # 交叉程度与溢价
    from collections import defaultdict
    byc = defaultdict(list)
    for r in feas:
        byc[r['crossing']].append(r['premium'])
    print('按交叉槽数分组的平均溢价:')
    for k in sorted(byc):
        v = byc[k]
        print('  交叉%d槽: n=%d  mean=%.2f' % (k, len(v), sum(v)/len(v)))
    json.dump(dict(configs=len(rows), feasible=len(feas),
                   premium_max=max(prem), premium_mean=sum(prem)/len(prem),
                   premium_zero=sum(1 for p in prem if p == 0),
                   counterexample_prem0_nodom=sum(1 for r in d0 if not r['dominates']),
                   counterexample_nodom_prem0=sum(1 for r in nd if r['premium'] == 0),
                   upper_gap_max=max(ug)),
              open(os.path.join(out, 'commitment_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
