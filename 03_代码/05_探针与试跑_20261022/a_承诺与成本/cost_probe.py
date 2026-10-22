# -*- coding: utf-8 -*-
"""
成本探针：把"必须全部按期"换成"允许延误但按量罚款"后，
最优准备量是否仍等于最坏情况所需 q_rob？

问题：一切按最坏准备，是不是不考虑成本？
答：成本在目标里，但若成本对准备量是线性的，最优解必然落在角点。
    本脚本用三种成本形态检验这一点。

场景（沿用第二轮网格）
  H=4, K=1, x∈{3,4}, L∈{1,2}, 恢复 R∈{1..R_max}, 日历 27 条
  q = 提前转出量；到达槽 L 后可被接收点处理
  本地可处理 K*(H-R)，接收点可用 A(C)=Σ_{s>=L} c_s
  短缺 shortfall(R,C,q) = max(0, x - K*(H-R) - min(q, A(C)))
  最坏短缺 S(q) = max over (R,C)

目标：min_q  cost(q) + λ*S(q)
"""
import csv, itertools, json, os

H, K = 4, 1.0

def q_rob(batch, rmax):
    return max(0.0, batch - K * max(0, H - rmax))

CALS = [(0,) + v for v in itertools.product([0, 1, 2], repeat=H - 1)]

def A(c, L):
    return sum(c[s] for s in range(L, H))

def worst_shortfall(x, L, Rmax, cals, q):
    s = 0.0
    for R in range(1, Rmax + 1):
        local = K * max(0, H - R)
        for c in cals:
            s = max(s, x - local - min(q, A(c, L)))
    return max(0.0, s)

# ---- 三种成本形态 ----
def cost_linear(q):
    return 1.0 * q

def cost_convex(q):
    return 1.0 * q + 0.5 * q * q

def cost_tiered(q):
    """阶梯递增单价：<=1 单价1.0；1..3 单价1.6；>3 单价2.6（邮政公告式的分档）"""
    if q <= 1:
        return 1.0 * q
    if q <= 3:
        return 1.0 + 1.6 * (q - 1)
    return 1.0 + 1.6 * 2 + 2.6 * (q - 3)

FORMS = {'linear': cost_linear, 'convex': cost_convex, 'tiered': cost_tiered}

def argmin_q(f, lam, x, L, Rmax, cals, grid=0.05):
    best, bq = None, None
    n = int(round(x / grid))
    for i in range(n + 1):
        q = i * grid
        v = f(q) + lam * worst_shortfall(x, L, Rmax, cals, q)
        if best is None or v < best - 1e-12:
            best, bq = v, q
    return bq, best

def main():
    lams = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    rows = []
    # 用与第二轮同构的日历对（全 351 对里抽样，控制运行时间：取全部 L/batch/rmax 组合 + 60 对）
    pairs = list(itertools.combinations(CALS, 2))
    sample = pairs[::6]  # 约 59 对
    for L in (1, 2):
        for x in (3, 4):
            for Rmax in (3, 4):
                qr = q_rob(x, Rmax)
                for a, b in sample:
                    cals = [a, b]
                    for name, f in FORMS.items():
                        for lam in lams:
                            q, v = argmin_q(f, lam, x, L, Rmax, cals)
                            rows.append(dict(L=L, x=x, Rmax=Rmax, q_rob=qr, form=name,
                                             lam=lam, q_star=q, interior=(0 < q < x),
                                             equals_worstcase=(abs(q - qr) < 1e-9)))
    out = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out, 'cost_probe_grid.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print('配置 %d' % len(rows))
    for name in FORMS:
        sub = [r for r in rows if r['form'] == name]
        n = len(sub)
        inte = sum(1 for r in sub if r['interior'])
        eqw = sum(1 for r in sub if r['equals_worstcase'])
        print('%-7s 内点解 %5.1f%%   等于最坏准备 %5.1f%%   平均 q*=%.2f'
              % (name, 100 * inte / n, 100 * eqw / n, sum(r['q_star'] for r in sub) / n))
    print()
    print('线性成本下 q* 的取值集合:', sorted({r['q_star'] for r in rows if r['form'] == 'linear'}))
    print('凸成本下 q* 的取值个数:', len({r['q_star'] for r in rows if r['form'] == 'convex'}))
    print('阶梯成本下 q* 的取值个数:', len({r['q_star'] for r in rows if r['form'] == 'tiered'}))
    print()
    print('按 λ 看"等于最坏准备"的比例（凸成本）:')
    for lam in lams:
        sub = [r for r in rows if r['form'] == 'convex' and r['lam'] == lam]
        print('  λ=%4.2f  等于最坏 %5.1f%%   平均 q*=%.2f'
              % (lam, 100 * sum(1 for r in sub if r['equals_worstcase']) / len(sub),
                 sum(r['q_star'] for r in sub) / len(sub)))
    json.dump({name: dict(
        interior=sum(1 for r in rows if r['form'] == name and r['interior']),
        equals_worst=sum(1 for r in rows if r['form'] == name and r['equals_worstcase']),
        n=sum(1 for r in rows if r['form'] == name)) for name in FORMS},
        open(os.path.join(out, 'cost_probe_summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
