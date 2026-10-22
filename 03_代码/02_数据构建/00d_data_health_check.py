# -*- coding: utf-8 -*-
"""
数据体检：只读检查，不改任何文件。
覆盖：完整性 / 唯一性 / 数值合理性 / 跨表一致性 / 单位一致性
"""
import os
import re
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, 'data')
F = os.path.join(BASE, 'data_fixed')

issues = []      # (级别, 文件, 问题)
notes = []       # 说明性信息


def iss(level, f, msg):
    issues.append((level, f, msg))


def rd(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        iss('ERR', os.path.basename(path), f'无法读取: {e}')
        return None


print('=' * 78)
print('A. 第一步修复产物 04b / 04c 复检')
print('=' * 78)

b = rd(os.path.join(F, '04b_武汉节点代理容量_真实GDP.csv'))
c = rd(os.path.join(F, '04c_武汉边代理属性_已修复.csv'))

# --- 04b ---
print(f'\n[04b] shape={b.shape}')
for col in ['node_id', '名称', '城市', '区县', 'district_gdp', 'capacity_index',
            'proxy_daily_load', 'overload_threshold']:
    if col not in b.columns:
        iss('致命', '04b', f'缺列 {col}')
    else:
        n = int(b[col].isna().sum())
        if n:
            iss('致命', '04b', f'{col} 缺失 {n} 行')
        else:
            print(f'  {col}: 无缺失')
if b['node_id'].duplicated().any():
    iss('致命', '04b', f'node_id 重复 {int(b["node_id"].duplicated().sum())} 条')
else:
    print('  node_id: 唯一 ✅')
print(f'  capacity_index 区间: {b.capacity_index.min():.4f} ~ {b.capacity_index.max():.4f}')
print(f'  district_gdp 区间: {b.district_gdp.min():.2f} ~ {b.district_gdp.max():.2f}')
# 容量与载荷一致性
chk = (b['proxy_daily_load'] - (50 + 950 * b['capacity_index'])).abs().max()
print(f'  proxy_daily_load 与公式最大偏差: {chk:.4f}')
chk2 = (b['overload_threshold'] - 1.25 * b['proxy_daily_load']).abs().max()
print(f'  overload_threshold 与公式最大偏差: {chk2:.4f}')
# 非武汉区县GDP唯一值
s = b[b['城市'] != '武汉市'].groupby('区县')['district_gdp'].first()
print(f'  非武汉区县 GDP 唯一值: {s.nunique()} / {len(s)}')

# --- 04c ---
print(f'\n[04c] shape={c.shape}')
print(f'  columns: {list(c.columns)}')
for col in ['Source', 'Target', 'Edge_capacity_proxy']:
    if col in c.columns:
        n = int(c[col].isna().sum())
        if n:
            iss('重大', '04c', f'{col} 缺失 {n} 行')
        else:
            print(f'  {col}: 无缺失 ✅')
    else:
        iss('重大', '04c', f'缺列 {col}')

print()
print('=' * 78)
print('B. 原 12 项问题逐条复检（对照 data/ 原始文件）')
print('=' * 78)

# --- P0-1 单位错误 ---
sup_raw = rd(os.path.join(D, '公开补充数据_城市物流指标.csv'))
sup_fix = rd(os.path.join(F, '公开补充数据_城市物流指标_已修正.csv'))
if sup_raw is not None:
    m = sup_raw[(sup_raw['指标'] == '货物周转量')]
    print('\n[P0-1 单位] 原文件 货物周转量 行:')
    bad = []
    for _, r in m.iterrows():
        u = str(r['单位']).strip()
        v = r['数值']
        tag = '  <-- 单位存疑' if u == '万吨公里' and v > 100000 else ''
        if tag:
            bad.append((r.get('城市', '?'), v, u))
        print(f"    {r.get('城市','?')}: {v} {u}{tag}")
    if bad:
        iss('致命(已修)', '公开补充数据_城市物流指标.csv',
            f'{len(bad)} 行"万吨公里"量级异常: {bad}')
if sup_fix is not None:
    m2 = sup_fix[sup_fix['指标'] == '货物周转量']
    units = m2['单位'].unique()
    print(f'  修正后单位取值: {units}')
    print(f'  修正后 货物周转量 区间: {m2["数值"].min()} ~ {m2["数值"].max()}')

# --- P0-2 节点容量缺失 ---
b_raw = rd(os.path.join(D, '04b_武汉节点代理容量.csv'))
if b_raw is not None:
    miss_raw = int(b_raw['capacity_index'].isna().sum()) if 'capacity_index' in b_raw.columns else -1
    print(f'\n[P0-2 容量缺失] 原 04b capacity_index 缺失: {miss_raw}')
    print(f'  修正后 04b 缺失: {int(b["capacity_index"].isna().sum())}')
    if miss_raw:
        iss('重大(已修)', '04b_武汉节点代理容量.csv', f'原缺失 {miss_raw} 条容量')
    # 节点名重复
    if '名称' in b_raw.columns:
        dup = int(b_raw['名称'].duplicated().sum())
        print(f'  原 04b 名称重复: {dup}  (唯一名称 {b_raw["名称"].nunique()}/{len(b_raw)})')
        iss('重大(已修)', '04b_武汉节点代理容量.csv',
            f'原 {dup} 条名称重复，会导致下游膨胀')

# --- P0-2b 边容量缺失 ---
c_raw = rd(os.path.join(D, '04c_武汉边代理属性.csv'))
if c_raw is not None:
    col = 'Edge_capacity_proxy'
    if col in c_raw.columns:
        miss = int(c_raw[col].isna().sum())
        print(f'  原 04c {col} 缺失: {miss} / {len(c_raw)}')
        if miss:
            iss('重大(已修)', '04c_武汉边代理属性.csv', f'原缺失 {miss} 条边容量')

# --- 完全图问题 ---
if c_raw is not None and 'Source' in c_raw.columns:
    nodes = set(c_raw['Source']) | set(c_raw['Target'])
    n = len(nodes)
    full = n * (n - 1) // 2
    print(f'\n[完全图] 下层节点 {n} 个；实际边 {len(c_raw)}；完全图应有 {full}')
    if len(c_raw) >= full * 0.99:
        iss('重大(未修)', '04c_武汉边代理属性.csv',
            f'下层网络是完全图（{len(c_raw)}≈{full} 条边），非稀疏网络')

print()
print('=' * 78)
print('C. 上游网络与节点表一致性')
print('=' * 78)

ll = rd(os.path.join(D, '04_lower_layer_nodes.csv'))
ul = rd(os.path.join(D, '04_upper_layer_nodes.csv'))
le = rd(os.path.join(D, '04_upper_layer_edges.csv'))
cl = rd(os.path.join(D, '04_cross_layer_edges.csv'))
for nm, df in [('04_lower_layer_nodes', ll), ('04_upper_layer_nodes', ul),
               ('04_upper_layer_edges', le), ('04_cross_layer_edges', cl)]:
    if df is not None:
        print(f'  {nm}: {df.shape}  cols={list(df.columns)[:8]}')

if ll is not None and 'id' in ll.columns:
    if ll['id'].duplicated().any():
        iss('重大', '04_lower_layer_nodes.csv', f'id 重复 {int(ll["id"].duplicated().sum())} 条')

# --- 城市指标估算占比 ---
g = rd(os.path.join(D, '04a_城市综合指标_含估算.csv'))
if g is not None:
    print(f'\n[04a] shape={g.shape}  cols={list(g.columns)}')
    for col in g.columns:
        if '估算' in col or '来源' in col:
            try:
                print(f'  {col} 取值分布:')
                print(g[col].value_counts().to_string())
            except Exception:
                pass
    # 找估算方式列
    est_cols = [x for x in g.columns if '方式' in x or '来源' in x]
    for ec in est_cols:
        vals = g[ec].astype(str)
        n_est = vals.str.contains('估算|回归|均摊|估计', na=False).sum()
        if n_est:
            iss('重大(部分未修)', '04a_城市综合指标_含估算.csv',
                f'{ec}: {n_est}/{len(g)} 行为估算值')

print()
print('=' * 78)
print('D. 硬编码 / 手工赋值 扫描（论文可信度风险点）')
print('=' * 78)

scan_files = ['04a_城市综合指标_含估算.csv', '04b_武汉节点代理容量.csv',
              '公开补充数据_城市物流指标.csv']
for fn in scan_files:
    df = rd(os.path.join(D, fn))
    if df is None:
        continue
    for col in df.columns:
        if df[col].dtype == object:
            vals = df[col].astype(str)
            # 是否包含"手工/硬编码/估算/经验"等字样
            hit = vals.str.contains('手工|硬编码|经验|拍|估计|估算', na=False).sum()
            if hit:
                print(f'  [{fn}] {col}: {hit} 行含估算/手工字样')

print()
print('=' * 78)
print('E. 下游结果文件抽查（数值合理性）')
print('=' * 78)

for fn in ['05_resilience_metrics.csv', '06b_cascade_failure_summary.csv',
           '07d_attack_comparison_summary.csv', '08_benchmark_improvement_vs_best_baseline.csv',
           '09b_optimization_algorithm_summary.csv']:
    df = rd(os.path.join(D, fn))
    if df is not None:
        print(f'\n  [{fn}] {df.shape}')
        print('   ' + df.head(6).to_string().replace('\n', '\n   '))

print()
print('=' * 78)
print('体检结论汇总')
print('=' * 78)
if not issues:
    print('未发现结构性问题。')
else:
    lvl_order = {'致命': 0, '致命(已修)': 1, '重大': 2, '重大(已修)': 3, '重大(部分未修)': 4, 'ERR': 5}
    issues.sort(key=lambda x: lvl_order.get(x[0], 9))
    for lv, f, msg in issues:
        print(f'  [{lv}] {f}: {msg}')
