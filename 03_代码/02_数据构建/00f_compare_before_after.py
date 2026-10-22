# -*- coding: utf-8 -*-
"""
回填修复版后：全流水线 前后结果对比
对比 compare_before/（原始脏数据结果）与 data/（修复后结果）
"""
from project_paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR, PAPER_DIR, TABLES_DIR, DOCS_DIR, SOURCES_DIR, BACKUP_DIR, LOGS_DIR
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, 'data')
B = os.path.join(BASE, 'data_backup_原始_20261011')   # 真正的原始备份（修复前）

OUT_MD = str(DOCS_DIR / '回填前后结果对比.md')
lines = []


def w(s=''):
    print(s)
    lines.append(str(s))


def num_delta(a, b):
    try:
        a, b = float(a), float(b)
        if a == 0:
            return '—'
        return f'{(b - a) / abs(a) * 100:+.2f}%'
    except Exception:
        return '—'


w('# 回填修复版后 · 全流水线结果对比')
w()
w('> 对比对象：`compare_before/`（原始含脏数据的结果） vs `data/`（修复后重跑结果）')
w('> 修复内容：①潜江单位错误 ②49节点/10246边容量缺失 ③节点主键重复 ④非武汉15区县真实GDP')
w()

# ---------- 1. 韧性总指标 ----------
w('## 1. 韧性总指标（05_resilience_metrics）')
w()
try:
    o = pd.read_csv(os.path.join(B, '05_resilience_metrics.csv'))
    n = pd.read_csv(os.path.join(D, '05_resilience_metrics.csv'))
    m = o.merge(n, on='指标', suffixes=('_旧', '_新'))
    w('| 指标 | 旧值 | 新值 | 变化 | 是否受影响 |')
    w('|---|---|---|---|---|')
    for _, r in m.iterrows():
        d = num_delta(r['数值_旧'], r['数值_新'])
        flag = '**变化**' if d not in ('—', '+0.00%', '-0.00%') else '不变'
        w(f"| {r['指标']} | {r['数值_旧']} | {r['数值_新']} | {d} | {flag} |")
except Exception as e:
    w(f'（读取失败：{e}）')
w()

# ---------- 2. 脆弱性排名 ----------
w('## 2. 节点脆弱性排名（06）')
w()
try:
    o = pd.read_csv(os.path.join(B, '06_ai_node_vulnerability_scores.csv'))
    n = pd.read_csv(os.path.join(D, '06_ai_node_vulnerability_scores.csv'))
    o = o.reset_index(drop=True)
    n = n.reset_index(drop=True)
    if len(o) == len(n):
        dscore = (n['svd_bridge_score'] - o['svd_bridge_score']).abs()
        drank = (n['svd_bridge_rank'] - o['svd_bridge_rank']).abs()
        w(f"- 打分最大变化：**{dscore.max():.4f}**")
        w(f"- 排名最大移动：**{int(drank.max())}** 位")
        w(f"- 完全未变的节点：**{int((dscore < 1e-9).sum())} / {len(o)}**")
    w('**旧 TOP10 最脆弱节点**')
    w()
    w('| 排名 | 名称 | 城市 | 分数 |')
    w('|---|---|---|---|')
    for _, r in o.nsmallest(10, 'svd_bridge_rank').iterrows():
        w(f"| {int(r['svd_bridge_rank'])} | {r['名称']} | {r['城市']} | {r['svd_bridge_score']} |")
    w()
    w('**新 TOP10 最脆弱节点**')
    w()
    w('| 排名 | 名称 | 城市 | 分数 |')
    w('|---|---|---|---|')
    for _, r in n.nsmallest(10, 'svd_bridge_rank').iterrows():
        w(f"| {int(r['svd_bridge_rank'])} | {r['名称']} | {r['城市']} | {r['svd_bridge_score']} |")
except Exception as e:
    w(f'（读取失败：{e}）')
w()

# ---------- 3. 攻击对比 ----------
w('## 3. 攻击策略效果（07d）')
w()
try:
    o = pd.read_csv(os.path.join(B, '07d_attack_comparison_summary.csv'))
    n = pd.read_csv(os.path.join(D, '07d_attack_comparison_summary.csv'))
    m = o.merge(n, on=['strategy', 'attack_kind'], suffixes=('_旧', '_新'))
    w('| 策略 | 攻击 | Damage_Index旧 | Damage_Index新 | 变化 |')
    w('|---|---|---|---|---|')
    for _, r in m.iterrows():
        w(f"| {r['strategy']} | {r['attack_kind']} | "
          f"{r['Damage_Index_旧']:.4f} | {r['Damage_Index_新']:.4f} | "
          f"{num_delta(r['Damage_Index_旧'], r['Damage_Index_新'])} |")
except Exception as e:
    w(f'（读取失败：{e}）')
w()

# ---------- 4. 恢复/优化 ----------
for fn, title in [('08_benchmark_improvement_vs_best_baseline.csv', '恢复策略（08）'),
                  ('09b_optimization_algorithm_summary.csv', '优化算法（09b）')]:
    w(f'## 4. {title}')
    w()
    try:
        o = pd.read_csv(os.path.join(B, fn))
        n = pd.read_csv(os.path.join(D, fn))
        w(f'旧：shape={o.shape}｜新：shape={n.shape}')
        w()
        w('```')
        w('[新结果]')
        w(n.to_string(index=False))
        w('```')
    except Exception as e:
        w(f'（读取失败：{e}）')
    w()

os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print()
print('对比报告已写入：', OUT_MD)
