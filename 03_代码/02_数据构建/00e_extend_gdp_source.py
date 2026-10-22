# -*- coding: utf-8 -*-
"""
第 2 步：修复源头 —— 让流水线自己能生成干净数据

根因：04_generate_quantified_datasets.py 依赖 data/武汉各区GDP.csv，
      但该表只有武汉各区，缺 15 个非武汉区县 → 49 个节点 GDP 为 NaN
      → capacity_index 与边容量连锁缺失。

做法（两层保险）：
  A) 扩展 data/武汉各区GDP.csv，补入本次爬取的 15 个非武汉区县真实 GDP
     （同时补 3 个同城但未被节点引用的：梁子湖区、华容区已有、安陆市、云梦县）
  B) 不覆盖原文件，先备份原表。

输出：
  data/武汉各区GDP.csv            （扩展后，含非武汉区县）
  data_backup_原始_20261011/武汉各区GDP.csv  （已由整体备份保存）
"""
import os
import shutil
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, 'data')
F = os.path.join(BASE, 'data_fixed')
BK = os.path.join(BASE, 'data_backup_原始_20261011')

gdp_path = os.path.join(D, '武汉各区GDP.csv')
crawl_path = os.path.join(F, '县市区GDP_爬取结果.csv')

g = pd.read_csv(gdp_path)
c = pd.read_csv(crawl_path)

print('原 GDP 表行数:', len(g))
print('爬取区县数:', len(c))

# 备份原表（若整体备份未含此文件）
bk_path = os.path.join(BK, '武汉各区GDP.csv')
if not os.path.exists(bk_path):
    shutil.copy2(gdp_path, bk_path)
    print('已备份原表 ->', bk_path)

add = []
for _, r in c.iterrows():
    add.append({
        '地区': str(r['区县']).strip(),
        '地区生产总值': float(r['地区生产总值_亿元']) if pd.notna(r['地区生产总值_亿元']) else None,
        '比上年增长 (%)': None,
        '第一产业': None,
        '第一产业比上年增长 (%)': None,
    })

g2 = pd.concat([g, pd.DataFrame(add)], ignore_index=True)
g2 = g2.drop_duplicates('地区', keep='first')
g2.to_csv(gdp_path, index=False, encoding='utf-8-sig')

print('扩展后行数:', len(g2))
print('新增区县:')
new = [a['地区'] for a in add if a['地区'] not in set(g['地区'])]
print(' ', new)
print()
print(g2.tail(20).to_string(index=False))
