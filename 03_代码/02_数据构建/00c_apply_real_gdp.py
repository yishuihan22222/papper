# -*- coding: utf-8 -*-
"""
第二步补充：用「爬取到的真实区县 GDP」替换 04b 中的「城市GDP均摊」估算值

背景：
  04b_武汉节点代理容量 中，非武汉节点（孝感/鄂州/黄冈）的 district_gdp
  一律取所在市的 GDP 后按该市节点数均摊，导致同城所有节点取值完全相同，
  无法区分同一市内不同区县的经济发展水平 —— 这是审稿人易质疑的"伪精度"。

本脚本：
  1) 读取 data_fixed/县市区GDP_爬取结果.csv（真实区县 GDP，来源可溯）
  2) 按「城市+区县」精确匹配到 04b 每一行，替换 district_gdp
  3) 重新归一化 district_gdp_norm（0-1，与武汉各区口径一致）
  4) 按论文原公式重算 capacity_index / proxy_daily_load / overload_threshold
  5) 保留原列以便对比，新增 *_真实GDP 列与数据来源

输出：
  data_fixed/04b_武汉节点代理容量_真实GDP.csv
  data_fixed/00c_gdp_replace_log.txt
"""
import os
import math
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'data_fixed')

LOG = []


def log(m):
    print(m)
    LOG.append(str(m))


def norm_minmax(s):
    lo, hi = s.min(), s.max()
    if hi == lo:
        return s * 0.0
    return (s - lo) / (hi - lo)


def main():
    log('=' * 74)
    log('用真实区县 GDP 替换均摊估算')
    log('=' * 74)

    nodes = pd.read_csv(os.path.join(OUT, '04b_武汉节点代理容量_已修复.csv'))
    gdp = pd.read_csv(os.path.join(OUT, '县市区GDP_爬取结果.csv'))

    log(f'节点表：{len(nodes)} 行；真实GDP表：{len(gdp)} 行')

    # 建立 (城市, 区县) -> (gdp, 来源, 年份) 映射
    gmap = {}
    for _, r in gdp.iterrows():
        if pd.isna(r['地区生产总值_亿元']):
            continue
        gmap[(str(r['城市']).strip(), str(r['区县']).strip())] = (
            float(r['地区生产总值_亿元']),
            str(r.get('数据来源', '')),
            str(r.get('公报年份', '')),
        )

    # 备份原列
    nodes['district_gdp_原均摊值'] = nodes['district_gdp']
    nodes['district_gdp_估算方式_原'] = nodes['district_gdp_估算方式']

    hit = 0
    missed = []
    new_gdp = []
    new_src = []
    for _, r in nodes.iterrows():
        if str(r['城市']).strip() == '武汉市':
            new_gdp.append(r['district_gdp'])          # 武汉各区保持原值
            new_src.append('武汉统计年鉴/区GDP（原有）')
            continue
        key = (str(r['城市']).strip(), str(r['区县']).strip())
        if key in gmap:
            g, src, yr = gmap[key]
            new_gdp.append(g)
            new_src.append(f'{src}（{yr}公报）')
            hit += 1
        else:
            new_gdp.append(r['district_gdp'])
            new_src.append(r['district_gdp_估算方式'])
            missed.append(key)

    nodes['district_gdp'] = new_gdp
    nodes['district_gdp_数据来源'] = new_src

    log(f'非武汉节点替换命中：{hit} 行')
    if missed:
        log(f'未命中（保留原估算）：{sorted(set(missed))}')
    else:
        log('未命中：无')

    # 重新归一化（全表口径，与武汉一致）
    nodes['district_gdp_norm'] = norm_minmax(nodes['district_gdp']).round(6)

    # 按论文原公式重算容量三件套
    nodes['capacity_index'] = (
        0.35 * nodes['type_weight']
        + 0.20 * nodes['district_gdp_norm']
        + 0.25 * nodes['weighted_degree_norm']
        + 0.20 * nodes['hub_weight']
    ).round(4)
    nodes['proxy_daily_load'] = (50 + 950 * nodes['capacity_index']).round(2)
    nodes['overload_threshold'] = (1.25 * nodes['proxy_daily_load']).round(2)

    out = os.path.join(OUT, '04b_武汉节点代理容量_真实GDP.csv')
    nodes.to_csv(out, index=False, encoding='utf-8-sig')

    log('')
    log(f'输出：{out}')
    log(f'capacity_index 区间：{nodes["capacity_index"].min()} ~ {nodes["capacity_index"].max()}')
    log(f'district_gdp 区间：{nodes["district_gdp"].min():.2f} ~ {nodes["district_gdp"].max():.2f}')
    log('')
    log('非武汉节点替换前后对照（去重）：')
    sub = nodes[nodes['城市'] != '武汉市'][
        ['城市', '区县', 'district_gdp_原均摊值', 'district_gdp',
         'district_gdp_数据来源']
    ].drop_duplicates(['城市', '区县']).sort_values(['城市', '区县'])
    log(sub.to_string(index=False))

    with open(os.path.join(OUT, '00c_gdp_replace_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))


if __name__ == '__main__':
    main()
