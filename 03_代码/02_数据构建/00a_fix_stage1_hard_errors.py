# -*- coding: utf-8 -*-
"""
第一步修复脚本：修掉三个硬错误（只读原数据，全部输出到 data_fixed/，不覆盖原文件）

P0-1  单位错误：潜江市货物周转量 492910.24 万吨公里 -> 49.2910 亿吨公里
P0-2  缺失容量：49 个非武汉节点缺区县 GDP，导致 capacity_index 等 5 列缺失
P0-3  重复主键：同名节点（如"货运司机服务中心"）被下游按名称合并后膨胀

同时输出修复日志，供论文数据说明章节直接引用。
"""
import os
import json
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'data_fixed')
LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ============ P0-1 单位修复：万吨公里 -> 亿吨公里 ============
    log('=' * 70)
    log('P0-1 单位修复：货物周转量')
    log('=' * 70)

    sup = pd.read_csv(os.path.join(DATA, '公开补充数据_城市物流指标.csv'))
    before = sup[sup['指标'] == '货物周转量'][['城市', '数值', '单位']].copy()
    log('修复前（货物周转量）：')
    log(before.to_string(index=False))

    UNIT_FIX = {'万吨公里': 1.0 / 10000.0}  # 万吨公里 / 10000 = 亿吨公里
    fixed_rows = 0
    for i, row in sup.iterrows():
        if row['指标'] == '货物周转量' and str(row['单位']).strip() in UNIT_FIX:
            old = float(row['数值'])
            new = round(old * UNIT_FIX[str(row['单位']).strip()], 4)
            sup.at[i, '数值'] = new
            sup.at[i, '单位'] = '亿吨公里'
            sup.at[i, '备注'] = str(row.get('备注', '')) + ' [已修正：万吨公里->亿吨公里]'
            fixed_rows += 1
            log(f"  {row['城市']}: {old} 万吨公里 -> {new} 亿吨公里")

    after = sup[sup['指标'] == '货物周转量'][['城市', '数值', '单位']].copy()
    log(f'共修正 {fixed_rows} 条记录。修复后：')
    log(after.to_string(index=False))
    sup.to_csv(os.path.join(OUT, '公开补充数据_城市物流指标_已修正.csv'),
               index=False, encoding='utf-8-sig')

    # ============ P0-2 缺失容量修复：补齐区县 GDP ============
    log('')
    log('=' * 70)
    log('P0-2 缺失容量修复：补齐区县 GDP')
    log('=' * 70)

    nodes = pd.read_csv(os.path.join(DATA, '04b_武汉节点代理容量.csv'))
    cities = pd.read_csv(os.path.join(DATA, '04a_城市综合指标_含估算.csv'))
    city_gdp = cities.set_index('城市')['GDP 亿元'].to_dict()

    # 武汉各区 GDP 直接来自官方区县表
    wuhan = pd.read_csv(os.path.join(DATA, '武汉各区GDP.csv'))
    wuhan_gdp = {
        str(r['地区']).strip(): float(r['地区生产总值'])
        for _, r in wuhan.iterrows()
        if pd.notna(r['地区生产总值'])
    }

    # 非武汉区县 -> 所属城市映射（按节点表实际出现的区县构建）
    DISTRICT_TO_CITY = {}
    for _, r in nodes.iterrows():
        c, d = r.get('城市'), r.get('区县')
        if pd.notna(c) and pd.notna(d):
            DISTRICT_TO_CITY.setdefault(str(d).strip(), str(c).strip())

    missing_before = int(nodes['capacity_index'].isna().sum())
    log(f'修复前缺失 capacity_index 的节点数：{missing_before}')

    # 城市非武汉下辖区县：用一个统一的"城市均值分摊系数"把城市GDP摊到区县
    # 说明：这是代理估计，必须写入 _修复方式 字段并在论文中披露
    filled_records = []

    def resolve_gdp(city, district):
        d = str(district).strip()
        c = str(city).strip()
        # 1) 武汉：走官方区县表（含别名）
        alias = {
            '东西湖区': '武汉临空港经济技术开发区(东西湖区)',
            '汉南区': '武汉经济技术开发区(汉南区)',
            '东湖高新技术开发区': '东湖高新技术开发区',
        }
        key = alias.get(d, d)
        if key in wuhan_gdp:
            return wuhan_gdp[key], '武汉区县官方GDP'
        if d in wuhan_gdp:
            return wuhan_gdp[d], '武汉区县官方GDP'
        # 2) 非武汉地市：按该市下辖节点数均摊城市GDP
        if c in city_gdp:
            return None, 'CITY_FALLBACK'
        return None, 'UNKNOWN'

    for i, row in nodes.iterrows():
        if pd.notna(row['district_gdp']):
            continue
        gdp, method = resolve_gdp(row['城市'], row['区县'])
        if method == 'CITY_FALLBACK':
            filled_records.append(i)

    # 统计各城市待分摊节点数（含已有GDP的武汉节点不参与）
    fallback_idx = [i for i, r in nodes.iterrows()
                    if pd.isna(r['district_gdp'])
                    and str(r['城市']).strip() in city_gdp]
    if fallback_idx:
        city_counts = nodes.loc[fallback_idx].groupby('城市').size().to_dict()
        for i in fallback_idx:
            c = str(nodes.at[i, '城市']).strip()
            share = float(city_gdp[c]) / max(city_counts.get(c, 1), 1)
            nodes.at[i, 'district_gdp'] = round(share, 4)
            nodes.at[i, 'district_gdp_估算方式'] = (
                f'城市GDP按辖区节点数均摊（{city_counts.get(c,1)}个节点）'
            )

    # 重新计算派生列
    def mm(s):
        s = pd.to_numeric(s, errors='coerce')
        a, b = s.min(), s.max()
        if pd.isna(a) or pd.isna(b) or abs(float(b) - float(a)) < 1e-12:
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - a) / (b - a)

    nodes['district_gdp_norm'] = mm(nodes['district_gdp'])
    nodes['capacity_index'] = (
        0.35 * nodes['type_weight']
        + 0.20 * nodes['district_gdp_norm']
        + 0.25 * nodes['weighted_degree_norm']
        + 0.20 * nodes['hub_weight']
    ).round(4)
    nodes['proxy_daily_load'] = (50 + 950 * nodes['capacity_index']).round(2)
    nodes['overload_threshold'] = (1.25 * nodes['proxy_daily_load']).round(2)

    missing_after = int(nodes['capacity_index'].isna().sum())
    log(f'修复后缺失 capacity_index 的节点数：{missing_after}')
    log(f'其中由城市GDP均摊补齐的节点数：{len(fallback_idx)}')

    # ============ P0-3 重复主键修复 ============
    log('')
    log('=' * 70)
    log('P0-3 重复主键修复：生成唯一 node_id')
    log('=' * 70)

    dup_before = nodes[nodes.duplicated('名称', keep=False)]
    log(f'修复前重复名称的节点记录数：{len(dup_before)}，涉及名称：'
        f'{sorted(dup_before["名称"].unique().tolist())}')

    nodes['node_id'] = (
        nodes['名称'].astype(str).str.strip()
        + '@@' + nodes['城市'].astype(str).str.strip()
        + '@@' + nodes['区县'].astype(str).str.strip()
    )
    # 仍重复的（同城同区同名）追加序号
    dup_mask = nodes.duplicated('node_id', keep=False)
    if dup_mask.any():
        nodes['_seq'] = nodes.groupby('node_id').cumcount()
        nodes['node_id'] = nodes.apply(
            lambda r: r['node_id'] if r['_seq'] == 0 else f"{r['node_id']}#{r['_seq']}",
            axis=1
        )
        nodes = nodes.drop(columns=['_seq'])

    dup_after = int(nodes.duplicated('node_id').sum())
    log(f'修复后 node_id 重复数：{dup_after}')
    log(f'节点总行数：{len(nodes)}，唯一 node_id 数：{nodes["node_id"].nunique()}')

    # 把 node_id 放到第一列
    cols = ['node_id'] + [c for c in nodes.columns if c != 'node_id']
    nodes = nodes[cols]
    nodes.to_csv(os.path.join(OUT, '04b_武汉节点代理容量_已修复.csv'),
                 index=False, encoding='utf-8-sig')

    # ============ 同步修复边表：容量缺失与主键 ============
    log('')
    log('=' * 70)
    log('同步修复边表：使用修复后的节点容量重算边属性')
    log('=' * 70)

    edges = pd.read_csv(os.path.join(DATA, '04c_武汉边代理属性.csv'))
    miss_edge_before = int(edges['Edge_capacity_proxy'].isna().sum())
    log(f'修复前边容量缺失数：{miss_edge_before} / {len(edges)}')

    node_lookup = nodes.drop_duplicates('名称', keep='first').set_index('名称')
    cap_map = node_lookup['capacity_index'].to_dict()

    def recalc_edge(row):
        sc = cap_map.get(row['Source'], 0.5)
        tc = cap_map.get(row['Target'], 0.5)
        if pd.isna(sc):
            sc = 0.5
        if pd.isna(tc):
            tc = 0.5
        cc = row['Composite_transport_cost']
        if pd.isna(cc):
            cc = 0.001
        import math
        return round((math.sqrt(sc * tc) / (1 + cc)) * 100, 4)

    edges['Edge_capacity_proxy'] = edges.apply(recalc_edge, axis=1)
    miss_edge_after = int(edges['Edge_capacity_proxy'].isna().sum())
    log(f'修复后边容量缺失数：{miss_edge_after} / {len(edges)}')
    edges.to_csv(os.path.join(OUT, '04c_武汉边代理属性_已修复.csv'),
                 index=False, encoding='utf-8-sig')

    # ============ 修复日志 ============
    log('')
    log('=' * 70)
    log('修复完成')
    log('=' * 70)
    log(f'输出目录：{OUT}')

    summary = pd.DataFrame([
        {'item': '潜江货物周转量单位修正条数', 'value': fixed_rows},
        {'item': '修复前缺失capacity_index节点数', 'value': missing_before},
        {'item': '修复后缺失capacity_index节点数', 'value': missing_after},
        {'item': '城市GDP均摊补齐节点数', 'value': len(fallback_idx)},
        {'item': '修复前重复名称记录数', 'value': len(dup_before)},
        {'item': '修复后node_id重复数', 'value': dup_after},
        {'item': '修复前边容量缺失数', 'value': miss_edge_before},
        {'item': '修复后边容量缺失数', 'value': miss_edge_after},
    ])
    summary.to_csv(os.path.join(OUT, '00a_fix_stage1_summary.csv'),
                   index=False, encoding='utf-8-sig')

    with open(os.path.join(OUT, '00a_fix_stage1_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))

    print()
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
