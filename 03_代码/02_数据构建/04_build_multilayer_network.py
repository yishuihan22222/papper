import os
import pandas as pd


PORTAL_KEYWORDS = {
    'airport': ['空港', '机场', '航港'],
    'rail': ['铁路', '火车站', '货运站', '站'],
    'port': ['港', '码头'],
    'bonded': ['保税'],
    'coldchain': ['冷链', '冷库'],
    'distribution': ['分拨', '快递', '配送']
}

CITY_TO_GATEWAY_TYPE = {
    '鄂州市': ['airport', 'port'],
    '黄冈市': ['port', 'distribution'],
    '黄石市': ['port'],
    '咸宁市': ['rail', 'distribution'],
    '孝感市': ['airport', 'distribution'],
    '仙桃市': ['distribution'],
    '潜江市': ['coldchain', 'distribution'],
    '天门市': ['distribution']
}


def detect_portal_family(row):
    family = str(row.get('family', '')).lower()
    node_type = str(row.get('类型', ''))
    node_name = str(row.get('名称', ''))
    text = f'{family} {node_type} {node_name}'
    # 已量化的功能类别是权威字段，避免再次被混合品牌或地名覆盖。
    if family in PORTAL_KEYWORDS:
        return family
    for key, words in PORTAL_KEYWORDS.items():
        if key in family:
            return key
        if any(w in text for w in words):
            return key
    return 'other'


if __name__ == '__main__':
    print('阶段 04: 双层耦合网络构建')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')

    city_path = os.path.join(data_dir, '03b_gravity_results.csv')
    node_path = os.path.join(data_dir, '04b_武汉节点代理容量.csv')

    out_upper_nodes = os.path.join(data_dir, '04_upper_layer_nodes.csv')
    out_upper_edges = os.path.join(data_dir, '04_upper_layer_edges.csv')
    out_lower_nodes = os.path.join(data_dir, '04_lower_layer_nodes.csv')
    out_cross_edges = os.path.join(data_dir, '04_cross_layer_edges.csv')
    out_summary = os.path.join(data_dir, '04_multilayer_network_summary.csv')

    city_df = pd.read_csv(city_path)
    node_df = pd.read_csv(node_path)

    lower_df = node_df.copy()
    lower_df['family'] = lower_df['family'].fillna('other') if 'family' in lower_df.columns else 'other'
    lower_df['portal_family'] = lower_df.apply(detect_portal_family, axis=1)

    upper_nodes = city_df[['城市', 'GDP 亿元', '综合时空阻抗', '需求强度指数', '产业匹配度', '枢纽加成alpha', '综合引力指数']].copy()
    upper_nodes = upper_nodes.rename(columns={'城市': 'upper_node'})
    upper_nodes['layer'] = 'upper'

    upper_edges = upper_nodes[['upper_node', '综合引力指数', '综合时空阻抗', '需求强度指数', '产业匹配度']].copy()
    upper_edges['source'] = '武汉市'
    upper_edges['target'] = upper_edges['upper_node']
    upper_edges['coupling_type'] = 'city_gravity_link'
    upper_edges['weight'] = (
        0.55 * upper_edges['综合引力指数'] / upper_edges['综合引力指数'].max() +
        0.25 * upper_edges['需求强度指数'] +
        0.20 * upper_edges['产业匹配度']
    ).round(4)
    upper_edges = upper_edges[['source', 'target', 'coupling_type', 'weight', '综合时空阻抗']]
    upper_nodes['node_role'] = 'peripheral_city'
    # 武汉中心仅补网络身份，不虚构未采集的城市指标；外围引力计算保持原样。
    hub = {c:pd.NA for c in upper_nodes.columns}
    hub.update({'upper_node':'武汉市','layer':'upper','node_role':'hub'})
    city_sources=pd.read_csv(os.path.join(data_dir,'公开补充数据_城市物流指标.csv'))
    hub_gdp=city_sources[(city_sources['城市']=='武汉市')&(city_sources['指标']=='地区生产总值')&(city_sources['单位']=='亿元')]
    if not hub_gdp.empty:hub['GDP 亿元']=float(hub_gdp.sort_values('年份').iloc[-1]['数值'])
    upper_nodes = pd.concat([pd.DataFrame([hub]), upper_nodes], ignore_index=True)

    lower_nodes = lower_df[['node_id', '名称', '城市', '区县', '类型', 'family', 'portal_family', 'proxy_daily_load', 'capacity_index', 'weighted_degree_norm', 'risk_exposure']].copy()
    lower_nodes['layer'] = 'lower'

    cross_records = []
    for city, families in CITY_TO_GATEWAY_TYPE.items():
        # 在全部有效门户中按论文式(13)预评分，而不是先把非目标类型全部过滤掉。
        # 这样功能匹配项 B_iv 才有 1.0/0.5 的真实区分度。
        candidates = lower_nodes[lower_nodes['portal_family'] != 'other'].copy()
        candidates['match_bonus'] = candidates['portal_family'].isin(families).map({True: 1.0, False: 0.5})
        candidates['preselect_score'] = (
            0.45 * candidates['capacity_index'] +
            0.25 * candidates['weighted_degree_norm'] +
            0.15 * candidates['match_bonus'] +
            0.15 * (1 - candidates['risk_exposure'])
        )
        candidates = candidates.sort_values(
            ['preselect_score', 'capacity_index', 'weighted_degree_norm'], ascending=False
        ).head(10)
        for _, row in candidates.iterrows():
            match_bonus = float(row['match_bonus'])
            coupling_weight = round(
                0.45 * float(row['capacity_index']) +
                0.25 * float(row['weighted_degree_norm']) +
                0.15 * float(match_bonus) +
                0.15 * (1 - float(row['risk_exposure'])),
                4
            )
            cross_records.append({
                'upper_node': city,
                'lower_node': row['名称'],
                'lower_node_id': row['node_id'],
                'upper_to_lower_type': row['portal_family'],
                'match_bonus': match_bonus,
                'coupling_weight': coupling_weight
            })

    # 用 node_id 去重（而非名称），避免同名节点被错误合并
    cross_edges = pd.DataFrame(cross_records).drop_duplicates(subset=['upper_node', 'lower_node_id'])

    summary = pd.DataFrame([
        {'item': 'upper_nodes', 'value': len(upper_nodes)},
        {'item': 'upper_peripheral_nodes', 'value': int(upper_nodes['node_role'].eq('peripheral_city').sum())},
        {'item': 'upper_edges', 'value': len(upper_edges)},
        {'item': 'lower_nodes', 'value': len(lower_nodes)},
        {'item': 'cross_edges', 'value': len(cross_edges)},
        {'item': 'identified_portal_nodes', 'value': int((lower_nodes['portal_family'] != 'other').sum())},
        {'item': 'avg_cross_layer_weight', 'value': round(cross_edges['coupling_weight'].mean(), 4) if not cross_edges.empty else 0.0}
    ])

    upper_nodes.to_csv(out_upper_nodes, index=False, encoding='utf-8-sig')
    upper_edges.to_csv(out_upper_edges, index=False, encoding='utf-8-sig')
    lower_nodes.to_csv(out_lower_nodes, index=False, encoding='utf-8-sig')
    cross_edges.to_csv(out_cross_edges, index=False, encoding='utf-8-sig')
    summary.to_csv(out_summary, index=False, encoding='utf-8-sig')

    print(summary.to_string(index=False))
    print(f'\n上层节点已保存至: {out_upper_nodes}')
    print(f'上层边已保存至: {out_upper_edges}')
    print(f'下层节点已保存至: {out_lower_nodes}')
    print(f'跨层耦合边已保存至: {out_cross_edges}')
    print(f'耦合网络摘要已保存至: {out_summary}')
