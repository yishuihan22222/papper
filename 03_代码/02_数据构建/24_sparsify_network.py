# -*- coding: utf-8 -*-
"""
24_sparsify_network.py —— 稠密候选图 -> 稀疏物流网络（外生引力阈值）

方法：
  1) 节点外生质量（不使用网络度，避免循环依赖）
       mass_i = (0.35*type_weight + 0.20*district_gdp_norm + 0.20*hub_weight) / 0.75
     该式由原 capacity_index 去掉网络内生项 weighted_degree_norm 后重归一化得到。
  2) 边引力：
       G_ij = sqrt(mass_i * mass_j) / (1 + generalized_impedance_ij)^1.5
     阻抗指数 1.5 与项目城市层引力模型保持一致；+1 避免近零阻抗奇异。
  3) 保留 G_ij 最高的前 ratio（默认 15%）边。
  4) 对每个节点从全部关联边中强制保留引力最高的 kmin 条（默认 3），
     避免偏远节点孤立。该保底按“关联边”计算，不受 Source/Target 行方向影响。

为什么不用 Composite_transport_cost 直接剪枝：
  该列含 risk/synergy，且部分端点量依赖完全图的 degree；直接用会形成
  “完全图度 -> 成本 -> 剪枝”的循环。外生质量法切断此循环，更适合论文解释。

输出：
  data/04c_武汉边代理属性_稀疏{ratio}.csv
  data/24_sparsify_summary_{ratio}.csv
  data/24_node_degree_compare_{ratio}.csv
  data/24_city_pair_edges_{ratio}.csv
若加 --activate，同时将稀疏边表写为 data/04c_武汉边代理属性.csv（下游实际输入）。

用法：
  python Code/24_sparsify_network.py --ratio=0.15 --kmin=3
  python Code/24_sparsify_network.py --ratio=0.15 --kmin=3 --activate
"""
import os
import math
import argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
DENSE_CANDIDATE = os.path.join(DATA, '04c_武汉边代理属性_候选全边.csv')
ACTIVE_EDGE = os.path.join(DATA, '04c_武汉边代理属性.csv')
NODE_FILE = os.path.join(DATA, '04b_武汉节点代理容量.csv')


def tag_of(ratio):
    return f'{int(round(ratio * 100))}pct'


def degree_series(edges, node_ids):
    cnt = pd.concat([edges['Source_id'], edges['Target_id']]).value_counts()
    return pd.Series({n: int(cnt.get(n, 0)) for n in node_ids})


def connected_components(edges, node_ids):
    adj = {n: set() for n in node_ids}
    for a, b in zip(edges['Source_id'], edges['Target_id']):
        if a in adj and b in adj and a != b:
            adj[a].add(b)
            adj[b].add(a)
    seen, comps = set(), []
    for n in node_ids:
        if n in seen:
            continue
        stack, comp = [n], 0
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp += 1
            stack.extend(adj[u] - seen)
        comps.append(comp)
    return sorted(comps, reverse=True)


def add_exogenous_gravity(edges, nodes, beta=1.5):
    required = {'type_weight', 'district_gdp_norm', 'hub_weight', 'node_id'}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f'节点表缺少外生质量字段: {sorted(missing)}')
    if 'Generalized_impedance' not in edges.columns:
        raise ValueError('候选边表缺少 Generalized_impedance')

    n = nodes.copy()
    for c in ['type_weight', 'district_gdp_norm', 'hub_weight']:
        n[c] = pd.to_numeric(n[c], errors='coerce').fillna(0.0)
    n['exogenous_mass'] = (
        0.35 * n['type_weight'] +
        0.20 * n['district_gdp_norm'] +
        0.20 * n['hub_weight']
    ) / 0.75
    mass = n.set_index('node_id')['exogenous_mass'].to_dict()

    d = edges.copy()
    imp = pd.to_numeric(d['Generalized_impedance'], errors='coerce')
    if imp.isna().any() or (imp < 0).any():
        raise ValueError('Generalized_impedance 存在缺失或负值')
    missing_ids = (set(d['Source_id']) | set(d['Target_id'])) - set(mass)
    if missing_ids:
        raise ValueError(f'边表含节点表不存在的 node_id: {len(missing_ids)} 个')

    d['Source_exogenous_mass'] = d['Source_id'].map(mass)
    d['Target_exogenous_mass'] = d['Target_id'].map(mass)
    d['Gravity_score_exogenous'] = [
        math.sqrt(float(mi) * float(mj)) / ((1.0 + float(z)) ** beta)
        for mi, mj, z in zip(d['Source_exogenous_mass'],
                             d['Target_exogenous_mass'], imp)
    ]
    return d, n[['node_id', 'exogenous_mass']]


def sparsify(edges, ratio, kmin):
    n_threshold = max(1, int(round(len(edges) * ratio)))
    threshold_idx = set(edges.nlargest(n_threshold, 'Gravity_score_exogenous').index)
    keep_idx = set(threshold_idx)

    # 对每个节点从所有“关联边”中取引力最高的 kmin 条，与行方向无关。
    incident = {}
    for idx, a, b in zip(edges.index, edges['Source_id'], edges['Target_id']):
        incident.setdefault(a, []).append(idx)
        incident.setdefault(b, []).append(idx)
    for _, idxs in incident.items():
        top = edges.loc[idxs].nlargest(kmin, 'Gravity_score_exogenous').index
        keep_idx.update(top)

    out = edges.loc[sorted(keep_idx)].copy()
    return out, len(threshold_idx), len(keep_idx) - len(threshold_idx)


def minmax(series):
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if abs(hi - lo) < 1e-12:
        return pd.Series([1.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def recompute_sparse_metrics(edges, nodes):
    """用稀疏网络重算网络内生节点量与依赖容量的边量。

    weighted_degree_proxy 使用 1/(1+impedance)，而不是原 1/impedance：
    保留“低阻抗边贡献更大”的单调性，同时避免 0.016 这类近零阻抗
    把单条边放大为 62.5、支配整个节点度分布。
    """
    e = edges.copy()
    n = nodes.copy()
    strength = 1.0 / (1.0 + pd.to_numeric(e['Generalized_impedance'], errors='coerce').fillna(0.0))
    src = pd.DataFrame({'node_id': e['Source_id'], 'edge_strength_sparse': strength})
    tgt = pd.DataFrame({'node_id': e['Target_id'], 'edge_strength_sparse': strength})
    degree = pd.concat([src, tgt], ignore_index=True).groupby('node_id')['edge_strength_sparse'].sum()
    n['weighted_degree_proxy'] = n['node_id'].map(degree).fillna(0.0)
    n['weighted_degree_norm'] = minmax(n['weighted_degree_proxy'])
    n['capacity_index'] = (
        0.35 * pd.to_numeric(n['type_weight'], errors='coerce').fillna(0.0) +
        0.20 * pd.to_numeric(n['district_gdp_norm'], errors='coerce').fillna(0.0) +
        0.25 * n['weighted_degree_norm'] +
        0.20 * pd.to_numeric(n['hub_weight'], errors='coerce').fillna(0.0)
    ).round(4)
    n['proxy_daily_load'] = (50 + 950 * n['capacity_index']).round(2)
    n['overload_threshold'] = (1.25 * n['proxy_daily_load']).round(2)

    cap = n.set_index('node_id')['capacity_index'].to_dict()
    sc = e['Source_id'].map(cap)
    tc = e['Target_id'].map(cap)
    cc = pd.to_numeric(e['Composite_transport_cost'], errors='coerce').fillna(0.0)
    e['Edge_capacity_proxy'] = ((sc * tc).pow(0.5) / (1.0 + cc) * 100).round(4)
    return e, n


def city_pair_table(edges, nodes):
    city = nodes.set_index('node_id')['城市'].astype(str).to_dict()
    d = edges[['Source_id', 'Target_id']].copy()
    d['Source_city'] = d['Source_id'].map(city)
    d['Target_city'] = d['Target_id'].map(city)
    d['city_pair'] = [
        '—'.join(sorted([a, b])) for a, b in zip(d['Source_city'], d['Target_city'])
    ]
    out = d.groupby('city_pair').size().reset_index(name='edge_count')
    out['is_intercity'] = ~out['city_pair'].apply(
        lambda x: len(set(x.split('—'))) == 1)
    return out.sort_values(['is_intercity', 'edge_count'], ascending=[False, False])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, default=0.15)
    ap.add_argument('--kmin', type=int, default=3)
    ap.add_argument('--beta', type=float, default=1.5)
    ap.add_argument('--activate', action='store_true')
    ap.add_argument('--input', default='')
    args = ap.parse_args()
    if not (0 < args.ratio < 1):
        raise ValueError('ratio 必须位于 (0, 1)')
    if args.kmin < 1:
        raise ValueError('kmin 必须 >= 1')

    input_path = args.input or (DENSE_CANDIDATE if os.path.exists(DENSE_CANDIDATE) else ACTIVE_EDGE)
    tag = tag_of(args.ratio)
    out_path = os.path.join(DATA, f'04c_武汉边代理属性_稀疏{tag}.csv')
    summary_path = os.path.join(DATA, f'24_sparsify_summary_{tag}.csv')
    degree_path = os.path.join(DATA, f'24_node_degree_compare_{tag}.csv')
    node_sparse_path = os.path.join(DATA, f'04b_武汉节点代理容量_稀疏{tag}.csv')
    city_path = os.path.join(DATA, f'24_city_pair_edges_{tag}.csv')

    print('=' * 70)
    print(f'外生引力稀疏化 ratio={args.ratio:.2f}, kmin={args.kmin}, beta={args.beta}')
    print(f'输入: {input_path}')
    print('=' * 70)

    raw = pd.read_csv(input_path)
    nodes = pd.read_csv(NODE_FILE)
    node_ids = sorted(nodes['node_id'].astype(str).unique())
    enriched, masses = add_exogenous_gravity(raw, nodes, beta=args.beta)
    sparse, n_threshold, n_bonus = sparsify(enriched, args.ratio, args.kmin)
    sparse, nodes_sparse = recompute_sparse_metrics(sparse, nodes)

    d0 = degree_series(raw, node_ids)
    d1 = degree_series(sparse, node_ids)
    comps = connected_components(sparse, node_ids)
    pair = city_pair_table(sparse, nodes)
    intercity_edges = int(pair.loc[pair['is_intercity'], 'edge_count'].sum())

    # 自洽硬校验：不满足则不写 active 文件。
    pair_key = sparse.apply(
        lambda r: '|'.join(sorted([str(r['Source_id']), str(r['Target_id'])])), axis=1)
    errors = []
    if pair_key.duplicated().any():
        errors.append('存在重复无向边')
    if (sparse['Source_id'] == sparse['Target_id']).any():
        errors.append('存在自环')
    if d1.min() < args.kmin:
        errors.append(f'最小度 {d1.min()} < kmin {args.kmin}')
    if len(comps) != 1:
        errors.append(f'网络不连通，连通分量 {len(comps)}')
    if set(sparse['Source_id']) | set(sparse['Target_id']) != set(node_ids):
        errors.append('稀疏边表未覆盖全部节点')
    if errors:
        raise RuntimeError('；'.join(errors))

    sparse.to_csv(out_path, index=False, encoding='utf-8-sig')
    nodes_sparse.to_csv(node_sparse_path, index=False, encoding='utf-8-sig')
    if args.activate:
        sparse.to_csv(ACTIVE_EDGE, index=False, encoding='utf-8-sig')
        nodes_sparse.to_csv(NODE_FILE, index=False, encoding='utf-8-sig')

    summary = pd.DataFrame([
        {'item': '方法', 'value': '外生节点质量-广义阻抗引力阈值'},
        {'item': '引力公式', 'value': 'sqrt(m_i*m_j)/(1+Generalized_impedance)^beta'},
        {'item': '稀疏加权度公式', 'value': 'sum[1/(1+Generalized_impedance)]'},
        {'item': 'beta', 'value': args.beta},
        {'item': '阈值目标比例', 'value': args.ratio},
        {'item': '候选边数', 'value': len(raw)},
        {'item': '阈值保留边数', 'value': n_threshold},
        {'item': '节点保底补充边数', 'value': n_bonus},
        {'item': '最终边数', 'value': len(sparse)},
        {'item': '实际保留比例', 'value': round(len(sparse) / len(raw), 4)},
        {'item': '原始平均度', 'value': round(d0.mean(), 2)},
        {'item': '稀疏平均度', 'value': round(d1.mean(), 2)},
        {'item': '稀疏最小度', 'value': int(d1.min())},
        {'item': '孤立节点数', 'value': int((d1 == 0).sum())},
        {'item': '连通分量数', 'value': len(comps)},
        {'item': '最大连通分量占比', 'value': round(comps[0] / len(node_ids), 4)},
        {'item': '城际边数', 'value': intercity_edges},
        {'item': '距离中位数_km', 'value': round(sparse['Distance_km'].median(), 3)},
        {'item': '距离P90_km', 'value': round(sparse['Distance_km'].quantile(0.90), 3)},
        {'item': '每节点保底边数', 'value': args.kmin},
        {'item': '激活为下游输入', 'value': bool(args.activate)},
    ])
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

    degree = masses.copy()
    degree['degree_before'] = degree['node_id'].map(d0).fillna(0).astype(int)
    degree['degree_after'] = degree['node_id'].map(d1).fillna(0).astype(int)
    degree['retained_ratio'] = (degree['degree_after'] / degree['degree_before']).round(4)
    degree.sort_values('degree_after', ascending=False).to_csv(
        degree_path, index=False, encoding='utf-8-sig')
    pair.to_csv(city_path, index=False, encoding='utf-8-sig')

    print(summary.to_string(index=False))
    print(f'\n稀疏候选: {out_path}')
    print(f'摘要: {summary_path}')
    print(f'节点度: {degree_path}')
    print(f'稀疏节点表: {node_sparse_path}')
    print(f'城市对: {city_path}')
    if args.activate:
        print(f'已激活为下游输入: {ACTIVE_EDGE}')
