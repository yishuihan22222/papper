import os
import math
import random
import pandas as pd

RANDOM_SEED = 42
MAX_LABEL_ITERS = 25


def build_graph(nodes_df, edges_df):
    key = 'node_id' if ('node_id' in nodes_df.columns and 'Source_id' in edges_df.columns) else '名称'
    skey = 'Source_id' if key == 'node_id' else 'Source'
    tkey = 'Target_id' if key == 'node_id' else 'Target'
    names = nodes_df[key].dropna().astype(str).tolist()
    g = {n: {} for n in names}
    for _, r in edges_df.iterrows():
        s, t = str(r[skey]), str(r[tkey])
        if s not in g or t not in g or s == t:
            continue
        w = float(pd.to_numeric(r.get('Edge_capacity_proxy', 0.0), errors='coerce'))
        if w <= 0:
            cost = float(pd.to_numeric(r.get('Composite_transport_cost', 1.0), errors='coerce'))
            w = 1.0 / max(cost, 0.001)
        g[s][t] = max(g[s].get(t, 0.0), w)
        g[t][s] = max(g[t].get(s, 0.0), w)
    return g


def weighted_degree(graph):
    return {n: round(sum(graph[n].values()), 4) for n in graph}


def approximate_betweenness(graph):
    nodes = list(graph.keys())
    scores = {n: 0.0 for n in nodes}
    for s in nodes:
        stack, preds = [], {v: [] for v in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        sigma[s] = 1.0
        dist = dict.fromkeys(nodes, math.inf)
        dist[s] = 0.0
        queue = [s]
        while queue:
            v = min(queue, key=lambda x: dist[x])
            queue.remove(v)
            stack.append(v)
            for w, wt in graph[v].items():
                alt = dist[v] + 1.0 / max(wt, 1e-6)
                if alt + 1e-12 < dist[w]:
                    dist[w] = alt
                    sigma[w] = sigma[v]
                    preds[w] = [v]
                    if w not in queue:
                        queue.append(w)
                elif abs(alt - dist[w]) <= 1e-12:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = dict.fromkeys(nodes, 0.0)
        while stack:
            w = stack.pop()
            for v in preds[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                scores[w] += delta[w]
    if len(nodes) > 2:
        scale = 1.0 / ((len(nodes) - 1) * (len(nodes) - 2))
        scores = {k: round(v * scale, 6) for k, v in scores.items()}
    return scores


def label_propagation(graph):
    random.seed(RANDOM_SEED)
    labels = {n: n for n in graph}
    nodes = list(graph.keys())
    for _ in range(MAX_LABEL_ITERS):
        changed = False
        random.shuffle(nodes)
        for n in nodes:
            nbrs = graph[n]
            if not nbrs:
                continue
            agg = {}
            for v, w in nbrs.items():
                agg[labels[v]] = agg.get(labels[v], 0.0) + w
            best = sorted(agg.items(), key=lambda x: (-x[1], str(x[0])))[0][0]
            if best != labels[n]:
                labels[n] = best
                changed = True
        if not changed:
            break
    community_id = {}
    for idx, lab in enumerate(sorted(set(labels.values())), start=1):
        community_id[lab] = f'C{idx:02d}'
    return {n: community_id[labels[n]] for n in graph}


def core_periphery_role(row):
    deg = float(row['weighted_degree_score'])
    btw = float(row['betweenness_score'])
    cross = float(row['cross_community_ratio'])
    cap = float(row['capacity_index'])
    if deg >= row['deg_q75'] and (btw >= row['btw_q75'] or cap >= row['cap_q75']):
        return 'core'
    if cross >= row['cross_q75'] or btw >= row['btw_q75']:
        return 'bridge'
    if deg <= row['deg_q25'] and cap <= row['cap_q25']:
        return 'periphery'
    return 'semi-periphery'


def zone_label(row):
    district = str(row['区县'])
    family = str(row['family'])
    name = str(row['名称'])
    text = f'{district}|{family}|{name}'
    if any(k in text for k in ['东西湖', '临空港', '吴家山']):
        return '东西湖-临空港综合物流组团'
    if any(k in text for k in ['黄陂', '空港', '机场']):
        return '黄陂空港枢纽组团'
    if any(k in text for k in ['新洲', '阳逻', '港', '码头']):
        return '新洲港口物流组团'
    if any(k in text for k in ['蔡甸', '经开', '汉南']):
        return '蔡甸-经开制造物流走廊'
    if any(k in text for k in ['江夏', '大花岭', '纸坊']):
        return '江夏南向货运组团'
    if family in ['bonded', 'airport', 'rail', 'port']:
        return '门户枢纽功能组团'
    return '综合物流一般组团'


if __name__ == '__main__':
    print('阶段 04e: 初始图结构识别与功能分区分析')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    nodes = pd.read_csv(os.path.join(data_dir, '04b_武汉节点代理容量.csv')).copy()
    edges = pd.read_csv(os.path.join(data_dir, '04c_武汉边代理属性.csv')).copy()
    cross = pd.read_csv(os.path.join(data_dir, '04_cross_layer_edges.csv')).copy()

    graph = build_graph(nodes, edges)
    deg = weighted_degree(graph)
    btw = approximate_betweenness(graph)
    comm = label_propagation(graph)
    _key = 'node_id' if ('node_id' in nodes.columns and 'Source_id' in edges.columns) else '名称'
    cross_map = cross.groupby('lower_node_id')['coupling_weight'].mean().to_dict() if ('lower_node_id' in cross.columns and not cross.empty) else (cross.groupby('lower_node')['coupling_weight'].mean().to_dict() if not cross.empty else {})

    df = nodes.copy()
    df['weighted_degree_score'] = df[_key].map(deg).fillna(0.0)
    df['betweenness_score'] = df[_key].map(btw).fillna(0.0)
    df['community_id'] = df[_key].map(comm).fillna('C00')
    df['cross_layer_weight'] = df[_key].map(cross_map).fillna(0.0)

    comm_map = df.set_index(_key)['community_id'].to_dict()
    cross_ratio = []
    _skey = 'Source_id' if _key == 'node_id' else 'Source'
    _tkey = 'Target_id' if _key == 'node_id' else 'Target'
    for _, r in edges.iterrows():
        s, t = str(r[_skey]), str(r[_tkey])
        cross_ratio.append((s, 1 if comm_map.get(s) != comm_map.get(t) else 0))
        cross_ratio.append((t, 1 if comm_map.get(s) != comm_map.get(t) else 0))
    cr = pd.DataFrame(cross_ratio, columns=[_key, 'is_cross']) if cross_ratio else pd.DataFrame(columns=[_key, 'is_cross'])
    cr = cr.groupby(_key)['is_cross'].mean().to_dict() if not cr.empty else {}
    df['cross_community_ratio'] = df[_key].map(cr).fillna(0.0)

    df['deg_q25'] = df['weighted_degree_score'].quantile(0.25)
    df['deg_q75'] = df['weighted_degree_score'].quantile(0.75)
    df['btw_q75'] = df['betweenness_score'].quantile(0.75)
    df['cap_q25'] = df['capacity_index'].quantile(0.25)
    df['cap_q75'] = df['capacity_index'].quantile(0.75)
    df['cross_q75'] = df['cross_community_ratio'].quantile(0.75)
    df['structural_role'] = df.apply(core_periphery_role, axis=1)
    df['functional_zone'] = df.apply(zone_label, axis=1)

    out_cols = (['node_id'] if 'node_id' in df.columns else []) + ['名称', '城市', '区县', '类型', 'family', 'capacity_index', 'proxy_daily_load', 'risk_exposure', 'weighted_degree_score', 'betweenness_score', 'community_id', 'cross_community_ratio', 'cross_layer_weight', 'structural_role', 'functional_zone']
    node_out = df[out_cols].sort_values(['structural_role', 'weighted_degree_score', 'betweenness_score'], ascending=[True, False, False]).copy()

    comm_summary = df.groupby('community_id').agg(
        node_count=('名称', 'size'),
        avg_capacity=('capacity_index', 'mean'),
        avg_load=('proxy_daily_load', 'mean'),
        avg_cross_layer_weight=('cross_layer_weight', 'mean')
    ).reset_index().sort_values(['node_count', 'avg_capacity'], ascending=False)

    role_summary = df.groupby('structural_role').agg(
        node_count=('名称', 'size'),
        avg_capacity=('capacity_index', 'mean'),
        avg_load=('proxy_daily_load', 'mean'),
        avg_cross_community_ratio=('cross_community_ratio', 'mean')
    ).reset_index().sort_values('node_count', ascending=False)

    zone_summary = df.groupby('functional_zone').agg(
        node_count=('名称', 'size'),
        avg_capacity=('capacity_index', 'mean'),
        avg_risk=('risk_exposure', 'mean'),
        core_count=('structural_role', lambda s: int((s == 'core').sum())),
        bridge_count=('structural_role', lambda s: int((s == 'bridge').sum()))
    ).reset_index().sort_values(['core_count', 'bridge_count', 'node_count'], ascending=False)

    p1 = os.path.join(data_dir, '04e_node_structural_roles.csv')
    p2 = os.path.join(data_dir, '04e_community_summary.csv')
    p3 = os.path.join(data_dir, '04e_core_periphery_summary.csv')
    p4 = os.path.join(data_dir, '04e_functional_zone_summary.csv')

    node_out.to_csv(p1, index=False, encoding='utf-8-sig')
    comm_summary.to_csv(p2, index=False, encoding='utf-8-sig')
    role_summary.to_csv(p3, index=False, encoding='utf-8-sig')
    zone_summary.to_csv(p4, index=False, encoding='utf-8-sig')

    print(role_summary.to_string(index=False))
    print(f'\n节点结构角色结果已保存至: {p1}')
    print(f'社区摘要已保存至: {p2}')
    print(f'核心-边缘摘要已保存至: {p3}')
    print(f'功能分区摘要已保存至: {p4}')
