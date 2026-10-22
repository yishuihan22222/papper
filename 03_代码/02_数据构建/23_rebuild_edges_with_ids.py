# -*- coding: utf-8 -*-
"""
23_rebuild_edges_with_ids.py —— 重建带 node_id 的完整边表

【为什么需要】
    原 02_build_distance_matrix.py 第65行按「名称」去重：
        df_nodes.drop_duplicates(subset=['名称'], keep='first')
    把 4 组同名但位置不同的真实设施误删，导致：
        边表仅有 242 个节点 / 29,161 条边（242*241/2）
    而按语义清洗版 246 个节点，完整应为：
        246*245/2 = 30,135 条边
    缺口 974 条，全部与这 4 个被丢弃的节点有关。

【策略】不重跑整个 02（那会重新抓 3 万条边的 API，且不可复现）
    1) 旧边表 29,161 条 —— 完全复用（其节点都是 242 个「首次出现」节点）
    2) 仅对 4 个新增节点，抓取其与全部其余节点的距离（约 978 条）
    3) 合并 → 完整 30,135 条边，每行带 Source_id / Target_id

【同名节点归属】旧边表中同名节点已按名称合并（共享同一套 241 条边），
    无法追溯归属。本脚本将它们视为「首次出现的那个节点」保留原有边；
    其余同名节点作为新增节点，单独抓边。这是唯一自洽的处理方式。

输出: data/武汉物流节点_连线权重集_含ID.csv
缓存: data/_geocode_cache_补边.json
"""
from project_paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR, PAPER_DIR, TABLES_DIR, DOCS_DIR, SOURCES_DIR, BACKUP_DIR, LOGS_DIR
import os
import time
import json
import urllib.request
import urllib.parse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')

API_KEY = os.environ.get('AMAP_API_KEY')
CLEAN = os.path.join(DATA, '武汉都市圈物流节点_语义清洗版.csv')
OLD_EDGES = os.path.join(DATA, '武汉物流节点_连线权重集.csv')
OUT = os.path.join(DATA, '武汉物流节点_连线权重集_含ID.csv')
CACHE = str(LOGS_DIR / 'cache' / '_geocode_cache_补边.json')


def load_nodes():
    df = pd.read_csv(CLEAN)
    df = df.dropna(subset=['经度(lng)', '纬度(lat)']).copy()
    df['_ck'] = df['经度(lng)'].round(5).astype(str) + '|' + df['纬度(lat)'].round(5).astype(str)
    n0 = len(df)
    df = df.drop_duplicates(subset=['_ck'], keep='first').drop(columns=['_ck'])
    if len(df) < n0:
        print(f'[坐标去重] 剔除 {n0 - len(df)} 个坐标重复节点')
    df['node_id'] = (df['名称'].astype(str).str.strip() + '@@'
                     + df['城市'].astype(str).str.strip() + '@@'
                     + df['区县'].astype(str).str.strip())
    dm = df.duplicated('node_id', keep=False)
    if dm.any():
        seq = df.groupby('node_id').cumcount()
        suf = seq.map(lambda k: '' if k == 0 else '#' + str(k + 1))
        df['node_id'] = df['node_id'] + suf.astype(str)
    return df.reset_index(drop=True)


def fetch_batch(orig_lnglat, dests):
    if not dests:
        return []
    url = 'https://restapi.amap.com/v3/distance'
    qs = urllib.parse.urlencode({
        'key': API_KEY,
        'origins': '|'.join(f'{d["经度(lng)"]},{d["纬度(lat)"]}' for d in dests),
        'destination': orig_lnglat,
        'type': '1',
        'output': 'json',
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f'{url}?{qs}', timeout=20) as r:
                js = json.loads(r.read().decode())
            if js.get('status') == '1':
                return js.get('results', [])
            print(f'  API异常: {js.get("info")}')
        except Exception as e:
            print(f'  请求失败(重试{attempt+1}): {e}')
            time.sleep(2)
    return []


if __name__ == '__main__':
    if not API_KEY:
        raise RuntimeError('缺少环境变量 AMAP_API_KEY')
    print('=' * 66)
    print('重建带 node_id 的完整边表')
    print('=' * 66)

    nodes = load_nodes()
    N = len(nodes)
    print(f'节点数: {N}（唯一 node_id: {nodes["node_id"].nunique()}）')
    print(f'完整完全图应有边: {N*(N-1)//2}')

    old = pd.read_csv(OLD_EDGES)
    print(f'已有边表: {len(old)} 条')

    # 旧边表覆盖的名称（仅名称）
    covered_names = set(old['Source']) | set(old['Target'])
    print(f'旧边表涉及唯一名称: {len(covered_names)}')

    # 确定「首次出现」节点 = 旧边表已覆盖的名称，取每个名称的第一个 node_id
    # 其余同名节点 = 新增节点
    first_occurrence = {}
    new_nodes = []
    for _, r in nodes.iterrows():
        nm = r['名称']
        if nm in covered_names and nm not in first_occurrence:
            first_occurrence[nm] = r['node_id']
        else:
            new_nodes.append(r['node_id'])
    # 旧边表中出现的名称若在 nodes 中找不到（极少），忽略
    covered_ids = set(first_occurrence.values())
    print(f'沿用旧边的节点: {len(covered_ids)}  新增节点: {len(new_nodes)}')
    print(f'新增节点明细:')
    for nid in new_nodes:
        row = nodes[nodes['node_id'] == nid].iloc[0]
        print(f'   {nid}  ({row["经度(lng)"]},{row["纬度(lat)"]})')

    coords = nodes.set_index('node_id')[['经度(lng)', '纬度(lat)', '名称']].to_dict('index')
    all_ids = nodes['node_id'].tolist()

    # ---- 1) 旧边 -> 带 ID（名称映射到 first_occurrence）----
    print('\n[1/3] 旧边表转换为 ID 形式 ...')
    rows = []
    for _, r in old.iterrows():
        sid = first_occurrence.get(r['Source'])
        tid = first_occurrence.get(r['Target'])
        if sid is None or tid is None:
            continue
        rows.append({'Source_id': sid, 'Target_id': tid,
                     'Source': r['Source'], 'Target': r['Target'],
                     'Distance_km': r['Distance_km'], 'Duration_mins': r['Duration_mins']})
    edges = pd.DataFrame(rows)
    print(f'   转换 {len(edges)} 条')

    # ---- 2) 新增节点补边（与所有其他节点）----
    print('\n[2/3] 为新增节点抓取距离 ...')
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f'   载入缓存 {len(cache)} 条')

    need = []
    for nid in new_nodes:
        for other in all_ids:
            if other == nid:
                continue
            key = f'{nid}|{other}'
            need.append((nid, other, key))
    print(f'   需抓取 {len(need)} 条')

    added = 0
    for i, (a, b, key) in enumerate(need):
        if key in cache:
            continue
        ca, cb = coords[a], coords[b]
        res = fetch_batch(f'{cb["经度(lng)"]},{cb["纬度(lat)"]}', [ca])
        if res and res[0].get('distance') and res[0].get('duration'):
            cache[key] = {'d': res[0]['distance'], 't': res[0]['duration']}
            added += 1
        else:
            cache[key] = {'d': '', 't': ''}
        if (i + 1) % 50 == 0:
            print(f'   进度 {i+1}/{len(need)}，本次新增 {added}')
            with open(CACHE, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
        time.sleep(0.12)
    with open(CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f)
    print(f'   抓取完成，新增 {added} 条')

    # 组装新边
    new_rows = []
    for a, b, key in need:
        c = cache.get(key, {})
        d, t = c.get('d', ''), c.get('t', '')
        if d in ('', None) or t in ('', None):
            continue
        new_rows.append({
            'Source_id': a, 'Target_id': b,
            'Source': coords[a]['名称'], 'Target': coords[b]['名称'],
            'Distance_km': round(float(d) / 1000, 2),
            'Duration_mins': round(float(t) / 60, 2),
        })
    print(f'   新边 {len(new_rows)} 条')

    # ---- 3) 合并 ----
    print('\n[3/3] 合并 ...')
    all_edges = pd.concat([edges, pd.DataFrame(new_rows)], ignore_index=True)
    # 去重（无向，按排序后的 id 对）
    all_edges['_k'] = all_edges.apply(lambda r: '|'.join(sorted([r['Source_id'], r['Target_id']])), axis=1)
    all_edges = all_edges.drop_duplicates(subset=['_k']).drop(columns=['_k'])
    all_edges = all_edges[(all_edges['Source_id'] != all_edges['Target_id'])
                          & (all_edges['Distance_km'] > 0)]
    print(f'   最终边数: {len(all_edges)}（目标 {N*(N-1)//2}）')

    all_edges = all_edges[['Source', 'Target', 'Source_id', 'Target_id', 'Distance_km', 'Duration_mins']]
    all_edges.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'\n已保存: {OUT}')
