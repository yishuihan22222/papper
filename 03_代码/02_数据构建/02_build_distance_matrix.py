import requests
import pandas as pd
import time
import os


def get_batch_distance(origin_lng_lat, dest_list, api_key):
    """
    调用高德 距离测量 API (v3/distance)
    由于高德规定 destination 只能有 1 个，origins 可以有 100 个。
    为了实现批量计算，我们将起点和终点倒置（计算 B 到 A 的距离，作为无向图的权重）。
    """
    url = 'https://restapi.amap.com/v3/distance'

    dest_str = '|'.join([f"{d['lng']},{d['lat']}" for d in dest_list])

    params = {
        'key': api_key,
        'origins': dest_str,
        'destination': origin_lng_lat,
        'type': '1',  # 1: 驾车路线
        'output': 'json'
    }

    try:
        response = requests.get(url, params=params)
        result = response.json()
        if result.get('status') == '1':
            return result.get('results', [])
        else:
            print(f"API 请求异常: {result.get('info')}")
            return []
    except Exception as e:
        print(f"网络请求崩溃: {e}")
        return []


if __name__ == "__main__":
    print("=" * 60)
    print("阶段 02: 真实路网距离抓取与空间自环清洗")
    print("=" * 60)

    # 1. 基础配置
    MY_API_KEY = os.environ.get('AMAP_API_KEY')
    if not MY_API_KEY:
        raise RuntimeError('缺少环境变量 AMAP_API_KEY')

    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_path))
    data_dir = os.path.join(project_root, 'data')

    input_file = os.path.join(data_dir, '武汉都市圈物流节点_语义清洗版.csv')
    if not os.path.exists(input_file):
        input_file = os.path.join(data_dir, '武汉物流枢纽节点数据_初稿.csv')

    try:
        df_nodes = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ 找不到输入文件: {input_file}")
        exit()

    # 清除没有坐标的空行
    df_nodes = df_nodes.dropna(subset=['经度(lng)', '纬度(lat)'])

    # 预处理：先按【坐标】去除真正重复采集的节点（同名但不同位置的设施必须保留）
    # 注意：早期版本按「名称」去重，会把同名但位置不同的真实设施误删，
    #      导致下游边表少算（246 节点应 30135 条边，此前仅 29161 条）。
    initial_node_count = len(df_nodes)
    df_nodes = df_nodes.copy()
    df_nodes['_coord_key'] = (df_nodes['经度(lng)'].round(5).astype(str) + '|'
                              + df_nodes['纬度(lat)'].round(5).astype(str))
    df_nodes = df_nodes.drop_duplicates(subset=['_coord_key'], keep='first').drop(columns=['_coord_key'])
    if len(df_nodes) < initial_node_count:
        print(f"[坐标清洗] 剔除了 {initial_node_count - len(df_nodes)} 个坐标完全重复的节点。")
    # 为每个节点生成唯一 node_id（名称@@城市@@区县，同名再加序号），贯穿全链路
    df_nodes['node_id'] = (df_nodes['名称'].astype(str).str.strip() + '@@'
                           + df_nodes['城市'].astype(str).str.strip() + '@@'
                           + df_nodes['区县'].astype(str).str.strip())
    _dm = df_nodes.duplicated('node_id', keep=False)
    if _dm.any():
        _seq = df_nodes.groupby('node_id').cumcount()
        _suffix = _seq.map(lambda k: '' if k == 0 else '#' + str(k + 1))
        df_nodes['node_id'] = df_nodes['node_id'] + _suffix.astype(str)

    nodes = df_nodes.to_dict('records')
    total_nodes = len(nodes)
    print(f"成功加载 {total_nodes} 个独立有效节点，准备调用高德 API...")

    # 2. 批量请求逻辑
    edge_list = []

    for i, origin in enumerate(nodes):
        origin_coords = f"{origin['经度(lng)']},{origin['纬度(lat)']}"
        origin_name = origin['名称']

        dest_candidates = nodes[i + 1:]

        if not dest_candidates:
            continue

        print(f"抓取进度: [{i + 1}/{total_nodes}] 正在获取 [{origin_name}] 辐射连线...")

        batch_size = 100
        for batch_start in range(0, len(dest_candidates), batch_size):
            batch_dests = dest_candidates[batch_start: batch_start + batch_size]
            formatted_dests = [{'lng': d['经度(lng)'], 'lat': d['纬度(lat)']} for d in batch_dests]

            results = get_batch_distance(origin_coords, formatted_dests, MY_API_KEY)

            for j, res in enumerate(results):
                dest_node = batch_dests[j]

                # 提取距离和时间，处理高德偶尔返回空串的边界情况
                if 'distance' in res and 'duration' in res and res['distance'] and res['duration']:
                    try:
                        dist_m = float(res['distance'])
                        dur_s = float(res['duration'])

                        edge_list.append({
                            'Source': origin_name,
                            'Target': dest_node['名称'],
                            'Source_id': origin['node_id'],
                            'Target_id': dest_node['node_id'],
                            'Distance_meters': dist_m,
                            'Duration_seconds': dur_s
                        })
                    except ValueError:
                        continue  # 转换数字失败，跳过这条脏数据

            time.sleep(0.3)  # 遵守高德 API 频控要求，防止被封 IP

    # 3. 核心清洗与保存网络边数据
    print("\n" + "-" * 60)
    print("开始执行拓扑结构清洗...")

    df_edges = pd.DataFrame(edge_list)
    raw_edges_count = len(df_edges)

    # 【论文核心动作】：非对称过滤，剔除自环 (Self-loops) 及 0km 连线
    # 物理意义：物流节点自己到自己的连线不仅无意义，还会导致图论算法崩溃
    # 改用 node_id 判自环，避免同名节点之间被误判为自环而删除
    df_clean = df_edges[
        (df_edges['Source_id'] != df_edges['Target_id']) &
        (df_edges['Distance_meters'] > 0)
        ].copy()

    removed_count = raw_edges_count - len(df_clean)
    print(f"   - 初始抓取连线: {raw_edges_count} 条")
    print(f"   - 剔除自环及 0km 脏数据: {removed_count} 条")
    print(f"   - 保留有效拓扑边: {len(df_clean)} 条")

    # 单位换算
    df_clean['Distance_km'] = (df_clean['Distance_meters'] / 1000).round(2)
    df_clean['Duration_mins'] = (df_clean['Duration_seconds'] / 60).round(2)

    # 提取最终字段（保留 node_id，供下游精确关联）
    df_final = df_clean[['Source', 'Target', 'Source_id', 'Target_id', 'Distance_km', 'Duration_mins']]

    # 导出文件
    output_file = os.path.join(data_dir, '武汉物流节点_连线权重集.csv')
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"真实路网抓取与清洗全部完成！")
    print(f"网络权重集已保存至: {output_file}")
    print("=" * 60)
