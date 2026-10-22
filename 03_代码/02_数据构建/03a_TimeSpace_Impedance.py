import requests
import pandas as pd
import time
import os


def get_location(address, api_key):
    """调用高德地理编码API，获取几何中心坐标"""
    url = 'https://restapi.amap.com/v3/geocode/geo'
    params = {'address': address, 'key': api_key, 'output': 'json'}
    try:
        res = requests.get(url, params=params).json()
        if res.get('status') == '1' and res.get('count') != '0':
            return res['geocodes'][0]['location']
    except Exception as e:
        print(f"坐标获取失败: {e}")
    return None


def get_time_space_metrics(origin, dest, api_key):
    """【核心修改】同时提取行驶里程与行驶时间"""
    url = 'https://restapi.amap.com/v3/distance'
    params = {'origins': origin, 'destination': dest, 'type': '1', 'key': api_key, 'output': 'json'}
    try:
        res = requests.get(url, params=params).json()
        if res.get('status') == '1' and res.get('results'):
            result = res['results'][0]
            dist_km = round(int(result['distance']) / 1000, 2)
            dur_mins = round(int(result['duration']) / 60, 2)
            return dist_km, dur_mins
    except Exception as e:
        print(f"路径计算失败: {e}")
    return None, None


if __name__ == "__main__":
    MY_API_KEY = os.environ.get('AMAP_API_KEY')
    if not MY_API_KEY:
        raise RuntimeError('缺少环境变量 AMAP_API_KEY')

    print("正在锁定武汉核心区坐标...")
    wuhan_loc = get_location('武汉市江汉区', MY_API_KEY)

    # 8 个卫星城与官方 GDP (2024年)
    cities_data = [
        {'城市': '鄂州市', 'GDP 亿元': 1341.30}, {'城市': '孝感市', 'GDP 亿元': 3258.54},
        {'城市': '黄冈市', 'GDP 亿元': 3216.65}, {'城市': '黄石市', 'GDP 亿元': 2305.81},
        {'城市': '咸宁市', 'GDP 亿元': 1944.57}, {'城市': '仙桃市', 'GDP 亿元': 1125.13},
        {'城市': '天门市', 'GDP 亿元': 785.40}, {'城市': '潜江市', 'GDP 亿元': 951.97}
    ]

    print("开始向高德服务器请求【时空双阻抗】数据...")
    for item in cities_data:
        dest_loc = get_location(item['城市'], MY_API_KEY)
        time.sleep(0.2)
        if dest_loc:
            dist_km, dur_mins = get_time_space_metrics(wuhan_loc, dest_loc, MY_API_KEY)

            if dist_km and dur_mins:
                item['物理距离_km'] = dist_km
                item['行驶时间_mins'] = dur_mins

                print(f"   -> {item['城市']}: {dist_km}km | {dur_mins}mins")
            time.sleep(0.2)

    # 论文公式(2)：Z=t'+0.5d'。使用 1/N 下界避免 min-max 后的零阻抗奇点。
    df_raw = pd.DataFrame(cities_data)
    for src, out in [('行驶时间_mins', '行驶时间_标准化'), ('物理距离_km', '物理距离_标准化')]:
        lo, hi = df_raw[src].min(), df_raw[src].max()
        df_raw[out] = 1.0 if hi == lo else (df_raw[src] - lo) / (hi - lo)
    z_floor = 1.0 / max(len(df_raw), 1)
    df_raw['综合时空阻抗'] = (
        df_raw['行驶时间_标准化'] + 0.5 * df_raw['物理距离_标准化']
    ).clip(lower=z_floor).round(6)

    # 保存 03a 数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', '03a_raw_impedance.csv')
    df_raw.to_csv(data_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print(f"时空阻抗数据已保存至: {data_path}")
