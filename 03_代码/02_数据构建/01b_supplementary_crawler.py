import requests
import pandas as pd
import time
import os


def get_supplementary_pois(cities, keywords, api_key):
    all_data = []
    for city in cities:
        print(f"正在补全城市: {city} ...")
        for kw in keywords:
            url = 'https://restapi.amap.com/v3/place/text'
            params = {
                'key': api_key,
                'keywords': kw,
                'city': city,
                'citylimit': 'true',
                'offset': 20,
                'page': 1,
                'output': 'json'
            }
            try:
                res = requests.get(url, params=params).json()
                if res.get('status') == '1':
                    for poi in res.get('pois', []):
                        lng, lat = poi.get('location', '').split(',')
                        all_data.append({
                            '名称': poi.get('name'),
                            '省份': poi.get('pname'),
                            '城市': poi.get('cityname'),
                            '区县': poi.get('adname'),
                            '详细地址': poi.get('address'),
                            '经度(lng)': lng,
                            '纬度(lat)': lat,
                            '类型': poi.get('type')
                        })
            except Exception as e:
                print(f"抓取 {city}-{kw} 失败: {e}")
            time.sleep(0.5)
    return all_data


if __name__ == "__main__":
    MY_API_KEY = os.environ.get('AMAP_API_KEY')
    if not MY_API_KEY:
        raise RuntimeError('缺少环境变量 AMAP_API_KEY')

    # 路径配置
    script_path = os.path.abspath(__file__)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(script_path)), 'data')

    # 1+8 圈层中最关键的三个外围城市
    target_cities = ['鄂州', '孝感', '黄冈']
    keywords = ['物流园', '机场货运', '分拨中心']

    # 1. 抓取新数据
    new_pois = get_supplementary_pois(target_cities, keywords, MY_API_KEY)
    df_new = pd.DataFrame(new_pois)

    # 2. 读取旧数据
    old_file = os.path.join(data_dir, '武汉物流枢纽节点数据_初稿.csv')
    df_old = pd.read_csv(old_file)

    # 3. 合并并彻底去重
    df_final = pd.concat([df_old, df_new], ignore_index=True)
    df_final.drop_duplicates(subset=['名称', '经度(lng)', '纬度(lat)'], inplace=True)

    # 4. 覆盖保存
    final_file = os.path.join(data_dir, '武汉都市圈物流节点_完善版.csv')
    df_final.to_csv(final_file, index=False, encoding='utf-8-sig')

    print(f"\n节点集合已生成！")
