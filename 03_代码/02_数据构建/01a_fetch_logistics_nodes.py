import requests
import pandas as pd
import time
import os


def get_gaode_poi(keyword, city, api_key):
    """
    调用高德API抓取指定城市和关键词的POI数据
    """
    print(f"开始抓取 {city} 的 【{keyword}】 数据...")
    poi_data = []
    page = 1

    while True:
        url = 'https://restapi.amap.com/v3/place/text'
        params = {
            'key': api_key,
            'keywords': keyword,
            'city': city,
            'citylimit': 'true',
            'offset': 20,
            'page': page,
            'output': 'json'
        }

        try:
            response = requests.get(url, params=params)
            result = response.json()
        except Exception as e:
            print(f"第 {page} 页请求失败: {e}")
            break

        if result.get('status') == '1' and result.get('count') != '0':
            pois = result.get('pois', [])
            if not pois: break

            for poi in pois:
                location = poi.get('location', '')
                lng, lat = location.split(',') if ',' in location else (None, None)

                poi_data.append({
                    '名称': poi.get('name'),
                    '省份': poi.get('pname'),
                    '城市': poi.get('cityname'),
                    '区县': poi.get('adname'),
                    '详细地址': poi.get('address'),
                    '经度(lng)': lng,
                    '纬度(lat)': lat,
                    '类型': poi.get('type')
                })

            print(f"成功抓取第 {page} 页...")
            page += 1
            time.sleep(0.5)
        else:
            break

    return poi_data


if __name__ == "__main__":
    MY_API_KEY = os.environ.get('AMAP_API_KEY')
    if not MY_API_KEY:
        raise RuntimeError('缺少环境变量 AMAP_API_KEY')
    target_city = '武汉'
    keywords_list = ['物流园', '快递分拨中心', '货运站', '综合保税区']
    script_path = os.path.abspath(__file__)
    code_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(code_dir)
    data_dir = os.path.join(project_root, 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"已在根目录下创建数据文件夹: {data_dir}")

    all_results = []
    for kw in keywords_list:
        data = get_gaode_poi(kw, target_city, MY_API_KEY)
        all_results.extend(data)

    df = pd.DataFrame(all_results)
    df.drop_duplicates(subset=['名称', '经度(lng)', '纬度(lat)'], inplace=True)

    csv_file_path = os.path.join(data_dir, '武汉物流枢纽节点数据_初稿.csv')
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 40)
    print(f"共获取 {len(df)} 个节点")
    print(f"文件位置: {csv_file_path}")
    print("=" * 40)
