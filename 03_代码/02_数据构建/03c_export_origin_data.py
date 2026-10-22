import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
input_file = os.path.join(project_root, 'data', '03b_gravity_results.csv')
output_file = os.path.join(project_root, 'data', '03c_Origin_Plot_Data.csv')


def prepare_origin_data():
    print(f"正在读取计算结果:\n   {input_file}")
    if not os.path.exists(input_file):
        print(f"\n报错：找不到文件！")
        return

    df = pd.read_csv(input_file, encoding='utf-8')

    if '综合引力指数' not in df.columns or '城市' not in df.columns:
        print("\n报错：数据里找不到 '综合引力指数' 或 '城市' 这两列")
        return

    df_sorted = df.sort_values(by='综合引力指数', ascending=False).reset_index(drop=True)
    df_sorted.insert(0, '排名', range(1, len(df_sorted) + 1))
    df_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    prepare_origin_data()