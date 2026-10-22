import os
import pandas as pd

if __name__ == "__main__":
    print("阶段 03b: 基于综合需求与时空阻抗的升级引力测算")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '..', 'data', '04a_城市综合指标_含估算.csv')
    output_path = os.path.join(script_dir, '..', 'data', '03b_gravity_results.csv')

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"找不到数据文件: {input_path}。请先运行 04_generate_quantified_datasets.py")
        raise SystemExit(1)

    required_cols = ['城市', 'GDP 亿元', '综合时空阻抗', '枢纽加成alpha', '需求强度指数', '产业匹配度', '综合引力指数_升级版']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"缺少必要字段: {missing}")
        raise SystemExit(1)

    df = df.copy()
    df['综合引力指数'] = df['综合引力指数_升级版'].round(4)
    df = df.sort_values(by='综合引力指数', ascending=False).reset_index(drop=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("升级引力计算完成！结果如下：\n")
    print(
        df[
            ['城市', 'GDP 亿元', '综合时空阻抗', '需求强度指数', '产业匹配度', '枢纽加成alpha', '综合引力指数']
        ].to_string(index=False)
    )
    print("\n" + "=" * 60)
