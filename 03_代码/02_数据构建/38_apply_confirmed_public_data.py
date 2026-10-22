"""Apply only verified entity/year/unit matches; preserve formula-derived inputs."""
from project_paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR, PAPER_DIR, TABLES_DIR, DOCS_DIR, SOURCES_DIR, BACKUP_DIR, LOGS_DIR
from pathlib import Path
from decimal import Decimal
import hashlib, json, shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data'
SOURCE=ROOT.parents[1]/'outputs'/'real_data_20261013'
SNAPSHOT=BACKUP_DIR/'真实数据接入前快照_20261013_161225'

def main():
    assert (SNAPSHOT/'data'/'公开补充数据_关键枢纽与风险指标.csv').exists(), 'Create the pre-change snapshot first'
    provenance=json.loads((SOURCE/'sources.json').read_text(encoding='utf-8'))
    for s in provenance:
        if 'raw_path' in s:
            assert hashlib.sha256((SOURCE/s['raw_path']).read_bytes()).hexdigest()==s['sha256'],s['source_id']
    SOURCES_DIR.mkdir(parents=True,exist_ok=True)
    DOCS_DIR.mkdir(parents=True,exist_ok=True)
    dest=SOURCES_DIR/'真实公开资料_20261013'
    shutil.copytree(SOURCE,dest,dirs_exist_ok=True)
    airport=pd.read_csv(dest/'民航局_湖北机场2023_2024.csv')
    actual=airport[airport['机场'].eq('鄂州/花湖')]
    assert len(actual)==1
    tons=Decimal(str(actual.iloc[0]['货邮2024吨']))
    value=float(tons/Decimal(10000))
    path=D/'公开补充数据_关键枢纽与风险指标.csv'
    frame=pd.read_csv(path)
    mask=frame['指标'].eq('花湖国际机场航空货邮吞吐量') & frame['统计时点'].astype(str).eq('2024') & frame['单位'].eq('万吨')
    assert mask.sum()==1, 'Need exactly one airport-level 2024 aviation cargo row'
    previous=pd.read_csv(SNAPSHOT/'data'/'公开补充数据_关键枢纽与风险指标.csv')
    old=float(previous.loc[mask,'数值'].iloc[0])
    source=next(s for s in provenance if s['url'].endswith('.xlsx'))
    frame.loc[mask,'数值']=str(value)
    frame.loc[mask,'来源标题']='2024年全国民用运输机场吞吐量排名（民航局原始Excel）'
    frame.loc[mask,'来源URL']=source['url']
    frame.loc[mask,'发布日期']='2025-03-14'
    frame.loc[mask,'备注']='民航局全机场航空货邮量865186.783吨，换算为86.5186783万吨；不是枢纽航空+卡车总量，不代入单个国际货站'
    port=frame['指标'].eq('阳逻港集装箱吞吐量') & frame['统计时点'].astype(str).eq('2024')
    frame.loc[port,'备注']='公开报道原值保留；另有264万标箱报道，口径尚未核实，不用于单设施参数替换'
    frame.to_csv(path,index=False,encoding='utf-8-sig')
    assert abs(float(pd.read_csv(path).loc[mask,'数值'].iloc[0])-value)<1e-12
    audit=[dict(字段='花湖国际机场航空货邮吞吐量',原值=old,新值=value,单位='万吨',统计期='2024',操作='公开统计背景字段精确替换',是否进入节点模拟参数='否',来源URL=source['url'],依据='同一全机场、同一2024自然年、吨转万吨；原值为报道约数')]
    for field,note in [
        ('proxy_daily_load / overload_threshold','当前单设施名称无法与公开运营量确认同设施同年匹配，保留论文定义的线性代理及倍率公式'),
        ('capacity_index / Edge_capacity_proxy','原有加权公式和几何平均公式保留；港口总量不赋给综保园区'),
        ('OD / flow redistribution / recovery curves','引力、阻抗、容量约束和恢复序列公式生成的模型量，保留原计算方法'),
        ('risk_exposure / Failure_probability_proxy','冰雪事件非设施失效概率样本；保留先验及概率公式'),
        ('花湖北货站首年吞吐量及规划量','跨2024/2025，吞吐量仅下界；规划量不是容量硬上限；未确认与现有POI为同设施'),
        ('班列83500标准集装箱','班列系统子业务量，不能替代吴家山站全部吨货量'),
        ('其他机场货邮量','天河无当前节点；其余省内机场不属于现有设施统计边界')]:
        audit.append(dict(字段=field,原值='',新值='',单位='',统计期='',操作='保留公式或仅存档',是否进入节点模拟参数='否',来源URL='',依据=note))
    pd.DataFrame(audit).to_csv(D/'38_real_data_application_audit.csv',index=False,encoding='utf-8-sig')
    config=dict(policy='confirmed_same_facility_same_year_same_unit_only',snapshot=str(SNAPSHOT),source_archive=str(dest),public_statistic_replacements=1,core_simulation_input_replacements=0,formula_derived_quantities_preserved=True,algorithm_changes=False,air_cargo_2024_tonnes=str(tons),air_cargo_2024_wan_tonnes=value,not_independent_validation=True)
    (D/'38_real_data_application.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    note=f'''# 真实资料接入说明（2026-10-13）

用户确认的原则：只替换已确认同设施、同年份、同单位的数据。

已把花湖国际机场2024年航空货邮吞吐量从报道约数 {old} 万吨，更新为民航局原表 {tons} 吨对应的 **{value} 万吨**。原始来源、原表行位置及SHA256存档在 `真实公开资料_20261013/`。该字段为全机场公开统计背景，并非单个国际货站负荷，原有模型没有消费这个背景字段。

本轮能严格替换的节点模拟输入为 **0项**。这不是用真实数据替换了全部模拟参数：北货站身份/年度不符，港口与园区范围不符，班列箱量与站点全部吨量不符，真实扰动记录也不是风险概率样本。按用户选择不作近似映射、不设新的校准假设。阳逻238.1万标箱原值保留并标记口径待核。

OD、代理日负荷、超载阈值、级联货量损失和恢复曲线等按既有公式生成，均可作为模型计算量保留；由真实统计推导的量也仍是推导值，不自动成为实测值。容量公式、风险先验、网络和算法参数均保持原配置。

完整重跑会刷新所有正式结果、补充实验和实验图。因为此次直接更新字段不进入核心模型，预计模型指标基本一致；任何差异要通过与快照的逐表比较确认，不能将重跑解释为外部真实运行验证。

备份：`{SNAPSHOT.as_posix()}`。执行记录见 `38_real_data_application_audit.csv` 和 `38_real_data_application.json`。后台重跑有单独状态、日志和完成后差异报告，失败时立即停止并标记失败，部分新输出不可冒充完整批次。
'''
    (DOCS_DIR/'真实数据接入说明_20261013.md').write_text(note,encoding='utf-8')
    print(json.dumps(config,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
