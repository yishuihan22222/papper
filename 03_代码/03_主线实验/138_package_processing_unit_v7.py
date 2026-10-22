"""Archive role evidence and package audited inputs, not empirical capacities."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import json
import shutil
import urllib.request
import datetime
import pandas as pd
import numpy as np

audit=importlib.import_module('136_audit_processing_units_v7')
OUT=audit.OUT


def fetch(record):
    try:
        request=urllib.request.Request(record['url'],headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(request,timeout=18) as response:
            payload=response.read();record['resolved_url']=response.url
        target=OUT/'evidence'/(record['id']+'.html')
        target.write_bytes(payload)
        record.update(downloaded=True,local_path=str(target),sha256=audit.sha(target),bytes=len(payload))
    except Exception as exc:
        record.update(downloaded=False,error=type(exc).__name__+': '+str(exc),retrieval_fallback='Public web tool was read in this session; no local original claimed')
    return record


def main():
    d=audit.read_json(OUT/'00_design.json');done=audit.read_json(OUT/'completion.json')
    assert done['completed'] and done['passed']==done['checks']
    records=[
        dict(id='HF2024',url='https://www.wuhan.gov.cn/sy/whyw/202411/t20241101_2478481.shtml',title='平急两用智慧物流园开工',publisher='武汉市人民政府',published='2024-11-01',supports='2024-10-30开工，不是当年全年运行的两个已知作业单元',scope='同名项目对应；不构成两个POI精确边界测绘'),
        dict(id='HF2025',url='https://www.huangpi.gov.cn/ywdt/jxdt/202507/t20250723_2623830.html',title='航发智慧物流园项目迎来新进展',publisher='黄陂区人民政府',published='2025-07-23',supports='后年仍施工、预计2026验收；与2024开工证据分开',scope='建设进程佐证，不将后年能力回填2024'),
        dict(id='POST2023',url='https://www.chinapost.com.cn/cn/report/2308/3704-1.htm',title='2023年武汉盛辉武汉黄金口十堰邮件处理中心内部处理业务外包',publisher='中国邮政集团',published='2023-08',supports='黄金口具有邮件进出口处理功能',scope='不证明原POI精确坐标、全部2024运营或实测K'),
        dict(id='ZTO2024',url='https://www.dxh.gov.cn/ZSYZ/XMTJ_16715/202406/t20240617_2416528.shtml',title='中通武汉转运中心升级扩建',publisher='东西湖区人民政府',published='2024-06-06',supports='武汉转运中心分拣装车功能',scope='品牌区域身份支持，微观POI边界未认证'),
        dict(id='JD2023',url='https://www.wuhan.gov.cn/ztzl/23zt/xczjc/202301/t20230119_2136808.shtml',title='京东武汉亚一分拣现场报道',publisher='武汉市人民政府',published='2023-01-19',supports='武汉亚一2期的分拣作业',scope='不能证明亚一新港两个POI同一处理线，因此隔离边界不清点'),
        dict(id='RECYCLE2025',url='https://gxs.wuhan.gov.cn/zzzq/zz_zszy/zz_zszy_gzdt/202506/t20250627_2602136.shtml',title='江夏再生资源分拣中心调研',publisher='武汉市供销合作总社',published='2025-06-27',supports='武再生盛庄分拣对象为再生资源',scope='2025角色证据，不能确认2024同类作业；用于相容性隔离'),
        dict(id='EDU_CURRENT',url='https://ezu.91wllm.cn/',title='鄂州职业大学就业信息网物流实习实训中心入口',publisher='鄂州职业大学',published='current',supports='校内物流实训功能线索',scope='原智慧物流POI精确对应和对外业务不明，隔离而非宣称纯教学'),
        dict(id='VAILOG_CURRENT',url='https://www.westbergpm.com/page62',title='项目风采武汉葛店物流园',publisher='威狮堡物业',published='undated_current',supports='武汉葛店物流园项目及地址',scope='不证明2号库的2024经营或容量，层级关系另据原POI名称')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(fetch,records))
    for r in records:r['accessed_date']='2026-10-21'
    prior=audit.SOURCES_DIR/'重点设施核验与校准_20261014/source_registry.json'
    prior_records=json.loads(prior.read_text(encoding='utf-8'))
    for r in prior_records:
        path=Path(r['local_path'])
        if path.exists():
            assert audit.sha(path)==r['sha256']
    audit.dump(dict(new_sources=records,prior_registry=str(prior),prior_registry_sha256=audit.sha(prior),prior_sources_rechecked=len(prior_records),source_scope='role evidence, not throughput-to-capacity calibration'),OUT/'17_source_registry.json')
    table=pd.read_csv(OUT/'01_all_246_node_decisions.csv').fillna('')
    model=OUT/'model_inputs';model.mkdir(exist_ok=True)
    n=pd.read_csv(OUT/'frozen_inputs/baseline_nodes.csv',float_precision='round_trip')
    n=n[n.node_id.isin(d['keep_ids'])].reset_index(drop=True)
    lookup=table.set_index('node_id')
    n['legacy_family']=n.family
    n['family']=lookup.loc[n.node_id,'reviewed_family'].to_numpy()
    n['identity_status']='processing_site_candidate_not_independently_certified'
    n['capacity_status']='legacy_proxy_formula_precision_not_empirical'
    audit.save(n,model/'nodes.csv')
    edges=pd.read_csv(OUT/'frozen_inputs/processing_candidate_edges.csv',float_precision='round_trip')
    audit.save(edges,model/'edges.csv')
    full=pd.read_csv(OUT/'frozen_inputs/original_nodes.csv')
    keep=full.node_id.isin(d['keep_ids']).to_numpy()
    payload=audit.read_json(OUT/'frozen_inputs/demands.json')
    audit.dump({key:dict(parts=np.asarray(v['parts'])[:,keep].tolist(),groups=v['groups'],node_ids=n.node_id.tolist(),not_full_city_demand=True) for key,v in payload.items()},model/'retained_source_demands.json')
    (model/'使用说明.md').write_text('''# 审查后模型输入

nodes.csv和edges.csv实际为122节点、1624条原边的诱导子图，family使用审查后的功能标签。基准K采用既有公式精度，L/K及边属性未重新标定。对应实验臂为candidate_corrected_family，而不是原246节点入口。

retained_source_demands.json保留原节点顺序下的122列需求成分；剔除来源任务不转嫁给其余节点，不代表都市圈全部任务。node_ids顺序必须与nodes.csv一致。没有重做城市需求拟合或为保持收益补边。

复算入口为Code/137_run_processing_unit_rectification_v7.py，它从冻结输入运行四组对照；原Code/data的默认入口保持历史复现用途，未静默改指向新样本。候选身份不等于逐点独立经营或2024运营认证，不能称为真实企业OD验证。
''',encoding='utf-8')
    # Guard correspondence and ensure this package is operationally consistent.
    assert n.node_id.tolist()==full.loc[keep,'node_id'].tolist()
    assert len(n)==122 and len(edges)==1624
    assert set(edges.Source_id)|set(edges.Target_id)<=set(n.node_id)
    assert table.node_id.nunique()==len(table)==246 and table.reason.notna().all()
    summaries=pd.read_csv(OUT/'12_policy_summary.csv')
    primary=summaries[summaries.scope.eq('candidate_corrected_family')]
    text=f'''# 节点处理单元整改与论文结论更新

本轮完成全部246条原始POI的角色处置、179组近邻或同名关系排查、3组园区与内部点的单一层级代表处理，并将结果接入新模型输入及6912条复算。这里的“完成”指审查、隔离与复算工作闭合，不代表246处或保留122处均取得企业级独立运营认证。

## 一 处置结果

- 处理场地候选122个；对应原图诱导子图1624条边。保留的仍是候选，不升级为实测处理网络。
- 明确角色不适合作为货物处理能力单元34条，包括道路、出口、客运电车站、宿舍、商业零售、办公、司机服务和监管服务点。它们可有物流支撑作用，不在当前层计独立处理能力。
- 证据未闭合87条隔离：公司与场地39条、边界19条、厂房或业务13条、运营日期10条、相容性6条。隔离不等于证明不存在或停业。
- 层级参考3条：维龙园区保留2号库、英山园区保留圆通分拣点、广浔园区保留园区场地候选而不另给德邦经营分部能力。没有相加代理K，也没有将近邻当成同一物理单元。严格意义的“已证实同一处理单元合并”为0。

## 二 新证据的作用

武汉市政府2024-11-01报道航发智慧物流园在10月30日开工，黄陂区2025报道继续记录建设。两条相邻同名POI从2024运营主样本隔离。没有把未来建设能力倒填为2024运行能力。

黄金口邮件处理中心的中国邮政外包公告、中通转运中心的东西湖区报道，支持其处理功能。亚一分期、综保区园区与投资公司、查验场与办公入口仍有微观边界问题：查验场保留实体功能候选但不改原坐标，亚一等重叠不清点隔离，不擅自断言别名相同。花湖货站保留用户先前“待核保留”的决定，南北身份不强配。

## 三 实际复算

固定三套容量、16组原冲击的共同投影、四档需求、三阈值、三政策，四组网络规格共6912条轨迹、2304个政策条件。914项数值与完整性检查、8次独立冷重启均通过。原246图的共同任务、122图沿用标签、122图固定原度数、122图修订功能标签分别保留，不能按结果择优选网络或风险分母。

修订功能标签的122点版本中，每套容量192条件，C−B总体平均差为{primary.C_B_mean_pp.min():.2f}—{primary.C_B_mean_pp.max():.2f}个百分点，前三档为{primary.first_three_mean_pp.min():.2f}—{primary.first_three_mean_pp.max():.2f}个百分点；总体差中位数均为0。576个条件合计提高{primary.C_B_better.sum()}、持平{primary.C_B_equal.sum()}、降低{primary.C_B_worse.sum()}。排序保持仍由模型假设支持，不构成新算法贡献。

共同任务原246图的C−B总体平均差仅0.41—0.42个百分点；它保留246图资源但仅承担122点任务，不能当作原完整城市任务结果。修订122图中C相对该共同任务对照的绝对服务变化为−38.35至+12.34个百分点。固定原度数分母的另一规格为−14.36至+88.00个百分点。大幅差异同时涉及承接资源、可达、暴露与归一化，不能写成净化带来的真实性提升量或纯拓扑因果效应。

本轮筛选把16组初始损伤从每组8点投影为2—6点，两边统一用同一投影，不补选损伤节点，也不与旧8点网格混成直接收益比较。剔除来源任务不再分摊给其余设施；这些比较只回答共同保留任务上的模型敏感性。

## 四 交付与尚未取得的事实

01_all_246_node_decisions.csv为全部逐点处置表；05_all_nearby_or_homonym_pairs.csv为关系台账；model_inputs为实际修订输入；10至16号文件为逐条件结果与核验。

已经完成角色筛选、层级去重处置、未决点隔离、新输入接入与必要复算。尚未取得的是保留122处的逐点独立经营边界、2024连续运营、同层任务接替权限以及同口径实测能力。公开检索没有提供这些事实时，不能以“逐点审查完成”代替“逐点事实认证完成”。论文应称为基于POI的场地候选网络与条件性模型研究，不能称为经企业台账验证的武汉实际调度系统。
'''
    (OUT/'节点整改结果与论文更新.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(archived=sum(r['downloaded'] for r in records),sources=len(records),model_nodes=len(n),model_edges=len(edges),inputs_written=True,real_K_replacements=0),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
