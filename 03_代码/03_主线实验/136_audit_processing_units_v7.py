"""Audit every legacy POI, freeze role decisions before revised simulations.

An accepted POI is a processing-site candidate, not a verified 2024 operating
unit. Administrative services, legal entities and parent POIs never inherit
warehouse capacity. Original inputs and completed experiments are read-only.
"""
from pathlib import Path
import importlib
import itertools
import json
import re
import hashlib
import numpy as np
import pandas as pd

from project_paths import DATA_DIR, RESULTS_DIR, SOURCES_DIR
from capacity_precision_v5 import formula_precision_nodes

core = importlib.import_module('83_dispatch_mainline_core')
OUT = RESULTS_DIR / 'processing_unit_rectification_v7_20261021'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save(value, path):
    pd.DataFrame(value).to_csv(path, index=False, encoding='utf-8-sig')


def names(text):
    return set(text.strip().splitlines())


NONPROCESSING = names('''
机场路
东湖综保区出口(G50沪渝高速东北向)
空轨综保区站(有轨电车站)
老货运公司宿舍
中百罗森(综保区移动终端产业园店)
极氪交付中心(武汉东湖综合保税区)
东湖保税交易中心
东湖综保区企业服务中心
东湖综合保税区申报大厅
中共武汉海关驻东湖综合保税区办事处
中共武汉经开综合保税区及港口物流园管理办公室工作委员会
武汉东湖综合保税区A塔楼
武汉东湖综合保税区B塔楼
武汉东湖综合保税区建设管理办公室
武汉新港空港综合保税区申报大厅
武汉经开综合保税区及港口物流园管理办公室
经开综合保税区卡口
阳逻保税区综合楼
中海物业电子厂房项目客户服务中心501室
中通快递武汉分拔中心财务部
新港永捷农产品物流园招商部
众驰货运司机招募中心
湖北零享货运司机服务中心
滴滴货运(武汉顺腾司服中心)
滴滴货运汉口金桥店司机服务中心
滴滴货运湖北九顺司机服务中心
货运司机服务中心
货运司机服务中心(翠鸟·盘龙城店)
通达车队货运司机服务中心
货运营业楼
江岸机务段南线运用车间
''')
PROJECTS = names('''
中冶交通红安商贸物流园项目经理部
中建六局物流园项目部
麻城市农产品综合物流园建设项目部
亚洲基础设施投资银行贷款项目湖北国际航空货运项目土方工程项目部
现代物流园二期3号4号库(应急物资储备库)项目
黄冈黑豹综合物流园(建设中)
蔡甸二期项目物流园区
航发智慧物流园
天津鼎宏物流(暂停营业)
''')
NONLOGISTICS = names('''
东湖高新综保区崇华芯通科技园1号楼
综保区移动终端产业园
综合保税区三号电子厂房
武汉信跃致科技有限公司
武汉创恒激光智能装备有限公司
武汉市拜瑞施生物科技有限公司
武汉康创生物科技有限公司
武汉智瑞捷电气技术有限公司
武汉瑞芯精密光通信设备股份有限公司
武汉米字能源科技有限公司
海微智造技术(武汉)有限公司
湖北兆元科技有限公司
瑞声科技有限公司
''')
BOUNDARY = names('''
东湖综保区第二卡口产业园
东湖综合保税区ice(综保区店)
武汉东湖综合保税区
武汉东湖综合保税区建设投资有限公司
武汉金宇综合保税发展有限公司
武汉临空港综合保税园区投资发展有限公司
保税物流园
武汉空港综合保税区
武汉经开综合保税区
武汉新港空港综合保税区阳逻港园区
京东物流园
京东亚一分拣中心
京东武汉新港分拣中心
物流园区
温口物流基地
武汉论道国际物流技术有限公司
武汉道格实业集团有限公司
湖北鑫畅通供应链管理有限公司
货喜物流分拨中心
德邦快递(孝感孝昌县广浔物流园经营分部)
''')
SPECIAL = names('''
循大师数字化绿色分拣中心
武再生盛庄-江夏分拣中心
食用菌分拣中心
鄂州职业大学智慧物流中心
武汉联盟商砼(货运站)运输车队
武汉军山造船厂联盟货运站
''')
EXTRA_SITES = names('''
中国邮政武汉市黄金口邮件处理中心
孝昌县邮政寄递物流共配中心
武钢运输部武钢车站
武汉慧欣集装箱货运堆场
百世快运东西湖基地
阳逻综保区冷库
花湖机场黄冈城市货站
君达货运码头
庆丰货运码头
葛化集团港口物流园仓储基地
铁路集装箱货运中心
武汉中远物流有限公司物流基地(临空港大道店)
东湖综保区进口肉类指定查验场
''')


def classify(row):
    name = row['名称']
    if name in NONPROCESSING:
        return 'exclude_role', 'POI所指为道路/客运/住宅/商业/办公/监管/司机或车辆服务等角色，不作为独立货物处理能力单元；不否认物流支撑作用'
    if name in PROJECTS:
        return 'quarantine_operation_date', '项目/施工/停业状态或启用时点未闭合；不把项目名称等同于2024已运营处理场地'
    if name in NONLOGISTICS:
        return 'quarantine_business', '原POI为厂房/制造科技企业或产业园，缺独立对外物流处理场地证据；不因位于综保区获得仓储能力'
    if name in BOUNDARY:
        return 'quarantine_boundary', '公司/整体园区/疑似别名或分期范围未唯一对应独立处理单元；主分析不重复赋能，保留原记录待证据闭合'
    if name in SPECIAL:
        return 'quarantine_compatibility', '再生资源/农产品/教学或专用工业货运可能有实体作业，但同层承接相同任务的业务相容性未闭合'
    if name in EXTRA_SITES or re.search(r'物流园|物流基地|分拨|分拣|转运|货运站|货运中心|仓储中心', name) or '货运火车站' in row['类型'] or '物流仓储场地' in row['类型'] or name == '湖北省鄂州花湖机场国际货站':
        return 'retain_site_candidate', '原POI名称、类别或地址明确指向仓库、分拨/集散、货站、堆场或货运铁路场地；仅候选资格，独立经营边界及2024持续运营未逐点证实'
    return 'quarantine_office_or_site', '货运/物流企业或营业网点名称能证明业务标签，不能单独证明该POI具有独立货物处理场地；隔离而非认定不存在'


def main():
    if OUT.exists():
        raise FileExistsError('Use the frozen outputs; new scientific design needs a new folder')
    OUT.mkdir(); (OUT/'frozen_inputs').mkdir(); (OUT/'shards').mkdir(); (OUT/'evidence').mkdir()
    n, e, demands, _ = core.load_study()
    poi = pd.read_csv(DATA_DIR/'武汉都市圈物流节点_语义清洗版.csv')
    poi['stem'] = poi['名称']+'@@'+poi['城市']+'@@'+poi['区县']
    poi['occurrence'] = poi.groupby('stem', sort=False).cumcount()+1
    poi['node_id'] = poi.stem + poi.occurrence.map(lambda k: '' if k == 1 else '#'+str(k))
    assert set(poi.node_id) == set(n.node_id) and len(poi) == len(n) == 246
    audit = n.merge(poi[['node_id','详细地址','经度(lng)','纬度(lat)']], on='node_id', validate='one_to_one')
    decisions = [classify(row) for _, row in audit.iterrows()]
    audit['decision'] = [d[0] for d in decisions]; audit['reason'] = [d[1] for d in decisions]
    audit['evidence_level'] = 'local_POI_only'
    audit['independent_unit_verified'] = False
    audit['operating_2024_verified'] = False
    audit['relation_group'] = ''; audit['representative_node_id'] = audit.node_id
    prior = json.loads((SOURCES_DIR/'宏观与设施补强_20261014/facility_review_records.json').read_text(encoding='utf-8'))
    by_id = {r['node_id']:r for r in prior}
    audit['prior_source_url'] = audit.node_id.map(lambda k:by_id.get(k,{}).get('核验来源',''))
    audit['prior_identity_note'] = audit.node_id.map(lambda k:by_id.get(k,{}).get('身份结论',''))
    # Explicit parent/tenant relations in source addresses; not proximity deduplication.
    # Selecting a child discards the parent's proxy resource rather than summing K.
    hierarchy = [
        ('H01','维龙物流园','维龙物流园2号库','鄂州市','华容区','同名园区与2号库，原坐标相距约23米；仅处理层级代表重叠，不宣称库区库存完全相同'),
        ('H02','鸿翔路广浔智慧物流园','德邦快递(孝感孝昌县广浔物流园经营分部)','孝感市','孝昌县','子点原地址明确为广浔物流园内；园区整体与内部处理网点不同时独立赋能'),
        ('H03','英山县综合物流园','英山圆通速递分拣中心','黄冈市','英山县','子点原地址明确英山综合物流园；保留分拣点，园区整体移为层级参考')]
    relations = []
    for gid,parent,child,city,district,note in hierarchy:
        pi = audit.index[(audit['名称']==parent)&(audit['城市']==city)&(audit['区县']==district)].item()
        ci = audit.index[(audit['名称']==child)&(audit['城市']==city)&(audit['区县']==district)].item()
        assert audit.loc[pi,'decision']=='retain_site_candidate'
        audit.loc[[pi,ci],'relation_group'] = gid
        if gid=='H02':
            audit.loc[ci,'decision']='reference_child';audit.loc[ci,'reason']='经营分部位于园区内，但独立处理场地不明；保留场地型园区候选，不另赋予经营分部容量'
            audit.loc[ci,'representative_node_id']=audit.loc[pi,'node_id']
            action='child_reference_only_parent_retained'
        else:
            assert audit.loc[ci,'decision']=='retain_site_candidate'
            audit.loc[pi,'decision']='reference_parent'; audit.loc[pi,'reason']=note
            audit.loc[pi,'representative_node_id']=audit.loc[ci,'node_id']
            action='parent_reference_only_child_retained'
        relations.append(dict(group_id=gid,parent_id=audit.loc[pi,'node_id'],child_id=audit.loc[ci,'node_id'],action=action,basis=note,capacity_sum=False,physical_duplicate_confirmed=False))
    special_evidence = {
        '航发智慧物流园':('HF2024','https://www.wuhan.gov.cn/sy/whyw/202411/t20241101_2478481.shtml','官方2024-11-01报道10月30日开工；两条同名相邻POI对应同一建设项目候选，不能算两个2024运行单元'),
        '中国邮政武汉市黄金口邮件处理中心':('POST2023','https://www.chinapost.com.cn/cn/report/2308/3704-1.htm','运营方2023内部处理外包公告支持邮件处理功能；不能证明逐日能力或全部2024运营状态'),
        '中通快递(武汉分拨中心)':('ZTO2024','https://www.dxh.gov.cn/ZSYZ/XMTJ_16715/202406/t20240617_2416528.shtml','2024官方报道东西湖走马岭武汉转运中心自动分拣及分拨装车；原POI精确场地对应仍需区分'),
        '京东亚一分拣中心':('JD2023','https://www.wuhan.gov.cn/ztzl/23zt/xczjc/202301/t20230119_2136808.shtml','官方报道武汉亚一2期分拣中心；不把两期或新港POI直接判为同一处理线'),
        '武再生盛庄-江夏分拣中心':('RECYCLE2025','https://gxs.wuhan.gov.cn/zzzq/zz_zszy/zz_zszy_gzdt/202506/t20250627_2602136.shtml','2025官方报道再生资源塑料分拣；支持业务区分，不回推2024同批任务相容'),
        '鄂州职业大学智慧物流中心':('EDU_CURRENT','https://ezu.91wllm.cn/','学校官网列有物流实习实训中心；与原POI精确对应及商业处理功能未闭合，不直接认定全部是纯教学'),
        '维龙物流园':('VAILOG_CURRENT','https://www.westbergpm.com/page62','物业方项目页列示武汉葛店物流园及电商大道8号；支持园区身份，不能证明2号库的2024处理量')}
    audit['new_evidence_id']=''; audit['new_source_url']=''; audit['new_evidence_note']=''
    for name,(eid,url,note) in special_evidence.items():
        ix=audit['名称'].eq(name)
        audit.loc[ix,['new_evidence_id','new_source_url','new_evidence_note']]=[eid,url,note]
        audit.loc[ix,'evidence_level']='public_role_support_micro_boundary_unverified'
    audit['source_row_complete']=True
    audit['main_included']=audit.decision.eq('retain_site_candidate')
    audit['status_uncertainty']='候选资格不等于实证独立场地认证；2024状态缺口不由当前网页倒推'
    keep = audit.main_included.to_numpy()
    assert 70 < keep.sum() < 200
    # A functional label for later reuse, separated from inherited model family.
    audit['reviewed_family']=audit.family
    corrections={'中国邮政武汉市黄金口邮件处理中心':'distribution','孝昌县邮政寄递物流共配中心':'distribution','武汉中远物流有限公司物流基地(临空港大道店)':'park','铁路集装箱货运中心':'rail','武汉慧欣集装箱货运堆场':'park','武钢运输部武钢车站':'rail','阳逻综保区冷库':'coldchain','花湖机场黄冈城市货站':'airport','鄂州花湖楚阳物流':'other','京东武汉新港分拣中心':'distribution','武汉新港空港综合保税区阳逻港园区':'bonded','武汉空港综合保税区':'bonded','武汉临空港综合保税园区投资发展有限公司':'other','武汉新港空港综合保税区申报大厅':'other'}
    for name,family in corrections.items():audit.loc[audit['名称'].eq(name),'reviewed_family']=family
    save(audit,OUT/'01_all_246_node_decisions.csv')
    save(audit[~audit.main_included],OUT/'02_removed_or_quarantined.csv')
    save(audit[audit.main_included],OUT/'03_processing_site_candidates.csv')
    save(relations,OUT/'04_parent_child_relations.csv')
    # All same-name and nearby pairs get a decision; closeness never triggers a merge.
    lon=np.deg2rad(audit['经度(lng)'].to_numpy());lat=np.deg2rad(audit['纬度(lat)'].to_numpy())
    distances=6371000*2*np.arcsin(np.sqrt(np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2))
    pairs=[]
    for i,j in itertools.combinations(range(len(audit)),2):
        a,b=audit.iloc[i],audit.iloc[j]
        if distances[i,j]>250 and a['名称']!=b['名称']:continue
        group=a.relation_group if a.relation_group==b.relation_group else ''
        if group:action='explicit_parent_child_single_representative'
        elif a['名称']==b['名称']=='航发智慧物流园':action='same_project_candidates_both_out_of_2024_main'
        elif a['名称']==b['名称']:action='homonym_different_addresses_no_merge'
        elif not a.main_included or not b.main_included:action='resolved_by_role_or_uncertainty_isolation_no_merge'
        else:action='distinct_site_candidates_retained_no_same_unit_evidence'
        pairs.append(dict(node_a=a.node_id,node_b=b.node_id,name_a=a['名称'],name_b=b['名称'],distance_m=round(float(distances[i,j]),3),address_a=a['详细地址'],address_b=b['详细地址'],action=action,confirmed_same_processing_unit=False))
    save(pairs,OUT/'05_all_nearby_or_homonym_pairs.csv')
    save(audit[audit.reviewed_family.ne(audit.family)],OUT/'06_function_label_corrections.csv')
    save(n,OUT/'frozen_inputs/original_nodes.csv');save(e,OUT/'frozen_inputs/original_edges.csv')
    clean=n.loc[keep].copy().reset_index(drop=True)
    # Keep old functional scoring frozen to isolate sample boundary in primary grid.
    # Export the corrected label separately; a declared variant tests its bonus effect.
    clean['reviewed_family']=audit.loc[keep,'reviewed_family'].to_numpy()
    clean['legacy_scoring_family']=clean.family
    clean['node_identity_status']='processing_site_candidate_not_fully_verified'
    save(clean,OUT/'frozen_inputs/processing_candidate_nodes.csv')
    clean_e=e[e.Source_id.isin(clean.node_id)&e.Target_id.isin(clean.node_id)].reset_index(drop=True)
    save(clean_e,OUT/'frozen_inputs/processing_candidate_edges.csv')
    fixed=json.loads((RESULTS_DIR/'dispatch_capacity_null_mainline_20261018/84_design.json').read_text(encoding='utf-8'))
    cases=[];projection=[]
    for c in fixed['cases']:
        common=sorted(set(c['initial_ids'])&set(clean.node_id))
        cases.append(dict(c,initial_ids=common))
        projection.append(dict(case_id=c['case_id'],original_initial_count=len(c['initial_ids']),common_initial_count=len(common),removed_ids=json.dumps(sorted(set(c['initial_ids'])-set(common)),ensure_ascii=False)))
    save(projection,OUT/'07_shock_projection.csv')
    demand_payload={}
    for key in fixed['demand_names']:
        parts,groups=demands[key]
        demand_payload[key]=dict(parts=np.asarray(parts).tolist(),groups=list(groups))
    dump(demand_payload,OUT/'frozen_inputs/demands.json')
    for cap in fixed['capacity_ids']:
        nn,_=formula_precision_nodes(core.capacity_nodes(n,cap),n)
        save(nn,OUT/f'frozen_inputs/{cap}_nodes.csv')
    protected=list(DATA_DIR.glob('*.csv'))+list(Path(__file__).parent.glob('*.py'))
    protected += [core.ROOT.parent/'中文论文初稿_节点边界与主张收口V6_20261021.docx']
    roles=['matched246','candidate_induced','candidate_fixed_degree','candidate_corrected_family']
    jobs=[dict(scope=s,capacity_id=c) for s,c in itertools.product(roles,fixed['capacity_ids'])]
    design=dict(schema='processing-unit-rectification-v7',date='2026-10-21',audit_nodes=246,
        candidate_nodes=int(keep.sum()),candidate_edges=len(clean_e),keep_ids=clean.node_id.tolist(),
        decision_counts=audit.decision.value_counts().to_dict(),cases=cases,thresholds=fixed['thresholds'],
        demand_names=fixed['demand_names'],capacity_ids=fixed['capacity_ids'],arms=['A','B','C'],
        scopes=roles,jobs=jobs,expected_runs=len(jobs)*16*4*3*3,
        preregistered_within_this_revision=True,prior_findings_known=True,
        primary_comparison='matched246 versus candidate_induced, same retained tasks and same projected shocks; resources and risk graph jointly change',
        variants={'candidate_fixed_degree':'same candidate graph and inherited family; original-246 degree denominator',
                  'candidate_corrected_family':'same candidate graph, induced degree; reviewed family only changes scenario bonus, L/K/edge proxies stay frozen'},
        no_sum_or_refit=True,no_new_edges=True,no_IPF_refit=True,actual_real_capacity_replacements=0,
        all_246_independently_verified=False,retained_are_candidates_not_certified=True,
        source_hashes={str(p):sha(p) for p in protected if p.exists()},
        audit_hashes={p.name:sha(p) for p in OUT.glob('*.csv')},
        stops=['Do not choose sample/representative from service outcomes','No missing evidence interpreted as nonexistence',
               'No nearest-neighbor automatic merge','No transfer of excluded proxy K or tasks to retained sites',
               'No claim all candidate POIs operated independently in 2024'])
    dump(design,OUT/'00_design.json');dump({'design_sha256':sha(OUT/'00_design.json')},OUT/'design_lock.json')
    dump(dict(rows=246,candidates=int(keep.sum()),edges=len(clean_e),decision_counts=design['decision_counts'],nearby_pairs=len(pairs),physical_duplicate_merges=0,hierarchy_single_representative_groups=len(relations),all_rows_disposed=True,all_independently_verified=False,grid_frozen_before_new_results=True,preflight='Prior no-outcome audit retained in processing_unit_rectification_v7_preflight_20261021; final criteria use explicit site words/source types instead of inherited family; H02 retains park because tenant handling boundary is unproven'),OUT/'08_audit_completion.json')
    print(json.dumps(read_json(OUT/'08_audit_completion.json'),ensure_ascii=False,indent=2))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


if __name__=='__main__':main()
