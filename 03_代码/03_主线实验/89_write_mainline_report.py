"""Evidence-bound research draft; expose demand and null-support limitations."""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from closed_loop_recovery_v3 import HERE
from submission_strengthening_common import module
core=module('83_dispatch_mainline_core');OUT=core.OUT

def md(df):
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))
def main():
    phases={name:json.loads((OUT/name).read_text(encoding='utf-8')) for name in ['84_completion.json','84b_control_completion.json','85_completion_v3.json','86_completion_v3.json','87_completion.json','88_completion.json']}
    if not all(v['completed'] for v in phases.values()):raise RuntimeError('Complete computations and QA first')
    for file,phase in [('84_run_capacity_policy_mainline.py','84_completion.json'),('84b_no_damage_controls.py','84b_control_completion.json'),('85c_generate_constrained_nulls_fixed_steps.py','85_completion_v3.json'),
                       ('86c_run_null_policy_fixed_steps.py','86_completion_v3.json'),('87_validate_mainline.py','87_completion.json'),('88_summarize_and_plot_mainline.py','88_completion.json')]:
        if core.sha(HERE/file)!=phases[phase]['source_sha256']:raise RuntimeError('Changed completed source '+file)
    for name in ['84_design.json','85_design_v3.json','86_design_v3.json','84b_control_design.json']:core.check_sources(json.loads((OUT/name).read_text(encoding='utf-8')))
    p=pd.read_csv(OUT/'88a_capacity_pairs.csv');summary=pd.read_csv(OUT/'88c_capacity_summary.csv');ds=pd.read_csv(OUT/'88d_demand_strata.csv')
    q=pd.read_csv(OUT/'88e_physical_metrics.csv');gs=pd.read_csv(OUT/'88g_null_graph_statistics.csv');positions=pd.read_csv(OUT/'88h_null_positions.csv')
    graph_support=pd.read_csv(OUT/'87b_null_support_diagnostics.csv').iloc[0];mix=pd.read_csv(OUT/'85b_swap_mixing_traces.csv')
    low=p[~p.demand.eq('external_plus150')].groupby('capacity_id').C_B_pp.mean();cb=summary[summary.contrast.eq('C-B')].set_index('capacity_id')
    primary=positions[positions.statistic.eq('primary_all_24')].iloc[0];nominal=positions[positions.statistic.eq('secondary_nominal_12')].iloc[0]
    cbtable=cb.reset_index()[['capacity_id','conditions','wins','ties','losses','mean_pp','median_pp']].round(6)
    wins=int(cb.wins.sum());ties=int(cb.ties.sum());losses=int(cb.losses.sum())
    later=mix[mix.proposal_steps_per_edge.ge(50)].groupby('graph_id').original_edge_overlap.agg(['min','max']);late_range=float((later['max']-later['min']).max())
    references='''1. Motter, A. E. & Lai, Y.-C. Cascade-based attacks on complex networks. Physical Review E 66, 065102 (2002). https://doi.org/10.1103/PhysRevE.66.065102 ; author manuscript: https://arxiv.org/abs/cond-mat/0301086
2. Motter, A. E. Cascade control and defense in complex networks. Physical Review Letters 93, 098701 (2004). https://doi.org/10.1103/PhysRevLett.93.098701 ; author manuscript: https://arxiv.org/abs/cond-mat/0401074
3. Kempe, D., Kleinberg, J. & Tardos, É. Maximizing the Spread of Influence through a Social Network. KDD (2003). Author manuscript: https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf
4. Ba, Q. & Savla, K. Computing Optimal Control of Cascading Failure in DC Networks. Author preprint, revised 2018. https://arxiv.org/abs/1712.06064
'''
    report=f'''# 容量异质性、A/B/C派入与受约束零模型：完整结果

日期：2026-10-18。三套预先指定容量×16个固定损伤×4种需求×3个风险阈值×3政策，共1728条主评价；50张空间与度数约束随机网络×4个固定损伤×2种需求×3个阈值×3政策，共3600条评价。另保存按风险场景匹配的无外部损伤控制{phases['84b_control_completion.json']['rows']}条。正式数据、物理规则及旧结果未改。

## 1. 主结果及必须同时呈现的边界

C相对B在576个容量/损伤/需求/阈值配对中更高{wins}、持平{ties}、更低{losses}；三套容量结果如下。差值为稳定已完成服务的百分点，不是派入量、损失金额或恢复AUC。

{md(cbtable)}

全网总能力、基准需求、节点身份、原图边、风险属性及需求分配矩阵固定。容量版本仅改变K的节点分布；额定能力是代理量而非已标定吨/日。类型成组版本没有宣称机场能力大于港口能力。原生成的81套容量只运行了事先固定的3套；未运行的78套不能写成已完成敏感性。

**约23个百分点的总体均值被最高需求档显著拉高，不能全部解释成外部损伤诱发的级联抑制。**这一档将外部需求乘2.5，因外部占25%，设施总需求约变为1.375倍；全网额定能力约为基准负荷的1.25倍。即使不损伤任何节点，A/B在本模型中也全网退出，C无节点失效并完成约90.9091%的需求。该结果反映最大应急承接与“超额即退出”的组合假设，不是观察到真实机场、港口或仓库集体停运。

排除最高需求档仅作明确标注的分层诊断，原全部结果仍保留。其余三档下C相对B平均差值：基准{low['baseline']:.6f}、节点异质{low['node_a0.20_s00']:.6f}、类型异质{low['family_a0.20_s00']:.6f}个百分点。各需求档完整结果如下：

{md(ds[['capacity_id','demand','conditions','wins','ties','losses','mean_pp']].round(6))}

## 2. 拒收、完成服务、触发渠道和运输成本

A：rho1.1，最大派入后最小成本。B：rho1.1，最大派入后最小总额定超额，再最小成本。C：rho1，额定能力内最大派入后最小成本。三者差异是派入政策本身；不能将C称为同样最大应急派入量约束下的比较。

表中未服务包括未派入及已派入未完成；二者原始量均在84a/86a。终态γ1的过载退出规则使尚存设施通常已无超额，因此只看终态未完成量可能掩盖之前的超额尝试，逐轮记录84b/86b必须一起保留。风险/过载触发为互斥标签，不等于可直接相加的因果贡献。

{md(q.round(5))}

运输成本为尝试转移流×代理成本的总和，无真实货币量纲。退出后无货可转的方案可能成本为0；这不表示它经济上更好。没有拒收罚损、等待成本或跨期排队，不能据此宣称C在真实业务总成本上最优。

## 3. 50张受约束零模型的结果

零模型严格保持246节点、每个节点的度数、4579边、简单图和全局路程分箱计数，节点位置与ID不变；最终图连通。距离分箱为0/5/10/20/40/80/无穷公里，仅保留粗粒度空间分布，不保证每个节点距离分布或精确边长不变。新边来自既有节点对候选资料，风险、跨区、成本按对应节点对取值，边能力按冻结原节点capacity_index重算；原图被准确重建后才开始随机化。需求及容量映射不随图重新拟合。

主零模型仅使用预先指定的节点异质容量版本、4个已生成冲击、2种需求和3阈值，每图24个条件。主统计量为每图C-B服务差的均值；名义需求12条件为透明的次级诊断。没有根据结果选择节点、阈值或容量种子。

{md(positions.round(6))}

主统计量原图{primary.original:.6f}个百分点；50图均值{primary.null_mean:.6f}，5%—95%经验分位范围{primary.null_q05:.6f}—{primary.null_q95:.6f}。名义需求子集原图{nominal.original:.6f}、零模型均值{nominal.null_mean:.6f}个百分点。图中分布不是现实灾害的置信区间。

**该零模型是局部扰动而非50个独立城市。**原边保留比例为{gs.original_edge_overlap.min():.4%}—{gs.original_edge_overlap.max():.4%}；50图共同保留的原边为{int(graph_support.common_original_edges_across_nulls)}/{int(graph_support.original_edges)}，即{graph_support.common_fraction:.4%}。后半链原边重叠率在每图内的最大波动范围为{late_range:.6f}，可作一项稳定诊断，但不能证明全空间混合充分。约束可能将可探索空间限制在原网络附近。

z仅为标准化位置描述；upper_tail_mc_position采用(1+超过或等于原值的图数)/51。它是近似有限链集合的尾部位置指标，**不是已校准的精确P值**，不据此写“显著优于所有城市”。C在当前规则下的弱占优已有结构性证明，零模型不能将同一保证重新当作独立方法创新；它补充的是改善幅度及其局部拓扑依赖。

## 4. 采样器修正与验证

最初全局均匀抽边提案在空间约束下接受率过低，触发停止，未产出完成图。第二版几何条件Metropolis提高了效率，但以成功交换数停止会抽到跳链；在查看策略成绩之前改为第三版固定提案步数，并保留两版代码、设计和试生成输出。

正式每链进行100×4579次提案，含拒绝与自环，独立种子202610180900—949。提案依次选择节点a、可改连邻居b、同路程箱非邻居c，再在c邻居中选满足第二条边分箱与非重复条件的d。a的候选邻居数和同箱非邻居数在此交换下不变；若d的正/反候选数为M_f/M_r，接受率min(1,M_f/M_r)纠正状态依赖提案。最终断连图按预设种子重试，未按策略效果挑图。

独立枚举的五节点状态转移矩阵检验了可逆性和行和；另核对50图全部度数、分箱、连通性、新边属性及唯一性。物理量守恒、触发分区、独立风险闭包、逐轮轨迹与{phases['87_completion.json']['cold_replays']}次冷启动复算均通过，共{phases['87_completion.json']['checks']}项主线独立检查。精确小例不代替大图混合时间证明。

## 5. 论文应该怎样写

核心可写成：在固定需求与风险、允许自由拒收、只评价终态已完成服务的代理模型中，额定能力内派入达到纯风险闭包并避免额外过载；这一条件性结论在两种等总能力异质性和局部随机拓扑中得到数值核对。它不是新证书算法，也不支持“现实应急超额派入必然有害”。

应在正文保留高需求无损伤控制以及零模型约92%原边保留率；二者都会改变读者对效应量和外推范围的理解，不能藏在SI。逐情景大表、交换诊断、全部来源哈希放SI或仓库。已有预算/恢复与保护试验可作后续讨论，但不能把本轮终态服务等同于恢复AUC。

## 6. 交付与复算

84a：1728条主结果；84b：主级联逐轮；84c：无损伤控制。85a/b：正式50图摘要与链诊断；null_graphs_v3：正式边表。86a/b：3600条零模型结果及逐轮。87：独立验证。88a—h：完整配对、分层、成本与零模型位置。图件含PNG/PDF/SVG/TIFF、绘图数据及布局/字体/碰撞QA；最终逐图目视核对记录另存，论文作者的科学审阅尚未完成。按用户最新指示，本轮不改写或新写论文正文，仅交付实验、图表和结果报告；不代表完成实测标定或投稿。
'''
    report_path=core.ROOT/'项目文档/容量异质性与派入政策主线完整结果_20261018.md'
    report_path.write_text(report,encoding='utf-8');(OUT/report_path.name).write_text(report,encoding='utf-8')
    visual_path=OUT/'figures/figure_visual_review.json'
    if not visual_path.exists():raise RuntimeError('Inspect final exported figures and save visual-review record before finalizing')
    visual=json.loads(visual_path.read_text(encoding='utf-8'))
    if not visual.get('all_figures_checked'):raise RuntimeError('Incomplete visual review')
    for name,digest in visual.get('preview_sha256',{}).items():
        if core.sha(OUT/'figures'/name)!=digest:raise RuntimeError('Changed figure after visual review '+name)
    sources=[HERE/n for n in ['83_dispatch_mainline_core.py','84_run_capacity_policy_mainline.py','84b_no_damage_controls.py','85_generate_constrained_nulls.py','85b_generate_constrained_nulls_mh.py','85c_generate_constrained_nulls_fixed_steps.py',
              '86_run_null_policy_mainline.py','86b_run_null_policy_mainline_mh.py','86c_run_null_policy_fixed_steps.py','87_validate_mainline.py','88_summarize_and_plot_mainline.py','89_write_mainline_report_and_draft.py','89_write_mainline_report.py']]
    artifacts=[p for p in OUT.rglob('*') if p.is_file() and p.name!='89_final_manifest.json']+sources+[report_path]
    for optional in [core.ROOT/'run_dispatch_mainline.ps1',HERE/'README_派入政策主线_20261018.md']:
        if optional.exists():artifacts.append(optional)
    # Verify all prior finalized work remains intact before issuing new manifest.
    for folder,manifest_name in [('protect_repair_v6b_training_aware_20261017','76_final_manifest.json'),('dispatch_theory_feasibility_20261018','79_final_manifest.json'),('dual_mechanism_triviality_audit_20261018','81_final_manifest.json')]:
        previous=json.loads((core.ROOT/'项目结果'/folder/manifest_name).read_text(encoding='utf-8'))
        for v in previous['artifacts']:
            if core.sha(core.ROOT/v['path'])!=v['sha256']:raise RuntimeError('Previous work changed '+v['path'])
    final=dict(completed=True,phases=phases,main_capacity_versions=3,generated_capacity_versions=81,formal_null_graphs=50,
               no_empirical_capacity_replacement=True,manuscript_status='Deferred at user request; no manuscript body created or edited',
               calibrated_p_values=False,broad_cross_city_validation=False,prior_finalized_work_preserved=True,
               artifacts=[dict(path=str(p.relative_to(core.ROOT)),sha256=core.sha(p)) for p in sorted(set(artifacts))])
    core.dump(final,OUT/'89_final_manifest.json')
    print('89 experiment report complete; manuscript body deferred at user request',flush=True);print(report_path,flush=True)

if __name__=='__main__':main()
