"""Independent reconstruction and reporting of the frozen gamma boundary grid."""
import importlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dual_mechanism_recovery_v5 import DispatchAllocatorV5, DualMechanismRecoveryV5
driver=importlib.import_module('114_gamma_demand_boundary')
OUT=driver.OUT
CHECKS=[]


def check(name,ok):
    CHECKS.append(dict(check=name,passed=bool(ok)))
    if not ok:raise AssertionError(name)


def md(frame,cols):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join(
        '| '+' | '.join(fmt(row[c]) for c in cols)+' |' for _,row in frame.iterrows())


def main():
    d=json.loads((OUT/'00_design.json').read_text(encoding='utf-8'));driver.check_sources(d)
    done=json.loads((OUT/'completion.json').read_text(encoding='utf-8'))
    check('run complete',done['completed'] and done['checks']==done['passed'])
    for p,h in done['files'].items():check('output hash '+p,driver.sha(OUT/p)==h)
    n=pd.read_csv(OUT/'frozen_inputs/nodes.csv');e=pd.read_csv(OUT/'frozen_inputs/edges.csv')
    demand=json.loads((OUT/'frozen_inputs/demand.json').read_text(encoding='utf-8'))
    base=np.asarray(demand['parts']);K=n.overload_threshold.to_numpy(float);ids=list(n.node_id);pos={k:i for i,k in enumerate(ids)}
    cases={c['case_id']:c for c in d['cases']}
    runs=pd.read_csv(OUT/'01_runs.csv');logs=pd.read_csv(OUT/'02_rounds.csv')
    keys=['theta','gamma','external_multiplier','case_id','arm']
    check('unique 840 run keys',len(runs)==840 and not runs.duplicated(keys).any() and runs.run_id.is_unique)
    check('480 shock plus 360 control',sum(~runs.control)==480 and sum(runs.control)==360)
    check('round run IDs complete',set(logs.run_id)==set(runs.run_id))
    # Separate implementation of the risk scoring equation, from frozen edge columns.
    edges=e.sort_values(['Source_id','Target_id']).reset_index(drop=True)
    src=np.r_[edges.Source_id.map(pos).to_numpy(int),edges.Target_id.map(pos).to_numpy(int)]
    dst=np.r_[edges.Target_id.map(pos).to_numpy(int),edges.Source_id.map(pos).to_numpy(int)]
    degree=np.bincount(src,minlength=len(n))
    edge_score=np.tile(.21+.16*edges.Failure_probability_proxy.to_numpy(float)
                       +.12*edges.Cross_district.to_numpy(float)
                       +.10*np.minimum(np.maximum(edges.Composite_transport_cost.to_numpy(float),.001)/.45,1.)
                       +.09*(1-np.minimum(edges.Edge_capacity_proxy.to_numpy(float)/55.,1.)),2)
    def pressure(failed,kind):
        removed=np.array([node in failed for node in ids]);mask=removed[dst]
        count=np.bincount(src[mask],minlength=len(n))
        value=np.bincount(src[mask],weights=edge_score[mask],minlength=len(n))/.68/np.maximum(degree,1)
        if kind=='spatial_cluster':value+=.1*n['区县'].isin(['东西湖区','新洲区','黄陂区']).to_numpy()
        if kind=='channel_disruption':value+=.1*n.family.isin(['airport','rail','port']).to_numpy()
        return removed,count,value
    risk_closures={}
    for case,theta in itertools.product(d['cases'],d['thresholds']):
        current=set(case['initial_ids'])
        while True:
            removed,count,value=pressure(current,case['kind'])
            new={ids[i] for i in np.flatnonzero((~removed)&(count>0)&(value>=theta))}
            if not new:break
            current|=new
        risk_closures[case['case_id'],theta]=current
    for row in runs.itertuples():
        case=cases[row.case_id];parts=base.copy();parts[1:]*=row.external_multiplier;D=parts.sum(axis=0)
        check('demand and capacity '+str(row.run_id),abs(row.Demand_Load-D.sum())<1e-6
              and abs(row.total_demand_multiplier-(.75+.25*row.external_multiplier))<1e-10)
        current=set(case['initial_ids']);totals=np.zeros(3,int)
        group=logs[logs.run_id==row.run_id].sort_values('round')
        check('consecutive rounds '+str(row.run_id),list(group['round'])==list(range(len(group))))
        for step in group.itertuples():
            util=np.asarray(json.loads(step.util_json));served=np.asarray(json.loads(step.source_completed_json))
            removed,count,p=pressure(current,row.kind)
            over=(~removed)&(util>row.gamma+1e-7)
            risk=(~removed)&(count>0)&(p>=row.theta)
            cause=[over&~risk,risk&~over,over&risk]
            new={ids[i] for i in np.flatnonzero(over|risk)}
            check('independent full trigger '+str((row.run_id,step.round)),new==set(json.loads(step.new_ids_json))
                  and [int(x.sum()) for x in cause]==[step.overload_only,step.risk_only,step.both]
                  and step.failed_count==len(current) and step.new_failures==len(new))
            check('independent receiver/source accounting '+str((row.run_id,step.round)),
                  np.isfinite(util).all() and np.isfinite(served).all()
                  and (util>=-1e-7).all() and (util<=(1. if row.arm=='C' else 1.1)+1e-7).all()
                  and (served>=-1e-7).all() and (served<=D+1e-6).all()
                  and np.max(abs(util[removed]),initial=0)<1e-7
                  and abs(np.minimum(util*K,K).sum()-served.sum())<1e-6
                  and abs(served.sum()-step.Served_Load)<1e-6
                  and abs(step.Service_Rate-served.sum()/D.sum())<1e-9
                  and abs(step.Attempted_Assigned_Load-(util*K).sum())<1e-6
                  and abs(step.Attempted_Assigned_Load-step.Served_Load-step.Uncompleted_Assigned_Load)<1e-6
                  and abs(step.Undispatched_Load+step.Uncompleted_Assigned_Load+step.Served_Load-D.sum())<1e-6)
            totals+=np.array([int(x.sum()) for x in cause]);current|=new
        check('stable last round '+str(row.run_id),group.iloc[-1].new_failures==0)
        check('terminal output consistency '+str(row.run_id),current==set(json.loads(row.final_ids_json))
              and len(current)==row.final_failed and bool(row.all_failed)==(len(current)==len(n))
              and abs(group.iloc[-1].Served_Load-row.Served_Load)<1e-6
              and list(totals)==[row.Overload_Only_Failures,row.Risk_Only_Failures,row.Both_Failures])
        theta=min(d['thresholds'],key=lambda x:abs(x-row.theta))
        if row.arm=='C':check('independent pure risk closure '+str(row.run_id),current==risk_closures[row.case_id,theta])
        if row.gamma>= (1. if row.arm=='C' else 1.1):check('structural endpoint '+str(row.run_id),totals[0]+totals[2]==0)
    pairkeys=['theta','gamma','external_multiplier','case_id','control','kind','total_demand_multiplier']
    B=runs[runs.arm=='B'];C=runs[runs.arm=='C']
    pairs=B.merge(C,on=pairkeys,suffixes=('_B','_C'),validate='one_to_one')
    pairs['C_minus_B_pp']=100*(pairs.Service_Rate_C-pairs.Service_Rate_B)
    check('420 pairs and total weak dominance',len(pairs)==420 and (pairs.C_minus_B_pp>=-1e-7).all())
    for row in pairs.itertuples():
        check('C failed subset B '+str(row.run_id_B),set(json.loads(row.final_ids_json_C))<=set(json.loads(row.final_ids_json_B)))
        if row.gamma==1.1:
            check('B/C equal at gamma=rho '+str(row.run_id_B),abs(row.C_minus_B_pp)<1e-7
                  and row.final_ids_json_B==row.final_ids_json_C)
    for key,g in C.groupby(['theta','external_multiplier','case_id']):
        check('C gamma invariant '+str(key),g.final_ids_json.nunique()==1 and g.Service_Rate.max()-g.Service_Rate.min()<1e-9)
    # Controls remain separate. DID is a model description, not an empirical causal estimate.
    controls=pairs[pairs.control].copy();shocks=pairs[~pairs.control].copy()
    ctl=controls[['theta','gamma','external_multiplier','kind','C_minus_B_pp','Service_Rate_B','Service_Rate_C']].rename(
        columns={'C_minus_B_pp':'no_damage_gap_pp','Service_Rate_B':'no_damage_B','Service_Rate_C':'no_damage_C'})
    shocks=shocks.merge(ctl,on=['theta','gamma','external_multiplier','kind'],validate='many_to_one')
    shocks['gap_minus_no_damage_pp']=shocks.C_minus_B_pp-shocks.no_damage_gap_pp
    summary=[]
    for key,g in shocks.groupby(['theta','gamma','external_multiplier','total_demand_multiplier']):
        summary.append(dict(zip(['theta','gamma','external_multiplier','total_demand_multiplier'],key),
            n_shocks=len(g),mean_gap_pp=g.C_minus_B_pp.mean(),median_gap_pp=g.C_minus_B_pp.median(),
            min_gap_pp=g.C_minus_B_pp.min(),max_gap_pp=g.C_minus_B_pp.max(),
            mean_gap_minus_no_damage_pp=g.gap_minus_no_damage_pp.mean(),
            B_all_failed=int(g.all_failed_B.sum()),wins=int((g.C_minus_B_pp>1e-7).sum()),
            ties=int((abs(g.C_minus_B_pp)<=1e-7).sum())))
    summary=pd.DataFrame(summary)
    adjacent=[]
    for key,g in runs[runs.arm=='B'].groupby(['theta','external_multiplier','case_id']):
        ordered=g.sort_values('gamma')
        for lo,hi in zip(ordered.itertuples(),list(ordered.itertuples())[1:]):
            adjacent.append(dict(theta=key[0],external_multiplier=key[1],case_id=key[2],
                                 control=lo.control,gamma_low=lo.gamma,gamma_high=hi.gamma,
                                 B_service_change_pp=100*(hi.Service_Rate-lo.Service_Rate),
                                 failures_change=hi.final_failed-lo.final_failed))
    adjacent=pd.DataFrame(adjacent)
    # Predeclared cold solves via the unconfigured legacy entry point.
    cold=[]
    core=importlib.import_module('83_dispatch_mainline_core');nn,ee,_,_=core.load_study()
    selected=runs[(runs.case_id=='test_random_00') & (((runs.arm=='B')&(runs.external_multiplier==1.5))
                   |((runs.arm=='C')&(runs.gamma==1.)&(runs.external_multiplier.isin([1.,2.5]))))]
    check('18 predeclared cold starts',len(selected)==18)
    for row in selected.itertuples():
        parts=base.copy();parts[1:]*=row.external_multiplier;case=cases[row.case_id]
        a=DispatchAllocatorV5(nn,ee,1. if row.arm=='C' else 1.1,'min_overload',2)
        en=DualMechanismRecoveryV5(a,ee,parts,demand['groups'],case['initial_ids'],case['kind'],row.theta,gamma=row.gamma)
        f,m,_=en.closure(en.initial)
        target=logs[logs.run_id==row.run_id].sort_values('round')
        ok=set(f)==set(json.loads(row.final_ids_json)) and abs(m['Served_Load']-row.Served_Load)<1e-6
        ok=ok and len(en.cascade_logs[en.initial])==len(target)
        current=set(en.initial)
        for step,stored in zip(en.cascade_logs[en.initial],target.itertuples()):
            state=en.state(current)
            ok=ok and set(step['new_ids'])==set(json.loads(stored.new_ids_json))
            ok=ok and np.allclose(state[1],json.loads(stored.util_json),atol=1e-7,rtol=0)
            ok=ok and np.allclose(state[2],json.loads(stored.source_completed_json),atol=1e-6,rtol=0)
            current.update(step['new_ids'])
        check('legacy cold solve '+str(row.run_id),ok)
        cold.append(dict(run_id=row.run_id,arm=row.arm,gamma=row.gamma,theta=row.theta,passed=bool(ok)))
        a.paths.cache_clear()
    for name,frame in [('04_pairs.csv',pairs),('05_shock_control_contrasts.csv',shocks),
                       ('06_shock_summary.csv',summary),('07_no_damage_controls.csv',controls),
                       ('08_gamma_adjacent_changes.csv',adjacent),('09_cold_validation.csv',pd.DataFrame(cold)),
                       ('10_independent_validation.csv',pd.DataFrame(CHECKS))]:driver.save(frame,name)
    findings=dict(completed=True,closures=840,shock_closures=480,control_closures=360,policy_pairs=420,
                  run_checks=done['checks'],independent_checks=len(CHECKS),independent_passed=sum(c['passed'] for c in CHECKS),
                  historical_gamma1_checks=72,cold_starts=len(cold),
                  B_service_decreases_when_gamma_increases=int((adjacent.B_service_change_pp < -1e-7).sum()),
                  C_weak_dominance=True,C_gamma_invariant=True,gamma_rho_B_C_equal=True,
                  all_parameters_empirically_calibrated=False,independent_real_samples=0)
    driver.dump(findings,OUT/'11_findings.json')
    report(summary,controls,shocks,adjacent,findings)
    driver.check_sources(d)
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='final_manifest.json']
    driver.dump(dict(**findings,files={str(p.relative_to(OUT)):driver.sha(p) for p in files}),OUT/'final_manifest.json')
    print(json.dumps(findings,ensure_ascii=False),flush=True)


def report(summary,controls,shocks,adjacent,findings):
    compact=summary.groupby(['gamma','external_multiplier','total_demand_multiplier'],as_index=False).agg(
        mean_gap_pp=('mean_gap_pp','mean'),mean_gap_minus_no_damage_pp=('mean_gap_minus_no_damage_pp','mean'),
        B_all_failed=('B_all_failed','sum'))
    # Equal weighting is a design summary, not a real-world demand/shock distribution.
    pivot=compact.pivot(index='external_multiplier',columns='gamma',values='mean_gap_pp').reset_index()
    pivot.columns=['外部需求倍率']+[f'gamma={x:g}' for x in pivot.columns[1:]]
    ctl=controls.groupby(['gamma','external_multiplier'],as_index=False).agg(
        B_service_pct=('Service_Rate_B',lambda x:100*x.mean()),C_service_pct=('Service_Rate_C',lambda x:100*x.mean()),
        B_all_failed=('all_failed_B','sum'))
    text='''# 退出阈值 × 需求压力边界实验

2026-10-20。本轮是现有静态代理模型的假设边界检查，不是现实运行标定或新算法。

## 设计与完整性

固定原图与基准容量，使用四个已被多轮查看的8节点损伤和三类无损伤对照，三个既有风险阈值，四个gamma，五个外部需求倍率，B/C两政策，共840次闭包。其中480有损伤、360无损伤。没有新增留出样本，也未追加容量、种子或更细网格。

外部倍率1/1.2/1.5/2/2.5，对应全网总需求倍率1/1.05/1.125/1.25/1.375。gamma只改变退出阈值，额定处理能力K和接纳上限rho不变；没有增加完成能力或跨期积压处理。

## C相对B的终态总服务差

下表为四损伤×三风险阈值的等权设计摘要，单位百分点。完整阈值分层见06_shock_summary.csv。此平均不是现实情景发生概率加权，表中最高需求不混入普通需求概括。

'''+md(pivot,list(pivot.columns))+'''

## 无初始损伤对照

以下每格包含三风险场景×三阈值共9条件，无初始失效时风险不能自行触发，场景bonus仍按原设计保留。已接收未完成量仍属未服务，不结转库存。

'''+md(ctl,list(ctl.columns))+'''

## 如何解释

gamma=1对应原来的“超过额定能力就退出”。gamma>=rho=1.1时，B的接纳约束已经阻止过载触发，因此B/C回到同一纯风险闭包，且本轮总服务相同。这是运行前已知的结构控制，不能作为新发现，也不能据此说现实系统应提高退出阈值。

所有条件均核对C>=B和C的失效集合不随gamma变化。这是原有假设下的一致性检查，不能重新计为最优性创新。中间gamma下的幅度才是本轮新增数值证据，不预设B服务随gamma单调。08表保留每个相邻gamma的正负变化。

05表额外给出“有损伤C-B差减去同风险场景无损伤C-B差”，仅作模型内控制后的描述；不是现实灾害因果效应。高需求、无损伤已经发生的能力缺口和内生退出，应与灾害诱发的额外损失区分。

## 验证

72个gamma=1既有条件回归旧84表的服务及失效集合；保存每轮利用率和来源完成，独立从冻结边表重算风险压力、全部触发、守恒与终态；另有18个预定条件用原始V5入口冷启动，比较逐轮新增失效、利用率及来源完成。

'''+f"运行校验{findings['run_checks']}项；独立校验{findings['independent_passed']}/{findings['independent_checks']}项通过。相邻gamma提高后B服务下降的条件数为{findings['B_service_decreases_when_gamma_increases']}。校验项不是独立现实样本。\n"+'''

## 投稿意义与边界

本轮能回答原结果有多依赖退出阈值，不能验证gamma的现实数值，不能消除自由拒收、不可逆退出、固定风险和未实测能力的限制。若优势集中在gamma接近1的区域，应明确表述为严格退出规则下的条件性结果，不能据此调整gamma去追求更大优势。

四个损伤曾被查看，设施承接相容性尚未核实，原图与基准容量范围有限。结果不能独立保证中科院二区或三区适配；原数据、生产源码、已有输出与论文正文保持不变。

## 文件与复算

00_design为运行前设计，design_lock记录SHA256；01_runs逐闭包，02_rounds逐轮完整画像，03运行校验；04全部配对，05匹配控制差，06有损伤分层摘要，07无损伤控制，08相邻gamma变化，09冷启动，10独立校验，11机器结论；configs保存配置；shards保存60个可恢复计算块；final_manifest为完整输出哈希。

用相同Python环境依次运行114_gamma_demand_boundary.py design、114_gamma_demand_boundary.py run、115_validate_gamma_boundary.py。已有设计不会覆盖；已有计算块只按设计哈希恢复，不改变结果。
'''
    (OUT/'退出阈值与需求压力_边界实验结果_20261020.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
