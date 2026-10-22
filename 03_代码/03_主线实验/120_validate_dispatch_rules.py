"""Independent physics, rule tests, regression and bounded study reporting."""
import importlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from model_config import ModelConfig
from configured_dual_mechanism_v5 import ConfiguredDispatchAllocator,ConfiguredDualMechanismRecovery
from dispatch_policy_rules import DispatchRuleAllocator,DispatchRuleSpec,dispatch_batch

driver=importlib.import_module('119_dispatch_rule_study');OUT=driver.OUT;CHECKS=[]


def check(name,ok):
    CHECKS.append(dict(check=name,passed=bool(ok)))
    if not ok:raise AssertionError(name)


def analytic():
    src=np.array([0,0]);dst=np.array([1,2]);rank=np.arange(3)
    for rule,wanted in [('remaining_capacity_share',[1.,2.]),('nearest_feasible',[2.,1.])]:
        f,_=dispatch_batch([3.,0.,0.],[0.,2.,4.],src,dst,[1.,2.],rank,rank,rule)
        check('hand computed proportional versus nearest '+rule,np.allclose(f,wanted,atol=1e-12,rtol=0))
    # Two origins compete for one receiver; the first also has an alternative.
    # A heuristic is allowed to lose accepted quantity relative to max flow.
    src=np.array([0,0,1]);dst=np.array([2,3,2]);rank=np.arange(4)
    for rule,wanted in [('remaining_capacity_share',[1/3,2/3,2/3]),('nearest_feasible',[1.,0.,0.])]:
        f,_=dispatch_batch([1.,1.,0.,0.],[0.,0.,1.,1.],src,dst,[1.,2.,3.],rank,rank,rule)
        check('known nonmaximal heuristic '+rule,np.allclose(f,wanted,atol=1e-12,rtol=0))
    rng=np.random.default_rng(202610200001)
    for trial in range(25):
        n=7;source=np.arange(n);s,t=np.nonzero(rng.random((n,n))<.3)
        D=rng.uniform(0,9,n);R=rng.uniform(0,8,n);cost=rng.uniform(1,3,len(s))
        A=np.zeros((2*n,len(s)))
        for arc,(i,j) in enumerate(zip(s,t)):A[i,arc]=1;A[n+j,arc]=1
        opt=linprog(-np.ones(len(s)),A_ub=A,b_ub=np.r_[D,R],bounds=(0,None),method='highs')
        check('independent max flow solved '+str(trial),opt.success)
        perm=rng.permutation(n);inv=np.argsort(perm)
        for rule in ['remaining_capacity_share','nearest_feasible']:
            f,_=dispatch_batch(D,R,s,t,cost,source,source,rule)
            check('random feasibility '+str((trial,rule)),(f>=0).all() and (A@f<=np.r_[D,R]+1e-9).all()
                  and f.sum()<=-opt.fun+1e-8)
            fp,_=dispatch_batch(D[perm],R[perm],inv[s],inv[t],cost,source[perm],source[perm],rule)
            check('label invariance '+str((trial,rule)),np.allclose(fp,f,atol=1e-9,rtol=0))
    nodes=pd.DataFrame(dict(node_id=['S','R1','R2'],名称=['S','R1','R2'],区县='toy',family='toy',
                            proxy_daily_load=[3.,0.,0.],overload_threshold=[10.,10.,10.]))
    edges=pd.DataFrame(dict(Source_id=['S','S'],Target_id=['R1','R2'],Composite_transport_cost=[1.,2.],
             Repair_cost_proxy=1.,Failure_probability_proxy=0.,Cross_district=0.,Edge_capacity_proxy=55.))
    cfg=ModelConfig(theta_risk=2.,theta_source='analytic only')
    shared={};names=[]
    for rule in driver.RULES:
        spec=DispatchRuleSpec(rule)
        a=DispatchRuleAllocator(nodes,edges,cfg,spec)
        en=ConfiguredDualMechanismRecovery(a,edges,[[3.,0.,0.]],['toy'],['S'],'random');en.cache=shared
        value=en.state({'S'});names.append(en.cache_namespace)
        check('cache isolated result '+rule,np.array_equal(value[1],a.evaluate({'S'},[3.,0.,0.])[1]))
        if rule=='lexicographic_lp':
            old=ConfiguredDispatchAllocator(nodes,edges,cfg)
            result=old.evaluate({'S'},[3.,0.,0.],details=True);new=a.evaluate({'S'},[3.,0.,0.],details=True)
            check('LP default metrics exact',result[0]==new[0] and np.array_equal(result[1],new[1])
                  and np.array_equal(result[2],new[2]) and result[3]['arcs'].equals(new[3]['arcs']))
        mask=np.eye(3,dtype=bool);mask[0,2]=True
        limited=DispatchRuleAllocator(nodes,edges,cfg,DispatchRuleSpec(rule,compatibility_scope='explicit_scenario'),mask)
        detail=limited.evaluate({'S'},[3.,0.,0.],details=True)
        nonzero=detail[3]['arcs'][detail[3]['arcs'].flow>1e-9]
        check('explicit permission respected '+rule,set(nonzero.target_id)=={'R2'} and abs(nonzero.flow.sum()-3)<1e-9)
        a.clear_path_cache();limited.clear_path_cache()
    check('rule cache namespaces differ',len(set(names))==3 and len(shared)==3)
    # Tie reversal changes only equal-cost priority, not access to information.
    f,_=dispatch_batch([1.,1.,0.],[0.,0.,1.],[0,1],[2,2],[1.,1.],[0,1,2],[0,1,2],'nearest_feasible')
    g,_=dispatch_batch([1.,1.,0.],[0.,0.,1.],[0,1],[2,2],[1.,1.],[0,1,2],[0,1,2],'nearest_feasible','reverse_sources')
    check('source tie reversal is active',np.array_equal(f,[1,0]) and np.array_equal(g,[0,1]))
    return len(CHECKS)


def md(frame,columns):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{0. if abs(x)<.00000005 else x:.5f}'
        return str(x)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join(
        '| '+' | '.join(fmt(row[c]) for c in columns)+' |' for _,row in frame.iterrows())


def main():
    d=json.loads((OUT/'00_design.json').read_text(encoding='utf-8'));driver.verify(d)
    done=json.loads((OUT/'completion.json').read_text(encoding='utf-8'))
    check('computation completed',done['completed'] and done['checks']==done['passed'])
    for p,h in done['files'].items():check('completed file '+p,driver.sha(OUT/p)==h)
    analytic()
    graph_meta={g['graph_id']:g for g in d['graphs']};graphs={};physics={};path_cache={}
    for name,meta in graph_meta.items():
        n=pd.read_csv(OUT/meta['nodes_path']);e=pd.read_csv(OUT/meta['edges_path']).sort_values(['Source_id','Target_id']).reset_index(drop=True)
        graphs[name]=(n,e);ids=list(n.node_id);pos={k:i for i,k in enumerate(ids)}
        a=e.Source_id.map(pos).to_numpy(int);b=e.Target_id.map(pos).to_numpy(int)
        src=np.r_[a,b];dst=np.r_[b,a];degree=np.bincount(src,minlength=len(n))
        score=np.full(len(src),.68) if meta['domain']=='synthetic' else np.tile(.21+.16*e.Failure_probability_proxy.to_numpy(float)
            +.12*e.Cross_district.to_numpy(float)+.10*np.minimum(np.maximum(e.Composite_transport_cost.to_numpy(float),.001)/.45,1.)
            +.09*(1-np.minimum(e.Edge_capacity_proxy.to_numpy(float)/55.,1.)),2)
        direct=np.full((len(n),len(n)),np.inf)
        for row in e.itertuples():
            i,j=pos[row.Source_id],pos[row.Target_id];direct[i,j]=direct[j,i]=max(float(row.Composite_transport_cost),.001)
        physics[name]=dict(ids=ids,pos=pos,src=src,dst=dst,degree=degree,score=score,direct=direct)
        if meta['domain']=='synthetic':
            pairs={tuple(sorted((int(i),int(j)))) for i,j in zip(a,b)}
            check('synthetic simple connected controlled graph '+name,len(n)==60 and len(e)==104 and len(pairs)==104
                  and all(i!=j for i,j in pairs) and driver.connected(pairs,60)
                  and abs(n.proxy_daily_load.sum()-6000)<1e-7 and abs(n.overload_threshold.sum()-7500)<1e-7)
    for rep in range(10):
        g=graphs[f'grid_{rep:02d}'][0];b=graphs[f'ba_{rep:02d}'][0]
        check('paired graph demands capacities '+str(rep),g.equals(b))
    check('one grid topology repeated',len({driver.sha(OUT/graph_meta[f'grid_{i:02d}']['edges_path']) for i in range(10)})==1)
    check('ten distinct BA realizations',len({driver.sha(OUT/graph_meta[f'ba_{i:02d}']['edges_path']) for i in range(10)})==10)
    inputs=json.loads((OUT/'inputs/demand_and_shocks.json').read_text(encoding='utf-8'))
    jobs={j['job_id']:j for j in d['jobs']};runs=pd.read_csv(OUT/'02_runs.csv');logs=pd.read_csv(OUT/'03_rounds.csv')
    arcs=pd.read_csv(OUT/'04_initial_arcs.csv');log_groups={k:v.sort_values('round') for k,v in logs.groupby('job_id')}
    arc_groups={k:v for k,v in arcs.groupby('job_id')}
    check('full unique job grid',len(runs)==2424 and runs.job_id.is_unique and set(log_groups)==set(runs.job_id))
    for row in runs.itertuples():
        j=jobs[row.job_id];n,e=graphs[j['graph_id']];p=physics[j['graph_id']];ids=p['ids'];pos=p['pos'];K=n.overload_threshold.to_numpy(float)
        data=inputs[j['input_id']];D=np.asarray(data['parts']).sum(axis=0);current=set(data['initial_ids'])
        trace=log_groups[row.job_id];check('consecutive rounds '+str(row.job_id),list(trace['round'])==list(range(len(trace))))
        for step in trace.itertuples():
            u=np.asarray(json.loads(step.util_json));s=np.asarray(json.loads(step.source_completed_json))
            removed=np.array([x in current for x in ids]);exposed=removed[p['dst']]
            count=np.bincount(p['src'][exposed],minlength=len(ids))
            pressure=np.bincount(p['src'][exposed],weights=p['score'][exposed],minlength=len(ids))/.68/np.maximum(p['degree'],1)
            if j['domain']!='synthetic':
                if j['kind']=='spatial_cluster':pressure+=.1*n['区县'].isin(['东西湖区','新洲区','黄陂区']).to_numpy()
                if j['kind']=='channel_disruption':pressure+=.1*n.family.isin(['airport','rail','port']).to_numpy()
            O=(~removed)&(u>j['gamma']+1e-7);R=(~removed)&(count>0)&(pressure>=j['theta'])
            new={ids[i] for i in np.flatnonzero(O|R)}
            check('independent triggers '+str((row.job_id,step.round)),new==set(json.loads(step.new_ids_json))
                  and [int((O&~R).sum()),int((R&~O).sum()),int((O&R).sum())]==[step.overload_only,step.risk_only,step.both]
                  and step.failed_count==len(current))
            check('independent service and admission '+str((row.job_id,step.round)),np.isfinite(u).all() and np.isfinite(s).all()
                  and u.min()>=-1e-7 and u.max()<=j['rho']+1e-7 and s.min()>=-1e-7 and (s<=D+1e-6).all()
                  and np.max(abs(u[removed]),initial=0)<1e-7 and abs(np.minimum(u*K,K).sum()-s.sum())<1e-6
                  and abs(s.sum()-step.Served_Load)<1e-6 and abs(s.sum()/D.sum()-step.Service_Rate)<1e-9
                  and abs((u*K).sum()-step.Attempted_Assigned_Load)<1e-6
                  and abs(step.Served_Load+step.Undispatched_Load+step.Uncompleted_Assigned_Load-D.sum())<1e-6)
            if step.round==0:
                own=np.where(~removed,np.minimum(D,K),0.);outgoing=np.zeros(len(ids));incoming=outgoing.copy();source_done=own.copy()
                key=(j['graph_id'],frozenset(current))
                if key not in path_cache:
                    cost=p['direct'].copy()
                    for mid in np.flatnonzero(~removed):np.minimum(cost,p['direct'][:,mid,None]+p['direct'][mid,None,:],out=cost)
                    cost[:,removed]=np.inf;np.fill_diagonal(cost,np.inf)
                    diagonal=np.flatnonzero(~removed);cost[diagonal,diagonal]=0.;path_cache[key]=cost
                cost=path_cache[key]
                a=arc_groups.get(row.job_id)
                if a is not None:
                    si=np.array([pos[x] for x in a.source_id]);ti=np.array([pos[x] for x in a.target_id]);flow=a.flow.to_numpy(float)
                    check('initial path cost feasible '+str(row.job_id),np.isfinite(cost[si,ti]).all()
                          and np.allclose(cost[si,ti],a.cost,atol=1e-10,rtol=0) and flow.min()>=-1e-7)
                    outgoing=np.bincount(si,weights=flow,minlength=len(ids));incoming=np.bincount(ti,weights=flow,minlength=len(ids))
                load=own+incoming;fraction=np.divide(np.minimum(load,K),load,out=np.zeros_like(load),where=load>0)
                source_done=own*fraction
                if a is not None:
                    check('initial completed arcs '+str(row.job_id),np.allclose(flow*fraction[ti],a.completed_flow,atol=1e-6,rtol=0))
                    source_done+=np.bincount(si,weights=flow*fraction[ti],minlength=len(ids))
                check('initial arc accounting '+str(row.job_id),np.allclose(load,u*K,atol=1e-6,rtol=0)
                    and np.allclose(source_done,s,atol=1e-6,rtol=0) and (outgoing<=D-own+1e-6).all())
            current|=new
        check('stable endpoint '+str(row.job_id),trace.iloc[-1].new_failures==0 and current==set(json.loads(row.final_ids_json))
              and len(current)==row.final_failed and abs(trace.iloc[-1].Service_Rate-row.Service_Rate)<1e-9)
        if j['topology']=='grid' and j['theta']==.2:
            check('known grid risk percolation control '+str(row.job_id),row.final_failed==60 and row.Service_Rate<1e-9)
    # Exact reference to the prior precision version for the unchanged LP branch.
    old=pd.read_csv(driver.PRECISION/'02_new_runs.csv');regressions=[]
    for row in runs[(runs.domain=='geographic_proxy')&(runs.rule=='lexicographic_lp')].itertuples():
        j=jobs[row.job_id];name=f"external_x{j['external_multiplier']:g}";arm='C' if j['rho']==1. else 'B'
        ref=old[(old.capacity_id=='baseline')&(old.case_id==j['case_id'])&(old.arm==arm)&(old.gamma==j['gamma'])
             &(old.demand==name)&np.isclose(old.theta,j['theta'],atol=1e-12,rtol=0)]
        check('one prior LP target '+str(row.job_id),len(ref)==1)
        ok=abs(ref.iloc[0].Service_Rate-row.Service_Rate)<1e-9 and ref.iloc[0].final_ids_json==row.final_ids_json
        check('unchanged LP regression '+str(row.job_id),ok);regressions.append(dict(job_id=row.job_id,passed=bool(ok)))
    check('168 LP reference checks',len(regressions)==168)
    keys=['graph_id','domain','topology','replicate','case_id','input_id','control','theta','gamma','rho','total_multiplier']
    primary=runs[runs.primary];base=primary[primary.rule=='lexicographic_lp'][keys+['Service_Rate','initial_assigned','Transfer_Cost','final_failed']]
    base=base.rename(columns={k:'lp_'+k for k in ['Service_Rate','initial_assigned','Transfer_Cost','final_failed']})
    comparison=primary[primary.rule!='lexicographic_lp'].merge(base,on=keys,validate='many_to_one')
    comparison['service_delta_vs_lp_pp']=100*(comparison.Service_Rate-comparison.lp_Service_Rate)
    comparison['initial_assignment_gap_vs_lp']=comparison.lp_initial_assigned-comparison.initial_assigned
    check('fixed initial LP maximum assignment bound',(comparison.initial_assignment_gap_vs_lp>=-1e-6).all())
    # Reverse only equal-cost source priorities, retaining the canonical run as reference.
    reverse=runs[~runs.primary];near=primary[primary.rule=='nearest_feasible']
    ties=reverse.merge(near[keys+['Service_Rate','final_ids_json']],on=keys,suffixes=('','_canonical'),validate='one_to_one')
    ties['reverse_minus_canonical_pp']=100*(ties.Service_Rate-ties.Service_Rate_canonical)
    check('480 tie sensitivity conditions',len(ties)==480)
    cold=[]
    for job_id in d['cold_job_ids']:
        j=jobs[job_id];nn,ee=graphs[j['graph_id']];data=inputs[j['input_id']];cfg=driver.config(j)
        spec=DispatchRuleSpec(j['rule'],j['tie_order'],graph_meta[j['graph_id']]['compatibility'])
        a=DispatchRuleAllocator(nn,ee,cfg,spec)
        en=ConfiguredDualMechanismRecovery(a,ee,data['parts'],data['groups'],data['initial_ids'],j['kind'])
        f,m,_=en.closure(en.initial);target=runs[runs.job_id==job_id].iloc[0];trace=log_groups[job_id]
        ok=set(f)==set(json.loads(target.final_ids_json)) and abs(m['Service_Rate']-target.Service_Rate)<1e-9
        ok=ok and len(trace)==len(en.cascade_logs[en.initial]);current=set(en.initial)
        for saved,actual in zip(trace.itertuples(),en.cascade_logs[en.initial]):
            state=en.state(current);ok=ok and set(actual['new_ids'])==set(json.loads(saved.new_ids_json))
            ok=ok and np.allclose(state[1],json.loads(saved.util_json),atol=1e-7,rtol=0)
            ok=ok and np.allclose(state[2],json.loads(saved.source_completed_json),atol=1e-6,rtol=0)
            current.update(actual['new_ids'])
        check('cold complete trace '+str(job_id),ok);cold.append(dict(job_id=job_id,passed=bool(ok)));a.clear_path_cache()
    check('12 fixed cold starts',len(cold)==12)
    summaries=[]
    for identity,g in comparison.groupby(['domain','topology','theta','gamma','rho','total_multiplier','control','rule']):
        summaries.append(dict(zip(['domain','topology','theta','gamma','rho','total_multiplier','control','rule'],identity),
            conditions=len(g),mean_delta_pp=g.service_delta_vs_lp_pp.mean(),median_delta_pp=g.service_delta_vs_lp_pp.median(),
            min_delta_pp=g.service_delta_vs_lp_pp.min(),max_delta_pp=g.service_delta_vs_lp_pp.max(),
            higher=int((g.service_delta_vs_lp_pp>1e-7).sum()),equal=int((abs(g.service_delta_vs_lp_pp)<=1e-7).sum()),
            lower=int((g.service_delta_vs_lp_pp< -1e-7).sum()),mean_initial_assignment_gap=g.initial_assignment_gap_vs_lp.mean()))
    summary=pd.DataFrame(summaries)
    replicate=comparison[comparison.domain=='synthetic'].groupby(['topology','replicate','rule','theta','rho'],as_index=False).agg(
        mean_delta_pp=('service_delta_vs_lp_pp','mean'),mean_initial_gap=('initial_assignment_gap_vs_lp','mean'))
    for name,value in [('06_rule_vs_lp.csv',comparison),('07_summary.csv',summary),('08_paired_synthetic_replicates.csv',replicate),
          ('09_tie_sensitivity.csv',ties),('10_lp_regression.csv',pd.DataFrame(regressions)),('11_cold_checks.csv',pd.DataFrame(cold)),
          ('12_independent_checks.csv',pd.DataFrame(CHECKS))]:driver.save(value,OUT/name)
    findings=dict(completed=True,closures=2424,run_checks=done['checks'],independent_checks=len(CHECKS),
       independent_passed=sum(x['passed'] for x in CHECKS),lp_regressions=168,cold_starts=12,
       tie_conditions_changed_service=int((abs(ties.reverse_minus_canonical_pp)>1e-7).sum()),
       max_abs_tie_service_change_pp=float(abs(ties.reverse_minus_canonical_pp).max()),
       external_business_evidence=True,quantitative_external_validation=False,wuhan_permissions_verified=False,
       synthetic_unique_grid_topologies=1,synthetic_unique_ba_topologies=10,empirical_samples=0)
    driver.dump(findings,OUT/'13_findings.json');report(comparison,summary,replicate,ties,findings)
    driver.verify(d)
    driver.dump(dict(**findings,files={str(p.relative_to(OUT)):driver.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final_manifest.json'}),OUT/'final_manifest.json')
    print(json.dumps(findings,ensure_ascii=False),flush=True)


def report(comparison,summary,replicate,ties,findings):
    compact=[]
    # Keep geography, synthetic topology, risk and admission limit distinct.
    for identity,g in comparison[~comparison.control].groupby(['domain','topology','theta','gamma','rho','rule']):
        compact.append(dict(zip(['domain','topology','theta','gamma','rho','rule'],identity),conditions=len(g),
              mean_delta_pp=g.service_delta_vs_lp_pp.mean(),min_delta_pp=g.service_delta_vs_lp_pp.min(),max_delta_pp=g.service_delta_vs_lp_pp.max(),
              higher=int((g.service_delta_vs_lp_pp>1e-7).sum()),equal=int((abs(g.service_delta_vs_lp_pp)<=1e-7).sum()),lower=int((g.service_delta_vs_lp_pp< -1e-7).sum())))
    compact=pd.DataFrame(compact)
    text='''# 同一派入层规则、外部业务证据与合成拓扑补充

2026-10-20。本轮补三项证据，定量外部验证尚未闭合。没有改稿件，没有新增UE、识别或预测主张。

## 业务对象与相容性

同一运营方、经授权、指定业务的设施间改派有运营方制度及河南/贵州实例。河南资料明确指定来源/目的地邮件由郑州调整到洛阳，以及特定邮路由驻马店调整至信阳。具体来源、权限和限制见“业务相容性与外部证据边界”及“外部业务关系登记”。
这些材料不提供武汉246节点互替许可。武汉图仍是地理资料支撑的原代理情景；合成图假定各节点处理同一类可拆分需求。未把同family、相邻或同企业自动视为充分许可。

## 同层比较设计

三规则只改变派入：剩余接纳空间比例同时提案/缩放/迭代；最近可行接收点按全弧成本及ID并列贪心；原最大派入—最小超额—最小成本LP。名义本地优先、比例完成、可达、风险、rho及gamma均按匹配条件一致。
启发式不保证最大派入，06表专列共同初态相对LP的派入缺口；不能把这部分缺口宣称为新算法发现。终态比较允许任一规则更高/持平/更低，不预设LP因逐轮最优就终态普遍最优。

504次武汉代理闭包；合成1440主闭包及480同成本并列反转。下表为同rho等条件下“启发式减LP”的服务百分点；需求分层和无损伤控制在07表完整保留。此摘要是设计等权，非真实发生概率。

'''+md(compact,list(compact.columns))+'''

## 合成拓扑验收

BA与6×10开边界格子均60节点104边，同一重复配对负荷、容量、损伤ID、单位边成本。BA从15节点路径开始、每新增节点按度连接两个旧节点，不作后续删边；十个不同生成种子。只有一张格子拓扑、十套配对输入，不能称十个独立格子网络。
风险中性化为失效邻居比例，无场景bonus；与武汉风险编码不同，不能将跨域幅度直接解释为纯网络类型因果。
格子度至多4，theta=.2时单个失效邻居即可触发风险，非空起点在连通图上全退是预先已知结构控制；不能将该格“BA胜格子”包装为现实发现。其余阈值和全部正负结果保留。

## 来源并列顺序

'''+f"480个预定最近规则并列反转条件中，有{findings['tie_conditions_changed_service']}格服务变化超过1e-7pp，最大绝对变化{findings['max_abs_tie_service_change_pp']:.6f}pp。"+'''
差异为同一规则实现中的并列敏感性，不增加另一种业务机制。09表包含全部条件，不仅保存变化最大的点。没有审计LP所有退化最优解。

## 外部验证状态

运营方与政府来源提供外部业务事实核验，支持权限、指定邮件改派和能力评估先行。公开分流实绩只登记为观测值，没有用它反推能力再拟合同一数字。
缺同期完整OD、可承接能力、许可和实际完成/失效结果，故本轮不能验收定量政策外部验证或真实失效预测准确性。没有用合成图、真实坐标或省级妥投率填补这一缺口。后续需同一事件和服务口径的输入与结果，业务相容性核验也不能由图模型自动补出。

## 验证与使用

'''+f"{findings['lp_regressions']}个原LP条件回归已核验容量精度版；{findings['independent_passed']}/{findings['independent_checks']}项独立检查通过，12次预定冷启动逐轮复现。"+'''
独立重建风险触发、接收与来源守恒、初始路径费用及派入弧；手算比例/最近差异、非最大流反例、节点重排、显式权限和缓存隔离均已检查。检查项不是独立样本。

原代码、数据与稿件保持不变。新入口dispatch_policy_rules.py要求显式DispatchRuleSpec，原ModelConfig及历史调用行为不变。新增只接受比例完成以隔离派入层，未把它命名为UE。
执行119_dispatch_rule_study.py design/run，再运行120_validate_dispatch_rules.py；已有设计不会覆盖。全部来源、输入、配置和结果由00设计、completion及final_manifest记录。
'''
    (OUT/'同层派入规则与外部证据及合成拓扑_结果_20261020.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()

