"""Independent accounting, trigger reconstruction, controls and bounded report."""
import importlib
import itertools
import json
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
from configured_dual_mechanism_v5 import ConfiguredDualMechanismRecovery
from dispatch_policy_rules import DispatchRuleAllocator,DispatchRuleSpec
from matched_admission_v5 import matched_plans,intervene

driver=importlib.import_module('122_matched_location_study')
OUT=driver.OUT;PRIOR=driver.PRIOR;CHECKS=[];TOL=1e-7


def check(name,ok):
    CHECKS.append(dict(check=name,passed=bool(ok)))
    if not ok:
        driver.save(CHECKS,OUT/'validation_failed.csv')
        raise AssertionError(name)


def physics(n,e,cfg):
    ids=list(n.node_id);pos={k:i for i,k in enumerate(ids)}
    i=e.Source_id.map(pos).to_numpy(int);j=e.Target_id.map(pos).to_numpy(int)
    src=np.r_[i,j];dst=np.r_[j,i];degree=np.bincount(src,minlength=len(n))
    v=cfg['values'];w=v['risk_weights']
    p=np.tile(e.Failure_probability_proxy.to_numpy(float),2)
    cross=np.tile(e.Cross_district.to_numpy(float),2)
    cost=np.tile(np.maximum(e.Composite_transport_cost.to_numpy(float),v['risk_cost_floor']),2)
    cap=np.tile(e.Edge_capacity_proxy.to_numpy(float),2)
    terms=[np.full(len(src),w[0]),w[1]*p,w[2]*cross,w[3]*np.minimum(cost/v['risk_cost_scale'],1.),
           w[4]*(1-np.minimum(cap/v['risk_capacity_scale'],1.))]
    direct=np.full((len(n),len(n)),np.inf)
    for r in e.itertuples():
        a,b=pos[r.Source_id],pos[r.Target_id];direct[a,b]=direct[b,a]=max(float(r.Composite_transport_cost),.001)
    return dict(ids=ids,pos=pos,src=src,dst=dst,degree=degree,terms=terms,direct=direct)


def trigger(info,n,failed,u,cfg,kind,mechanism):
    v=cfg['values'];removed=np.array([s in failed for s in info['ids']]);mask=removed[info['dst']]
    count=np.bincount(info['src'][mask],minlength=len(n))
    terms=[np.bincount(info['src'][mask],weights=x[mask],minlength=len(n))/np.maximum(info['degree'],1)
           /v['pressure_normalizer'] for x in info['terms']]
    pressure=np.sum(terms,axis=0)
    if kind=='spatial_cluster':pressure+=v['bonus_amplitude']*n['区县'].isin(v['spatial_districts']).to_numpy()
    if kind=='channel_disruption':pressure+=v['bonus_amplitude']*n.family.isin(v['channel_families']).to_numpy()
    over_enabled,risk_enabled=driver.MECHANISMS[mechanism]
    over=(~removed)&over_enabled&(u>v['gamma']+v['overload_epsilon'])
    risk=(~removed)&risk_enabled&(count>0)&(pressure>=v['theta_risk'])
    ids=lambda z:sorted(info['ids'][i] for i in np.flatnonzero(z))
    return dict(new=ids(over|risk),overload_only=ids(over&~risk),risk_only=ids(risk&~over),both=ids(over&risk))


def paths(info,failed,hops):
    active=np.array([s not in failed for s in info['ids']])
    direct=info['direct'];out=direct.copy() if hops else np.full_like(direct,np.inf)
    for _ in range(1,hops):
        previous=out.copy()
        for mid in np.flatnonzero(active):out=np.minimum(out,previous[:,mid,None]+direct[mid,None,:])
    out[:,~active]=np.inf;np.fill_diagonal(out,np.inf);ix=np.flatnonzero(active);out[ix,ix]=0.
    return out


def rational_witness():
    # Exact rational enumeration separates the mathematical theta=2/3 boundary
    # from binary .68 normalization in legacy implementation.
    adjacency={'s':{'a','b'},'a':{'s','b','c'},'b':{'s','a'},'c':{'a'}}
    def close(seed,theta):
        F=set(seed)
        while True:
            new={j for j in adjacency if j not in F and len(adjacency[j]&F)>0
                 and Fraction(len(adjacency[j]&F),len(adjacency[j]))>=theta}
            if not new:return F
            F|=new
    for t in [Fraction(1,2),Fraction(2,3),Fraction(3,4),Fraction(1,1)]:
        ra=close({'s','a'},t);rb=close({'s','b'},t)
        check('exact rational risk seed a '+str(t),len(ra)==4)
        check('exact rational risk seed b '+str(t),len(rb)==(4 if t<=Fraction(2,3) else 2))
    for q in [Fraction(5,4),Fraction(3,2),Fraction(7,4),Fraction(2,1)]:
        check('exact initial matching '+str(q),27+q-(q-1)==28 and 18+q<=20)


def md(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    d=json.loads((OUT/'00_design.json').read_text(encoding='utf-8'));driver.verify(d)
    completed=json.loads((OUT/'completion.json').read_text(encoding='utf-8'))
    check('full run completed',completed['completed'] and completed['conditions']==648 and completed['interventions']==10368)
    for path,h in completed['files'].items():check('frozen result '+path,driver.old.sha(OUT/path)==h)
    rational_witness()
    rows=pd.read_csv(OUT/'03_interventions.csv');initial=pd.read_csv(OUT/'04_initial_plans.csv')
    rounds=pd.read_csv(OUT/'05_continuation_rounds.csv')
    check('full unique interventions',len(rows)==10368 and not rows.duplicated(['condition_id','mechanism','plan']).any())
    check('full unique first plans',len(initial)==2592 and not initial.duplicated(['condition_id','plan']).any())
    trace={key:v.sort_values('round') for key,v in rounds.groupby(['condition_id','mechanism','plan'])}
    graphs={g['graph_id']:(pd.read_csv(PRIOR/g['nodes_path']),pd.read_csv(PRIOR/g['edges_path']),g) for g in d['graphs']}
    prior_runs=pd.read_csv(PRIOR/'02_runs.csv').set_index('job_id')
    prior_arcs=pd.read_csv(PRIOR/'04_initial_arcs.csv');prior_arcs={k:g for k,g in prior_arcs.groupby('job_id')}
    inputs=json.loads((PRIOR/'inputs/demand_and_shocks.json').read_text(encoding='utf-8'))
    allplans={};jobmap={j['condition_id']:j for j in d['jobs']};firstsets={}
    for j in d['jobs']:
        cid=j['condition_id'];n,e,meta=graphs[j['graph_id']];K=n.overload_threshold.to_numpy(float)
        payload=json.loads((OUT/'plans'/f'{cid:04d}.json').read_text(encoding='utf-8'));allplans[cid]=payload
        cfg=payload['config'];info=physics(n,e,cfg);D=np.asarray(payload['demand']);F0=set(payload['initial_ids'])
        active=np.array([s not in F0 for s in info['ids']]);own=np.where(active,np.minimum(D,K),0.)
        avail=paths(info,F0,cfg['values']['max_hops']);summaries=[]
        old=prior_runs.loc[j['job_id']];oq=np.zeros(len(n))
        if j['job_id'] in prior_arcs:
            for r in prior_arcs[j['job_id']].itertuples():oq[info['pos'][r.source_id]]+=r.flow
        for name,p in payload['plans'].items():
            src=np.asarray(p['src'],int);dst=np.asarray(p['dst'],int);flow=np.asarray(p['flow']);nnode=len(n)
            outgoing=np.bincount(src,weights=flow,minlength=nnode);incoming=np.bincount(dst,weights=flow,minlength=nnode)
            load=own+incoming;fraction=np.divide(np.minimum(load,K),load,out=np.zeros(nnode),where=load>0)
            source=own*fraction+np.bincount(src,weights=flow*fraction[dst],minlength=nnode)
            E=float(np.maximum(load-K,0.).sum());C=float(np.minimum(load,K).sum());A=float(own.sum()+flow.sum())
            tag=str((cid,name))
            check('first feasibility '+tag,flow.min(initial=0.)>=-1e-9
                and (outgoing<=D-own+1e-5).all() and (load<=j['rho']*K+1e-5).all()
                and np.isfinite(avail[src[flow>1e-8],dst[flow>1e-8]]).all())
            check('first independent reconstruction '+tag,np.allclose(p['load'],load,atol=1e-7,rtol=0)
                and np.allclose(p['own'],own,atol=1e-7,rtol=0) and np.allclose(p['q'],outgoing,atol=1e-7,rtol=0)
                and np.allclose(p['completed_source'],source,atol=1e-7,rtol=0)
                and np.allclose(p['completed_receiver'],np.minimum(load,K),atol=1e-7,rtol=0))
            check('first objective matching '+tag,abs(p['assigned']-A)<1e-6 and abs(p['served']-C)<1e-6
                and abs(p['excess']-E)<1e-6 and abs(C-(A-E))<1e-6
                and abs(E-old.initial_nominal_excess)<1e-5 and abs(A-old.initial_assigned)<1e-5
                and np.max(abs(outgoing-oq))<1e-5)
            valid=flow>1e-12
            check('first path cost '+tag,abs(float(np.dot(flow[valid],avail[src[valid],dst[valid]]))-p['cost'])<1e-5)
            rank=np.array([sorted(info['ids']).index(v) for v in info['ids']])
            score=float(np.dot(flow,rank[dst]/max(1,len(n)-1)))
            summaries.append(dict(plan=name,peak=float((load/K).max()),score=score,cost=p['cost']))
            for mechanism in driver.MECHANISMS:
                result=rows[(rows.condition_id==cid)&(rows.plan==name)&(rows.mechanism==mechanism)].iloc[0]
                first=trigger(info,n,F0,load/K,cfg,j['kind'],mechanism)
                for label in ['new','overload_only','risk_only','both']:
                    check('first cause '+str((cid,name,mechanism,label)),first[label]==json.loads(result['first_'+label+'_json']))
                F1=sorted(F0|set(first['new']));firstsets[(cid,mechanism,name)]=F1
                check('first update '+str((cid,name,mechanism)),F1==json.loads(result.continuation_start_json))
                current=set(F1);logs=trace[(cid,mechanism,name)]
                check('consecutive continuation '+str((cid,name,mechanism)),list(logs['round'])==list(range(len(logs))))
                for step in logs.itertuples():
                    u=np.asarray(json.loads(step.util_json));s=np.asarray(json.loads(step.source_completed_json))
                    removed=np.array([v in current for v in info['ids']]);new=trigger(info,n,current,u,cfg,j['kind'],mechanism)
                    check('continuation trigger '+str((cid,name,mechanism,step.round)),sorted(current)==json.loads(step.current_ids_json)
                        and new['new']==json.loads(step.new_ids_json) and step.new_failures==len(new['new'])
                        and step.overload_only==len(new['overload_only']) and step.risk_only==len(new['risk_only'])
                        and step.both==len(new['both']))
                    check('continuation physical '+str((cid,name,mechanism,step.round)),u.min()>=-1e-8
                        and u.max()<=j['rho']+1e-7 and np.max(abs(u[removed]),initial=0.)<1e-7
                        and s.min()>=-1e-7 and (s<=D+1e-5).all()
                        and abs(np.minimum(u*K,K).sum()-s.sum())<1e-5
                        and abs(s.sum()-step.Served_Load)<1e-5
                        and abs(s.sum()/D.sum()-step.Service_Rate)<1e-9)
                    current|=set(new['new'])
                last=logs.iloc[-1]
                check('stable terminal '+str((cid,name,mechanism)),json.loads(last.new_ids_json)==[]
                    and sorted(current)==json.loads(result.final_failed_json)
                    and abs(last.Service_Rate-result.final_service)<1e-10
                    and abs(last.Served_Load-result.final_served)<1e-6)
        summary=pd.DataFrame(summaries).set_index('plan')
        check('objective directions '+str(cid),summary.loc['nearest_cost','cost']<=summary.cost.min()+1e-5
            and summary.loc['balanced_util','peak']<=summary.peak.min()+1e-7
            and summary.loc['low_id_probe','score']<=summary.score.min()+1e-5
            and summary.loc['high_id_probe','score']>=summary.score.max()-1e-5)
    # Verify the state-sufficiency result for all six plan pairs, not just the
    # two contrasts selected for the results table.
    for (cid,mechanism),group in rows.groupby(['condition_id','mechanism']):
        for (_,a),(_,b) in itertools.combinations(group.iterrows(),2):
            same_start=a.continuation_start_json==b.continuation_start_json
            check('state sufficiency '+str((cid,mechanism,a.plan,b.plan)),not same_start or
                (a.final_failed_json==b.final_failed_json and abs(a.final_service-b.final_service)<TOL))
        j=jobmap[cid];E=allplans[cid]['plans']['nearest_cost']['excess']
        if mechanism in ['risk_only','neither'] or j['rho']<=j['gamma']+1e-7 or E<1e-7:
            check('no-effect theorem '+str((cid,mechanism)),group.final_service.max()-group.final_service.min()<TOL
                and group.final_failed_json.nunique()==1 and group.continuation_start_json.nunique()==1)
    # Fixed cold starts cover first and last condition in both domains and all
    # four initial plans (16 comparisons). Rebuild both the optimal face and LP.
    cold=[]
    for domain in ['geographic_proxy','synthetic']:
        subset=[j for j in d['jobs'] if j['domain']==domain]
        for j in [subset[0],subset[-1]]:
            cid=j['condition_id'];n,e,meta=graphs[j['graph_id']];data=inputs[j['input_id']]
            cfg=driver.config_for(j,'both');a=DispatchRuleAllocator(n,e,cfg,DispatchRuleSpec(compatibility_scope=meta['compatibility']))
            plans,_=matched_plans(a,set(data['initial_ids']),np.asarray(data['parts']).sum(axis=0))
            en=ConfiguredDualMechanismRecovery(a,e,data['parts'],data['groups'],data['initial_ids'],j['kind'])
            for name,p in plans.items():
                new=intervene(en,p);stored=rows[(rows.condition_id==cid)&(rows.mechanism=='both')&(rows.plan==name)].iloc[0]
                ok=new['final_failed']==json.loads(stored.final_failed_json) and abs(new['final_service']-stored.final_service)<TOL
                check('cold start '+str((cid,name)),ok);cold.append(dict(condition_id=cid,plan=name,passed=bool(ok)))
            a.clear_path_cache()
    driver.save(cold,OUT/'06_cold_starts.csv')
    # All signed effects are second minus first; no best-of-probe selection.
    effects=[]
    indexcols=['condition_id','old_job_id','graph_id','domain','topology','replicate','case_id','control','total_multiplier','gamma','rho','theta','mechanism']
    for contrast,(left,right) in d['contrasts'].items():
        aa=rows[rows.plan==left];bb=rows[rows.plan==right]
        merged=aa.merge(bb,on=indexcols,suffixes=('_left','_right'),validate='one_to_one')
        for r in merged.to_dict('records'):
            delta=100*(r['final_service_right']-r['final_service_left'])
            cid=r['condition_id'];payload=allplans[cid]
            l=payload['plans'][left];h=payload['plans'][right]
            effects.append(dict(**{k:r[k] for k in indexcols},contrast=contrast,left=left,right=right,
                delta_pp=delta,changed=abs(delta)>100*TOL,
                first_seed_changed=r['continuation_start_json_left']!=r['continuation_start_json_right'],
                final_set_changed=r['final_failed_json_left']!=r['final_failed_json_right'],
                initial_excess=l['excess'],initial_served=l['served'],initial_assigned=l['assigned'],
                first_load_maxdiff=float(np.max(abs(np.asarray(l['load'])-np.asarray(h['load'])))),
                final_left_pp=100*r['final_service_left'],final_right_pp=100*r['final_service_right']))
    effects=pd.DataFrame(effects);driver.save(effects,OUT/'07_paired_effects.csv')
    summary=[]
    for keys,g in effects.groupby(['contrast','domain','mechanism'],sort=False):
        summary.append(dict(contrast=keys[0],domain=keys[1],mechanism=keys[2],pairs=len(g),
            improved=int((g.delta_pp>100*TOL).sum()),tied=int((abs(g.delta_pp)<=100*TOL).sum()),
            worsened=int((g.delta_pp<-100*TOL).sum()),mean_delta_pp=float(g.delta_pp.mean()),
            max_abs_delta_pp=float(abs(g.delta_pp).max()),first_seed_different=int(g.first_seed_changed.sum()),
            final_set_different=int(g.final_set_changed.sum())))
    summary=pd.DataFrame(summary);driver.save(summary,OUT/'08_effect_summary.csv')
    strata=effects.groupby(['contrast','domain','topology','rho','gamma','theta','total_multiplier','mechanism'],dropna=False).agg(
        pairs=('changed','size'),changed=('changed','sum'),mean_delta_pp=('delta_pp','mean'),min_delta_pp=('delta_pp','min'),max_delta_pp=('delta_pp','max')).reset_index()
    driver.save(strata,OUT/'09_stratified_effects.csv')
    repeat=effects[effects.domain=='synthetic'].groupby(['contrast','topology','replicate','rho','mechanism']).agg(
        conditions=('changed','size'),changed=('changed','sum'),mean_delta_pp=('delta_pp','mean')).reset_index()
    driver.save(repeat,OUT/'10_synthetic_paired_replicates.csv')
    inter=effects.pivot(index=['condition_id','contrast'],columns='mechanism',values='delta_pp').reset_index()
    inter['risk_interaction_pp']=inter['both']-inter['overload_only']-inter['risk_only']+inter['neither']
    driver.save(inter,OUT/'11_feedback_interaction.csv')
    # Select the first positive/negative primary condition by frozen condition
    # order, not largest effect. This is an illustrative trace, not a held-out test.
    examples=[]
    for contrast in ['primary','secondary']:
        changed=effects[(effects.contrast==contrast)&(effects.mechanism=='both')&effects.changed].sort_values('condition_id')
        for sign in [1,-1]:
            subset=changed[sign*changed.delta_pp>100*TOL]
            if len(subset):examples.append(subset.iloc[0].to_dict())
    driver.dump(examples,OUT/'12_trace_examples.json')
    driver.save(CHECKS,OUT/'13_independent_checks.csv')
    p=effects[(effects.contrast=='primary')&(effects.mechanism=='both')]
    s=effects[(effects.contrast=='secondary')&(effects.mechanism=='both')]
    findings=dict(completed=True,conditions=648,interventions=10368,initial_plans=2592,
        analytic_checks=364,witness_runs=368,independent_checks=len(CHECKS),passed=sum(c['passed'] for c in CHECKS),cold_starts=len(cold),
        primary_changed=int(p.changed.sum()),primary_mean_delta_pp=float(p.delta_pp.mean()),primary_max_abs_delta_pp=float(abs(p.delta_pp).max()),
        secondary_changed=int(s.changed.sum()),secondary_max_abs_delta_pp=float(abs(s.delta_pp).max()),
        primary_changed_by_domain=p.groupby('domain').changed.sum().astype(int).to_dict(),
        all_no_feedback_controls_zero=bool((effects[effects.mechanism.isin(['risk_only','neither'])].changed==False).all()),
        comparison='Only first-step positions differ; same per-source maximum admission and minimum excess; common legacy LP continuation',
        empirical_validation=False,novelty_established=False)
    driver.dump(findings,OUT/'14_findings.json')
    shown=summary[summary.mechanism.isin(['both','overload_only'])].copy()
    report=f'''# 等接纳量的位置效应：完整结果与使用边界

## 本轮回答了什么

四节点族证明位置可改变终态，且首次逐来源接纳、总完成、总超额和成本均相同。该例初态超额可避免，不是最小超额最优面证明。完整命题、参数区间、反例及精确算术边界见同目录证明文档。

更严格对照沿用上一轮全部648个初态。固定原LP最大接纳的逐来源向量，再固定最小超额，因此首次总完成量也一致。四位置方案×四机制开关共10368个干预轨迹；2592个首次计划。首次失效更新后全部采用同一原LP，只研究单次位置干预。

## 冻结对照结果

主对照为峰值均衡减最近成本；次对照为高ID减低ID（任意位置敏感性探针）。正负数不代表现实发生概率或一般政策优劣。

{md(shown)}

主对照双机制下648条件有{findings['primary_changed']}个非零服务差，最大绝对差{findings['primary_max_abs_delta_pp']:.6f}个百分点；次对照有{findings['secondary_changed']}个，最大绝对差{findings['secondary_max_abs_delta_pp']:.6f}个百分点。全部正负与零结果见07表。按需求、阈值、rho、gamma及拓扑分层见09表，合成重复按10表计数，不把一张格子的十套输入说成十张格子。

只风险/都关闭的全部对照均为零。rho<=gamma+epsilon及初态最小超额为零的条件均没有位置干预效应，符合状态充分性证明。不同首次种子也可能在共同后续算法下汇合；首次过载位置不同不是终态服务差的充分条件。

## 机制交互如何解释

11表为同一条件的位置效应在四开关下的差中差，即双机制−仅过载−仅风险＋都关闭。这里后两项为零，所以数值等于双机制位置差减仅过载位置差。它可以正、负或零，不能预设“风险必然放大”。该交互是确定性配对仿真的模型内对照，不能直接解释为真实灾害中的因果效应。

## 已排除和未排除的解释

排除了首次多接货、换来源构成、首次多完成及初态超额总量不同。共同可达、能力、风险和后续算法均未更改。后续接纳量没有强行固定，它是反馈造成的结果；不声称每轮或累计工作量一致。首次来源完成归属未匹配，但它不进入V5后续状态。

两个ID目标仅展示最优面位置敏感性，不是可部署的调度策略。峰值均衡仍是标准优化目标；即使出现差异，也不自动构成新算法或新理论。武汉仍是246混合设施代理联系，不是实测相容邮政网；合成检验不替代外部定量验证。

## 验收

四节点族368轨迹、364项核验；完整实验{len(CHECKS)}项独立检查全部通过，16次固定冷启动一致。独立重建首次流量守恒、逐来源匹配、总超额与完成、路径成本、逐轮风险与过载触发，并检查同首次失效集必同终态。原数据、旧源码及V5稿件的冻结哈希均保持一致。

是否足以支撑新的论文贡献，还要结合效应分布和已有文献判断；本轮未作创新性已成立或中科院分区可录用的承诺。
'''
    (OUT/'等接纳量位置反馈_结果与论文边界_20261020.md').write_text(report,encoding='utf-8')
    driver.verify(d)
    driver.dump(dict(completed=True,original_files_unchanged=True,**{k:v for k,v in findings.items() if k!='completed'},
        files={str(p.relative_to(OUT)):driver.old.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['delivery_manifest.json']},
        sources={str(p):driver.old.sha(p) for p in [Path(__file__),driver.CODE/'matched_admission_v5.py',Path(driver.__file__)]}),
        OUT/'delivery_manifest.json')
    print(json.dumps(findings,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
