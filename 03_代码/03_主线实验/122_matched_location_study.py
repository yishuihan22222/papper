"""Matched-admission one-step interventions, analytic witness and frozen grid."""
import argparse
from dataclasses import replace
import importlib
import itertools
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import scipy
from model_config import ModelConfig
from configured_dual_mechanism_v5 import ConfiguredDualMechanismRecovery
from dispatch_policy_rules import DispatchRuleAllocator, DispatchRuleSpec
from matched_admission_v5 import PLAN_NAMES, evaluate_flow, matched_plans, intervene

old=importlib.import_module('119_dispatch_rule_study')
CODE=Path(__file__).parent;ROOT=CODE.parents[1]
OUT=ROOT/'项目结果/matched_location_feedback_20261020'
PRIOR=old.OUT
MECHANISMS=dict(both=(True,True),overload_only=(True,False),risk_only=(False,True),neither=(False,False))


def dump(v,p):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save(v,p):pd.DataFrame(v).to_csv(p,index=False,encoding='utf-8-sig')


def witness_tables(q=2.):
    ids=['s','a','b','c'];D=np.array([q,9.,9.,9.])
    n=pd.DataFrame(dict(node_id=ids,名称=ids,区县='witness',family='same_service',
                        proxy_daily_load=D,overload_threshold=[10.,10.,10.,20.]))
    e=pd.DataFrame([dict(Source_id=i,Target_id=j,Composite_transport_cost=1.,Failure_probability_proxy=0.,
        Cross_district=0.,Edge_capacity_proxy=55.,Repair_cost_proxy=1.) for i,j in [('s','a'),('s','b'),('a','b'),('a','c')]])
    return n,e,D


def analytic():
    OUT.mkdir(parents=True,exist_ok=True)
    n,e,D=witness_tables();save(n,OUT/'witness_nodes.csv');save(e,OUT/'witness_edges.csv')
    results=[];checks=[]
    def check(name,ok):
        checks.append(dict(check=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    # A parameter family, including boundaries and saturation controls.
    for q,gamma,theta in itertools.product([.5,1.,1.25,1.5,1.75,2.],
            [1.,1.025,1.05,1.075,1.1],[.5,2/3,.68,.75,1.,1.01]):
        nn,ee,DD=witness_tables(q)
        cfg=ModelConfig(theta_risk=theta,gamma=gamma,risk_weights=(.68,0.,0.,0.,0.),bonus_amplitude=0.)
        a=DispatchRuleAllocator(nn,ee,cfg,DispatchRuleSpec(compatibility_scope='synthetic_same_service'))
        en=ConfiguredDualMechanismRecovery(a,ee,[DD],['witness'],['s'],'random')
        pair=[]
        for target in ['a','b']:
            p=evaluate_flow(a,{'s'},DD,[0],[a.pos[target]],[q],target);r=intervene(en,p)
            pair.append((p,r));results.append(dict(q=q,gamma=gamma,theta=theta,mechanism='both',target=target,
                 initial_assigned=p.assigned,initial_served=p.served,initial_excess=p.excess,initial_cost=p.cost,**r))
        p0,r0=pair[0];p1,r1=pair[1]
        check('witness matched '+str((q,gamma,theta)),np.array_equal(p0.q,p1.q)
              and p0.assigned==p1.assigned and p0.served==p1.served and p0.excess==p1.excess and p0.cost==p1.cost)
        crossing=.9+q/10>gamma+cfg.overload_epsilon
        if 2/3<theta<=1 and crossing:
            check('positive family '+str((q,gamma,theta)),r0['final_served']==0 and abs(r1['final_served']-DD.sum())<1e-7)
        if not crossing:
            check('no initial crossing '+str((q,gamma,theta)),r0['final_failed']==r1['final_failed']
                  and abs(r0['final_served']-r1['final_served'])<1e-7)
        # The exact equality theta=2/3 is a numerical boundary diagnostic:
        # legacy summation of .68 terms can yield the adjacent lower float.
        # Do not change its comparison tolerance or call it an exact-rational test.
        if theta<2/3-1e-12:
            check('risk saturation '+str((q,gamma,theta)),len(r0['final_failed'])==len(r1['final_failed'])==4)
        a.clear_path_cache()
    # At the main witness point, independently switch both failure channels.
    for mechanism,(over,risk) in MECHANISMS.items():
        cfg=ModelConfig(theta_risk=.75,gamma=1.05,risk_weights=(.68,0.,0.,0.,0.),bonus_amplitude=0.,
                        overload_enabled=over,risk_enabled=risk)
        a=DispatchRuleAllocator(n,e,cfg,DispatchRuleSpec(compatibility_scope='synthetic_same_service'))
        en=ConfiguredDualMechanismRecovery(a,e,[D],['witness'],['s'],'random')
        values=[]
        for target in ['a','b']:
            p=evaluate_flow(a,{'s'},D,[0],[a.pos[target]],[2.],target);r=intervene(en,p);values.append(r['final_served'])
            results.append(dict(q=2.,gamma=1.05,theta=.75,mechanism=mechanism,target=target,
                 initial_assigned=p.assigned,initial_served=p.served,initial_excess=p.excess,initial_cost=p.cost,**r))
        expected=dict(both=[0.,29.],overload_only=[18.,29.],risk_only=[29.,29.],neither=[29.,29.])[mechanism]
        check('main witness ablation '+mechanism,np.allclose(values,expected,atol=1e-7,rtol=0))
        a.clear_path_cache()
    # A different graph shows that distinct overload seeds can lead to the same
    # final failed set but different load allocations at the first step are not
    # enough: the common continuation erases all history of the first plan.
    save([dict(r,continuation_logs=json.dumps(r['continuation_logs'],ensure_ascii=False)) for r in results],OUT/'01_witness_runs.csv')
    save(checks,OUT/'02_witness_checks.csv')
    dump(dict(checks=len(checks),passed=sum(c['passed'] for c in checks),runs=len(results)),OUT/'analytic_complete.json')
    print(json.dumps(dict(analytic_checks=len(checks),witness_runs=len(results),all_passed=True)),flush=True)


def design():
    if (OUT/'00_design.json').exists():raise FileExistsError('Design already frozen')
    analytic()
    d=json.loads((PRIOR/'00_design.json').read_text(encoding='utf-8'));old.verify(d)
    jobs=[dict(j,condition_id=i) for i,j in enumerate(j for j in d['jobs'] if j['primary'] and j['rule']=='lexicographic_lp')]
    assert len(jobs)==648
    for folder in ['shards','plans']:(OUT/folder).mkdir(exist_ok=True)
    protected={str(p):old.sha(p) for p in CODE.glob('*.py')}
    protected.update({str(p):old.sha(p) for p in [PRIOR/'00_design.json',PRIOR/'design_lock.json',
        PRIOR/'02_runs.csv',PRIOR/'delivery_manifest.json',
        ROOT.parent/'中文论文初稿_全图美化与正文优化V5_20261019.docx']})
    inputfiles={str(p):old.sha(p) for p in (PRIOR/'inputs').iterdir() if p.is_file()}
    design=dict(schema='matched-first-step-location-1',jobs=jobs,conditions=648,plans=list(PLAN_NAMES),
        mechanisms={k:list(v) for k,v in MECHANISMS.items()},intervention_runs=648*4*4,
        question='Can first-step receiver location change closure after matching every source admission and minimum total excess?',
        matching=['initial failed set, D, K, rho, gamma, risk, paths, completion',
                  'baseline LP maximum admission VECTOR fixed by source',
                  'minimum initial nominal excess fixed, hence total initial completed service fixed',
                  'all variants use one common legacy min_overload LP after first synchronous failure update'],
        contrasts=dict(primary=['nearest_cost','balanced_util'],secondary=['low_id_probe','high_id_probe']),
        semantics='Static same-slot counterfactual intervention; later admissions are outcomes, not matched cumulative work',
        probes='ID rank objectives are arbitrary sensitivity probes; no implementable-policy superiority or structural novelty claimed',
        graphs=d['graphs'],runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        protected=protected,inputfiles=inputfiles,
        controls=['risk_only and neither must erase receiver-location effects after the common continuation',
                  'rho<=gamma+epsilon implies no overload and identical continuation',
                  'zero minimum excess implies no initial overload and identical continuation'],
        stop=['Do not select favorable graphs, scenarios or objective probes',
              'If equal-volume minimum-excess contrasts vanish, retain that null result',
              'A 4-node existence witness does not establish general novelty or real logistics validity'])
    dump(design,OUT/'00_design.json')
    note='''# 等接纳量、等最小超额的首次接收位置干预（运行前冻结）

全部沿用上一轮648个原LP初态条件，武汉168、合成480；不筛选有差异的图或冲击。每条件四个初始派入解，四种反馈开关，10368个配对干预轨迹。后续只用同一个原LP，不再沿用初始探针目标。

先取原最大接纳LP的逐来源接纳向量q，固定其每一分量；再最小化名义超额E，固定E*。因此总接纳、来源构成和首次总完成量同时一致。四个位置解分别为最小运输成本、最小峰值利用率后最小成本、按接收点ID升序/降序优先后最小成本。后两者仅为任意位置极端探针，不是业务政策或新方法。

只对首次派入进行干预。首次同步失效更新后，由同一LP重算直至闭包。后续接纳变化属于干预后果，并未强行声称每轮接纳量或累计处理任务都相同。风险仅依赖固定属性与失效集合，静态模型无首次流量跨期记忆。

共同来源接纳取自原LP，不从三启发式最小值降量；保持原初态最大接纳量。E*按固定q重新求解并与原LP核对。E相同时完成总量相同，来源完成归属可以不同，故逐来源首次完成不是匹配条件。接收负荷不同是唯一物理干预，成本不同不进入风险或触发。

四节点手算例另列。该例两计划同来源接纳、同超额、同完成、同成本，但不是最小超额计划，因此不能拿其正结果冒充最优面上的位置效应。严格最优面对照允许得出零结果。

主对照最近成本vs峰值均衡；次对照低/高ID，全部正负结果保留。比较双机制、仅过载、仅风险、都关闭。仅风险/都关闭必须无位置效应；rho<=gamma+epsilon或E*=0也应无首次位置效应。不得将合成输出视为实证样本，武汉仍是原代理相容关系。
'''
    (OUT/'00_design.md').write_text(note,encoding='utf-8')
    dump(dict(design=old.sha(OUT/'00_design.json'),note=old.sha(OUT/'00_design.md')),OUT/'design_lock.json')
    print(json.dumps(dict(frozen_conditions=648,planned_interventions=10368)),flush=True)


def verify(d):
    lock=json.loads((OUT/'design_lock.json').read_text(encoding='utf-8'))
    assert old.sha(OUT/'00_design.json')==lock['design'] and old.sha(OUT/'00_design.md')==lock['note']
    for group in ['protected','inputfiles']:
        for path,h in d[group].items():
            if old.sha(path)!=h:raise AssertionError('Frozen input/source changed '+path)


def config_for(j,mechanism):
    over,risk=MECHANISMS[mechanism]
    return replace(old.config(j),overload_enabled=over,risk_enabled=risk)


def plan_record(a,p):
    return dict(name=p.name,src=p.src.tolist(),dst=p.dst.tolist(),flow=p.flow.tolist(),q=p.q.tolist(),own=p.own.tolist(),
        load=p.load.tolist(),completed_source=p.completed_source.tolist(),completed_receiver=p.completed_receiver.tolist(),
        assigned=p.assigned,served=p.served,excess=p.excess,cost=p.cost,excess_optimum=p.excess_optimum,
        peak_util_optimum=p.peak_util_optimum,max_error=p.max_error)


def run():
    d=json.loads((OUT/'00_design.json').read_text(encoding='utf-8'));verify(d)
    graphs={g['graph_id']:(pd.read_csv(PRIOR/g['nodes_path']),pd.read_csv(PRIOR/g['edges_path']),g) for g in d['graphs']}
    inputs=json.loads((PRIOR/'inputs/demand_and_shocks.json').read_text(encoding='utf-8'))
    rows=[];logs=[];planrows=[];started=time.perf_counter()
    for j in d['jobs']:
        cid=j['condition_id'];path=OUT/'shards'/f'{cid:04d}.json'
        if path.exists():
            b=json.loads(path.read_text(encoding='utf-8'));assert b['design']==old.sha(OUT/'00_design.json')
            rows.extend(b['rows']);logs.extend(b['logs']);planrows.extend(b['planrows']);continue
        nn,ee,meta=graphs[j['graph_id']];data=inputs[j['input_id']];parts=np.asarray(data['parts']);D=parts.sum(axis=0)
        cfg=config_for(j,'both');scope=meta['compatibility'];spec=DispatchRuleSpec(compatibility_scope=scope)
        a=DispatchRuleAllocator(nn,ee,cfg,spec)
        plans,baseline=matched_plans(a,frozenset(data['initial_ids']),D)
        pair=list(plans.values())
        if max(np.max(abs(p.q-pair[0].q)) for p in pair)>1e-5 or max(p.served for p in pair)-min(p.served for p in pair)>1e-5:
            raise AssertionError('Unmatched first-step quantities')
        baseline_excess=baseline[0]['Nominal_Excess_Load']
        if max(abs(p.excess-baseline_excess) for p in pair)>1e-5:raise AssertionError('Lost minimum excess')
        payload=dict(condition_id=cid,ids=a.ids,demand=D.tolist(),initial_ids=data['initial_ids'],
                     plans={k:plan_record(a,p) for k,p in plans.items()},baseline_metrics=baseline[0],config=cfg.snapshot())
        dump(payload,OUT/'plans'/f'{cid:04d}.json')
        batchrows=[];batchlogs=[];batchplans=[]
        for name,p in plans.items():
            batchplans.append(dict(condition_id=cid,plan=name,initial_assigned=p.assigned,initial_served=p.served,
                initial_excess=p.excess,initial_cost=p.cost,initial_peak_util=float((p.load/a.nominal_capacity).max()),
                max_error=p.max_error,excess_optimum=p.excess_optimum,
                baseline_assigned=baseline[0]['Attempted_Assigned_Load'],baseline_excess=baseline_excess,
                q_max_gap=float(np.max(abs(p.q-pair[0].q)))))
        cache={}
        for mechanism in MECHANISMS:
            mcfg=config_for(j,mechanism);aa=DispatchRuleAllocator(nn,ee,mcfg,spec)
            en=ConfiguredDualMechanismRecovery(aa,ee,parts,data['groups'],data['initial_ids'],j['kind'])
            for name,p in plans.items():
                result=intervene(en,p,cache);trace=result.pop('continuation_logs')
                tag=dict(condition_id=cid,old_job_id=j['job_id'],graph_id=j['graph_id'],domain=j['domain'],topology=j['topology'],
                    replicate=j['replicate'],case_id=j['case_id'],control=j['control'],total_multiplier=j['total_multiplier'],
                    gamma=j['gamma'],rho=j['rho'],theta=j['theta'],mechanism=mechanism,plan=name)
                for key in ['first_new','first_overload_only','first_risk_only','first_both','continuation_start','final_failed']:
                    result[key+'_json']=json.dumps(result.pop(key),ensure_ascii=False)
                batchrows.append(dict(**tag,**result))
                current=frozenset(json.loads(result['continuation_start_json']))
                for step in trace:
                    metric,u,completed=en.state(current)
                    batchlogs.append(dict(condition_id=cid,mechanism=mechanism,plan=name,round=step['round'],
                        current_ids_json=json.dumps(sorted(current)),new_ids_json=json.dumps(step['new_ids']),
                        util_json=json.dumps(u.tolist()),source_completed_json=json.dumps(completed.tolist()),
                        **{k:v for k,v in step.items() if k not in ['round','new_ids']}))
                    current=current|frozenset(step['new_ids'])
                if sorted(current)!=json.loads(result['final_failed_json']):raise AssertionError('Trace/final mismatch')
            aa.clear_path_cache()
        a.clear_path_cache()
        b=dict(design=old.sha(OUT/'00_design.json'),rows=batchrows,logs=batchlogs,planrows=batchplans)
        dump(b,path);rows.extend(batchrows);logs.extend(batchlogs);planrows.extend(batchplans)
        if (cid+1)%24==0 or cid==647:
            print(json.dumps(dict(conditions=cid+1,planned=648,interventions=len(rows),elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    save(rows,OUT/'03_interventions.csv');save(planrows,OUT/'04_initial_plans.csv');save(logs,OUT/'05_continuation_rounds.csv')
    verify(d)
    dump(dict(completed=True,conditions=len(d['jobs']),interventions=len(rows),plan_count=len(planrows),
        elapsed_seconds=time.perf_counter()-started,
        files={str(p.relative_to(OUT)):old.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json'}),OUT/'completion.json')
    print(json.dumps(dict(completed=True,interventions=len(rows))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['analytic','design','run']);args=p.parse_args()
    globals()[args.phase]()
