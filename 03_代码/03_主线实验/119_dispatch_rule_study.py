"""Frozen same-layer policy comparison on Wuhan proxies and controlled graphs."""
import argparse
import importlib
import itertools
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from model_config import ModelConfig
from dispatch_policy_rules import DispatchRuleSpec,DispatchRuleAllocator
from configured_dual_mechanism_v5 import ConfiguredDualMechanismRecovery
from capacity_precision_v5 import formula_precision_nodes

core=importlib.import_module('83_dispatch_mainline_core')
CODE=Path(__file__).parent;ROOT=CODE.parents[1]
OUT=ROOT/'项目结果/dispatch_rules_external_synthetic_20261020'
PRECISION=ROOT/'项目结果/capacity_precision_audit_20261020'
RULES=['lexicographic_lp','remaining_capacity_share','nearest_feasible']


def sha(p):return core.sha(p)
def dump(v,p):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save(v,p):pd.DataFrame(v).to_csv(p,index=False,encoding='utf-8-sig')


def connected(pairs,n):
    a=np.zeros((n,n),int)
    for i,j in pairs:a[i,j]=a[j,i]=1
    return connected_components(csr_matrix(a),directed=False)[0]==1


def ba_edges(seed):
    """Degree-proportional m=2 growth from a 15-node path, then relabel.

    A connected initial graph is part of the BA design. It gives exactly the
    same 104 edges as a 6x10 open grid, without post-generation edge changes.
    """
    rng=np.random.default_rng(seed);edges=[(i,i+1) for i in range(14)]
    degree=np.zeros(60,float)
    for i,j in edges:degree[i]+=1;degree[j]+=1
    for new in range(15,60):
        targets=rng.choice(new,2,replace=False,p=degree[:new]/degree[:new].sum())
        for target in targets:edges.append((int(target),new));degree[target]+=1
        degree[new]=2
    perm=rng.permutation(60)
    return sorted({tuple(sorted((int(perm[i]),int(perm[j])))) for i,j in edges})


def grid_edges():
    return [(r*10+c,(r+1)*10+c) for r in range(5) for c in range(10)]+[(r*10+c,r*10+c+1) for r in range(6) for c in range(9)]


def synthetic_tables(pairs,load):
    ids=[f's{i:03d}' for i in range(60)]
    nodes=pd.DataFrame(dict(node_id=ids,名称=ids,城市='synthetic',区县='synthetic',family='same_service',
        proxy_daily_load=load,overload_threshold=1.25*load))
    edges=pd.DataFrame([dict(Source_id=ids[i],Target_id=ids[j],Composite_transport_cost=1.,
         Failure_probability_proxy=0.,Cross_district=0.,Edge_capacity_proxy=55.,Repair_cost_proxy=1.) for i,j in pairs])
    return nodes,edges


def write_graph(label,n,e,metadata):
    n.to_csv(OUT/f'inputs/{label}_nodes.csv',index=False,encoding='utf-8-sig')
    e.to_csv(OUT/f'inputs/{label}_edges.csv',index=False,encoding='utf-8-sig')
    return dict(graph_id=label,**metadata,nodes_path=f'inputs/{label}_nodes.csv',edges_path=f'inputs/{label}_edges.csv')


def design():
    if (OUT/'00_design.json').exists():raise FileExistsError('Design already frozen')
    for directory in ['inputs','shards','configs']:(OUT/directory).mkdir(parents=True,exist_ok=True)
    n,e,demands,_=core.load_study();nodes,_=formula_precision_nodes(n,n)
    old=json.loads((ROOT/'项目结果/gamma_demand_boundary_20261020/00_design.json').read_text(encoding='utf-8'))
    graphs=[write_graph('wuhan_proxy',nodes,e,dict(domain='geographic_proxy',topology='wuhan',replicate=0,
        compatibility='original_proxy_assumption',observed_operational_validation=False))]
    inputs={};jobs=[]
    def add_job(graph,case,parts,groups,mult,theta,gamma,rho,rule,tie='canonical'):
        input_id=graph['graph_id']+'|'+case['case_id']+f'|x{mult:g}'
        if input_id not in inputs:inputs[input_id]=dict(parts=np.asarray(parts).tolist(),groups=list(groups),initial_ids=case['initial_ids'])
        jobs.append(dict(job_id=len(jobs),graph_id=graph['graph_id'],domain=graph['domain'],topology=graph['topology'],
            replicate=graph['replicate'],case_id=case['case_id'],kind=case.get('kind','random'),control=case.get('control',False),
            input_id=input_id,total_multiplier=mult,theta=theta,gamma=gamma,rho=rho,rule=rule,tie_order=tie,
            primary=tie=='canonical'))
    cases=old['cases'];thresholds=[old['thresholds'][0],.42]
    base,groups=demands['nominal']
    for case,external,theta,gamma,rho,rule in itertools.product(cases,[1.,1.2,2.5],thresholds,[1.,1.05],[1.,1.1],RULES):
        parts=np.asarray(base).copy();parts[1:]*=external
        add_job(graphs[0],case,parts,groups,.75+.25*external,theta,gamma,rho,rule)
        jobs[-1]['external_multiplier']=external
    assert len(jobs)==504
    graph_meta=[]
    for replicate in range(10):
        seed=202610200100+replicate;rng=np.random.default_rng(seed)
        load=rng.lognormal(0.,.25,60);load*=6000/load.sum()
        shock=np.sort(rng.choice(60,4,replace=False))
        syn_cases=[dict(case_id='uniform4',initial_ids=[f's{i:03d}' for i in shock]),
                   dict(case_id='geometric_block4',initial_ids=['s022','s023','s032','s033'])]
        for topology,pairs in [('grid',grid_edges()),('ba',ba_edges(seed+100000))]:
            assert len(pairs)==104 and len({tuple(sorted(p)) for p in pairs})==104 and connected(pairs,60)
            nn,ee=synthetic_tables(pairs,load)
            graph=write_graph(f'{topology}_{replicate:02d}',nn,ee,dict(domain='synthetic',topology=topology,
                replicate=replicate,seed=seed,ba_generation_seed=seed+100000 if topology=='ba' else None,
                compatibility='synthetic_same_service',observed_operational_validation=False))
            graphs.append(graph)
            graph_meta.append(dict(graph_id=graph['graph_id'],topology=topology,replicate=replicate,nodes=60,edges=104,
                 total_load=float(load.sum()),total_capacity=float(1.25*load.sum()),
                 topology_hash=sha(OUT/graph['edges_path']),ba_m=2 if topology=='ba' else None,
                 ba_initial_nodes=15 if topology=='ba' else None,ba_initial_graph='path' if topology=='ba' else None))
            for case,mult,theta,rho,rule in itertools.product(syn_cases,[1.,1.05,1.375],[.2,.42],[1.,1.1],RULES):
                add_job(graph,case,[mult*load],['synthetic_same_service'],mult,theta,1.,rho,rule)
            for case,mult,theta,rho in itertools.product(syn_cases,[1.,1.05,1.375],[.2,.42],[1.,1.1]):
                add_job(graph,case,[mult*load],['synthetic_same_service'],mult,theta,1.,rho,'nearest_feasible','reverse_sources')
    assert len(jobs)==2424
    dump(inputs,OUT/'inputs/demand_and_shocks.json');save(graph_meta,OUT/'01_synthetic_graphs.csv')
    protected=list(CODE.glob('*.py'))+list((CODE.parent/'data').glob('*.csv'))
    protected += [PRECISION/'02_new_runs.csv',PRECISION/'delivery_manifest.json',
        ROOT.parent/'中文论文初稿_全图美化与正文优化V5_20261019.docx',
        ROOT.parent/'论文初稿配套图表_V5_20261019/初稿源文件/manuscript.md']
    cold=[]
    for domain in ['geographic_proxy','synthetic']:
        for rule in RULES:
            subset=[j for j in jobs if j['domain']==domain and j['rule']==rule and j['primary']]
            cold.extend([subset[0]['job_id'],subset[-1]['job_id']])
    d=dict(schema='same-layer-dispatch-1',date='2026-10-20',jobs=jobs,graphs=graphs,
       closures=2424,wuhan_proxy_closures=504,synthetic_primary_closures=1440,synthetic_tie_sensitivity=480,
       cold_job_ids=cold,lp_reference_real_closures=168,
       semantics='Facilities process the same eligible divisible service; no observed Wuhan permissions inferred',
       common_inputs=['nominal own priority','rho','gamma','completion proportional','fixed risk and demand','two-hop paths'],
       synthetic_controls='60 nodes,104 edges,paired load/K/shock IDs and unit edge costs; 10 BA realizations, one grid with 10 paired input replicates',
       risk_synthetic='neutral neighbor-failure fraction, bonus=0, weights=(.68,0,0,0,0)',
       primary_comparison='Compare rules at the same rho, gamma, demand, shock and graph; no universal LP terminal dominance assumption',
       external_validation='Documented dispatch authority and named diversion cases; quantitative prediction/policy validation not established',
       limitations=['prior-explored Wuhan shocks','Wuhan is a proxy scenario, not observed homogeneous service network',
                    'synthetic replicates are not real disaster samples','grid topology is repeated with new inputs, not ten independent grids',
                    'heuristics need not maximize assignment; report initial assignment gap to LP explicitly',
                    'no UE, identification, prediction claim or new mechanical blocking rule'],
       stop=['No selecting favorable graphs/shocks or retuning gamma','No fitting capacity from reported throughput or diversion',
             'Any integrity failure is diagnosed before interpretation','No real external validation claim without matched inputs and outcomes'],
       protected={str(p):sha(p) for p in protected},
       frozen={str(p.relative_to(OUT)):sha(p) for p in (OUT/'inputs').iterdir()},
       evidence={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.md','.csv']},
       runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__))
    dump(d,OUT/'00_design.json')
    note='''# 同一派入层规则与合成拓扑（运行前冻结）

2026-10-20。先保存业务证据边界与具体关系登记，未建立武汉246节点实测相容图。数学对象为授权同类可拆分处理需求；武汉沿原代理联系作情景检验，不能作为现实定量验证。

三规则：剩余接纳空间比例（同时提出、接收点比例缩放、扣减后迭代）；最近可行接收点（全弧按成本、来源ID、接收ID排序贪心）；原LP（最大接纳、最小超额、最小成本）。三者共用名义本地优先、比例完成、可达算子、能力、需求、风险和退出规则。
rho=1/1.1交叉；在相同rho内比较规则，单独报告拒收、已接收未完成、服务、失效和成本。启发式不保证最大派入，不预设LP终态总优；以共同初态LP派入量报告启发式接纳缺口。

武汉基准容量使用已审计的1.25L版本，四既有损伤和三无损伤对照，两个阈值、三需求、两个gamma、两个rho、三规则，共504闭包。现有LP的168条件回归上一轮精度版。

合成：6×10无周期边界格子104边；BA按度优先连接，从15节点路径图开始，每次加节点连接两旧节点，达到60节点104边后仅随机重标号。不后删边，不宣称60节点实证证明幂律。用SciPy检查连通和简单图；没有安装额外包。
BA十个独立生成种子；格子只有一个拓扑、十套配对负荷/损伤。两类每套输入的总负荷6000、K=1.25L、节点对应负荷/容量和受损ID一致，边成本全为1，风险为中性失效邻居比例且无bonus。
格子节点度至多4，因此theta=.2下只要有失效邻居就满足风险条件；在连通格子上任何非空起点都会经纯风险闭包全退。此格是事先可推导的结构控制，不算新发现，不把它拿来宣称BA在现实中更安全。
两个阈值、三总需求倍率、两个rho、两种四节点损伤、三规则，合计1440主闭包；另对最近规则预定反转同成本弧来源并列顺序，480闭包。总2424。表中不能把输入重复称十张独立格子网。

保存全部逐轮利用率和来源完成、初始派入弧及权限元数据；独立重建压力/触发/守恒，校验LP原路径精确一致、玩具反例、节点重排与缓存命名空间。12个固定条件冷启动。
不作现实概率显著性检验；报告完整胜/平/负与十个配对重复，不选优势参数。外部材料只能验收业务机制对应，缺同期OD、能力及服务结果时必须保留定量验证缺口。
'''
    (OUT/'00_design.md').write_text(note,encoding='utf-8')
    dump(dict(design=sha(OUT/'00_design.json'),note=sha(OUT/'00_design.md')),OUT/'design_lock.json')
    print(json.dumps(dict(design_frozen=True,closures=2424),ensure_ascii=False),flush=True)


def verify(d):
    lock=json.loads((OUT/'design_lock.json').read_text(encoding='utf-8'))
    assert sha(OUT/'00_design.json')==lock['design'] and sha(OUT/'00_design.md')==lock['note']
    for path,h in d['protected'].items():
        if sha(path)!=h:raise RuntimeError('Changed protected file '+path)
    for path,h in d['frozen'].items():
        if sha(OUT/path)!=h:raise RuntimeError('Changed frozen input '+path)
    for path,h in d['evidence'].items():
        if sha(OUT/path)!=h:raise RuntimeError('Changed prior evidence '+path)


def config(job):
    extra=dict(bonus_amplitude=0.,risk_weights=(.68,0.,0.,0.,0.)) if job['domain']=='synthetic' else {}
    return ModelConfig(theta_risk=job['theta'],rho=job['rho'],gamma=job['gamma'],
        theta_source='uncalibrated fixed same-layer study 2026-10-20',**extra)


def run():
    d=json.loads((OUT/'00_design.json').read_text(encoding='utf-8'));verify(d)
    graph_info={v['graph_id']:v for v in d['graphs']}
    graphs={name:(pd.read_csv(OUT/g['nodes_path']),pd.read_csv(OUT/g['edges_path'])) for name,g in graph_info.items()}
    inputs=json.loads((OUT/'inputs/demand_and_shocks.json').read_text(encoding='utf-8'))
    rows=[];logs=[];arcs=[];checks=[];started=time.perf_counter()
    def check(name,ok):
        checks.append(dict(check=name,passed=bool(ok)))
        if not ok:save(checks,OUT/'validation_failed.csv');raise AssertionError(name)
    for start in range(0,len(d['jobs']),24):
        path=OUT/'shards'/f'{start//24:03d}.json'
        if path.exists():
            b=json.loads(path.read_text(encoding='utf-8'));assert b['design']==sha(OUT/'00_design.json')
            rows.extend(b['rows']);logs.extend(b['logs']);arcs.extend(b['arcs']);checks.extend(b['checks']);continue
        batchrows=[];batchlogs=[];batcharcs=[];check_start=len(checks)
        for job in d['jobs'][start:start+24]:
            nn,ee=graphs[job['graph_id']];data=inputs[job['input_id']];parts=np.asarray(data['parts']);D=parts.sum(axis=0)
            cfg=config(job);scope=graph_info[job['graph_id']]['compatibility']
            rule=DispatchRuleSpec(job['rule'],job['tie_order'],scope)
            cfg.write_snapshot(OUT/'configs'/f'{cfg.fingerprint}.json')
            specpath=OUT/'configs'/f'rule_{rule.fingerprint}.json'
            if not specpath.exists():dump(rule.snapshot(),specpath)
            a=DispatchRuleAllocator(nn,ee,cfg,rule)
            en=ConfiguredDualMechanismRecovery(a,ee,parts,data['groups'],data['initial_ids'],job['kind'])
            failed,m,iterations=en.closure(en.initial);initial=a.evaluate(en.initial,D,details=True)
            tag=dict(job,model_config_id=cfg.fingerprint,rule_config_id=rule.fingerprint,input_fingerprint=a.input_fingerprint)
            batchrows.append(dict(**tag,**m,final_failed=len(failed),initial_count=len(en.initial),iterations=iterations,
                 final_ids_json=json.dumps(sorted(failed),ensure_ascii=False),
                 initial_assigned=initial[0]['Attempted_Assigned_Load'],initial_served=initial[0]['Served_Load'],
                 initial_cost=initial[0]['Transfer_Cost'],initial_nominal_excess=initial[0]['Nominal_Excess_Load']))
            for arc in initial[3]['arcs'].itertuples():
                if abs(arc.flow)>1e-9:
                    batcharcs.append(dict(job_id=job['job_id'],source_id=arc.source_id,target_id=arc.target_id,
                        flow=float(arc.flow),completed_flow=float(arc.completed_flow),cost=float(arc.cost)))
            current=set(en.initial)
            for step in en.cascade_logs[en.initial]:
                mm,u,s=en.state(current)
                check('round physical '+str((job['job_id'],step['round'])),mm['Max_LP_Violation']<1e-5
                    and mm['Service_Conservation_Error']<1e-6 and mm['Assignment_Conservation_Error']<1e-6
                    and abs(np.minimum(u*a.nominal_capacity,a.nominal_capacity).sum()-s.sum())<1e-6)
                record=dict(step);record['new_ids_json']=json.dumps(record.pop('new_ids'),ensure_ascii=False)
                batchlogs.append(dict(job_id=job['job_id'],**record,util_json=json.dumps(u.tolist()),source_completed_json=json.dumps(s.tolist())))
                current.update(step['new_ids'])
            check('terminal accounting '+str(job['job_id']),current==set(failed)
                and m['Overload_Only_Failures']+m['Risk_Only_Failures']+m['Both_Failures']==len(failed)-len(en.initial))
            if job['rho']==1.:
                check('safe admission no overload '+str(job['job_id']),m['Overload_Only_Failures']==m['Both_Failures']==0)
            a.clear_path_cache()
        dump(dict(design=sha(OUT/'00_design.json'),rows=batchrows,logs=batchlogs,arcs=batcharcs,
                  checks=checks[check_start:]),path)
        rows.extend(batchrows);logs.extend(batchlogs);arcs.extend(batcharcs)
        if len(rows)%240==0 or len(rows)==2424:
            print(json.dumps(dict(closures=len(rows),planned=2424,elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    check('full grid',len(rows)==2424 and len({r['job_id'] for r in rows})==2424)
    save(rows,OUT/'02_runs.csv');save(logs,OUT/'03_rounds.csv');save(arcs,OUT/'04_initial_arcs.csv');save(checks,OUT/'05_run_checks.csv')
    verify(d)
    dump(dict(completed=True,closures=len(rows),checks=len(checks),passed=sum(c['passed'] for c in checks),
        elapsed_seconds=time.perf_counter()-started,
        files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json'}),OUT/'completion.json')
    print(json.dumps(dict(completed=True,closures=len(rows),checks=len(checks))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['design','run']);a=parser.parse_args()
    design() if a.phase=='design' else run()
