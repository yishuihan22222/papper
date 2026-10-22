"""Frozen-demand A/B/C comparisons and pair-consistent null edge attributes."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from closed_loop_recovery_v3 import HERE,load_inputs,DemandModel
from dual_mechanism_recovery_v5 import DispatchAllocatorV5,DualMechanismRecoveryV5
from protect_repair_v6 import ProtectRepairStudy,BoundedCache

ROOT=HERE.parent.parent;DATA=HERE.parent/'data'
OUT=ROOT/'项目结果/dispatch_capacity_null_mainline_20261018'
CAP=ROOT/'项目结果/capacity_heterogeneity_design_20261018'
OLD=ROOT/'项目结果/protect_repair_v6_20261017'
CAP_IDS=['baseline','node_a0.20_s00','family_a0.20_s00']
NULL_CASE_IDS=['test_random_00','test_random_01','test_spatial_00','test_channel_00']
NULL_DEMANDS=['nominal','external_plus150']
DISTANCE_BINS=[0.,5.,10.,20.,40.,80.,float('inf')]
CONSUMED=['Failure_probability_proxy','Cross_district','Composite_transport_cost','Edge_capacity_proxy','Repair_cost_proxy']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(v,p):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def save(v,p):pd.DataFrame(v).to_csv(p,index=False,encoding='utf-8-sig')
def source_hashes(paths):return {str(Path(p).relative_to(ROOT)):sha(p) for p in paths}
def check_sources(design):
    for p,h in design['sources'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('Changed study input/source '+p)
def completion(folder,files,extra):
    dump(dict(completed=True,files={name:sha(folder/name) for name in files},**extra),folder/'completion.json')
def valid_completion(folder):
    p=folder/'completion.json'
    if not p.exists():return False
    v=json.loads(p.read_text(encoding='utf-8'))
    for name,h in v['files'].items():
        if sha(folder/name)!=h:raise RuntimeError('Changed completed output '+str(folder/name))
    return v['completed']
def load_study():
    n,e,ci,cross=load_inputs();dm=DemandModel(n,e,ci,cross);s=ProtectRepairStudy(n,e,dm)
    d=json.loads((OLD/'71_design.json').read_text(encoding='utf-8'))
    return n,e,s.demands,d
def capacity_nodes(n,cap_id):
    table=pd.read_csv(CAP/'82a_assumed_capacity_scenarios.csv');g=table[table.scenario.eq(cap_id)].set_index('node_id')
    if len(g)!=len(n):raise ValueError('Incomplete capacity version')
    result=n.copy();result['overload_threshold']=g.loc[result.node_id,'assumed_capacity'].to_numpy(float)
    if abs(result.overload_threshold.sum()-n.overload_threshold.sum())>1e-7:raise AssertionError('Total capacity changed')
    return result
def candidate_edges(n):
    e=pd.read_csv(DATA/'04c_武汉边代理属性_候选全边.csv');pos={k:i for i,k in enumerate(n.node_id)}
    e['i']=e.Source_id.map(pos);e['j']=e.Target_id.map(pos)
    if e[['i','j']].isna().any().any():raise ValueError('Unknown candidate endpoint')
    e['pair_key']=[tuple(sorted((int(i),int(j)))) for i,j in zip(e.i,e.j)]
    cap=n.set_index('node_id').capacity_index
    e['Edge_capacity_proxy']=(np.sqrt(e.Source_id.map(cap)*e.Target_id.map(cap))/(1+e.Composite_transport_cost)*100).round(4)
    return e
def distance_bin(values):return np.searchsorted(np.asarray(DISTANCE_BINS[1:]),values,side='right')
def run_grid(n,e,demands,cases,thresholds,demand_names,tag,*,progress=None):
    allocators={arm:DispatchAllocatorV5(n,e,1. if arm=='C' else 1.1,'min_cost' if arm=='A' else 'min_overload',2) for arm in ['A','B','C']}
    states={(arm,d):BoundedCache(2048) for arm in allocators for d in demand_names}
    pressures={};rows=[];rounds=[];checks=[]
    for theta in thresholds:
        for case in cases:
            for demand in demand_names:
                parts,groups=demands[demand];fsets={};metrics={}
                for arm,allocator in allocators.items():
                    en=DualMechanismRecoveryV5(allocator,e,parts,groups,case['initial_ids'],case['kind'],theta,gamma=1.)
                    en.cache=states[arm,demand];en.pressure_cache=pressures.setdefault(case['kind'],BoundedCache(2048))
                    f,m,iterations=en.closure(en.initial);initial_metric=en.state(en.initial)[0]
                    key=dict(**tag,theta=theta,case_id=case['case_id'],shock_scope=case['shock_scope'],kind=case['kind'],demand=demand,arm=arm)
                    rows.append(dict(**key,**m,initial_count=len(en.initial),final_failed=len(f),failure_fraction=len(f)/len(n),
                                     cascade_iterations=iterations,initial_attempted_service=initial_metric['Attempted_Service_Rate'],
                                     initial_completed_service=initial_metric['Service_Rate'],initial_excess=initial_metric['Nominal_Excess_Load'],
                                     final_failed_indices_json=json.dumps(sorted(allocator.pos[k] for k in f))))
                    for event in en.cascade_logs[en.initial]:
                        record=dict(event);record['new_ids_json']=json.dumps(record.pop('new_ids'),ensure_ascii=False)
                        rounds.append(dict(**key,**record))
                    ok=(len(f)>=len(en.initial) and set(case['initial_ids'])<=f and abs(m['Service_Conservation_Error'])<1e-7 and
                        abs(m['Assignment_Conservation_Error'])<1e-7 and m['Max_LP_Violation']<1e-5 and
                        m['Overload_Only_Failures']+m['Risk_Only_Failures']+m['Both_Failures']==len(f)-len(en.initial))
                    if arm=='C':ok=ok and m['Overload_Only_Failures']==0 and m['Both_Failures']==0 and m['Max_Utilization']<=1+1e-7
                    checks.append(dict(**key,check='physical accounting and trigger partition',passed=bool(ok)))
                    if not ok:raise AssertionError(str(key))
                    fsets[arm]=f;metrics[arm]=m
                for arm in ['A','B']:
                    ok=fsets['C']<=fsets[arm] and metrics['C']['Service_Rate']>=metrics[arm]['Service_Rate']-1e-9
                    checks.append(dict(**tag,theta=theta,case_id=case['case_id'],demand=demand,arm=arm,check='conditional C dominance',passed=bool(ok)))
                    if not ok:raise AssertionError('Dominance outside expected assumptions')
            if progress:progress(theta,case['case_id'])
    return rows,rounds,checks
