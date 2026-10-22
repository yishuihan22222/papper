"""Run the V7 predeclared sample boundary, degree and label comparisons."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import importlib
import json
import time
import numpy as np
import pandas as pd

from configured_dual_mechanism_v5 import ConfiguredDispatchAllocator, ConfiguredDualMechanismRecovery
from model_config import ModelConfig

audit = importlib.import_module('136_audit_processing_units_v7')
OUT = audit.OUT


def inputs(job):
    design = audit.read_json(OUT/'00_design.json')
    n=pd.read_csv(OUT/f"frozen_inputs/{job['capacity_id']}_nodes.csv",float_precision='round_trip')
    original=pd.read_csv(OUT/'frozen_inputs/original_edges.csv',float_precision='round_trip')
    selected=n.node_id.isin(design['keep_ids']).to_numpy()
    nn=n if job['scope']=='matched246' else n.loc[selected].reset_index(drop=True)
    e=original if job['scope']=='matched246' else original[original.Source_id.isin(nn.node_id)&original.Target_id.isin(nn.node_id)].reset_index(drop=True)
    if job['scope']=='candidate_corrected_family':
        records=pd.read_csv(OUT/'01_all_246_node_decisions.csv').set_index('node_id')
        nn=nn.copy();nn['family']=records.loc[nn.node_id,'reviewed_family'].to_numpy()
    degrees=pd.concat([original.Source_id,original.Target_id]).value_counts().reindex(nn.node_id).fillna(0).to_numpy(float)
    return design,nn,e,selected,degrees


def engine(nn,e,parts,groups,case,config,degrees=None):
    a=ConfiguredDispatchAllocator(nn,e,config)
    en=ConfiguredDualMechanismRecovery(a,e,parts,groups,case['initial_ids'],case['kind'],config)
    if degrees is not None:
        en.degree=degrees.copy();en.cache_namespace+='|original246_degree_reference'
    return en


def cell(job):
    d,nn,e,selected,degrees=inputs(job)
    payload=audit.read_json(OUT/'frozen_inputs/demands.json')
    rows=[]
    for theta in d['thresholds']:
        configs={arm:ModelConfig.for_arm(arm,theta_risk=theta,theta_source='Frozen V7 boundary study; not empirical') for arm in d['arms']}
        allocators={arm:ConfiguredDispatchAllocator(nn,e,configs[arm]) for arm in d['arms']}
        states={};pressures={}
        for case in d['cases']:
            for name in d['demand_names']:
                parts=np.array(payload[name]['parts'],float)
                if job['scope']=='matched246':parts[:,~selected]=0.
                else:parts=parts[:,selected]
                for arm in d['arms']:
                    en=ConfiguredDualMechanismRecovery(allocators[arm],e,parts,payload[name]['groups'],case['initial_ids'],case['kind'],configs[arm])
                    if job['scope']=='candidate_fixed_degree':
                        en.degree=degrees.copy();en.cache_namespace+='|original246_degree_reference'
                    en.cache=states.setdefault(arm,{})
                    en.pressure_cache=pressures.setdefault(arm,{})
                    f,m,rounds=en.closure(en.initial)
                    if m['Max_LP_Violation']>1e-5 or m['Service_Conservation_Error']>1e-7 or m['Assignment_Conservation_Error']>1e-7:
                        raise AssertionError('Allocation/accounting violation')
                    if m['Overload_Only_Failures']+m['Risk_Only_Failures']+m['Both_Failures'] != len(f)-len(en.initial):
                        raise AssertionError('Exit causes do not close')
                    rows.append(dict(**job,theta=theta,case_id=case['case_id'],kind=case['kind'],demand=name,arm=arm,
                        initial_count=len(en.initial),nodes=len(nn),edges=len(e),rounds=rounds,final_failed=len(f),
                        failure_fraction=len(f)/len(nn),final_ids_json=json.dumps(sorted(f),ensure_ascii=False),**m))
    path=OUT/'shards'/f"{job['scope']}_{job['capacity_id']}.csv"
    audit.save(rows,path)
    return str(path)


def main():
    start=time.perf_counter();d=audit.read_json(OUT/'00_design.json')
    assert audit.sha(OUT/'00_design.json')==audit.read_json(OUT/'design_lock.json')['design_sha256']
    assert all(audit.sha(p)==h for p,h in d['source_hashes'].items())
    assert all(audit.sha(OUT/p)==h for p,h in d['audit_hashes'].items())
    audit.dump(dict(source_sha256=audit.sha(__file__),workers=3,grid_changed=False),OUT/'09_execution.json')
    paths=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(cell,job):job for job in d['jobs']}
        for future in as_completed(futures):
            paths.append(future.result());print('completed',len(paths),'/',len(futures),futures[future],flush=True)
    runs=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths],ignore_index=True)
    checks=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:
            audit.save(checks,OUT/'failed_checks.csv');raise AssertionError(name)
    keys=['scope','capacity_id','theta','case_id','kind','demand','arm']
    check('exact frozen trajectory count',len(runs)==d['expected_runs']==6912)
    check('unique experiment keys',len(runs[keys].drop_duplicates())==len(runs))
    check('finite outcomes',np.isfinite(runs[['Service_Rate','Demand_Load','Served_Load']].to_numpy()).all())
    check('feasible and conserving',runs.Max_LP_Violation.max()<=1e-5 and runs.Service_Conservation_Error.max()<=1e-7 and runs.Assignment_Conservation_Error.max()<=1e-7)
    pairs=runs.pivot(index=keys[:-1],columns='arm',values='Service_Rate').reset_index()
    pairs['C_B_pp']=100*(pairs.C-pairs.B);pairs['C_A_pp']=100*(pairs.C-pairs.A)
    check('conditional safe policy ordering',pairs.C_B_pp.min()>=-1e-7 and pairs.C_A_pp.min()>=-1e-7)
    summaries=[]
    for (scope,cap),g in pairs.groupby(['scope','capacity_id']):
        low=g[~g.demand.eq('external_plus150')]
        summaries.append(dict(scope=scope,capacity_id=cap,conditions=len(g),B_mean=100*g.B.mean(),C_mean=100*g.C.mean(),C_B_mean_pp=g.C_B_pp.mean(),C_B_median_pp=g.C_B_pp.median(),C_B_better=int((g.C_B_pp>1e-7).sum()),C_B_equal=int((g.C_B_pp.abs()<=1e-7).sum()),C_B_worse=int((g.C_B_pp< -1e-7).sum()),first_three_mean_pp=low.C_B_pp.mean()))
    for frame in [runs]:frame['theta_key']=frame.theta.map(lambda x:format(x,'.12g'))
    index=['capacity_id','theta_key','case_id','kind','demand','arm']
    ref=runs[runs.scope.eq('matched246')].set_index(index).sort_index()
    effects=[]
    for scope in d['scopes'][1:]:
        part=runs[runs.scope.eq(scope)].set_index(index).sort_index()
        check('same tasks '+scope,float((ref.Demand_Load-part.Demand_Load).abs().max())<1e-7)
        change=part[[]].copy();change['scope']=scope;change['service_change_pp']=100*(part.Service_Rate-ref.Service_Rate)
        change['reference_service']=ref.Service_Rate;change['candidate_service']=part.Service_Rate
        change['demand_denominator_difference']=part.Demand_Load-ref.Demand_Load
        effects.append(change.reset_index())
    fixed=runs[runs.scope.eq('candidate_fixed_degree')&runs.arm.eq('C')].set_index(index)
    for key,row in fixed.iterrows():
        expected=set(json.loads(ref.loc[key,'final_ids_json']))&set(d['keep_ids'])
        check('fixed reference C risk closure subset '+str(key),set(json.loads(row.final_ids_json))<=expected)
    resources=[];payload=audit.read_json(OUT/'frozen_inputs/demands.json')
    for job in d['jobs']:
        _,nn,e,selected,degrees=inputs(job)
        for name in d['demand_names']:
            parts=np.array(payload[name]['parts']);parts=parts[:,selected]
            resources.append(dict(**job,demand=name,nodes=len(nn),edges=len(e),total_capacity=float(nn.overload_threshold.sum()),retained_base_load=float(nn.proxy_daily_load.sum()),task_denominator=float(parts.sum())))
    cold=[]
    for job in [dict(scope=s,capacity_id=d['capacity_ids'][0]) for s in d['scopes']]:
        _,nn,e,selected,degrees=inputs(job)
        case=d['cases'][-1];name=d['demand_names'][0];parts=np.array(payload[name]['parts'])
        if job['scope']=='matched246':parts[:,~selected]=0.
        else:parts=parts[:,selected]
        for arm in ['B','C']:
            cfg=ModelConfig.for_arm(arm,theta_risk=d['thresholds'][-1],theta_source='Cold V7 verification')
            en=engine(nn,e,parts,payload[name]['groups'],case,cfg,degrees if job['scope']=='candidate_fixed_degree' else None)
            f,m,_=en.closure(en.initial)
            original=runs[(runs.scope==job['scope'])&(runs.capacity_id==job['capacity_id'])&(runs.case_id==case['case_id'])&(runs.demand==name)&(runs.arm==arm)&(runs.theta_key==format(d['thresholds'][-1],'.12g'))].iloc[0]
            equal=abs(m['Service_Rate']-original.Service_Rate)<1e-12 and set(f)==set(json.loads(original.final_ids_json))
            check('cold restart '+str(job)+arm,equal);cold.append(dict(**job,arm=arm,identical=equal))
    for p,h in d['source_hashes'].items():check('unchanged '+p,audit.sha(p)==h)
    runs=runs.sort_values(keys).reset_index(drop=True)
    effect=pd.concat(effects,ignore_index=True)
    for rows,name in [(runs,'10_model_runs.csv'),(pairs,'11_policy_pairs.csv'),(summaries,'12_policy_summary.csv'),(effect,'13_matched_task_effects.csv'),(resources,'14_resources.csv'),(cold,'15_cold_restarts.csv'),(checks,'16_checks.csv')]:audit.save(rows,OUT/name)
    c=effect[effect.arm.eq('C')]
    ranges={s:dict(min_pp=float(g.service_change_pp.min()),max_pp=float(g.service_change_pp.max()),mean_pp=float(g.service_change_pp.mean()),decrease=int((g.service_change_pp< -1e-7).sum()),tie=int((g.service_change_pp.abs()<=1e-7).sum()),increase=int((g.service_change_pp>1e-7).sum())) for s,g in c.groupby('scope')}
    completion=dict(completed=True,runs=len(runs),conditions=len(pairs),checks=len(checks),passed=sum(r['passed'] for r in checks),cold_restarts=len(cold),candidate_nodes=d['candidate_nodes'],candidate_edges=d['candidate_edges'],C_boundary_changes=ranges,actual_real_capacity_replacements=0,all_nodes_empirically_verified=False,elapsed_seconds=time.perf_counter()-start)
    audit.dump(completion,OUT/'completion.json')
    print(json.dumps(completion,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
