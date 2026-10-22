"""Post-result diagnostic: keep original-degree risk denominators after exclusion."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import time

import numpy as np
import pandas as pd

from capacity_precision_v5 import formula_precision_nodes
from configured_dual_mechanism_v5 import ConfiguredDispatchAllocator, ConfiguredDualMechanismRecovery
from model_config import ModelConfig
from public_data_inputs_v1 import sha

study = importlib.import_module('132_node_boundary_and_scope_study')
core = study.core
OUT = study.OUT


def cell(cap):
    shard = OUT/'degree_shards'/f'{cap}.csv'
    if shard.exists():
        cached=pd.read_csv(shard,float_precision='round_trip')
        if len(cached)!=384:raise ValueError('Incomplete diagnostic shard')
        return cached.to_dict('records')
    d = json.loads((OUT/'00_design.json').read_text(encoding='utf-8'))
    n, full_e, _, _ = core.load_study()
    keep = ~n.node_id.isin(d['excluded']).to_numpy()
    nn, _ = formula_precision_nodes(core.capacity_nodes(n,cap),n)
    nn = nn.loc[keep].reset_index(drop=True)
    e = full_e[full_e.Source_id.isin(nn.node_id)&full_e.Target_id.isin(nn.node_id)].reset_index(drop=True)
    degrees = pd.concat([full_e.Source_id,full_e.Target_id]).value_counts().reindex(nn.node_id).to_numpy(float)
    demand = json.loads((OUT/'frozen_inputs/demands.json').read_text(encoding='utf-8'))
    rows=[]
    for theta in d['thresholds']:
        cfg={arm:ModelConfig.for_arm(arm,theta_risk=theta,theta_source='Historical grid; post hoc normalization diagnostic') for arm in ['B','C']}
        allocators={arm:ConfiguredDispatchAllocator(nn,e,cfg[arm]) for arm in cfg}
        caches={};pressures={}
        for case in d['cases']:
            for name in d['demand_names']:
                parts=np.array(demand[name]['parts'])[:,keep]
                for arm in cfg:
                    en=ConfiguredDualMechanismRecovery(allocators[arm],e,parts,demand[name]['groups'],case['initial_ids'],case['kind'],cfg[arm])
                    en.degree=degrees.copy()
                    en.cache_namespace += '|full_graph_degree_reference'
                    en.cache=caches.setdefault((arm,name),{})
                    en.pressure_cache=pressures.setdefault((arm,case['kind']),{})
                    f,m,rounds=en.closure(en.initial)
                    if m['Max_LP_Violation']>1e-5:raise AssertionError('LP accounting')
                    rows.append(dict(capacity_id=cap,theta=theta,case_id=case['case_id'],kind=case['kind'],
                        demand=name,arm=arm,nodes=242,final_failed=len(f),rounds=rounds,
                        final_ids_json=json.dumps(sorted(f),ensure_ascii=False),**m))
    study.save(rows,shard)
    return rows


def main():
    start=time.perf_counter()
    (OUT/'degree_shards').mkdir(exist_ok=True)
    existing_design=(OUT/'11_degree_reference_design.json').exists()
    d=json.loads((OUT/'00_design.json').read_text(encoding='utf-8'))
    prior_hash=sha(OUT/'04_model_runs.csv')
    design=dict(post_hoc_diagnostic=True,reason='Large service changes after role exclusion may include degree renormalization',
        expected_runs=1152,arms=['B','C'],capacity_ids=d['capacity_ids'],cases=d['cases'],thresholds=d['thresholds'],
        demand_names=d['demand_names'],fixed='Same 242 nodes, edges, tasks and capacities as primary exclusion; only risk degree denominators use full-246 reference',
        no_new_empirical_validation=True,source_sha256=sha(__file__),primary_results_sha256=prior_hash)
    if not existing_design:
        study.dump(design,OUT/'11_degree_reference_design.json')
    else:
        prior=json.loads((OUT/'11_degree_reference_design.json').read_text(encoding='utf-8'))
        assert prior['primary_results_sha256']==prior_hash and prior['expected_runs']==1152
        study.dump(dict(numerical_experiment_unchanged=True,source_sha256=sha(__file__),
            reason='Canonical 12-significant-digit theta join key avoids CSV parsing mismatch; preserve original theta for computations',
            source_design_sha256=sha(OUT/'11_degree_reference_design.json')),OUT/'11b_csv_join_fix.json')
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(cell,cap):cap for cap in d['capacity_ids']}
        for f in as_completed(futures):
            rows.extend(f.result());print('degree diagnostic',futures[f],flush=True)
    r=pd.DataFrame(rows)
    original=pd.read_csv(OUT/'04_model_runs.csv',float_precision='round_trip')
    for frame in [r,original]:frame['theta_key']=frame.theta.map(lambda value:format(value,'.12g'))
    keys=['capacity_id','theta_key','case_id','kind','demand','arm']
    r=r.sort_values(keys).reset_index(drop=True)
    old=original[original.scope.eq('matched_tasks246')].set_index(keys).sort_index()
    induced=original[original.scope.eq('reduced242')].set_index(keys).sort_index()
    fixed=r.set_index(keys).sort_index()
    effect=fixed[[]].copy()
    effect['fixed_degree_change_pp']=100*(fixed.Service_Rate-old.Service_Rate)
    effect['recomputed_degree_change_pp']=100*(induced.Service_Rate-old.Service_Rate)
    effect['normalization_contrast_pp']=100*(fixed.Service_Rate-induced.Service_Rate)
    checks=[]
    def check(k,v):
        checks.append(dict(check=k,passed=bool(v)))
        if not v:raise AssertionError(k)
    check('all 1152 predeclared diagnostic trajectories',len(r)==1152)
    check('same task denominators',float((fixed.Demand_Load-old.Demand_Load).abs().max())<1e-7)
    check('accounting',r.Max_LP_Violation.max()<=1e-5 and r.Service_Conservation_Error.max()<=1e-7)
    for index,row in fixed[fixed.index.get_level_values('arm')=='C'].iterrows():
        old_ids=set(json.loads(old.loc[index,'final_ids_json']))-set(d['excluded'])
        check('fixed-reference C risk closure subset '+str(index),set(json.loads(row.final_ids_json))<=old_ids)
    pivot=r.pivot(index=keys[:-1],columns='arm',values='Service_Rate')
    check('safe-policy bound with fixed degree',(pivot.C-pivot.B).min()>=-1e-9)
    check('primary results unchanged',sha(OUT/'04_model_runs.csv')==prior_hash)
    for data,name in [(r,'12_fixed_degree_runs.csv'),(effect.reset_index(),'13_degree_normalization_effect.csv'),(checks,'14_degree_checks.csv')]:study.save(data,OUT/name)
    c=effect.reset_index().query('arm == "C"')
    findings=dict(completed=True,post_hoc_diagnostic=True,runs=len(r),checks=len(checks),passed=sum(v['passed'] for v in checks),
        C_recomputed_degree_min_change_pp=float(c.recomputed_degree_change_pp.min()),
        C_fixed_degree_min_change_pp=float(c.fixed_degree_change_pp.min()),
        C_fixed_degree_max_change_pp=float(c.fixed_degree_change_pp.max()),
        C_normalization_max_contrast_pp=float(c.normalization_contrast_pp.max()),
        elapsed_seconds=time.perf_counter()-start)
    study.dump(findings,OUT/'15_degree_findings.json')
    print(json.dumps(findings,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
