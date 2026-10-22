"""Exploratory audit: keep all three existing LP objectives unchanged.

This does not claim novelty. Equivalent optimal dispatches affecting cascades
are already studied in power-grid literature. Only model relevance is tested.
"""
import importlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,csr_matrix,eye,hstack,vstack
from dispatch_policy_rules import DispatchRuleAllocator,DispatchRuleSpec
from configured_dual_mechanism_v5 import ConfiguredDualMechanismRecovery
from matched_admission_v5 import evaluate_flow,intervene

prior=importlib.import_module('122_matched_location_study')
ind=importlib.import_module('123_validate_matched_location')
CODE=Path(__file__).parent;OUT=prior.ROOT/'项目结果/research_value_reaudit_20261020';OLD=prior.OUT


def alternatives(a,payload):
    p=payload['plans']['nearest_cost'];src=np.asarray(p['src']);dst=np.asarray(p['dst']);m=len(src);n=len(a.ids)
    K=a.nominal_capacity;own=np.asarray(p['own']);D=np.asarray(payload['demand']);F=payload['initial_ids']
    R=coo_matrix((np.ones(m),(dst,np.arange(m))),shape=(n,m)).tocsr()
    S=coo_matrix((np.ones(m),(src,np.arange(m))),shape=(n,m)).tocsr()
    paths=a.paths(frozenset(F)).copy();ix=np.flatnonzero([s not in F for s in a.ids]);paths[ix,ix]=0.
    costs=paths[src,dst]
    A=vstack([hstack([R,csr_matrix((n,n))]),hstack([R,-eye(n)])],format='csr')
    b=np.r_[a.dispatch_limit-own,K-own]
    E=vstack([hstack([S,csr_matrix((n,n))]),csr_matrix(np.r_[np.zeros(m),np.ones(n)][None,:]),
        csr_matrix(np.r_[costs,np.zeros(n)][None,:])],format='csr')
    rhs=np.r_[p['q'],p['excess'],p['cost']]
    result={'baseline':evaluate_flow(a,F,D,src,dst,p['flow'],'baseline')}
    for name in ['face_minpeak','face_low_id','face_high_id']:
        aa=A;bb=b;ee=E;cc=np.r_[a.canonical_rank[dst]/max(n-1,1),np.zeros(n)]
        if name=='face_high_id':cc=-cc
        if name=='face_minpeak':
            aa=vstack([hstack([A,csr_matrix((2*n,1))]),hstack([R,csr_matrix((n,n)),-csr_matrix(K[:,None])])],format='csr')
            bb=np.r_[b,-own];ee=hstack([E,csr_matrix((E.shape[0],1))],format='csr');cc=np.r_[np.zeros(m+n),1.]
        sol=linprog(cc,A_ub=aa,b_ub=bb,A_eq=ee,b_eq=rhs,bounds=(0,None),method='highs')
        if not sol.success:raise RuntimeError('Optimal face '+name+': '+sol.message)
        result[name]=evaluate_flow(a,F,D,src,dst,sol.x[:m],name)
        q=result[name]
        if (np.max(abs(q.q-np.asarray(p['q'])))>1e-5 or abs(q.excess-p['excess'])>1e-5
            or abs(q.cost-p['cost'])>1e-5 or abs(q.served-p['served'])>1e-5):raise AssertionError('Objective face mismatch')
    return result


def main():
    OUT.mkdir(exist_ok=True);(OUT/'face_shards').mkdir(exist_ok=True)
    design=json.loads((OLD/'00_design.json').read_text(encoding='utf-8'));prior.verify(design)
    pairs=pd.read_csv(OLD/'07_paired_effects.csv')
    selected=pairs[(pairs.contrast=='primary')&(pairs.mechanism=='both')&(pairs.initial_excess>1e-7)&(pairs.rho>pairs.gamma+1e-7)]
    ids=sorted(selected.condition_id.astype(int));assert len(ids)==150
    jobmap={j['condition_id']:j for j in design['jobs']};graphs={g['graph_id']:g for g in design['graphs']}
    lockfile=OUT/'00_face_screen_design.json'
    if not lockfile.exists():
        prior.dump(dict(schema='full-objective-face-screen-1',selected_ids=ids,selection='All prior conditions with E*>0 and rho>gamma+epsilon, before checking the new probes',
            interventions=600,plans=['baseline','face_minpeak','face_low_id','face_high_id'],
            matching='Same per-source maximum admissions, globally minimum excess, and original minimum transport cost',
            interpretation='Exploratory sample of the full optimal face; absence of a sampled difference is not a global uniqueness proof',
            protected={str(p):prior.old.sha(p) for p in [Path(__file__),OLD/'final_delivery_manifest.json',OLD/'03_interventions.csv',OLD/'04_initial_plans.csv',OLD/'00_design.json']}),lockfile)
    lock=json.loads(lockfile.read_text(encoding='utf-8'));assert lock['selected_ids']==ids
    for p,h in lock['protected'].items():assert prior.old.sha(p)==h
    rows=[];checks=[];started=time.perf_counter()
    def check(name,ok):
        checks.append(dict(check=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    for index,cid in enumerate(ids):
        shard=OUT/'face_shards'/f'{cid:04d}.json'
        if shard.exists():
            v=json.loads(shard.read_text(encoding='utf-8'));assert v['design']==prior.old.sha(lockfile)
            rows.extend(v['rows']);checks.extend(v['checks']);continue
        start=len(checks);j=jobmap[cid];g=graphs[j['graph_id']]
        nn=pd.read_csv(prior.PRIOR/g['nodes_path']);ee=pd.read_csv(prior.PRIOR/g['edges_path'])
        payload=json.loads((OLD/'plans'/f'{cid:04d}.json').read_text(encoding='utf-8'));D=np.asarray(payload['demand'])
        cfg=prior.config_for(j,'both');a=DispatchRuleAllocator(nn,ee,cfg,DispatchRuleSpec(compatibility_scope=g['compatibility']))
        plans=alternatives(a,payload);en=ConfiguredDualMechanismRecovery(a,ee,[D],['audit'],payload['initial_ids'],j['kind'])
        info=ind.physics(nn,ee,cfg.snapshot());batch=[];plan_data={}
        for name,p in plans.items():
            result=intervene(en,p);logs=result.pop('continuation_logs')
            current=set(result['continuation_start'])
            for log in logs:
                metric,u,done=en.state(current);cause=ind.trigger(info,nn,current,u,cfg.snapshot(),j['kind'],'both')
                check('independent update '+str((cid,name,log['round'])),cause['new']==log['new_ids'])
                check('continuation balance '+str((cid,name,log['round'])),abs(np.minimum(u*a.nominal_capacity,a.nominal_capacity).sum()-done.sum())<1e-5)
                current.update(cause['new'])
            check('stable end '+str((cid,name)),sorted(current)==result['final_failed'] and not logs[-1]['new_ids'])
            base=plans['baseline']
            check('all objectives fixed '+str((cid,name)),np.max(abs(p.q-base.q))<1e-5 and abs(p.assigned-base.assigned)<1e-5
                and abs(p.excess-base.excess)<1e-5 and abs(p.served-base.served)<1e-5 and abs(p.cost-base.cost)<1e-5)
            batch.append(dict(condition_id=cid,domain=j['domain'],topology=j['topology'],case_id=j['case_id'],rho=j['rho'],gamma=j['gamma'],theta=j['theta'],
                total_multiplier=j['total_multiplier'],plan=name,initial_assigned=p.assigned,initial_served=p.served,initial_excess=p.excess,initial_cost=p.cost,
                initial_peak_util=float((p.load/a.nominal_capacity).max()),final_service=result['final_service'],
                first_ids_json=json.dumps(result['continuation_start'],ensure_ascii=False),final_ids_json=json.dumps(result['final_failed'],ensure_ascii=False)))
            plan_data[name]=dict(src=p.src.tolist(),dst=p.dst.tolist(),flow=p.flow.tolist(),load=p.load.tolist(),q=p.q.tolist(),result=result,trace=logs)
        prior.dump(dict(design=prior.old.sha(lockfile),rows=batch,plans=plan_data,checks=checks[start:]),shard)
        rows.extend(batch);a.clear_path_cache()
        if (index+1)%25==0:print(json.dumps(dict(face_conditions=index+1,planned=150,elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    prior.save(rows,OUT/'01_full_objective_face_runs.csv');prior.save(checks,OUT/'02_face_screen_checks.csv')
    frame=pd.DataFrame(rows);summary=[]
    for cid,g in frame.groupby('condition_id'):
        summary.append(dict(condition_id=cid,domain=g.domain.iloc[0],first_sets=g.first_ids_json.nunique(),final_sets=g.final_ids_json.nunique(),
            service_range_pp=100*(g.final_service.max()-g.final_service.min()),changed=bool(g.final_service.max()-g.final_service.min()>1e-7)))
    summary=pd.DataFrame(summary);prior.save(summary,OUT/'03_face_screen_summary.csv')
    for p,h in lock['protected'].items():assert prior.old.sha(p)==h
    prior.dump(dict(completed=True,conditions=150,trajectories=600,checks=len(checks),passed=sum(c['passed'] for c in checks),
        changed_conditions=int(summary.changed.sum()),changed_first_sets=int((summary.first_sets>1).sum()),max_service_range_pp=float(summary.service_range_pp.max()),
        independent_global_uniqueness_proof=False),OUT/'04_face_screen_findings.json')
    print((OUT/'04_face_screen_findings.json').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':main()
