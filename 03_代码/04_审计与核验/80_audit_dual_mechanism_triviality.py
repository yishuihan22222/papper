"""Test whether risk-first certification adds anything to strict dispatch C.

No production rule/data edits. C uses the existing V5 allocator with rho=1.
Risk closure is recomputed independently of the production pressure function.
"""
import hashlib,itertools,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
from closed_loop_recovery_v3 import HERE,load_inputs,DemandModel
from dual_mechanism_recovery_v5 import DispatchAllocatorV5,DualMechanismRecoveryV5
from protect_repair_v6 import ProtectRepairStudy,BoundedCache

ROOT=HERE.parent.parent;OUT=ROOT/'项目结果/dual_mechanism_triviality_audit_20261018'
OLD=ROOT/'项目结果/protect_repair_v6_20261017'
SEED=2026101802
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(v,name):pd.DataFrame(v).to_csv(OUT/name,index=False,encoding='utf-8-sig')

class IndependentRisk:
    def __init__(self,n,e):
        self.ids=n.node_id.tolist();self.pos={k:i for i,k in enumerate(self.ids)};z=len(n)
        self.w=np.zeros((z,z));self.adj=np.zeros((z,z),dtype=bool);self.degree=np.zeros(z)
        for edge in e.itertuples():
            i=self.pos[edge.Source_id];j=self.pos[edge.Target_id]
            value=(.21+.16*edge.Failure_probability_proxy+.12*edge.Cross_district+
                   .10*min(max(edge.Composite_transport_cost,.001)/.45,1)+.09*(1-min(edge.Edge_capacity_proxy/55,1)))/.68
            self.w[i,j]+=value;self.w[j,i]+=value
            self.adj[i,j]=True;self.adj[j,i]=True;self.degree[i]+=1;self.degree[j]+=1
        self.w/=np.maximum(self.degree,1)[:,None]
        self.bonus={
            'spatial_cluster':.1*n['区县'].isin(['东西湖区','新洲区','黄陂区']).to_numpy(float),
            'channel_disruption':.1*n.family.isin(['airport','rail','port']).to_numpy(float)}
    def scores(self,f,kind):
        mask=np.array([k in f for k in self.ids]);return self.adj[:,mask].sum(axis=1),self.w[:,mask].sum(axis=1)+self.bonus.get(kind,0.)
    def closure(self,f,kind,theta):
        current=frozenset(f)
        for _ in range(len(self.ids)+1):
            count,p=self.scores(current,kind)
            new={self.ids[i] for i in range(len(self.ids)) if self.ids[i] not in current and count[i]>0 and p[i]>=theta}
            if not new:return current
            current=current|new
        raise RuntimeError('Independent risk closure did not terminate')

def compare(n,e,parts,groups,seeds,kind,theta,tag,allocators,states,risk):
    tag=dict(tag,theta=theta)
    outputs=[];checks=[];engines={};values={};fr=risk.closure(seeds,kind,theta)
    def check(label,ok):
        checks.append(dict(**tag,check=label,passed=bool(ok)))
        if not ok:raise AssertionError(str(tag)+' '+label)
    for arm in ['A','B','C']:
        en=DualMechanismRecoveryV5(allocators[arm],e,parts,groups,seeds,kind,theta,gamma=1.)
        en.cache=states[arm];engines[arm]=en
        failed,metric,_=en.closure(seeds);values[arm]=(failed,metric)
        outputs.append(dict(**tag,arm=arm,initial_count=len(seeds),risk_closure_count=len(fr),
                            failed=len(failed),service=metric['Service_Rate'],overload_only=metric['Overload_Only_Failures'],
                            risk_only=metric['Risk_Only_Failures'],both=metric['Both_Failures'],
                            failed_json=json.dumps(sorted(failed),ensure_ascii=False)))
    fc,mc=values['C'];cert_metric=engines['C'].state(fr)[0]
    outputs.append(dict(**tag,arm='certificate',initial_count=len(seeds),risk_closure_count=len(fr),
                        failed=len(fr),service=cert_metric['Service_Rate'],overload_only=0,risk_only=len(fr)-len(seeds),both=0,
                        failed_json=json.dumps(sorted(fr),ensure_ascii=False)))
    check('C failed set equals independent risk closure',fc==fr)
    check('certificate equals C terminal service',abs(cert_metric['Service_Rate']-mc['Service_Rate'])<1e-10)
    check('C has no overload triggers',mc['Overload_Only_Failures']==0 and mc['Both_Failures']==0)
    for f in [frozenset(seeds),fr]:
        count,p=risk.scores(f,kind);c0,p0=engines['C'].pressure(f)
        check('independent pressure matches '+str(len(f)),np.array_equal(count,c0) and np.allclose(p,p0,atol=1e-12,rtol=0))
    for arm in ['A','B']:
        f,m=values[arm]
        check('risk closure forced in '+arm,fr<=f)
        check('C weakly dominates terminal service '+arm,mc['Service_Rate']>=m['Service_Rate']-1e-9)
    return outputs,checks,engines['C'],fr

def allocs(n,e):return {arm:DispatchAllocatorV5(n,e,1. if arm=='C' else 1.1,'min_cost' if arm=='A' else 'min_overload',2) for arm in ['A','B','C']}
def fresh_states():return {arm:BoundedCache(2048) for arm in ['A','B','C']}

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    original=json.loads((OLD/'71_design.json').read_text(encoding='utf-8'))
    sources=[Path(__file__),HERE/'dual_mechanism_recovery_v5.py',HERE/'closed_loop_recovery_v3.py',HERE/'v4_parameter_audit_core.py',
             HERE/'protect_repair_v6.py',OLD/'71_design.json']+[HERE.parent/'data'/v for v in ['04b_武汉节点代理容量.csv','04c_武汉边代理属性.csv','04a_城市综合指标_含估算.csv','04_cross_layer_edges.csv']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    cases=original['training_cases']+original['evaluation_cases'];thresholds=sorted({float(v['theta']) for v in original['models']})
    design=dict(seed=SEED,scope='Structural audit and A/B/C pilot on unchanged uniform-utilization capacities; not the heterogeneous-capacity main study',
                theory_conditions='Fixed exogenous shock and demand; nonnegative flow-independent risk; gamma>=1; free rejection; no queues, new shocks, repair or intervention; fixed capacities; total completed service objective',
                cases=cases,thresholds=thresholds,demand_names=original['training_demands']+original['evaluation_demands'],sources=hashes,toy_networks=50)
    (OUT/'80_design.json').write_text(json.dumps(design,ensure_ascii=False,indent=2),encoding='utf-8')
    rng=np.random.default_rng(SEED);toy=[];checks=[];upper=[]
    for case in range(50):
        count=int(rng.integers(3,7));ids=[f'n{i}' for i in range(count)]
        n=pd.DataFrame(dict(node_id=ids,overload_threshold=rng.integers(1,15,count).astype(float),proxy_daily_load=rng.integers(1,13,count).astype(float),family=['other']*count,区县=['toy']*count))
        pairs={(i,i+1) for i in range(count-1)}
        pairs|={(i,j) for i in range(count) for j in range(i+1,count) if rng.random()<.3}
        e=pd.DataFrame([dict(Source_id=ids[i],Target_id=ids[j],Failure_probability_proxy=float(rng.random()),Cross_district=int(rng.integers(0,2)),
                             Composite_transport_cost=float(rng.uniform(.01,.6)),Edge_capacity_proxy=float(rng.uniform(1,80)),Repair_cost_proxy=1.) for i,j in sorted(pairs)])
        parts=n.proxy_daily_load.to_numpy()[None,:];seeds={ids[int(rng.integers(count))]};theta=float(rng.uniform(.1,1.05));risk=IndependentRisk(n,e)
        rows,ck,c,fr=compare(n,e,parts,['toy'],seeds,'toy',theta,dict(scope='toy',case_id=str(case),demand='toy'),allocs(n,e),fresh_states(),risk)
        toy+=rows;checks+=ck;remaining=sorted(set(ids)-fr);v=c.state(fr)[0]['Service_Rate']
        for bits in itertools.product([False,True],repeat=len(remaining)):
            f=fr|{k for k,b in zip(remaining,bits) if b};m=c.state(f)[0]
            ok=m['Service_Rate']<=v+1e-10
            assert ok
            upper.append(dict(case=case,failed_count=len(f),risk_closure_count=len(fr),terminal_bound=v,subset_service=m['Service_Rate'],passed=ok))
    save(toy,'80a_toy_policy_results.csv');save(upper,'80b_enumerated_terminal_bounds.csv')
    print('80 toy networks and terminal-set enumeration PASS',flush=True)
    n,e,ci,cross=load_inputs();dm=DemandModel(n,e,ci,cross);study=ProtectRepairStudy(n,e,dm);risk=IndependentRisk(n,e)
    allocators=allocs(n,e);states={d:fresh_states() for d in design['demand_names']};real=[]
    for theta in thresholds:
        for case in cases:
            for demand in design['demand_names']:
                parts,groups=study.demands[demand]
                tag=dict(scope=case['shock_scope'],case_id=case['case_id'],demand=demand,theta=theta)
                rows,ck,_,_=compare(n,e,parts,groups,set(case['initial_ids']),case['kind'],theta,tag,allocators,states[demand],risk)
                real+=rows;checks+=ck
            print('80 original network theta',theta,'case',case['case_id'],'PASS',flush=True)
    save(real,'80c_original_network_policy_results.csv');save(checks,'80_validation.csv')
    table=pd.DataFrame(real);wide=table.pivot(index=['theta','case_id','demand','scope'],columns='arm',values='service').reset_index()
    for arm in ['A','B','C']:wide['certificate_minus_'+arm+'_pp']=100*(wide.certificate-wide[arm])
    save(wide,'80d_certificate_policy_pairs.csv')
    def counts(s):return dict(wins=int((s>1e-8).sum()),ties=int((s.abs()<=1e-8).sum()),losses=int((s< -1e-8).sum()),mean_pp=float(s.mean()),min_pp=float(s.min()),max_pp=float(s.max()))
    for p,h in hashes.items():assert sha(ROOT/p)==h,p
    result=dict(completed=True,toy_networks=50,toy_terminal_subsets=len(upper),original_conditions=len(wide),original_policy_rows=len(table),
                checks=len(checks),passed=sum(v['passed'] for v in checks),subset_bounds_passed=len(upper),
                comparisons={arm:counts(wide['certificate_minus_'+arm+'_pp']) for arm in ['A','B','C']},
                elapsed_seconds=time.perf_counter()-start,python=platform.python_version(),source_sha256=sha(__file__),
                conclusion='Under the audited static model certificate dispatch equals C, so beating B alone cannot establish an additional certificate-method contribution.',
                numerical_checks_not_proofs=True,no_new_capacity_data=True,production_sources_unchanged=True)
    (OUT/'80_completion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
