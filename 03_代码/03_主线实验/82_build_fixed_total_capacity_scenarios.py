"""Versioned assumed heterogeneity; never replace capacities with throughput."""
import hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'项目结果/capacity_heterogeneity_design_20261018'
SOURCE=ROOT/'数据及代码/data/04b_武汉节点代理容量.csv'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);before=sha(SOURCE)
    n=pd.read_csv(SOURCE).sort_values('node_id').reset_index(drop=True)
    load=n.proxy_daily_load.to_numpy(float);base=n.overload_threshold.to_numpy(float);families=sorted(n.family.unique())
    rows=[];checks=[];summary=[]
    def record(label,placement,amplitude,seed,capacity):
        ratio=capacity/load;util=load/capacity
        ok=bool(np.isfinite(capacity).all() and (capacity>load).all() and abs(capacity.sum()-base.sum())<1e-7 and
                np.max(np.abs(ratio-base/load))<=amplitude+1e-12)
        checks.append(dict(scenario=label,passed=ok,total_capacity_error=float(capacity.sum()-base.sum())))
        if not ok:raise AssertionError(label)
        for i,node in n.iterrows():
            rows.append(dict(scenario=label,placement=placement,amplitude=amplitude,seed=seed,node_id=node.node_id,family=node.family,
                             baseline_proxy_load=load[i],baseline_proxy_capacity=base[i],assumed_capacity=float(capacity[i]),
                             capacity_load_ratio=float(ratio[i]),nominal_utilization=float(util[i]),
                             evidence_class='original_proxy' if placement=='baseline' else 'synthetic_fixed_total',
                             empirically_calibrated=False,unit='existing_proxy_unit_per_slot'))
        summary.append(dict(scenario=label,placement=placement,amplitude=amplitude,seed=seed,total_capacity=float(capacity.sum()),
                            min_ratio=float(ratio.min()),max_ratio=float(ratio.max()),min_utilization=float(util.min()),
                            median_utilization=float(np.median(util)),max_utilization=float(util.max())))
    record('baseline','baseline',0.,-1,base)
    for seed_index in range(20):
        seed=202610180300+seed_index
        for placement in ['node','family']:
            rng=np.random.default_rng(seed)
            if placement=='node':z=rng.uniform(-1.,1.,len(n))
            else:
                values=dict(zip(families,rng.uniform(-1.,1.,len(families))));z=n.family.map(values).to_numpy(float,copy=True)
            z-=np.dot(load,z)/load.sum();z/=max(float(np.abs(z).max()),1e-12)
            for amplitude in [.1,.2]:
                capacity=base+amplitude*load*z
                label=f'{placement}_a{amplitude:.2f}_s{seed_index:02d}'
                record(label,placement,amplitude,seed,capacity)
    pd.DataFrame(rows).to_csv(OUT/'82a_assumed_capacity_scenarios.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(OUT/'82b_scenario_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(OUT/'82_validation.csv',index=False,encoding='utf-8-sig')
    assert sha(SOURCE)==before
    result=dict(completed=True,scenarios=len(summary),node_rows=len(rows),checks=len(checks),passed=sum(v['passed'] for v in checks),
                seed_range=[202610180300,202610180319],original_input_sha256=before,source_sha256=sha(__file__),python=platform.python_version(),
                real_capacity_replacements=0,all_scenarios_same_total_capacity=True,baseline_loads_unchanged=True,
                family_order='Random family slack contrasts, not airport>port or any asserted empirical ranking',
                future_primary_pilot=['baseline','node_a0.20_s00','family_a0.20_s00'],
                caveat='Scenario uncertainty, not fitted capacity distributions or observed confidence intervals.')
    (OUT/'82_completion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
