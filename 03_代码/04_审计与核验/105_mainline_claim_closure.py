"""Close existing evidence into bounded claims; no model/manuscript edits or LPs."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'项目结果/mainline_claim_closure_20261018'
MAIN=ROOT/'项目结果/dispatch_capacity_null_mainline_20261018'
EDGE=ROOT/'项目结果/dual_mechanism_edge_trace_20261018'
QUEUE=ROOT/'项目结果/reversible_blocking_prototype_20261018'
PEAK=ROOT/'项目结果/reversible_blocking_peak_horizon_20261018'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(x,p):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def save(x,name):pd.DataFrame(x).to_csv(OUT/name,index=False,encoding='utf-8-sig')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'105_completion.json').exists():raise RuntimeError('Preserve completed outputs; create a new version')
    checks=[];refs={}
    for manifest,base,key in [(MAIN/'89_final_manifest.json',ROOT,'artifacts'),
                              (EDGE/'104_completion.json',EDGE,'files'),
                              (QUEUE/'94_completion.json',ROOT,'files'),
                              (PEAK/'95_completion.json',ROOT,'files')]:
        value=json.loads(manifest.read_text(encoding='utf-8'))
        assert value['completed']
        pairs=[(r['path'],r['sha256']) for r in value[key]] if key=='artifacts' else value[key].items()
        for name,h in pairs:
            p=base/name;actual=sha(p)
            assert actual==h,str(p)
            refs[str(p)]=h
        refs[str(manifest)]=sha(manifest)
        checks.append(dict(check='Existing manifest '+manifest.name,passed=True))
    a=pd.read_csv(MAIN/'84a_capacity_policy_results.csv')
    assert len(a)==1728
    keys=['capacity_id','theta','case_id','demand']
    p=a.pivot(index=keys,columns='arm',values='Service_Rate').reset_index()
    p['C_minus_B_pp']=100*(p.C-p.B)
    assert len(p)==576 and p.C_minus_B_pp.min()>=-1e-7
    old=pd.read_csv(MAIN/'88a_capacity_pairs.csv')
    q=p.merge(old[keys+['C_B_pp']],on=keys,validate='one_to_one')
    assert np.allclose(q.C_minus_B_pp,q.C_B_pp,atol=1e-10,rtol=0)
    summary=[]
    for cap,z in p.groupby('capacity_id'):
        for layer,g in [('all_demands',z),('excluding_external_plus150',z[z.demand.ne('external_plus150')])]:
            summary.append(dict(capacity_id=cap,layer=layer,conditions=len(g),
                                wins=int(g.C_minus_B_pp.gt(1e-7).sum()),
                                ties=int(g.C_minus_B_pp.abs().le(1e-7).sum()),
                                losses=int(g.C_minus_B_pp.lt(-1e-7).sum()),
                                mean_pp=float(g.C_minus_B_pp.mean()),median_pp=float(g.C_minus_B_pp.median())))
    save(summary,'105_capacity_policy_recalculation.csv')
    c=a[a.arm.eq('C')]
    assert c.Overload_Only_Failures.eq(0).all() and c.Both_Failures.eq(0).all()
    assert a[a.arm.eq('A')].Service_Rate.eq(0).all()
    controls=pd.read_csv(MAIN/'84c_no_damage_results.csv')
    high=controls[controls.demand.eq('external_plus150')]
    assert high[high.arm.isin(['A','B'])].final_failed.eq(246).all()
    assert high[high.arm.eq('C')].final_failed.eq(0).all()
    assert np.allclose(high[high.arm.eq('C')].Service_Rate,1/1.1,atol=1e-6)
    graphs=pd.read_csv(MAIN/'85a_null_graph_summary.csv')
    null=pd.read_csv(MAIN/'86a_null_policy_results.csv')
    assert len(graphs)==50 and len(null)==3600
    assert graphs.connected.all() and graphs.degree_preserved.all() and graphs.distance_hist_preserved.all()
    qp=pd.read_csv(QUEUE/'94b_paired_results.csv')
    pp=pd.read_csv(PEAK/'95b_paired_results.csv')
    assert len(qp)==5000 and len(pp)==240
    rates=[]
    for label,data,fields in [('94_full_grid',qp,[('buffer_strict_gain','buffer_gain_ticks'),
                                                ('max_accept_suboptimal','max_accept_regret_ticks'),
                                                ('rolling_2_suboptimal','rolling_regret_ticks'),
                                                ('fifo_order_cost','order_cost_ticks')]),
                              ('95_targeted_peak_development',pp,[('buffer_strict_gain','direct_gain'),
                                                                 ('max_accept_suboptimal','max_accept_regret'),
                                                                 ('rolling_2_suboptimal','regret_2'),
                                                                 ('fifo_order_cost','fifo_order_cost')])]:
        for measure,col in fields:
            count=int(data[col].gt(0).sum())
            rates.append(dict(design=label,measure=measure,numerator=count,denominator=len(data),
                              percent=100*count/len(data),
                              interpretation='Synthetic design frequency; not real regime probability or innovation success probability'))
    save(rates,'105_prototype_frequency_audit.csv')
    # Record the source of a tempting 0.6% denominator, without attributing it.
    strat=pd.read_csv(QUEUE/'94j_stratified_summary.csv')
    save(strat,'105_original_94_strata.csv')
    cap=pd.read_csv(ROOT/'项目结果/capacity_heterogeneity_design_20261018/82b_scenario_summary.csv')
    assert len(cap)==81 and np.allclose(cap.total_capacity,cap.total_capacity.iloc[0],atol=1e-6)
    edge=json.loads((EDGE/'101_completion.json').read_text(encoding='utf-8'))
    result=dict(main_rows=len(a),main_pairs=len(p),wins=int(p.C_minus_B_pp.gt(1e-7).sum()),
                ties=int(p.C_minus_B_pp.abs().le(1e-7).sum()),losses=int(p.C_minus_B_pp.lt(-1e-7).sum()),
                no_damage_controls=len(controls),capacity_versions_generated=81,capacity_versions_run=3,
                main_A_zero_service_rows=int(a[a.arm.eq('A')].Service_Rate.eq(0).sum()),
                null_graphs=50,null_policy_rows=len(null),
                original_edge_retention_min=float(graphs.original_edge_overlap.min()),
                original_edge_retention_max=float(graphs.original_edge_overlap.max()),
                edge_runs=edge['runs'],edge_trace_rows=729431,
                prototype_rates=rates,prototype_0_6_percent_as_general_regime_rate_supported=False,
                manuscript_modified=False,production_rules_modified=False,new_simulations_run=False,
                scientific_toolkit='Existing CSV reaggregation and frozen hash verification only',
                python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__)
    checks.extend([dict(check=x,passed=True) for x in [
        '1728 rows -> 576 paired conditions matches saved 88 table',
        'C has no overload triggers in all mainline rows',
        'A zero-service degeneration retained, not used as sole method baseline',
        'Extreme demand no-damage collapse and safe-service control verified',
        '50 null graphs and 3600 policy rows retain declared constraints',
        '81 capacity designs preserve total capacity; only 3 evaluated',
        '5000 and 240 prototype configurations reaggregated separately']])
    save(checks,'105_validation.csv')
    dump(result,OUT/'105_verified_numbers.json')
    dump(dict(completed=True,unique_prior_files_verified=len(refs),
              sources=refs,source_code_sha256=sha(__file__),
              outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}),
         OUT/'105_completion.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('Prior frozen files verified:',len(refs))


if __name__=='__main__':main()
