"""Independent geometry, accounting, risk-closure and sampler kernel checks."""
import argparse,itertools,json,random
from pathlib import Path
import numpy as np
import pandas as pd
from closed_loop_recovery_v3 import HERE
from submission_strengthening_common import module
core=module('83_dispatch_mainline_core');OUT=core.OUT

def sampler_check():
    generator=module('85c_generate_constrained_nulls_fixed_steps');rng=np.random.default_rng(2026101804)
    rows=[]
    for trial in range(8):
        count=5;allpairs=list(itertools.combinations(range(count),2));bins=np.full((count,count),-1,dtype=int)
        for a,b in allpairs:bins[a,b]=bins[b,a]=int(rng.integers(0,2))
        original=set(allpairs[i] for i in rng.choice(len(allpairs),size=5,replace=False));degree=np.bincount(np.array(sorted(original)).ravel(),minlength=count)
        hist=np.bincount([bins[e] for e in original],minlength=2);states=[]
        for comb in itertools.combinations(allpairs,len(original)):
            if np.array_equal(np.bincount(np.array(comb).ravel(),minlength=count),degree) and np.array_equal(np.bincount([bins[e] for e in comb],minlength=2),hist):states.append(frozenset(comb))
        lookup={s:i for i,s in enumerate(states)};transition=np.zeros((len(states),len(states)))
        for ix,state in enumerate(states):
            adj=[set() for _ in range(count)]
            for a,b in state:adj[a].add(b);adj[b].add(a)
            for a in range(count):
                eligible=[b for b in adj[a] if any(bins[a,c]==bins[a,b] and c not in adj[a] for c in range(count))]
                for b in eligible:
                    choices=[c for c in range(count) if bins[a,c]==bins[a,b] and c not in adj[a]]
                    for c in choices:
                        ds=[d for d in adj[c] if d not in {a,b,c} and d not in adj[b] and bins[b,d]>=0 and bins[b,d]==bins[c,d]]
                        for d in ds:
                            nb=(adj[b]-{a})|{d};nc=(adj[c]-{d})|{a}
                            rev=[v for v in nb if v not in {a,b,c} and v not in nc and bins[c,v]>=0 and bins[c,v]==bins[b,v]]
                            alpha=min(1.,len(ds)/len(rev))
                            new=(set(state)-{tuple(sorted((a,b))),tuple(sorted((c,d)))})|{tuple(sorted((a,c))),tuple(sorted((b,d)))}
                            probability=1/count/len(eligible)/len(choices)/len(ds)*alpha
                            transition[ix,lookup[frozenset(new)]]+=probability
            transition[ix,ix]+=1-transition[ix].sum()
        error=float(np.abs(transition-transition.T).max());ok=error<1e-12 and np.allclose(transition.sum(axis=1),1,atol=1e-12) and transition.min()>=-1e-12
        if not ok:raise AssertionError('Independently enumerated kernel not reversible')
        pairs,trace,attempts=generator.chain(sorted(original),bins,count,202610180400+trial,1000)
        actual=np.bincount(np.asarray(pairs).ravel(),minlength=count)
        if attempts!=1000 or not np.array_equal(actual,degree) or not np.array_equal(np.bincount([bins[e] for e in pairs],minlength=2),hist):raise AssertionError('Sampler implementation constraint failure')
        rows.append(dict(trial=trial,enumerated_states=len(states),maximum_transition_asymmetry=error,fixed_proposals=attempts,passed=ok))
    core.save(rows,OUT/'87a_exact_sampler_kernel_checks.csv')
    core.dump(dict(completed=True,checks=len(rows),passed=len(rows),sampler_sha256=core.sha(HERE/'85c_generate_constrained_nulls_fixed_steps.py'),
                   validator_sha256=core.sha(__file__),scope='Exact five-node constrained transition matrices and implementation invariant checks; not a mixing-time proof'),OUT/'87_sampler_completion.json')
    print('87 exact sampler kernel checks PASS',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sampler-only',action='store_true');args=parser.parse_args()
    sampler_check()
    if args.sampler_only:return
    for name in ['84_completion.json','85_completion_v3.json','86_completion_v3.json']:
        record=json.loads((OUT/name).read_text(encoding='utf-8'))
        if not record['completed']:raise RuntimeError('Incomplete stage '+name)
    design=json.loads((OUT/'84_design.json').read_text(encoding='utf-8'));nd=json.loads((OUT/'85_design_v3.json').read_text(encoding='utf-8'));rd=json.loads((OUT/'86_design_v3.json').read_text(encoding='utf-8'))
    for d in [design,nd,rd]:core.check_sources(d)
    n,e,demands,old=core.load_study();pos={k:i for i,k in enumerate(n.node_id)};candidate=core.candidate_edges(n)
    lookup={tuple(sorted((row.Source_id,row.Target_id))):row for row in candidate.itertuples()}
    degree0=pd.concat([e.Source_id,e.Target_id]).value_counts().reindex(n.node_id,fill_value=0)
    hist0=np.bincount(core.distance_bin(e.Distance_km.to_numpy()),minlength=6);checks=[]
    def check(label,ok):
        checks.append(dict(check=label,passed=bool(ok)))
        if not ok:raise AssertionError(label)
    edge_hashes=[];common=set(tuple(sorted((a,b))) for a,b in zip(e.Source_id,e.Target_id));original=set(common)
    generator=module('85c_generate_constrained_nulls_fixed_steps')
    for index in range(50):
        graph_id=f'null_{index:02d}';folder=OUT/'null_graphs_v3'/graph_id;core.valid_completion(folder);ee=pd.read_csv(folder/'edges.csv')
        degree=pd.concat([ee.Source_id,ee.Target_id]).value_counts().reindex(n.node_id,fill_value=0);pairs={tuple(sorted((a,b))) for a,b in zip(ee.Source_id,ee.Target_id)}
        check('exact null degree and simple edge count '+graph_id,np.array_equal(degree,degree0) and len(ee)==len(e)==len(pairs) and all(a!=b for a,b in pairs))
        check('exact spatial bins '+graph_id,np.array_equal(np.bincount(core.distance_bin(ee.Distance_km.to_numpy()),minlength=6),hist0))
        check('connected final null '+graph_id,generator.connected([(pos[a],pos[b]) for a,b in pairs],len(n)))
        for row in ee.itertuples():
            target=lookup[tuple(sorted((row.Source_id,row.Target_id)))]
            if any(abs(float(getattr(row,c))-float(getattr(target,c)))>1e-10 for c in core.CONSUMED+['Distance_km','Duration_mins']):raise AssertionError('Incorrect reconstructed pair attributes')
        check('all pair-specific null attributes '+graph_id,True);common&=pairs;edge_hashes.append(core.sha(folder/'edges.csv'))
        plan=OUT/'null_policy_cells_v3'/graph_id;core.valid_completion(plan)
        recorded=json.loads((plan/'completion.json').read_text(encoding='utf-8'));check('policy uses frozen null edges '+graph_id,recorded['edge_sha256']==edge_hashes[-1])
    check('fifty distinct nulls',len(set(edge_hashes))==50)
    mainr=pd.read_csv(OUT/'84a_capacity_policy_results.csv');nullr=pd.read_csv(OUT/'86a_null_policy_results.csv')
    allr=pd.concat([mainr,nullr],ignore_index=True);check('main and null full grid',len(mainr)==1728 and len(nullr)==3600 and not allr.duplicated(['capacity_id','graph_id','theta','case_id','demand','arm']).any())
    demand_totals={name:float(parts.sum()) for name,(parts,_) in demands.items()};cases={v['case_id']:v for v in old['training_cases']+old['evaluation_cases']}
    for index,row in enumerate(allr.itertuples()):
        failed=set(json.loads(row.final_failed_indices_json));seeds={pos[k] for k in cases[row.case_id]['initial_ids']}
        ok=(len(failed)==row.final_failed and seeds<=failed and row.Overload_Only_Failures+row.Risk_Only_Failures+row.Both_Failures==len(failed)-len(seeds) and
            abs(row.Demand_Load-demand_totals[row.demand])<1e-7 and abs(row.Served_Load+row.Unserved_Load-row.Demand_Load)<1e-7 and
            abs(row.Attempted_Assigned_Load+row.Undispatched_Load-row.Demand_Load)<1e-7 and
            abs(row.Uncompleted_Assigned_Load-(row.Attempted_Assigned_Load-row.Served_Load))<1e-7 and
            row.Served_Load<=row.Attempted_Assigned_Load+1e-7 and row.Attempted_Assigned_Load<=row.Demand_Load+1e-7)
        check('independent physical accounting '+str(index),ok)
    # Independent pressure implementation from the already-preserved structural audit.
    risk_module=module('80_audit_dual_mechanism_triviality')
    for graph_id,g in allr[allr.arm.eq('C')].groupby('graph_id'):
        ee=e if graph_id=='original' else pd.read_csv(OUT/'null_graphs_v3'/graph_id/'edges.csv');risk=risk_module.IndependentRisk(n,ee)
        for row in g.itertuples():
            fr=risk.closure(cases[row.case_id]['initial_ids'],row.kind,row.theta)
            check('independent C risk closure '+str(row.Index),{pos[k] for k in fr}==set(json.loads(row.final_failed_indices_json)))
    rounds=pd.concat([pd.read_csv(OUT/'84b_cascade_rounds.csv'),pd.read_csv(OUT/'86b_null_cascade_rounds.csv')],ignore_index=True)
    keys=['capacity_id','graph_id','theta','case_id','demand','arm'];indexed=allr.set_index(keys)
    for key,g in rounds.groupby(keys):
        target=indexed.loc[key];g=g.sort_values('round')
        check('round-to-terminal accounting '+str(key),g.failed_count.iloc[0]==target.initial_count and g.new_failures.iloc[-1]==0 and
              g.failed_count.iloc[-1]==target.final_failed and g.new_failures.sum()==target.final_failed-target.initial_count and
              abs(g.Service_Rate.iloc[-1]-target.Service_Rate)<1e-10 and (g.failed_count.diff().dropna()>=0).all())
    oldr=pd.read_csv(core.ROOT/'项目结果/dual_mechanism_triviality_audit_20261018/80c_original_network_policy_results.csv')
    baseline=mainr[mainr.capacity_id.eq('baseline')].merge(oldr[oldr.arm.isin(['A','B','C'])],on=['theta','case_id','demand','arm'],suffixes=('_new','_old'),validate='one_to_one')
    check('baseline pilot unchanged',len(baseline)==576 and np.allclose(baseline.Service_Rate,baseline.service,atol=1e-11,rtol=0))
    control_design=json.loads((OUT/'84b_control_design.json').read_text(encoding='utf-8'));core.check_sources(control_design)
    controls=pd.read_csv(OUT/'84c_no_damage_results.csv')
    expected_controls=3*len(control_design['kinds'])*3*4*3
    check('matched no-damage grid complete',len(controls)==expected_controls and controls.initial_count.eq(0).all())
    for row in controls.itertuples():
        ok=(abs(row.Served_Load+row.Unserved_Load-row.Demand_Load)<1e-7 and
            abs(row.Attempted_Assigned_Load+row.Undispatched_Load-row.Demand_Load)<1e-7 and
            row.Overload_Only_Failures+row.Risk_Only_Failures+row.Both_Failures==row.final_failed)
        if row.arm=='C':ok=ok and row.final_failed==0 and row.Max_Utilization<=1+1e-7
        check('independent no-damage control accounting '+str(row.Index),ok)
    cold=[]
    for cap in core.CAP_IDS:
        for arm in ['A','B','C']:
            for demand in ['nominal','external_plus150']:cold.append(('original',cap,arm,demand,'test_random_00'))
    for graph in ['null_00','null_24','null_49']:
        for arm in ['A','B','C']:cold.append((graph,nd['target_capacity'],arm,'external_plus150','test_channel_00'))
    from dual_mechanism_recovery_v5 import DispatchAllocatorV5,DualMechanismRecoveryV5
    for graph,cap,arm,demand,case_id in cold:
        nn=core.capacity_nodes(n,cap);ee=e if graph=='original' else pd.read_csv(OUT/'null_graphs_v3'/graph/'edges.csv')
        allocator=DispatchAllocatorV5(nn,ee,1. if arm=='C' else 1.1,'min_cost' if arm=='A' else 'min_overload',2)
        parts,groups=demands[demand];case=cases[case_id];engine=DualMechanismRecoveryV5(allocator,ee,parts,groups,case['initial_ids'],case['kind'],.2,gamma=1.)
        failed,m,_=engine.closure(engine.initial);target=indexed.loc[(cap,graph,.2,case_id,demand,arm)]
        check('cold uncached replay '+str((graph,cap,arm,demand)),abs(m['Service_Rate']-target.Service_Rate)<1e-10 and
              {pos[k] for k in failed}==set(json.loads(target.final_failed_indices_json)))
    backup=Path(design['backup']);records=json.loads((backup/'backup_manifest.json').read_text(encoding='utf-8'))
    for v in records:check('preserved source and backup '+v['path'],core.sha(core.ROOT/v['path'])==v['sha256'] and core.sha(backup/v['path'])==v['sha256'])
    core.save(checks,OUT/'87_validation.csv');core.save([dict(original_edges=len(original),common_original_edges_across_nulls=len(common),common_fraction=len(common)/len(original))],OUT/'87b_null_support_diagnostics.csv')
    core.dump(dict(completed=True,checks=len(checks),passed=sum(v['passed'] for v in checks),cold_replays=len(cold),preserved_files=len(records),
                   common_original_edges=len(common),common_fraction=len(common)/len(original),source_sha256=core.sha(__file__),mixing_time_proven=False),OUT/'87_completion.json')
    print('87 independent mainline validation complete',flush=True)

if __name__=='__main__':main()
