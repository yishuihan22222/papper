"""Three capacity versions, identical fixed demand and initial shocks."""
import argparse,gc,json,shutil,time
from datetime import datetime
from pathlib import Path
import pandas as pd
from closed_loop_recovery_v3 import HERE
from submission_strengthening_common import module
core=module('83_dispatch_mainline_core');OUT=core.OUT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    n,e,demands,old=core.load_study();thresholds=sorted({float(v['theta']) for v in old['models']});cases=old['training_cases']+old['evaluation_cases']
    paths=[Path(__file__),HERE/'83_dispatch_mainline_core.py',HERE/'dual_mechanism_recovery_v5.py',HERE/'closed_loop_recovery_v3.py',HERE/'v4_parameter_audit_core.py',
           HERE/'protect_repair_v6.py',HERE/'80_audit_dual_mechanism_triviality.py',HERE/'82_build_fixed_total_capacity_scenarios.py',core.OLD/'71_design.json',
           core.CAP/'82a_assumed_capacity_scenarios.csv',core.CAP/'82_completion.json']
    paths +=[core.DATA/v for v in ['04b_武汉节点代理容量.csv','04c_武汉边代理属性.csv','04c_武汉边代理属性_候选全边.csv','04a_城市综合指标_含估算.csv','04_cross_layer_edges.csv']]
    design=dict(capacity_ids=core.CAP_IDS,thresholds=thresholds,cases=cases,demand_names=old['training_demands']+old['evaluation_demands'],
                policies={'A':'rho1.1 maximum assignment then min cost','B':'rho1.1 maximum assignment then min excess then min cost','C':'rho1 strict capacity maximum assignment then min cost'},
                gamma=1.,capacity_unit='existing proxy unit per slot; synthetic heterogeneity',real_capacity_replacements=0,
                demand_rule='Original demand components frozen across capacity versions and rewired graphs; no rebalancing after capacity changes',
                edge_rule='Original edges and risk attributes fixed for capacity comparisons; node capacity_index frozen as risk-attribute covariate',sources=core.source_hashes(paths))
    path=OUT/'84_design.json'
    if path.exists():
        existing=json.loads(path.read_text(encoding='utf-8'))
        if existing['sources']!=design['sources']:raise RuntimeError('Changed capacity study source')
        design=existing
    else:
        backup=core.ROOT/'历史存档/pass_data'/('主线容量零模型前快照_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        backup.mkdir(parents=True);records=[]
        for p in paths:
            rel=p.relative_to(core.ROOT);target=backup/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
            if core.sha(target)!=core.sha(p):raise RuntimeError('Backup mismatch')
            records.append(dict(path=str(rel),sha256=core.sha(p)))
        core.dump(records,backup/'backup_manifest.json');design['backup']=str(backup);core.dump(design,path)
    core.check_sources(design);start=time.perf_counter()
    for cap_id in core.CAP_IDS:
        folder=OUT/'capacity_cells'/cap_id;folder.mkdir(parents=True,exist_ok=True)
        if args.resume and core.valid_completion(folder):continue
        nn=core.capacity_nodes(n,cap_id)
        def progress(theta,case):
            core.dump(dict(current=cap_id,theta=theta,case=case,elapsed_seconds=time.perf_counter()-start),OUT/'84_progress.json')
            print('84',cap_id,theta,case,'complete',flush=True)
        rows,rounds,checks=core.run_grid(nn,e,demands,cases,thresholds,design['demand_names'],dict(capacity_id=cap_id,graph_id='original'),progress=progress)
        for data,name in [(rows,'results.csv'),(rounds,'rounds.csv'),(checks,'validation.csv')]:core.save(data,folder/name)
        core.completion(folder,['results.csv','rounds.csv','validation.csv'],dict(sources=design['sources'],rows=len(rows),checks=len(checks),passed=sum(v['passed'] for v in checks)))
        gc.collect()
    for name,target in [('results.csv','84a_capacity_policy_results.csv'),('rounds.csv','84b_cascade_rounds.csv'),('validation.csv','84_validation.csv')]:
        frames=[pd.read_csv(OUT/'capacity_cells'/cap/name) for cap in core.CAP_IDS];core.save(pd.concat(frames,ignore_index=True),OUT/target)
    core.check_sources(design);v=pd.read_csv(OUT/'84_validation.csv');r=pd.read_csv(OUT/'84a_capacity_policy_results.csv')
    core.dump(dict(completed=True,capacity_versions=3,policy_rows=len(r),conditions=len(r)//3,checks=len(v),passed=int(v.passed.sum()),source_sha256=core.sha(__file__)),OUT/'84_completion.json')
    print('84 capacity A/B/C complete',flush=True)

if __name__=='__main__':main()
