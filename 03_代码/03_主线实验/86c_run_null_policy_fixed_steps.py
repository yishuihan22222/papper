"""Stream completed constrained nulls through the prespecified A/B/C grid."""
import argparse,gc,json,time
from pathlib import Path
import pandas as pd
from closed_loop_recovery_v3 import HERE
from submission_strengthening_common import module
core=module('83_dispatch_mainline_core');OUT=core.OUT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--wait-for-graphs',action='store_true');args=ap.parse_args()
    nd=json.loads((OUT/'85_design_v3.json').read_text(encoding='utf-8'));core.check_sources(nd)
    n,_,demands,old=core.load_study();n=core.capacity_nodes(n,nd['target_capacity']);cases=[v for v in old['training_cases']+old['evaluation_cases'] if v['case_id'] in nd['cases']]
    if len(cases)!=4:raise ValueError('Missing prespecified null shock')
    paths=[Path(__file__),HERE/'83_dispatch_mainline_core.py',HERE/'dual_mechanism_recovery_v5.py',HERE/'closed_loop_recovery_v3.py',HERE/'v4_parameter_audit_core.py',
           HERE/'protect_repair_v6.py',OUT/'85_design_v3.json',core.CAP/'82a_assumed_capacity_scenarios.csv',core.OLD/'71_design.json']
    paths+=[core.DATA/v for v in ['04b_武汉节点代理容量.csv','04c_武汉边代理属性.csv','04a_城市综合指标_含估算.csv','04_cross_layer_edges.csv']]
    design=dict(graphs=50,capacity_id=nd['target_capacity'],cases=cases,thresholds=nd['thresholds'],demands=nd['demands'],sources=core.source_hashes(paths),
                nominal_graph_comparator='Original graph, same capacity version and 24 conditions selected from 84a; all node demands frozen')
    p=OUT/'86_design_v3.json'
    if p.exists():
        existing=json.loads(p.read_text(encoding='utf-8'))
        if existing['sources']!=design['sources']:raise RuntimeError('Changed null-policy source')
        design=existing
    else:core.dump(design,p)
    core.check_sources(design);start=time.perf_counter()
    for index in range(50):
        graph_id=f'null_{index:02d}';folder=OUT/'null_policy_cells_v3'/graph_id;folder.mkdir(parents=True,exist_ok=True)
        if args.resume and core.valid_completion(folder):continue
        graph=OUT/'null_graphs_v3'/graph_id;waiting=time.perf_counter()
        while not (graph/'completion.json').exists():
            if not args.wait_for_graphs:raise RuntimeError('Generate null graph first '+graph_id)
            if time.perf_counter()-waiting>900:raise RuntimeError('Null generator did not supply graph within 15 minutes '+graph_id)
            time.sleep(1.)
        core.valid_completion(graph);e=pd.read_csv(graph/'edges.csv')
        def progress(theta,case):
            core.dump(dict(completed_graphs=index,current=graph_id,theta=theta,case=case,elapsed_seconds=time.perf_counter()-start),OUT/'86_progress_v3.json')
        rows,rounds,checks=core.run_grid(n,e,demands,cases,design['thresholds'],design['demands'],dict(graph_id=graph_id,capacity_id=design['capacity_id']),progress=progress)
        for v,name in [(rows,'results.csv'),(rounds,'rounds.csv'),(checks,'validation.csv')]:core.save(v,folder/name)
        core.completion(folder,['results.csv','rounds.csv','validation.csv'],dict(sources=design['sources'],edge_sha256=core.sha(graph/'edges.csv'),rows=len(rows)))
        core.dump(dict(completed_graphs=index+1,total=50,elapsed_seconds=time.perf_counter()-start),OUT/'86_progress_v3.json')
        print('86',graph_id,'A/B/C complete',flush=True);gc.collect()
    for name,target in [('results.csv','86a_null_policy_results.csv'),('rounds.csv','86b_null_cascade_rounds.csv'),('validation.csv','86_validation.csv')]:
        frames=[pd.read_csv(OUT/'null_policy_cells_v3'/f'null_{i:02d}'/name) for i in range(50)];core.save(pd.concat(frames,ignore_index=True),OUT/target)
    core.check_sources(design);core.check_sources(nd);r=pd.read_csv(OUT/'86a_null_policy_results.csv');v=pd.read_csv(OUT/'86_validation.csv')
    core.dump(dict(completed=True,graphs=50,policy_rows=len(r),conditions=len(r)//3,checks=len(v),passed=int(v.passed.sum()),source_sha256=core.sha(__file__)),OUT/'86_completion_v3.json')
    print('86 fifty null policy grids complete',flush=True)

if __name__=='__main__':main()
