"""Degree-preserving swaps with exact global road-distance-bin counts."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import pandas as pd
from closed_loop_recovery_v3 import HERE
from submission_strengthening_common import module
core=module('83_dispatch_mainline_core');OUT=core.OUT

def connected(edges,count):
    neighbors=[set() for _ in range(count)]
    for i,j in edges:neighbors[i].add(j);neighbors[j].add(i)
    seen={0};stack=[0]
    while stack:
        for j in neighbors[stack.pop()]-seen:seen.add(j);stack.append(j)
    return len(seen)==count

def chain(original,bins,count,seed,target):
    rng=random.Random(seed);edges=list(original);present=set(edges);index={e:i for i,e in enumerate(edges)}
    adjacency=[set() for _ in range(count)];bin_counts=np.zeros((count,6),dtype=int)
    for a,b in edges:
        adjacency[a].add(b);adjacency[b].add(a);k=int(bins[a,b]);bin_counts[a,k]+=1;bin_counts[b,k]+=1
    pools={(a,k):set(np.flatnonzero(bins[a]==k).tolist()) for a in range(count) for k in range(6)}
    sizes=np.array([[len(pools[a,k]) for k in range(6)] for a in range(count)])
    def propose():
        a=rng.randrange(count)
        eligible=[b for b in adjacency[a] if bin_counts[a,int(bins[a,b])]<sizes[a,int(bins[a,b])]]
        if not eligible:return False
        b=rng.choice(eligible);x=int(bins[a,b]);choices=pools[a,x]-adjacency[a];c=rng.choice(tuple(choices))
        ds=[v for v in adjacency[c] if v not in {a,b,c} and v not in adjacency[b] and bins[b,v]>=0 and bins[b,v]==bins[c,v]]
        if not ds:return False
        d=rng.choice(ds);nb=adjacency[b]-{a}|{d};nc=adjacency[c]-{d}|{a}
        reverse=[v for v in nb if v not in {a,b,c} and v not in nc and bins[c,v]>=0 and bins[c,v]==bins[b,v]]
        if d not in reverse:raise AssertionError('No reverse proposal')
        forward_q=1/count/len(eligible)/len(choices)/len(ds);reverse_q=1/count/len(eligible)/len(choices)/len(reverse)
        alpha=min(1.,len(ds)/len(reverse));reverse_alpha=min(1.,len(reverse)/len(ds))
        if abs(forward_q*alpha-reverse_q*reverse_alpha)>1e-14:raise AssertionError('Detailed balance correction')
        if rng.random()>alpha:return False
        old1=tuple(sorted((a,b)));old2=tuple(sorted((c,d)));first=tuple(sorted((a,c)));second=tuple(sorted((b,d)))
        ix=index.pop(old1);iy=index.pop(old2);present.remove(old1);present.remove(old2);present.add(first);present.add(second)
        edges[ix]=first;edges[iy]=second;index[first]=ix;index[second]=iy
        y=int(bins[c,d]);bin_counts[b,x]-=1;bin_counts[b,y]+=1;bin_counts[c,y]-=1;bin_counts[c,x]+=1
        adjacency[a].remove(b);adjacency[a].add(c);adjacency[b].remove(a);adjacency[b].add(d)
        adjacency[c].remove(d);adjacency[c].add(a);adjacency[d].remove(c);adjacency[d].add(b)
        return True
    accepted=0;trace=[];m=len(edges);original_set=set(original)
    for attempts in range(1,target+1):
        accepted+=int(propose())
        if attempts%(5*m)==0 or attempts==target:
            trace.append(dict(accepted_swaps=accepted,swaps_per_edge=accepted/m,attempts=attempts,proposal_steps_per_edge=attempts/m,original_edge_overlap=len(present&original_set)/m))
    return edges,trace,target

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    n,e,_,old=core.load_study();candidate=core.candidate_edges(n);pos={k:i for i,k in enumerate(n.node_id)}
    original=sorted(tuple(sorted((pos[a],pos[b]))) for a,b in zip(e.Source_id,e.Target_id));m=len(original)
    if len(set(original))!=m or not connected(original,len(n)):raise ValueError('Original graph must be simple and connected')
    lookup={key:i for i,key in enumerate(candidate.pair_key)};bins=np.full((len(n),len(n)),-1,dtype=np.int16)
    for row,b in zip(candidate.itertuples(),core.distance_bin(candidate.Distance_km.to_numpy())):bins[row.i,row.j]=bins[row.j,row.i]=b
    baseline=candidate.iloc[[lookup[k] for k in original]].set_index(['Source_id','Target_id'])
    # Edge-capacity is recomputed with frozen original node covariates. Every
    # consumed original edge attribute must be reproduced before rewiring.
    original_lookup={tuple(sorted((r.Source_id,r.Target_id))):r for r in e.itertuples()}
    for row in baseline.reset_index().itertuples():
        previous=original_lookup[tuple(sorted((row.Source_id,row.Target_id)))]
        if not all(abs(float(getattr(row,k))-float(getattr(previous,k)))<1e-10 for k in core.CONSUMED):raise AssertionError('Pair attribute reconstruction differs')
    paths=[Path(__file__),HERE/'85_generate_constrained_nulls.py',OUT/'85_design.json',HERE/'83_dispatch_mainline_core.py',core.DATA/'04b_武汉节点代理容量.csv',core.DATA/'04c_武汉边代理属性.csv',core.DATA/'04c_武汉边代理属性_候选全边.csv']
    coord=core.DATA/'武汉都市圈物流节点_语义清洗版.csv'
    if coord.exists():paths.append(coord)
    design=dict(graphs=50,seed_base=202610180900,proposal_steps_per_edge=100,distance_bin_edges=core.DISTANCE_BINS,
                constraints=['exact individual node degrees','exact edge count','simple graph','fixed node identities and geometry','exact global road-distance-bin histogram','connected final graph','new pairs must have existing candidate data'],
                spatial_boundary='Global coarse road-distance bins, not exact distance or per-node length distributions; connected final states conditioned by rejection',
                sampler_amendment='Original uniform-edge proposal stopped below acceptance threshold before any graph or policy outcome; same constraints retained, geometry-conditioned Metropolis proposal with reverse-proposal correction',
                fixed_step_amendment='V2 accepted-swap stopping was replaced before primary null results were inspected; use fixed proposal count including self-loops to avoid jump-chain bias; v2 outputs retained',
                sampling='50 independently seeded fixed-length Metropolis chains, 100*m proposals each, early 50*m for burn-in diagnostics; approximate constrained ensemble, no claim of proven mixing or exact p-values',
                target_capacity='node_a0.20_s00',cases=core.NULL_CASE_IDS,demands=core.NULL_DEMANDS,thresholds=sorted({float(v['theta']) for v in old['models']}),
                statistic='Mean C-minus-B completed service (percentage points) over 4 shocks x 2 demands x 3 thresholds; B-minus-A secondary descriptive',
                sources=core.source_hashes(paths))
    path=OUT/'85_design_v3.json'
    if path.exists():
        existing=json.loads(path.read_text(encoding='utf-8'))
        if existing['sources']!=design['sources']:raise RuntimeError('Changed null-generation source')
        design=existing
    else:core.dump(design,path)
    core.check_sources(design);degree0=np.bincount(np.asarray(original).ravel(),minlength=len(n));hist0=np.bincount([bins[k] for k in original],minlength=6)
    summaries=[];all_traces=[];start=time.perf_counter()
    for index in range(50):
        graph_id=f'null_{index:02d}';folder=OUT/'null_graphs_v3'/graph_id;folder.mkdir(parents=True,exist_ok=True)
        if args.resume and core.valid_completion(folder):
            summaries.append(json.loads((folder/'summary.json').read_text(encoding='utf-8')));all_traces+=pd.read_csv(folder/'mixing_trace.csv').to_dict('records');continue
        for retry in range(3):
            seed=design['seed_base']+index+1000000*retry
            pairs,trace,attempts=chain(original,bins,len(n),seed,100*m)
            if connected(pairs,len(n)):break
        else:raise RuntimeError('Could not generate connected constrained null')
        pairs=sorted(pairs);degree=np.bincount(np.asarray(pairs).ravel(),minlength=len(n));hist=np.bincount([bins[k] for k in pairs],minlength=6)
        if not np.array_equal(degree,degree0) or not np.array_equal(hist,hist0) or len(set(pairs))!=m:raise AssertionError('Null constraints violated')
        generated=candidate.iloc[[lookup[k] for k in pairs]].drop(columns=['i','j','pair_key'])
        core.save(generated,folder/'edges.csv')
        trace=[dict(graph_id=graph_id,**v) for v in trace];core.save(trace,folder/'mixing_trace.csv');all_traces+=trace
        summary=dict(graph_id=graph_id,seed=seed,retry=retry,edges=m,nodes=len(n),successful_swaps=trace[-1]['accepted_swaps'],attempts=attempts,
                     acceptance_rate=trace[-1]['accepted_swaps']/attempts,original_edge_overlap=len(set(pairs)&set(original))/m,connected=True,degree_preserved=True,distance_hist_preserved=True,
                     mean_distance=float(generated.Distance_km.mean()),source_sha256=core.sha(__file__))
        core.dump(summary,folder/'summary.json');summaries.append(summary)
        core.completion(folder,['edges.csv','summary.json','mixing_trace.csv'],dict(sources=design['sources']))
        core.dump(dict(completed=index+1,total=50,elapsed_seconds=time.perf_counter()-start),OUT/'85_progress_v3.json')
        print('85v3',graph_id,'generated; overlap',round(summary['original_edge_overlap'],3),'acceptance',round(summary['acceptance_rate'],3),flush=True)
    core.save(summaries,OUT/'85a_null_graph_summary.csv');core.save(all_traces,OUT/'85b_swap_mixing_traces.csv');core.check_sources(design)
    core.dump(dict(completed=True,graphs=50,constraint_checks_passed=True,source_sha256=core.sha(__file__),approximate_ensemble=True),OUT/'85_completion_v3.json')

if __name__=='__main__':main()
