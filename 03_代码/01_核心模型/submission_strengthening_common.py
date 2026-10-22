"""20261015补强实验公共工具；保留正式模型，输出到独立补强目录。"""
from pathlib import Path
import importlib.util, json, sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
D=HERE.parent/'data'
OUT=HERE.parent.parent/'项目结果'/'submission_strengthening_20261015'
OUT.mkdir(parents=True,exist_ok=True)

def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(df,name):df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def mm(s):
    s=pd.to_numeric(s,errors='coerce').fillna(0)
    return (s-s.min())/(s.max()-s.min()) if s.max()!=s.min() else pd.Series(1.,index=s.index)

def score_nodes(nodes,edges,variant='full',dimensions=2):
    """与06同公式的独立消融计算；不修改正式评分文件。"""
    cols=['Failure_probability_proxy','Repair_cost_proxy','Composite_transport_cost','Cross_district','Functional_synergy']
    stack=pd.concat([edges[['Source_id']+cols].rename(columns={'Source_id':'node_id'}),edges[['Target_id']+cols].rename(columns={'Target_id':'node_id'})])
    f=stack.groupby('node_id').agg(edge_risk_mean=('Failure_probability_proxy','mean'),edge_risk_max=('Failure_probability_proxy','max'),repair_cost_mean=('Repair_cost_proxy','mean'),transport_cost_mean=('Composite_transport_cost','mean'),cross_district_ratio=('Cross_district','mean'),synergy_mean=('Functional_synergy','mean'),incident_edge_count=('Failure_probability_proxy','size'))
    n=nodes.merge(f,on='node_id',how='left').fillna(0)
    feat=n[['capacity_index','weighted_degree_norm','risk_exposure','proxy_daily_load','edge_risk_mean','edge_risk_max','repair_cost_mean','transport_cost_mean','cross_district_ratio','synergy_mean','incident_edge_count']].copy()
    feat.insert(4,'load_pressure',mm(n.proxy_daily_load))
    if variant=='deduplicated':feat=feat.drop(columns=['proxy_daily_load','load_pressure'])
    x=feat.apply(mm);z=(x-x.mean())/x.std(ddof=0).replace(0,1);z=z.fillna(0)
    _,sing,vt=np.linalg.svd(z.values,full_matrices=False)
    emb=z.values@vt[:dimensions].T;distance=np.linalg.norm(emb-emb.mean(axis=0),axis=1)
    struct=(.22*mm(n.capacity_index)+.18*mm(n.weighted_degree_norm)+.16*mm(n.proxy_daily_load)+.10*mm(n.risk_exposure)+.10*mm(n.incident_edge_count)+.08*mm(n.synergy_mean))/.84
    if variant=='deduplicated':struct=(struct*.84-.16*mm(n.proxy_daily_load))/.68
    bridge=.28*mm(n.cross_district_ratio)+.22*mm(n.edge_risk_mean)+.20*mm(n.edge_risk_max)+.15*mm(n.repair_cost_mean)+.10*mm(n.transport_cost_mean)+.05*mm(n.proxy_daily_load)
    weights=np.array([.38,.37,.25])
    if variant=='no_svd':weights[2]=0
    if variant=='no_bridge':weights[1]=0
    if variant=='no_structure':weights[0]=0
    weights/=weights.sum()
    result=n[['node_id','名称','城市','区县','类型','family','edge_risk_mean','cross_district_ratio']].copy()
    result[['edge_risk_mean','cross_district_ratio']]=result[['edge_risk_mean','cross_district_ratio']].round(4)
    result['structural_importance']=np.round(struct,4);result['bridge_risk_score']=np.round(bridge,4)
    result['svd_bridge_score']=np.round(weights[0]*struct+weights[1]*bridge+weights[2]*mm(pd.Series(distance,index=n.index)),4)
    result['svd_bridge_rank']=result.svd_bridge_score.rank(ascending=False,method='dense').astype(int)
    result.attrs['explained_variance_ratio']=float(np.sum(sing[:dimensions]**2)/np.sum(sing**2)) if np.sum(sing**2) else 0
    return result

class FlowEvaluator:
    """复用05c路径选择/评分/分配，省去每状态的DataFrame构造；需等价性验收。"""
    def __init__(self,nodes,edges,emergency=1.10):
        self.f=module('05c_dynamic_flow_resilience');self.f.EMERGENCY_OVERLOAD_FACTOR=emergency
        n=nodes.copy();e=edges.copy();n['名称']=n.node_id;e['Source']=e.Source_id;e['Target']=e.Target_id
        self.n,self.e,self.adj=self.f.prep(n,e)
        self.info=self.n.set_index('名称').to_dict('index');self.ids=sorted(self.info)
        self.positions={k:i for i,k in enumerate(self.n['名称'])}
        self.loads=self.n.proxy_daily_load.to_numpy(float);self.total=float(self.loads.sum())
        self.src=e.Source.map(self.positions).to_numpy(int);self.dst=e.Target.map(self.positions).to_numpy(int)
        self.efficiency=1/e.Composite_transport_cost.clip(lower=.001).to_numpy(float)
        self.emergency=emergency
    def evaluate(self,removed):
        removed=set(removed);active=set(self.ids)-removed;alloc={k:0. for k in active}
        rerouted=unsat=weighted=0.
        for src in sorted(removed):
            candidates=[]
            for c in self.f.collect_candidates(src,self.adj,active,self.info):
                ni=self.info[c['node']];res=max(self.emergency*float(ni['overload_threshold'])-float(ni['proxy_daily_load'])-alloc[c['node']],0.)
                if res<=1e-9:continue
                c['score']=self.f.candidate_score(c,ni,res);c['residual']=res;candidates.append(c)
            candidates.sort(key=lambda c:(-c['score'],-c['residual'],str(c['node'])))
            left,rr,wc=self.f.allocate_by_share(float(self.info[src]['proxy_daily_load']),candidates,alloc,self.info,self.f._DisabledLog(),src)
            rerouted+=rr;weighted+=wc;unsat+=max(left,0.)
        utilization={};over=0
        for k in sorted(active):
            row=self.info[k];thr=float(row['overload_threshold']);u=(float(row['proxy_daily_load'])+alloc[k])/thr if thr>0 else 1.
            utilization[k]=round(u,4);over+=int(u>1)
        flags=np.array([k in active for k in self.n['名称']]);alive_load=float(self.loads[flags].sum())
        sat=round(rerouted/(rerouted+unsat),4) if rerouted+unsat>0 else 1.
        loss=round(unsat/(rerouted+unsat),4) if rerouted+unsat>0 else 0.
        ov=round(over/len(active),4) if active else 0.;cost=round(weighted/rerouted,4) if rerouted>0 else 0.
        lcc=self.f.component_ratio(self.ids,self.adj,removed,len(self.ids));reach=round(alive_load/self.total,4) if self.total else 0.
        dyn=round(.22*lcc+.20*reach+.24*sat+.12*max(0,1-loss)+.10*max(0,1-ov)+.12/(1+cost),4)
        if rerouted<=1e-9 and rerouted+unsat>0:dyn=round(dyn-.12,4)
        if not active:dyn=0.
        metrics=dict(LCC=lcc,Reach=reach,Flow_Satisfaction=sat,Unsatisfied_Demand=round(unsat,4),Rerouted_Load=round(rerouted,4),Mean_Redistribution_Cost=cost,Overload_Rate=ov,Service_Loss=loss,Dynamic_Flow_Resilience=dyn,Network_Service_Rate=round((alive_load+rerouted)/self.total,6) if self.total else 0.,Failed_Demand=round(rerouted+unsat,4),Processing_Order_Policy='node_id_ascending')
        return metrics,utilization
    def damage(self,removed,metrics):
        flags=np.array([k not in removed for k in self.n['名称']]);mask=flags[self.src]&flags[self.dst]
        ew=round(float(self.efficiency[mask].sum())/len(self.e),4) if len(self.e) else 0.
        base=round(float(self.efficiency.sum())/len(self.e),4) if len(self.e) else 0.
        loss=max(0,1-min(ew/base,1)) if base else 1.
        return .24*(1-metrics['LCC'])+.18*loss+.18*(1-metrics['Reach'])+.24*(1-metrics['Flow_Satisfaction'])+.16*len(removed)/len(self.n)

def cascade(nodes,edges,ai,seeds,kind,threshold=.45,emergency=1.1):
    m=module('07_baseline_attack_comparison');n=nodes.copy();e=edges.copy();n['名称']=n.node_id;e['Source']=e.Source_id;e['Target']=e.Target_id
    n=m.addp(n,e,ai);nb=m.neigh(e);flow=FlowEvaluator(nodes,edges,emergency);m.T=threshold
    failed=set(seeds);history=[]
    for step in range(81):
        metric,util=flow.evaluate(failed);new=m.cas(n,nb,failed,kind,util)-failed
        history.append({'step':step,'failed_nodes':len(failed),'new_failed':len(new),**metric})
        if not new:return failed,history,flow
        failed.update(new)
    raise RuntimeError('Supplementary cascade did not converge')
