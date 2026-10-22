"""09 的等价数值评价器：只减少 DataFrame 重建，不改变目标、权重和精度。"""
from collections import OrderedDict
import numpy as np
import pandas as pd


class RecoveryEvaluator:
    def __init__(self, nodes, edges, cache_limit=8192):
        self.nodes, self.edges = nodes, edges
        self.ids = nodes.node_id.astype(str).tolist()
        self.index = {k:i for i,k in enumerate(self.ids)}
        self.size = len(self.ids)
        self.src = edges.Source.map(self.index).to_numpy(dtype=int)
        self.dst = edges.Target.map(self.index).to_numpy(dtype=int)
        self.cost = 1 / edges.Composite_transport_cost.clip(lower=.01).to_numpy(dtype=float)
        self.full_eff = float(self.cost.sum())
        self.loads = pd.to_numeric(nodes.proxy_daily_load, errors='coerce').fillna(0).to_numpy(dtype=float)
        threshold = pd.to_numeric(nodes.overload_threshold, errors='coerce')
        self.pressure = (pd.Series(self.loads,index=nodes.index) / threshold.where(threshold.ne(0))).astype('float64').fillna(1).clip(upper=1.5).to_numpy()
        self.total_load = float(self.loads.sum())
        self.risk = edges.Failure_probability_proxy.clip(0,1).to_numpy(dtype=float)
        self.repair = edges.Repair_cost_proxy.to_numpy(dtype=float)
        self.carbon = (.55 * edges.repair_norm + .45 * edges.cross_norm).to_numpy(dtype=float)
        self.adj = [set() for _ in self.ids]
        for a,b in zip(self.src,self.dst):
            self.adj[a].add(b); self.adj[b].add(a)
        self.cache = OrderedDict()
        self.cache_limit = cache_limit

    def evaluate(self, restored, base_active=None):
        restored = frozenset(map(str,restored))
        active = frozenset(map(str,base_active or [])) | restored
        key = (active, restored)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        if not active:
            return {k:0.0 for k in ['Ew','Reach','Demand_Satisfaction','Connectivity','Recovery_Score','Mean_Repair_Cost','Mean_Carbon_Cost','J','Pareto_F1','Pareto_F2','Pareto_F3']}
        flags = np.zeros(self.size,dtype=bool)
        flags[[self.index[k] for k in active]] = True
        recovered = np.zeros(self.size,dtype=bool)
        recovered[[self.index[k] for k in restored]] = True
        mask = flags[self.src] & flags[self.dst]
        ew = float(self.cost[mask].sum()) / self.full_eff if mask.any() and self.full_eff>0 else 0.0
        reach = float(self.loads[flags].sum()) / self.total_load if self.total_load else 0.0
        if mask.any():
            avg_risk = float(self.risk[mask].mean())
            repaired = mask & (recovered[self.src] | recovered[self.dst])
            use = repaired if repaired.any() else mask
            avg_repair = float(self.repair[use].mean())
            carbon = float(self.carbon[use].mean())
        else:
            avg_risk, avg_repair, carbon = 1.0, 0.0, 0.0
        overload = float(self.pressure[flags].mean())
        sat = max((len(active)/max(self.size,1))*(1-avg_risk)*max(0,1-.36*overload),0.0)
        remaining = set(np.flatnonzero(flags))
        largest = 0
        while remaining:
            stack=[remaining.pop()]; size=0
            while stack:
                current=stack.pop(); size+=1
                nxt=self.adj[current] & remaining
                remaining.difference_update(nxt); stack.extend(nxt)
            largest=max(largest,size)
        connectivity = largest/self.size if self.size else 0.0
        f1=min(ew,1.0); f2=.5*reach+.5*sat
        f3=1-(.6*min(avg_repair/120,1)+.4*min(carbon,1))
        score=.32*reach+.24*sat+.22*connectivity+.12*f1+.10*max(0,1-avg_repair/120)
        objective=.34*f1+.28*reach+.22*sat-.10*min(avg_repair/120,1)-.06*min(carbon,1)
        values={'Ew':ew,'Reach':reach,'Demand_Satisfaction':sat,'Connectivity':connectivity,'Recovery_Score':score,'Mean_Repair_Cost':avg_repair,'Mean_Carbon_Cost':carbon,'J':objective,'Pareto_F1':f1,'Pareto_F2':f2,'Pareto_F3':f3}
        result={k:round(v,4) for k,v in values.items()}
        self.cache[key]=result
        if len(self.cache)>self.cache_limit: self.cache.popitem(last=False)
        return result.copy()
