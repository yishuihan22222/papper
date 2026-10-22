"""独立候选模型；不覆盖正式参数。负荷/能力仍为代理量，无真实OD含义。"""
from submission_strengthening_common import *
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


class StrictReassignment:
    """相同两跳候选路径上的节点剩余能力LP；先最大服务，再最小转移成本。"""
    def __init__(self, nodes, edges):
        self.flow = FlowEvaluator(nodes, edges, emergency=1.)

    def evaluate(self, removed):
        f = self.flow; removed = set(removed); active = set(f.ids)-removed
        if any(float(f.info[k]['proxy_daily_load']) > float(f.info[k]['overload_threshold'])+1e-8 for k in active):
            raise ValueError('此原型要求存活节点初始负荷不超过额定能力；不得静默丢弃初始超载需求')
        sources = sorted(removed); receivers = sorted(active)
        si = {k:i for i,k in enumerate(sources)}; ri = {k:i for i,k in enumerate(receivers)}
        arcs = []
        for s in sources:
            for c in f.f.collect_candidates(s,f.adj,active,f.info):
                arcs.append((s,c['node'],float(c['cost'])))
        demand = np.array([float(f.info[k]['proxy_daily_load']) for k in sources])
        residual = np.array([max(float(f.info[k]['overload_threshold'])-float(f.info[k]['proxy_daily_load']),0.) for k in receivers])
        allocation=np.zeros(len(arcs)); optimum=0.; violation=0.
        if arcs:
            cols=np.repeat(np.arange(len(arcs)),2)
            rows=np.array([[si[s],len(sources)+ri[t]] for s,t,c in arcs]).ravel()
            a=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(sources)+len(receivers),len(arcs))).tocsr()
            b=np.r_[demand,residual]
            first=linprog(-np.ones(len(arcs)),A_ub=a,b_ub=b,bounds=(0,None),method='highs')
            if not first.success: raise RuntimeError(first.message)
            optimum=float(first.x.sum())
            second=linprog(np.array([c for s,t,c in arcs]),A_ub=a,b_ub=b,
                           A_eq=np.ones((1,len(arcs))),b_eq=[optimum],bounds=(0,None),method='highs')
            if not second.success: raise RuntimeError(second.message)
            allocation=second.x; violation=max(float(np.max(a@allocation-b)),abs(allocation.sum()-optimum),max(0.,-float(allocation.min())))
        incoming={k:0. for k in receivers}; served_source={k:0. for k in sources}
        for (s,t,c),x in zip(arcs,allocation): incoming[t]+=x; served_source[s]+=x
        alive=sum(float(f.info[k]['proxy_daily_load']) for k in receivers)
        cost=sum(c*x for (s,t,c),x in zip(arcs,allocation))
        loads=[dict(node_id=k,load_after=float(f.info[k]['proxy_daily_load'])+incoming[k],capacity=float(f.info[k]['overload_threshold']),utilization=(float(f.info[k]['proxy_daily_load'])+incoming[k])/float(f.info[k]['overload_threshold'])) for k in receivers]
        metrics=dict(Network_Service_Rate=(alive+optimum)/f.total,Rerouted_Load=optimum,
                     Unsatisfied_Demand=float(demand.sum())-optimum,Total_Transfer_Cost=cost,
                     Max_Constraint_Violation=violation,Candidate_Arcs=len(arcs),
                     Mass_Balance_Error=abs(alive+optimum+float(demand.sum())-optimum-f.total))
        return metrics,pd.DataFrame(loads),served_source


class PersistentRecovery:
    """逐动作持续风险；重新失效必须保留，不能因出现在已修复名单中自动复活。"""
    def __init__(self,nodes,edges,ai,kind,initial,threshold=.45):
        self.n=nodes.reset_index(drop=True);self.e=edges;self.kind=kind;self.initial=frozenset(initial);self.threshold=threshold
        self.flow=FlowEvaluator(nodes,edges);self.ids=self.n.node_id.tolist();self.pos={k:i for i,k in enumerate(self.ids)}
        a=ai.set_index('node_id').loc[self.ids];self.svd=a.svd_bridge_score.to_numpy(float);self.bridge=a.bridge_risk_score.to_numpy(float)
        self.src=np.r_[edges.Source_id.map(self.pos).to_numpy(int),edges.Target_id.map(self.pos).to_numpy(int)]
        self.dst=np.r_[edges.Target_id.map(self.pos).to_numpy(int),edges.Source_id.map(self.pos).to_numpy(int)]
        self.degree=np.bincount(self.src,minlength=len(self.ids))
        self.risk=np.tile(edges.Failure_probability_proxy.to_numpy(float),2)
        self.cross=np.tile(edges.Cross_district.to_numpy(float),2)
        self.cost=np.tile(edges.Composite_transport_cost.clip(lower=.001).to_numpy(float),2)
        self.cap=np.tile(edges.Edge_capacity_proxy.to_numpy(float),2)
        self.bonus=np.zeros(len(self.ids))
        if kind=='spatial_cluster': self.bonus=.10*self.n['区县'].isin(['东西湖区','新洲区','黄陂区']).to_numpy()
        if kind=='channel_disruption': self.bonus=.10*self.n.family.isin(['airport','rail','port']).to_numpy()
        self.states={};self.closures={};self.prefix={};self.flow_calls=0
        self.repair={k:float(edges.loc[edges.Source_id.eq(k)|edges.Target_id.eq(k),'Repair_cost_proxy'].mean()) for k in self.ids}
        self.repair={k:(v if np.isfinite(v) else 0.) for k,v in self.repair.items()}
        self.cost_scale=max(self.repair.values()) or 1.

    def state(self,removed):
        key=frozenset(removed)
        if key not in self.states:
            self.states[key]=self.flow.evaluate(key);self.flow_calls+=1
        return self.states[key]

    def newly_failed(self,removed,utilization):
        rm=np.array([k in removed for k in self.ids]);active=~rm
        if not active.any():return set()
        mask=rm[self.dst];cnt=np.bincount(self.src[mask],minlength=len(self.ids));den=np.maximum(cnt,1)
        def mean(values): return np.bincount(self.src[mask],weights=values[mask],minlength=len(self.ids))/den
        sc=.21*cnt/np.maximum(self.degree,1)+.16*mean(self.risk)+.12*mean(self.cross)+.10*np.minimum(mean(self.cost)/.45,1)+.09*(1-np.minimum(mean(self.cap)/55.,1))
        sc+=.15*self.svd/max(float(self.svd[active].max()),1.)+.10*self.bridge/max(float(self.bridge[active].max()),1.)
        sc+=.07*np.minimum([float(utilization.get(k,0.)) for k in self.ids],1.6)+self.bonus
        return {self.ids[i] for i in np.flatnonzero(active&(cnt>0)&(sc>=self.threshold))}

    def closure(self,removed):
        key=frozenset(removed)
        if key not in self.closures:
            failed=set(key)
            for _ in range(len(self.ids)+1):
                metric,util=self.state(failed);new=self.newly_failed(failed,util)
                if not new:break
                failed.update(new)
            else:raise RuntimeError('级联未收敛')
            self.closures[key]=(frozenset(failed),metric)
        return self.closures[key]

    def trajectory(self,sequence):
        key=tuple(sequence)
        if key not in self.prefix:
            if not key:
                failed,metric=self.closure(self.initial)
                self.prefix[key]=(failed,[dict(step=0,failed_nodes=len(failed),refailed_repair=0,**metric)],0.)
            else:
                failed,old,cost=self.trajectory(key[:-1]);node=key[-1]
                # 允许重复失败，但本轮实验限制每节点仅一次修复尝试；修复前已活跃的尝试仍计成本。
                remaining=set(failed)-{node};end,metric=self.closure(remaining)
                row=dict(step=len(key),failed_nodes=len(end),refailed_repair=int(node in end),additional_failures=len(end-remaining),**metric)
                self.prefix[key]=(end,old+[row],cost+self.repair[node]/self.cost_scale)
        return self.prefix[key]

    def objective(self,sequence):
        _,rows,cost=self.trajectory(sequence);v=[x['Network_Service_Rate'] for x in rows]
        auc=float(np.mean((np.array(v[:-1])+v[1:])/2)) if len(v)>1 else v[0]
        return auc,cost/max(len(sequence),1)
