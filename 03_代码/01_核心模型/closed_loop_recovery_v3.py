"""需求—额定能力分配—持续风险恢复—累计预算：独立版本，数量均为代理量。"""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from submission_strengthening_common import D, HERE, module


@dataclass(frozen=True)
class DemandScenario:
    name: str
    multipliers: tuple = ()
    mix: float = .25
    coupled: bool = True


class DemandModel:
    """75%本地代理需求+25%外部8城需求；IPF保存名义设施边际，不是OD标定。"""
    def __init__(self, nodes, edges, city_data, cross):
        mapping = module('41_cross_layer_demand_validation')
        self.nodes = nodes.reset_index(drop=True).copy()
        self.cities, raw = mapping.assignment_matrix(self.nodes, edges, cross)
        self.matrix = mapping.balance_proxy_margins(self.nodes, city_data, self.cities, raw)
        self.base = self.nodes.proxy_daily_load.to_numpy(float)
        weights = np.array([1+float(city_data.set_index('城市').loc[c, '需求强度指数']) for c in self.cities])
        self.q = weights/weights.sum()*self.base.sum()

    def components(self, scenario):
        if not 0 <= scenario.mix <= 1:
            raise ValueError('需求混合比例必须在[0,1]')
        factors = dict(scenario.multipliers)
        if set(factors)-set(self.cities) or any(not np.isfinite(v) or v < 0 for v in factors.values()):
            raise ValueError('未知城市或非法需求倍数')
        q = self.q*np.array([factors.get(c, 1.) for c in self.cities])
        matrix = self.matrix if scenario.coupled else np.tile(self.base/self.base.sum(), (len(self.cities),1))
        parts = np.vstack([(1-scenario.mix)*self.base, scenario.mix*q[:,None]*matrix])
        expected = (1-scenario.mix)*self.base.sum()+scenario.mix*q.sum()
        if not np.isclose(parts.sum(),expected,atol=1e-7,rtol=0):
            raise AssertionError('需求质量不守恒')
        return parts, ['local_proxy']+self.cities


class CapacityAllocator:
    """本地优先，失败源与活跃节点溢出需求联合重分配；所有接收点严格不超过K。

    两跳中转点须存活，但未给中转流加吞吐上限；这是设施承接代理模型，非真实道路多商品流。
    """
    def __init__(self, nodes, edges):
        self.nodes = nodes.reset_index(drop=True).copy()
        self.ids = self.nodes.node_id.tolist()
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('node_id不唯一')
        self.pos = {v:i for i,v in enumerate(self.ids)}
        self.capacity = self.nodes.overload_threshold.to_numpy(float)
        if not np.isfinite(self.capacity).all() or (self.capacity < 0).any():
            raise ValueError('能力必须有限且非负')
        self.direct = np.full((len(self.ids),len(self.ids)), np.inf)
        for r in edges.itertuples():
            i,j = self.pos[r.Source_id],self.pos[r.Target_id]
            cost = max(float(r.Composite_transport_cost), .001)
            self.direct[i,j] = min(self.direct[i,j],cost)
            self.direct[j,i] = min(self.direct[j,i],cost)
        self.solve_calls = 0

    @lru_cache(maxsize=512)
    def paths(self, failed):
        """最便宜的1/2跳路径；所有源/终点顺序固定，避免输入行顺序影响目标值。"""
        active = np.array([k not in failed for k in self.ids])
        cost = self.direct.copy()
        for mid in np.flatnonzero(active):
            np.minimum(cost, self.direct[:,mid,None]+self.direct[mid,None,:], out=cost)
        cost[:,~active] = np.inf
        np.fill_diagonal(cost,np.inf)
        return cost

    def evaluate(self, failed, demand, *, details=False):
        failed = frozenset(failed)
        if not failed <= set(self.ids):
            raise ValueError('失效节点不在网络中')
        demand = np.asarray(demand,float)
        if demand.shape != self.capacity.shape or not np.isfinite(demand).all() or (demand < 0).any() or demand.sum() <= 0:
            raise ValueError('非法节点需求')
        self.solve_calls += 1
        active = np.array([k not in failed for k in self.ids])
        own = np.where(active, np.minimum(demand,self.capacity),0.)
        displaced = demand-own
        residual = np.where(active,np.maximum(self.capacity-own,0.),0.)
        path = self.paths(failed)
        feasible = (displaced[:,None]>1e-9)&(residual[None,:]>1e-9)&np.isfinite(path)
        src,dst = np.nonzero(feasible)
        flow = np.zeros(len(src));optimum=0.;max_residual=0.;gap=0.
        if len(src):
            count=len(src);cols=np.r_[np.arange(count),np.arange(count)]
            rows=np.r_[src,len(demand)+dst]
            a=coo_matrix((np.ones(2*count),(rows,cols)),shape=(2*len(demand),count)).tocsr()
            b=np.r_[displaced,residual]
            first=linprog(-np.ones(count),A_ub=a,b_ub=b,bounds=(0,None),method='highs')
            if not first.success:raise RuntimeError('服务LP失败：'+first.message)
            optimum=float(first.x.sum())
            second=linprog(path[src,dst],A_ub=a,b_ub=b,A_eq=np.ones((1,count)),b_eq=[optimum],bounds=(0,None),method='highs')
            if not second.success:raise RuntimeError('成本LP失败：'+second.message)
            flow=second.x;gap=abs(float(flow.sum())-optimum)
            max_residual=max(0.,float(np.max(a@flow-b)),float(-flow.min()),gap)
        outflow=np.bincount(src,weights=flow,minlength=len(demand))
        incoming=np.bincount(dst,weights=flow,minlength=len(demand))
        served=own+outflow;load=own+incoming;unserved=demand-served
        util=np.divide(load,self.capacity,out=np.zeros_like(load),where=self.capacity>0)
        if (served > demand+1e-6).any() or (unserved < -1e-6).any() or (load > self.capacity+1e-6).any():
            raise AssertionError('服务或能力约束违反')
        result=dict(Service_Rate=float(served.sum()/demand.sum()),Served_Load=float(served.sum()),
                    Unserved_Load=float(unserved.sum()),Demand_Load=float(demand.sum()),
                    Transferred_Load=float(flow.sum()),Transfer_Cost=float(np.dot(flow,path[src,dst])),
                    Initial_Active_Overflow=float(np.maximum(demand[active]-self.capacity[active],0.).sum()),
                    Max_Utilization=float(util.max()),Max_LP_Violation=max_residual,
                    Service_Conservation_Error=abs(float(served.sum()+unserved.sum()-demand.sum())),
                    Receiver_Conservation_Error=abs(float(load.sum()-served.sum())),
                    Transfer_Service_Gap=gap)
        extra=None
        if details:
            extra=dict(served=served,load=load,unserved=unserved,
                       arcs=pd.DataFrame(dict(source_id=[self.ids[i] for i in src],target_id=[self.ids[i] for i in dst],flow=flow,cost=path[src,dst])))
        return result,util,served,extra


class ClosedLoopRecovery:
    """外部损伤固定，持续风险闭包与严格分配相互作用；没有待评SVD/中心性评分。

    风险公式是原07剩余6项按权重和0.75重新归一，仍是假设代理模型，不是实测失效概率。
    """
    def __init__(self, allocator, edges, components, group_names, initial_failed, kind,
                 hazard='ended', threshold=.55):
        self.a=allocator;self.parts=np.asarray(components,float);self.groups=group_names
        self.demand=self.parts.sum(axis=0);self.initial=frozenset(initial_failed)
        self.hazard=hazard;self.threshold=float(threshold);self.kind=kind
        if hazard not in ['ended','continuing']:
            raise ValueError('未知风险模式')
        self.cache={};self.closed={};self.prefix={}
        self.src=np.r_[edges.Source_id.map(allocator.pos).to_numpy(int),edges.Target_id.map(allocator.pos).to_numpy(int)]
        self.dst=np.r_[edges.Target_id.map(allocator.pos).to_numpy(int),edges.Source_id.map(allocator.pos).to_numpy(int)]
        self.degree=np.bincount(self.src,minlength=len(allocator.ids))
        self.attributes=[np.tile(edges[c].to_numpy(float),2) for c in ['Failure_probability_proxy','Cross_district','Composite_transport_cost','Edge_capacity_proxy']]
        self.attributes[2]=np.maximum(self.attributes[2],.001)
        self.bonus=np.zeros(len(allocator.ids))
        if kind=='spatial_cluster':self.bonus=.10*allocator.nodes['区县'].isin(['东西湖区','新洲区','黄陂区']).to_numpy()
        if kind=='channel_disruption':self.bonus=.10*allocator.nodes.family.isin(['airport','rail','port']).to_numpy()
        # 每设施一次修复尝试成本=其入射边成本代理均值/全网中位数；不是把所有边当独立设施修复。
        stacked=pd.concat([edges[['Source_id','Repair_cost_proxy']].rename(columns={'Source_id':'node_id'}),edges[['Target_id','Repair_cost_proxy']].rename(columns={'Target_id':'node_id'})])
        costs=stacked.groupby('node_id').Repair_cost_proxy.mean().reindex(allocator.ids)
        median=float(costs.dropna().median())
        if not math.isfinite(median) or median<=0:raise ValueError('修复成本代理无效')
        self.cost=costs.fillna(median).to_numpy(float)/median
        if (self.cost<=0).any() or not np.isfinite(self.cost).all():raise ValueError('修复成本须为有限正数')

    def state(self, failed):
        key=frozenset(failed)
        if key not in self.cache:self.cache[key]=self.a.evaluate(key,self.demand)[:3]
        return self.cache[key]

    def newly_failed(self, failed, util):
        if self.hazard=='ended':return frozenset()
        rm=np.array([k in failed for k in self.a.ids]);mask=rm[self.dst]
        count=np.bincount(self.src[mask],minlength=len(rm));den=np.maximum(count,1)
        avg=[np.bincount(self.src[mask],weights=x[mask],minlength=len(rm))/den for x in self.attributes]
        risk=(.21*count/np.maximum(self.degree,1)+.16*avg[0]+.12*avg[1]+.10*np.minimum(avg[2]/.45,1)+.09*(1-np.minimum(avg[3]/55,1))+.07*np.minimum(util,1.6))/.75+self.bonus
        return frozenset(self.a.ids[i] for i in np.flatnonzero((~rm)&(count>0)&(risk>=self.threshold)))

    def closure(self, failed):
        key=frozenset(failed)
        if key not in self.closed:
            current=key;rounds=0
            for _ in range(len(self.a.ids)+1):
                metric,util,served=self.state(current);new=self.newly_failed(current,util)
                if not new:break
                current=current|new;rounds+=1
            else:raise RuntimeError('风险闭包未收敛')
            self.closed[key]=(current,metric,rounds)
        return self.closed[key]

    def trajectory(self, sequence):
        key=tuple(sequence)
        if len(key)!=len(set(key)) or not set(key)<=set(self.a.ids):
            raise ValueError('每设施最多修复一次，且必须是网络内设施')
        if key not in self.prefix:
            if not key:
                failed,metric,rounds=self.closure(self.initial)
                self.prefix[key]=(failed,[dict(step=0,cumulative_cost=0.,refailed_attempt=0,new_failures=len(failed-self.initial),cascade_rounds=rounds,failed_count=len(failed),**metric)],0.)
            else:
                previous,old,cost=self.trajectory(key[:-1]);node=key[-1];remaining=previous-{node}
                failed,metric,rounds=self.closure(remaining);cost+=float(self.cost[self.a.pos[node]])
                row=dict(step=len(key),node_id=node,cumulative_cost=cost,refailed_attempt=int(node in failed),new_failures=len(failed-remaining),cascade_rounds=rounds,failed_count=len(failed),**metric)
                self.prefix[key]=(failed,old+[row],cost)
        return self.prefix[key]

    def evaluate(self, sequence, horizon, budget, details=False):
        failed,rows,cost=self.trajectory(sequence)
        if horizon<1 or len(sequence)>horizon or cost>budget+1e-9:
            raise ValueError('修复行动超出步数或累计资源预算')
        service=np.array([x['Service_Rate'] for x in rows]+[rows[-1]['Service_Rate']]*(horizon-len(sequence)))
        auc=float(np.sum((service[:-1]+service[1:])/2)/horizon)
        result=dict(AUC_Service=auc,Final_Service=rows[-1]['Service_Rate'],Initial_Service=rows[0]['Service_Rate'],
                    Cumulative_Cost=cost,Attempts=len(sequence),Refailed_Attempts=sum(r['refailed_attempt'] for r in rows),
                    Final_Failed=len(failed),Budget=budget,Horizon=horizon)
        if details:
            # 明确标注：源设施内部按各城代理需求占比归属服务，不是观测到的城市货物流。
            _,_,served=self.state(failed)
            fractions=np.divide(served,self.demand,out=np.zeros_like(served),where=self.demand>0)
            supplied=self.parts@fractions;totals=self.parts.sum(axis=1)
            city=pd.DataFrame(dict(group=self.groups,demand=totals,served=supplied,service_rate=np.divide(supplied,totals,out=np.ones_like(totals),where=totals>0)))
            result['City_Service_Conservation_Error']=abs(float(city.served.sum()-served.sum()))
            padded=rows+[dict(rows[-1],step=i,node_id='',refailed_attempt=0,new_failures=0,cascade_rounds=0) for i in range(len(rows),horizon+1)]
            return result,pd.DataFrame(padded),city
        return result


class RecoveryPlanner:
    """共同情景目标、固定时间范围、累计硬预算；缓存仅省计算，不改评价调用计数。"""
    def __init__(self, engines, horizon=6, budget=4.):
        self.engines=engines;self.horizon=horizon;self.budget=budget
        self.cost=engines[0].cost;self.ids=engines[0].a.ids;self.pos=engines[0].a.pos
        if any(not np.allclose(e.cost,self.cost) for e in engines):raise ValueError('跨情景成本必须一致')
        self.pool=sorted(set().union(*(e.closure(e.initial)[0] for e in engines)))
        self.calls=0;self.scores={}

    def affordable(self,sequence):
        return len(sequence)<=self.horizon and len(sequence)==len(set(sequence)) and all(k in self.pool for k in sequence) and sum(self.cost[self.pos[k]] for k in sequence)<=self.budget+1e-9

    def score(self,sequence):
        self.calls+=1;key=tuple(sequence)
        if not self.affordable(key):return (-np.inf,-np.inf)
        if key not in self.scores:
            values=[e.evaluate(key,self.horizon,self.budget) for e in self.engines]
            auc=[v['AUC_Service'] for v in values]
            self.scores[key]=(float(.5*np.mean(auc)+.5*np.min(auc)),-float(values[0]['Cumulative_Cost']))
        return self.scores[key]

    def greedy(self):
        seq=[]
        while len(seq)<self.horizon:
            current=self.score(seq);best=current;choice=None
            for node in self.pool:
                if node in seq or not self.affordable(seq+[node]):continue
                value=self.score(seq+[node])
                if value[0]>best[0]+1e-12 or (abs(value[0]-best[0])<=1e-12 and value[1]>best[1]+1e-12):
                    best=value;choice=node
            if choice is None:break
            seq.append(choice)
        return seq

    def local_search(self, seed_sequence, rng, proposals=40):
        best=list(seed_sequence);value=self.score(best);accepted=[]
        for iteration in range(proposals):
            candidate=best.copy();op=rng.randrange(4)
            if op==0 and len(candidate)>1:
                i,j=rng.sample(range(len(candidate)),2);candidate[i],candidate[j]=candidate[j],candidate[i]
            elif op==1 and candidate:
                candidate.pop(rng.randrange(len(candidate)))
            elif op==2 and len(candidate)<self.horizon:
                available=[k for k in self.pool if k not in candidate]
                if available:candidate.insert(rng.randrange(len(candidate)+1),rng.choice(available))
            elif candidate:
                available=[k for k in self.pool if k not in candidate]
                if available:candidate[rng.randrange(len(candidate))]=rng.choice(available)
            if not self.affordable(candidate):
                self.calls+=1 # 违反硬预算的提案仍消耗一次搜索机会。
                continue
            trial=self.score(candidate)
            if trial[0]>value[0]+1e-12 or (abs(trial[0]-value[0])<=1e-12 and trial[1]>value[1]+1e-12):
                accepted.append(dict(iteration=iteration,before=value[0],after=trial[0],cost_before=-value[1],cost_after=-trial[1]))
                best,value=candidate,trial
        return best,accepted


def load_inputs():
    n=pd.read_csv(D/'04b_武汉节点代理容量.csv').sort_values('node_id').reset_index(drop=True)
    e=pd.read_csv(D/'04c_武汉边代理属性.csv')
    ci=pd.read_csv(D/'04a_城市综合指标_含估算.csv');cross=pd.read_csv(D/'04_cross_layer_edges.csv')
    return n,e,ci,cross
