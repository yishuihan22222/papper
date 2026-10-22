"""Finite risk hardening before an external shock; ordinary V5 repairs after it.

Hardening attenuates local propagated pressure, including the fixed exposure
bonus, by (1-eta). It does not remove direct exogenous damage, prevent overload,
increase nominal throughput, or exempt repaired nodes from immediate failure.
Cost and effectiveness are scenario assumptions, not measured investments.
"""
from collections import OrderedDict
import json
import time
import numpy as np
import pandas as pd
from closed_loop_recovery_v3 import DemandScenario,RecoveryPlanner
from dual_mechanism_recovery_v5 import DispatchAllocatorV5,DualMechanismRecoveryV5


class BoundedCache(OrderedDict):
    def __init__(self,limit):super().__init__();self.limit=limit
    def __getitem__(self,key):
        value=super().__getitem__(key);self.move_to_end(key);return value
    def __setitem__(self,key,value):
        super().__setitem__(key,value);self.move_to_end(key)
        while len(self)>self.limit:self.popitem(last=False)


class RiskHardenedRecoveryV6(DualMechanismRecoveryV5):
    def __init__(self,*args,protected=(),eta=.5,**kwargs):
        super().__init__(*args,**kwargs)
        self.protected=frozenset(protected);self.eta=float(eta)
        if not self.protected<=set(self.a.ids):raise ValueError('Unknown protected node')
        if not np.isfinite(eta) or not 0<=eta<1:raise ValueError('Finite eta must satisfy 0 <= eta < 1')
        self.attenuation=np.array([1-self.eta if k in self.protected else 1. for k in self.a.ids])

    def pressure(self,failed):
        count,raw=super().pressure(failed)
        if self.eta==0 or not self.protected:return count,raw
        return count,raw*self.attenuation


def incident_repair_cost(nodes,edges):
    z=pd.concat([edges[['Source_id','Repair_cost_proxy']].rename(columns={'Source_id':'node_id'}),
                 edges[['Target_id','Repair_cost_proxy']].rename(columns={'Target_id':'node_id'})])
    c=z.groupby('node_id').Repair_cost_proxy.mean().reindex(nodes.node_id)
    return c.fillna(c.median())/c.median()


class ProtectRepairStudy:
    """Four candidate ex-ante portfolios; full-pool strong greedy recourse.

    Repairs observe the realized external shock IDs, but use only the same two
    planning demands in all arms. Ex-ante portfolios use training shocks only.
    A shared planning signature may reuse a repair sequence only when both
    stable initial failed sets, rules, protection, budget and horizon coincide.
    """
    def __init__(self,nodes,edges,dm):
        self.n=nodes;self.e=edges;self.dm=dm;self.ids=nodes.node_id.tolist();self.pos={k:i for i,k in enumerate(self.ids)}
        self.cost=incident_repair_cost(nodes,edges);self.cost_by_id=self.cost.to_dict()
        self.demands={name:dm.components(spec) for name,spec in [
            ('nominal',DemandScenario('nominal')),
            ('external_plus20',DemandScenario('external_plus20',tuple((c,1.2) for c in dm.cities))),
            ('ezhou_plus20',DemandScenario('ezhou_plus20',(('鄂州市',1.2),))),
            ('external_plus150',DemandScenario('external_plus150',tuple((c,2.5) for c in dm.cities))) ]}
        self.training_demands=['nominal','external_plus20']
        self.allocators={};self.states={};self.raw_pressures={};self.repair_plans={}
        degree=pd.concat([edges.Source_id,edges.Target_id]).value_counts().reindex(self.ids,fill_value=0)
        load=nodes.set_index('node_id').proxy_daily_load
        values={'degree_per_cost':degree/self.cost,'load_per_cost':load/self.cost,
                'hybrid_per_cost':(.5*degree/max(float(degree.max()),1)+.5*load/max(float(load.max()),1))/self.cost}
        self.rankings={label:sorted(self.ids,key=lambda k:(-float(v.loc[k]),k)) for label,v in values.items()}
        self.plan_searches=0;self.plan_reuses=0;self.upper_bound_stops=0

    def engine(self,model,protected,case,demand,eta_override=None):
        rho=float(model['rho'])
        if rho not in self.allocators:self.allocators[rho]=DispatchAllocatorV5(self.n,self.e,rho,'min_overload',2)
        parts,names=self.demands[demand]
        en=RiskHardenedRecoveryV6(self.allocators[rho],self.e,parts,names,case['initial_ids'],case['kind'],
                                model['theta'],gamma=1.,protected=protected,
                                eta=model['eta'] if eta_override is None else eta_override)
        en.cache=self.states.setdefault((rho,demand),BoundedCache(4096))
        en.pressure_cache=self.raw_pressures.setdefault(case['kind'],BoundedCache(2048))
        return en

    def protection_cost(self,protected,cost_ratio):
        return float(cost_ratio*sum(float(self.cost_by_id[k]) for k in sorted(protected)))

    def candidates(self,allowance,cost_ratio):
        # These rankings use graph/data attributes only, never future shock IDs.
        result=[dict(label='none',protected=(),cost=0.)];seen={frozenset()}
        for label,order in self.rankings.items():
            selected=[];spent=0.
            for node in order:
                cost=cost_ratio*float(self.cost_by_id[node])
                if spent+cost<=allowance+1e-9:selected.append(node);spent+=cost
            key=frozenset(selected)
            if key not in seen:
                seen.add(key);result.append(dict(label=label,protected=tuple(sorted(key)),cost=self.protection_cost(key,cost_ratio)))
        return result

    def repair(self,model,protected,case,remaining_budget,horizon):
        engines=[self.engine(model,protected,case,d) for d in self.training_demands]
        closed=tuple(en.closure(en.initial)[0] for en in engines)
        effective_eta=float(model['eta']) if protected else 0.
        key=(model['rho'],model['theta'],effective_eta,tuple(sorted(protected)),case['kind'],closed,float(remaining_budget),horizon)
        if key in self.repair_plans:
            self.plan_reuses+=1
            return dict(self.repair_plans[key],reused=True,search_seconds=0.,score_calls=0)
        self.plan_searches+=1;start=time.perf_counter()
        if all(en.closure(en.initial)[1]['Service_Rate']>=1-1e-12 for en in engines):
            sequence=[];calls=0;certificate='completed service upper bound; zero cost wins ties';self.upper_bound_stops+=1
        else:
            planner=RecoveryPlanner(engines,horizon,remaining_budget);sequence=planner.greedy();calls=planner.calls;certificate='full-pool strong greedy'
        item=dict(sequence=sequence,score_calls=calls,search_seconds=time.perf_counter()-start,certificate=certificate,reused=False)
        self.repair_plans[key]=dict(item)
        return item

    def evaluate(self,model,protected,case,demand,sequence,total_budget,horizon,cost_ratio,*,details=False,eta_override=None):
        pcost=self.protection_cost(protected,cost_ratio);remaining=total_budget-pcost
        if remaining< -1e-9:raise ValueError('Ex-ante hardening exceeds total budget')
        en=self.engine(model,protected,case,demand,eta_override)
        metric,trace,city=en.evaluate(sequence,horizon,max(0.,remaining),details=True)
        failed=en.trajectory(sequence)[0];initial=en.trajectory([])[0]
        result=dict(**metric,Protection_Cost=pcost,Repair_Cost=metric['Cumulative_Cost'],
                    Total_Cost=pcost+metric['Cumulative_Cost'],Total_Budget=total_budget,
                    Protected_Nodes=len(protected),Protected_Exogenous_Overlap=len(set(protected)&set(case['initial_ids'])),
                    Initial_Stable_Failed=len(initial),Protected_Initial_Failed=len(set(protected)&initial),
                    Protected_Final_Failed=len(set(protected)&failed),
                    initial_stable_failed_indices_json=json.dumps(sorted(self.pos[k] for k in initial)),
                    final_failed_indices_json=json.dumps(sorted(self.pos[k] for k in failed)))
        if result['Total_Cost']>total_budget+1e-9:raise AssertionError('Joint budget exceeded')
        if details:return result,trace,city
        return result

    def select_protection(self,model,training_cases,total_budget,horizon,cost_ratio,with_repair):
        allowance=total_budget/2 if with_repair else total_budget
        trials=[];winner=None;best=(-np.inf,-np.inf)
        for candidate in self.candidates(allowance,cost_ratio):
            values=[];spent=[]
            for case in training_cases:
                seq=self.repair(model,candidate['protected'],case,total_budget-candidate['cost'],horizon)['sequence'] if with_repair else []
                for demand in self.training_demands:
                    metric=self.evaluate(model,candidate['protected'],case,demand,seq,total_budget,horizon,cost_ratio)
                    values.append(metric['AUC_Service']);spent.append(metric['Total_Cost'])
            score=(float(.5*np.mean(values)+.5*np.min(values)),-float(np.mean(spent)))
            trials.append(dict(label=candidate['label'],protected=list(candidate['protected']),protection_cost=candidate['cost'],
                               objective=score[0],mean_total_cost=-score[1],training_values=values))
            if score[0]>best[0]+1e-12 or (abs(score[0]-best[0])<=1e-12 and score[1]>best[1]+1e-12):
                best=score;winner=candidate
        return winner,trials
