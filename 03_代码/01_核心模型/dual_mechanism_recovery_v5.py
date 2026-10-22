"""V5: dispatch admission, nominal completed service, and two failure channels.

All capacities/loads are proxies. Excess dispatched demand is NOT completed
throughput. A fixed-slot proportional completion rule enforces nominal capacity;
unfinished assigned demand is reported, not carried as a measured time queue.
Immediate overload shutdown is an explicit hypothesis, not an empirical fact.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack
from closed_loop_recovery_v3 import ClosedLoopRecovery
from v4_parameter_audit_core import HopLimitedAllocator

RISK_WEIGHTS = np.array([.21,.16,.12,.10,.09])
OVERLOAD_EPSILON = 1e-7


class DispatchAllocatorV5(HopLimitedAllocator):
    """Nominal-priority dispatch, optional emergency admission, and completion.

Stage1 maximizes assigned demand, stage2 optionally minimizes nominal excess,
stage3 minimizes transport cost preserving earlier optima. Emergency own-demand
admission is a zero-cost self arc; own nominal load is served first in dispatch.
Completion at a receiver is proportional across assigned origins and capped at K.
No transit throughput or physical edge capacities are invented.
"""
    def __init__(self,nodes,edges,rho=1.1,policy='min_overload',max_hops=2):
        super().__init__(nodes,edges,max_hops)
        if not np.isfinite(rho) or rho<1 or policy not in ['min_overload','min_cost']:
            raise ValueError('Invalid dispatch configuration')
        if (self.capacity<=0).any():raise ValueError('V5 requires positive nominal capacities')
        self.nominal_capacity=self.capacity.copy()
        self.rho=float(rho);self.policy=policy
        self.dispatch_limit=self.rho*self.nominal_capacity
        # LP row/column order is a documented tie-resolution convention. Keep
        # caller array order for demand/output, use canonical IDs inside the LP.
        self.canonical_order=np.array(sorted(range(len(self.ids)),key=lambda i:self.ids[i]))
        self.canonical_rank=np.empty(len(self.ids),int)
        self.canonical_rank[self.canonical_order]=np.arange(len(self.ids))

    def evaluate(self,failed,demand,*,details=False):
        failed=frozenset(failed);demand=np.asarray(demand,float)
        if not failed<=set(self.ids):raise ValueError('Unknown failed node')
        if demand.shape!=self.capacity.shape or not np.isfinite(demand).all() or (demand<0).any() or demand.sum()<=0:
            raise ValueError('Invalid demand')
        self.solve_calls+=1;n=len(demand)
        active=np.array([k not in failed for k in self.ids])
        own=np.where(active,np.minimum(demand,self.nominal_capacity),0.)
        displaced=demand-own
        residual=np.where(active,np.maximum(self.dispatch_limit-own,0.),0.)
        paths=self.paths(failed).copy()
        # A self arc represents additional local admission, not a transport edge.
        diagonal=np.flatnonzero(active)
        paths[diagonal,diagonal]=0.
        feasible=(displaced[:,None]>1e-9)&(residual[None,:]>1e-9)&np.isfinite(paths)
        src,dst=np.nonzero(feasible)
        arc_order=np.lexsort((self.canonical_rank[dst],self.canonical_rank[src]))
        src,dst=src[arc_order],dst[arc_order];m=len(src);flow=np.zeros(m)
        maximum_assignment=0.;lp_error=0.;excess_optimum=0.;excess_gap=0.
        if m:
            cols=np.r_[np.arange(m),np.arange(m)]
            rows=np.r_[self.canonical_rank[src],n+self.canonical_rank[dst]]
            constraint=coo_matrix((np.ones(2*m),(rows,cols)),shape=(2*n,m)).tocsr()
            rhs=np.r_[displaced[self.canonical_order],residual[self.canonical_order]]
            first=linprog(-np.ones(m),A_ub=constraint,b_ub=rhs,bounds=(0,None),method='highs')
            if not first.success:raise RuntimeError('Assignment LP: '+first.message)
            maximum_assignment=float(first.x.sum())
            objective=paths[src,dst]
            if self.policy=='min_cost':
                last=linprog(objective,A_ub=constraint,b_ub=rhs,A_eq=np.ones((1,m)),b_eq=[maximum_assignment],bounds=(0,None),method='highs')
                if not last.success:raise RuntimeError('Dispatch cost LP: '+last.message)
                flow=last.x
            else:
                receiver=constraint[n:,:]
                top=hstack([constraint,csr_matrix((2*n,n))],format='csr')
                over=hstack([receiver,-eye(n,format='csr')],format='csr')
                a=vstack([top,over],format='csr')
                b=np.r_[rhs,(self.nominal_capacity-own)[self.canonical_order]]
                equality=np.r_[np.ones(m),np.zeros(n)][None,:]
                second=linprog(np.r_[np.zeros(m),np.ones(n)],A_ub=a,b_ub=b,A_eq=equality,b_eq=[maximum_assignment],bounds=(0,None),method='highs')
                if not second.success:raise RuntimeError('Excess LP: '+second.message)
                excess_optimum=max(0.,float(second.x[m:].sum()))
                eq2=np.vstack([equality,np.r_[np.zeros(m),np.ones(n)][None,:]])
                last=linprog(np.r_[objective,np.zeros(n)],A_ub=a,b_ub=b,A_eq=eq2,b_eq=[maximum_assignment,excess_optimum],bounds=(0,None),method='highs')
                if not last.success:raise RuntimeError('Lexicographic cost LP: '+last.message)
                flow=last.x[:m]
                lp_error=max(lp_error,float(np.maximum(a@last.x-b,0).max()),abs(float(last.x[m:].sum())-excess_optimum))
            lp_error=max(lp_error,float(np.maximum(constraint@flow-rhs,0).max()),max(0.,float(-flow.min())),abs(float(flow.sum())-maximum_assignment))
        outgoing=np.bincount(src,weights=flow,minlength=n)
        incoming=np.bincount(dst,weights=flow,minlength=n)
        load=own+incoming
        assigned=own+outgoing
        completion_fraction=np.divide(np.minimum(load,self.nominal_capacity),load,out=np.zeros_like(load),where=load>0)
        completed_flow=flow*completion_fraction[dst]
        completed=own*completion_fraction+np.bincount(src,weights=completed_flow,minlength=n)
        receiver_completed=load*completion_fraction
        unserved=demand-completed;undispatched=demand-assigned
        util=load/self.nominal_capacity
        nominal_excess=float(np.maximum(load-self.nominal_capacity,0.).sum())
        if self.policy=='min_overload':excess_gap=abs(nominal_excess-excess_optimum)
        transfer=src!=dst
        errors=[lp_error,excess_gap,max(0.,float(np.max(load-self.dispatch_limit))),
                max(0.,float(np.max(completed-demand))),max(0.,float(-unserved.min())),
                max(0.,float(np.max(receiver_completed-self.nominal_capacity)))]
        if max(errors)>1e-5:raise AssertionError(('V5 allocation invariant',errors))
        metrics=dict(Service_Rate=float(completed.sum()/demand.sum()),
                     Attempted_Service_Rate=float(assigned.sum()/demand.sum()),
                     Served_Load=float(completed.sum()),Unserved_Load=float(unserved.sum()),Demand_Load=float(demand.sum()),
                     Attempted_Assigned_Load=float(assigned.sum()),Undispatched_Load=float(undispatched.sum()),
                     Uncompleted_Assigned_Load=float(assigned.sum()-completed.sum()),
                     Transferred_Load=float(completed_flow[transfer].sum()),
                     Attempted_Transferred_Load=float(flow[transfer].sum()),
                     Emergency_Self_Admission=float(flow[~transfer].sum()),
                     Transfer_Cost=float(np.dot(flow,paths[src,dst])),
                     Completed_Transfer_Cost=float(np.dot(completed_flow,paths[src,dst])),
                     Initial_Active_Overflow=float(np.maximum(demand[active]-self.nominal_capacity[active],0.).sum()),
                     Nominal_Excess_Load=nominal_excess,Max_Utilization=float(util.max()),
                     Max_LP_Violation=max(errors),Excess_Optimality_Gap=excess_gap,
                     Service_Conservation_Error=abs(float(completed.sum()+unserved.sum()-demand.sum())),
                     Assignment_Conservation_Error=abs(float(assigned.sum()+undispatched.sum()-demand.sum())),
                     Receiver_Conservation_Error=abs(float(receiver_completed.sum()-completed.sum())),
                     Receiver_Assignment_Error=abs(float(load.sum()-assigned.sum())),
                     Transfer_Service_Gap=abs(float(flow.sum())-maximum_assignment))
        extra=None
        if details:
            extra=dict(served=completed,load=load,unserved=unserved,assigned=assigned,
                       completed_receiver=receiver_completed,completion_fraction=completion_fraction,
                       arcs=pd.DataFrame(dict(source_id=[self.ids[i] for i in src],target_id=[self.ids[i] for i in dst],
                                              flow=flow,completed_flow=completed_flow,cost=paths[src,dst],is_transfer=transfer)))
        return metrics,util,completed,extra


class DualMechanismRecoveryV5(ClosedLoopRecovery):
    """Two separately evaluated channels; shared state does not imply independence.

Overload remains enabled when risk propagation is disabled. Failure is synchronous
and irreversible until an explicit repair; service for recovery is measured only
after closure. Every new endogenous failure has a disjoint trigger label.
"""
    def __init__(self,allocator,edges,components,group_names,initial_failed,kind,
                 theta_risk,gamma=1.,overload_enabled=True,risk_enabled=True,
                 bonus_amplitude=.10,risk_weights=None,epsilon=OVERLOAD_EPSILON):
        # Canonical edge order also fixes summation and incident-cost ordering.
        edges=edges.sort_values(['Source_id','Target_id']).reset_index(drop=True)
        super().__init__(allocator,edges,components,group_names,initial_failed,kind,'continuing',theta_risk)
        if not isinstance(allocator,DispatchAllocatorV5):raise TypeError('V5 allocator required')
        if not np.isfinite(gamma) or gamma<1 or not np.isfinite(theta_risk) or theta_risk<0:
            raise ValueError('Invalid failure thresholds')
        if not np.isfinite(epsilon) or epsilon<0 or not np.isfinite(bonus_amplitude) or bonus_amplitude<0:
            raise ValueError('Invalid tolerance/bonus')
        self.gamma=float(gamma);self.theta_risk=float(theta_risk)
        self.overload_enabled=bool(overload_enabled);self.risk_enabled=bool(risk_enabled)
        self.epsilon=float(epsilon);self.bonus=self.bonus*(bonus_amplitude/.10)
        w=RISK_WEIGHTS.copy() if risk_weights is None else np.array(risk_weights,float)
        if w.shape!=(5,) or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),.68):
            raise ValueError('Five nonnegative weights must sum to .68')
        self.risk_weights=w
        p,c,cost,capacity=self.attributes
        if not all(np.isfinite(x).all() for x in self.attributes) or (p<0).any() or (p>1).any() or (c<0).any() or (c>1).any() or (capacity<0).any():
            raise ValueError('Invalid risk attributes')
        self.edge_terms=[np.full(len(self.src),w[0]),w[1]*p,w[2]*c,
                         w[3]*np.minimum(cost/.45,1.),w[4]*(1-np.minimum(capacity/55.,1.))]
        self.edge_exposure=np.sum(self.edge_terms,axis=0)
        self.pressure_cache={};self.cascade_logs={};self.cascade_events={}

    def pressure(self,failed):
        key=frozenset(failed)
        if key not in self.pressure_cache:
            removed=np.array([k in key for k in self.a.ids]);mask=removed[self.dst]
            count=np.bincount(self.src[mask],minlength=len(removed))
            terms=[np.bincount(self.src[mask],weights=t[mask],minlength=len(removed))/np.maximum(self.degree,1)/.68 for t in self.edge_terms]
            self.pressure_cache[key]=(count,np.sum(terms,axis=0)+self.bonus)
        return self.pressure_cache[key]

    def classify(self,failed,util):
        rm=np.array([k in failed for k in self.a.ids]);util=np.asarray(util,float)
        if util.shape!=rm.shape or not np.isfinite(util).all():raise ValueError('Invalid utilization')
        count,pressure=self.pressure(failed)
        over=(~rm)&self.overload_enabled&(util>self.gamma+self.epsilon)
        risk=(~rm)&self.risk_enabled&(count>0)&(pressure>=self.theta_risk)
        def ids(mask):return frozenset(self.a.ids[i] for i in np.flatnonzero(mask))
        return dict(overload_only=ids(over&~risk),risk_only=ids(risk&~over),both=ids(over&risk),
                    new=ids(over|risk),pressure=pressure,utilization=util)

    def newly_failed(self,failed,util):return self.classify(failed,util)['new']

    def closure(self,failed):
        key=frozenset(failed)
        if key not in self.closed:
            current=key;rounds=0;logs=[];events=[];counts=dict(overload_only=0,risk_only=0,both=0)
            for step in range(len(self.a.ids)+1):
                metric,util,_=self.state(current);cause=self.classify(current,util)
                for label in counts:
                    counts[label]+=len(cause[label])
                    for node in sorted(cause[label]):
                        ix=self.a.pos[node]
                        events.append(dict(round=step,node_id=node,cause=label,utilization=float(util[ix]),
                                           pressure=float(cause['pressure'][ix]),gamma=self.gamma,theta_risk=self.theta_risk))
                logs.append(dict(round=step,failed_count=len(current),new_failures=len(cause['new']),
                                 overload_only=len(cause['overload_only']),risk_only=len(cause['risk_only']),both=len(cause['both']),
                                 new_ids=sorted(cause['new']),**metric))
                if not cause['new']:break
                current=current|cause['new'];rounds+=1
            else:raise RuntimeError('V5 closure did not converge')
            stable=dict(metric,Overload_Only_Failures=counts['overload_only'],Risk_Only_Failures=counts['risk_only'],
                        Both_Failures=counts['both'],Secondary_Failures=len(current-key),
                        Stable_Service_Rate=metric['Service_Rate'])
            if sum(counts.values())!=len(current-key):raise AssertionError('Failure attribution does not partition new failures')
            self.closed[key]=(current,stable,rounds);self.cascade_logs[key]=logs;self.cascade_events[key]=events
        return self.closed[key]
