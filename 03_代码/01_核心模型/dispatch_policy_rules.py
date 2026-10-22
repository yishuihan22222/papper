"""Same-layer dispatch comparators, with the existing V5 completion and closure.

These are transparent policy baselines, not newly claimed optimization methods.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd
from configured_dual_mechanism_v5 import ConfiguredDispatchAllocator


@dataclass(frozen=True)
class DispatchRuleSpec:
    rule: str = 'lexicographic_lp'
    tie_order: str = 'canonical'
    compatibility_scope: str = 'original_proxy_assumption'
    evidence_ids: tuple = ()

    def __post_init__(self):
        if self.rule not in ['lexicographic_lp','remaining_capacity_share','nearest_feasible']:
            raise ValueError('Unknown dispatch rule')
        if self.tie_order not in ['canonical','reverse_sources'] or (self.tie_order!='canonical' and self.rule!='nearest_feasible'):
            raise ValueError('Source tie reversal is only defined for nearest_feasible')
        if self.compatibility_scope not in ['original_proxy_assumption','synthetic_same_service','explicit_scenario']:
            raise ValueError('Unknown compatibility scope')
        object.__setattr__(self,'evidence_ids',tuple(self.evidence_ids))
        if any(not isinstance(v,str) or not v for v in self.evidence_ids):raise ValueError('Invalid evidence IDs')

    def snapshot(self):
        return dict(schema='dispatch-rule-1',rule=self.rule,tie_order=self.tie_order,
                    compatibility_scope=self.compatibility_scope,evidence_ids=list(self.evidence_ids),
                    semantics='Same-service divisible batch; permissions are separate input assumptions')

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.snapshot(),sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def dispatch_batch(displaced,residual,src,dst,cost,source_rank,target_rank,rule,tie_order='canonical'):
    """Return accepted arc quantities, conserving each source/receiver budget.

    Proportional sharing proposes simultaneously to each feasible receiver using
    its remaining admission space as weight. Oversubscribed receivers scale all
    proposals equally; repeat after deduction. No source is processed first.
    Nearest feasible processes all arcs by (cost, source ID, receiver ID).
    """
    displaced=np.asarray(displaced,float);residual=np.asarray(residual,float)
    src=np.asarray(src,int);dst=np.asarray(dst,int);cost=np.asarray(cost,float)
    if rule not in ['remaining_capacity_share','nearest_feasible']:raise ValueError('Expected heuristic rule')
    if (not np.isfinite(np.r_[displaced,residual,cost]).all() or min(displaced.min(),residual.min())<0
        or displaced.shape!=residual.shape or len(src)!=len(dst) or len(src)!=len(cost)):
        raise ValueError('Invalid dispatch batch')
    n=len(displaced);remaining=displaced.copy();space=residual.copy();flow=np.zeros(len(src));iterations=0
    if not len(src):return flow,0
    if min(src.min(),dst.min())<0 or max(src.max(),dst.max())>=n:raise ValueError('Invalid endpoint')
    if rule=='nearest_feasible':
        rank=np.asarray(source_rank)[src]
        if tie_order=='reverse_sources':rank=-rank
        for arc in np.lexsort((np.asarray(target_rank)[dst],rank,cost)):
            i,j=src[arc],dst[arc];x=min(remaining[i],space[j])
            flow[arc]=x;remaining[i]=max(0.,remaining[i]-x);space[j]=max(0.,space[j]-x)
        return flow,1
    # At least one source or receiver is exhausted in every nonzero iteration.
    for _ in range(2*n+2):
        available=(remaining[src]>1e-9)&(space[dst]>1e-9)
        weight=np.where(available,space[dst],0.)
        denom=np.bincount(src,weights=weight,minlength=n)
        proposal=np.divide(remaining[src]*weight,denom[src],out=np.zeros_like(weight),where=denom[src]>0)
        total=np.bincount(dst,weights=proposal,minlength=n)
        fraction=np.divide(space,total,out=np.ones_like(space),where=total>0)
        accepted=proposal*np.minimum(fraction[dst],1.)
        if accepted.sum()<=1e-9:break
        flow+=accepted
        remaining=np.maximum(0.,remaining-np.bincount(src,weights=accepted,minlength=n))
        space=np.maximum(0.,space-np.bincount(dst,weights=accepted,minlength=n));iterations+=1
    else:raise RuntimeError('Proportional dispatch did not exhaust a source/receiver')
    return flow,iterations


class DispatchRuleAllocator(ConfiguredDispatchAllocator):
    def __init__(self,nodes,edges,config,rule_spec,compatibility=None):
        if not isinstance(rule_spec,DispatchRuleSpec):raise TypeError('Expected DispatchRuleSpec')
        if config.completion_rule!='proportional':raise ValueError('Study fixes completion at proportional')
        super().__init__(nodes,edges,config)
        self._rule_spec=rule_spec
        if compatibility is None:
            if rule_spec.compatibility_scope=='explicit_scenario':raise ValueError('Explicit compatibility matrix required')
            self._compatibility=None;permission_hash='all_original_potential_arcs'
        else:
            matrix=np.asarray(compatibility)
            if matrix.dtype!=bool or matrix.shape!=(len(self.ids),len(self.ids)) or not matrix.diagonal().all():
                raise ValueError('Boolean source/receiver permissions with true local diagonal required')
            self._compatibility=matrix.copy();self._compatibility.setflags(write=False)
            permission_hash=hashlib.sha256(self.compatibility.tobytes()).hexdigest()
        self.input_fingerprint=hashlib.sha256((self.input_fingerprint+rule_spec.fingerprint+permission_hash).encode()).hexdigest()

    @property
    def rule_spec(self):return self._rule_spec

    @property
    def compatibility(self):return self._compatibility

    def paths(self,failed):
        value=super().paths(failed)
        if self.compatibility is None:return value
        result=value.copy();result[~self.compatibility]=np.inf
        return result

    def clear_path_cache(self):
        # The underlying inherited cache is shared by method identity.
        from v4_parameter_audit_core import HopLimitedAllocator
        HopLimitedAllocator.paths.cache_clear()

    def evaluate(self,failed,demand,*,details=False):
        if self.rule_spec.rule=='lexicographic_lp':return super().evaluate(failed,demand,details=details)
        failed=frozenset(failed);D=np.asarray(demand,float);n=len(self.ids)
        if not failed<=set(self.ids) or D.shape!=(n,) or not np.isfinite(D).all() or D.min()<0 or D.sum()<=0:
            raise ValueError('Invalid failed set or demand')
        self.solve_calls+=1
        active=np.array([k not in failed for k in self.ids]);K=self.nominal_capacity
        own=np.where(active,np.minimum(D,K),0.);displaced=D-own
        residual=np.where(active,np.maximum(self.dispatch_limit-own,0.),0.)
        paths=self.paths(failed).copy();diagonal=np.flatnonzero(active);paths[diagonal,diagonal]=0.
        feasible=(displaced[:,None]>1e-9)&(residual[None,:]>1e-9)&np.isfinite(paths)
        src,dst=np.nonzero(feasible);order=np.lexsort((self.canonical_rank[dst],self.canonical_rank[src]))
        src,dst=src[order],dst[order]
        flow,iterations=dispatch_batch(displaced,residual,src,dst,paths[src,dst],self.canonical_rank,self.canonical_rank,
                                      self.rule_spec.rule,self.rule_spec.tie_order)
        outgoing=np.bincount(src,weights=flow,minlength=n);incoming=np.bincount(dst,weights=flow,minlength=n)
        load=own+incoming;assigned=own+outgoing
        fraction=np.divide(np.minimum(load,K),load,out=np.zeros_like(load),where=load>0)
        done=flow*fraction[dst];completed=own*fraction+np.bincount(src,weights=done,minlength=n)
        receiver=load*fraction;util=load/K;transfer=src!=dst
        errors=[max(0.,float(np.max(outgoing-displaced))),max(0.,float(np.max(load-self.dispatch_limit))),
                max(0.,float(np.max(completed-D))),max(0.,float(-flow.min(initial=0.))),
                abs(float(receiver.sum()-completed.sum()))]
        error=max(errors)
        if error>1e-5:raise AssertionError(('Dispatch feasibility',errors))
        m=dict(Service_Rate=float(completed.sum()/D.sum()),Attempted_Service_Rate=float(assigned.sum()/D.sum()),
            Served_Load=float(completed.sum()),Unserved_Load=float((D-completed).sum()),Demand_Load=float(D.sum()),
            Attempted_Assigned_Load=float(assigned.sum()),Undispatched_Load=float((D-assigned).sum()),
            Uncompleted_Assigned_Load=float(assigned.sum()-completed.sum()),Transferred_Load=float(done[transfer].sum()),
            Attempted_Transferred_Load=float(flow[transfer].sum()),Emergency_Self_Admission=float(flow[~transfer].sum()),
            Transfer_Cost=float(np.dot(flow,paths[src,dst])),Completed_Transfer_Cost=float(np.dot(done,paths[src,dst])),
            Initial_Active_Overflow=float(np.maximum(D[active]-K[active],0.).sum()),
            Nominal_Excess_Load=float(np.maximum(load-K,0.).sum()),Max_Utilization=float(util.max()),
            Max_LP_Violation=error,Excess_Optimality_Gap=None,Transfer_Service_Gap=None,
            Service_Conservation_Error=abs(float(completed.sum()+(D-completed).sum()-D.sum())),
            Assignment_Conservation_Error=abs(float(assigned.sum()+(D-assigned).sum()-D.sum())),
            Receiver_Conservation_Error=abs(float(receiver.sum()-completed.sum())),
            Receiver_Assignment_Error=abs(float(load.sum()-assigned.sum())),Dispatch_Iterations=iterations)
        extra=None
        if details:
            extra=dict(served=completed,load=load,unserved=D-completed,assigned=assigned,completed_receiver=receiver,
                completion_fraction=fraction,arcs=pd.DataFrame(dict(source_id=[self.ids[i] for i in src],
                target_id=[self.ids[i] for i in dst],flow=flow,completed_flow=done,cost=paths[src,dst],is_transfer=transfer)))
        return m,util,completed,extra
