"""V4 candidate: fixed-neighborhood nonnegative exposure; V3 remains unchanged.

This is an uncalibrated exposure-threshold surrogate, not a failure probability
or a physical overload cascade. The strict allocator and recovery state machine
are inherited unchanged. Global monotonicity is not asserted when LP utilization
changes with failures.
"""
import numpy as np
from closed_loop_recovery_v3 import ClosedLoopRecovery as V3Recovery


class ClosedLoopRecoveryV4(V3Recovery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p, cross, cost, capacity = self.attributes
        if not all(np.isfinite(x).all() for x in self.attributes):
            raise ValueError('Nonfinite exposure attributes')
        if ((p < 0) | (p > 1)).any() or ((cross < 0) | (cross > 1)).any() or (capacity < 0).any():
            raise ValueError('Exposure attributes outside assumed bounds')
        # Transform each edge BEFORE aggregation; all contributions nonnegative.
        self.edge_terms = [np.full(len(self.src), .21), .16*p, .12*cross,
                           .10*np.minimum(cost/.45, 1.),
                           .09*(1.-np.minimum(capacity/55., 1.))]
        self.edge_exposure = np.sum(self.edge_terms, axis=0)

    def risk_components(self, failed, util):
        util = np.asarray(util, float)
        if util.shape != self.demand.shape or not np.isfinite(util).all() or (util < 0).any():
            raise ValueError('Invalid utilization')
        mask = np.array([k in failed for k in self.a.ids])[self.dst]
        count = np.bincount(self.src[mask], minlength=len(util))
        den = np.maximum(self.degree, 1)  # fixed ALL-neighbor denominator
        terms = [np.bincount(self.src[mask], weights=x[mask], minlength=len(util))/den/.75
                 for x in self.edge_terms]
        utilization = .07*np.minimum(util, 1.6)/.75
        return dict(failed_neighbor_count=count, neighbor_fraction=terms[0],
                    edge_risk=terms[1], cross_district=terms[2], transport_cost=terms[3],
                    capacity_fragility=terms[4], utilization=utilization, bonus=self.bonus,
                    neighbor_pressure=np.sum(terms, axis=0),
                    risk=np.sum(terms, axis=0)+utilization+self.bonus)

    def newly_failed(self, failed, util):
        if self.hazard == 'ended':
            return frozenset()
        c = self.risk_components(failed, util)
        active = np.array([k not in failed for k in self.a.ids])
        return frozenset(self.a.ids[i] for i in np.flatnonzero(
            active & (c['failed_neighbor_count'] > 0) & (c['risk'] >= self.threshold)))
