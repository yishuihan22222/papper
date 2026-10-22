"""Independent sensitivity adapters. Production V3/V4 files are not edited."""
from functools import lru_cache
import numpy as np
from closed_loop_recovery_v3 import CapacityAllocator
from closed_loop_recovery_v4 import ClosedLoopRecoveryV4

BASE_WEIGHTS = np.array([.21, .16, .12, .10, .09, .07])
WEIGHT_NAMES = ['neighbor_fraction','edge_risk','cross_district','transport_cost','capacity_fragility','utilization']


class HopLimitedAllocator(CapacityAllocator):
    """Cheapest path with <=h edges and surviving internal/receiving nodes.

The failed origin is allowed to dispatch displaced demand. No physical edge or
intermediate throughput capacities are invented. h=0 is no-transfer control.
"""
    def __init__(self, nodes, edges, max_hops=2):
        super().__init__(nodes, edges)
        if max_hops not in [0, 1, 2, 3]:
            raise ValueError('Supported diagnostic range: 0..3 hops')
        self.max_hops = max_hops

    @lru_cache(maxsize=48)
    def paths(self, failed):
        active = np.array([x not in failed for x in self.ids])
        cost = self.direct.copy() if self.max_hops else np.full_like(self.direct, np.inf)
        for _ in range(1, self.max_hops):
            previous = cost.copy()
            for mid in np.flatnonzero(active):
                np.minimum(cost, previous[:, mid, None]+self.direct[mid, None, :], out=cost)
        cost[:, ~active] = np.inf
        np.fill_diagonal(cost, np.inf)
        return cost


class ParameterizedV4(ClosedLoopRecoveryV4):
    """Nonnegative risk weights with fixed total .75; separate bonus scale."""
    def __init__(self, *args, weights=None, bonus_scale=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = BASE_WEIGHTS.copy() if weights is None else np.asarray(weights, float)
        if self.weights.shape != (6,) or not np.isfinite(self.weights).all() or (self.weights < 0).any() or not np.isclose(self.weights.sum(), .75):
            raise ValueError('Six nonnegative weights must sum to .75')
        if not np.isfinite(bonus_scale) or bonus_scale < 0:
            raise ValueError('Invalid bonus scale')
        self.bonus = self.bonus*bonus_scale
        p, cross, cost, capacity = self.attributes
        self.edge_terms = [np.full(len(self.src), self.weights[0]), self.weights[1]*p,
                           self.weights[2]*cross, self.weights[3]*np.minimum(cost/.45, 1.),
                           self.weights[4]*(1-np.minimum(capacity/55., 1.))]
        self.edge_exposure = np.sum(self.edge_terms, axis=0)

    def risk_components(self, failed, util):
        c = super().risk_components(failed, util)
        c['utilization'] = self.weights[5]*np.minimum(np.asarray(util), 1.6)/.75
        c['risk'] = c['neighbor_pressure']+c['utilization']+c['bonus']
        return c


def configurations():
    items = [dict(name='baseline', axis='baseline', capacity_ratio=1.25, hops=2,
                  weights=BASE_WEIGHTS.tolist(), bonus_scale=1.)]
    for ratio in [1., 1.1, 1.5]:
        items.append(dict(items[0], name=f'capacity_{ratio:.2f}', axis='capacity_oat', capacity_ratio=ratio))
    for hops in [0, 1, 3]:
        items.append(dict(items[0], name=f'hops_{hops}', axis='hops_oat', hops=hops))
    for j, label in enumerate(WEIGHT_NAMES):
        for factor in [.8, 1.2]:
            w = BASE_WEIGHTS.copy(); w[j] *= factor; w *= .75/w.sum()
            items.append(dict(items[0], name=f'weight_{label}_{factor:.1f}', axis='weights_oat',
                              weights=w.tolist(), varied_weight=label, raw_weight_factor=factor))
    for factor in [0., .5, 1.5]:
        items.append(dict(items[0], name=f'bonus_{factor:.1f}', axis='bonus_oat', bonus_scale=factor))
    for ratio in [1., 1.1, 1.5]:
        for hops in [1, 3]:
            items.append(dict(items[0], name=f'capacity_{ratio:.2f}_hops_{hops}',
                              axis='capacity_hops_joint', capacity_ratio=ratio, hops=hops))
    assert len(items) == 28 and len({x['name'] for x in items}) == 28
    return items
