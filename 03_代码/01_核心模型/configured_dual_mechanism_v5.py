"""Versioned configuration/completion extension; frozen V5 solver is reused."""
import hashlib
import numpy as np
import pandas as pd
from completion_rules import complete_batch
from model_config import ModelConfig
from dual_mechanism_recovery_v5 import DispatchAllocatorV5, DualMechanismRecoveryV5


def frame_hash(frame):
    h = hashlib.sha256()
    h.update('|'.join(map(str, frame.columns)).encode('utf-8'))
    h.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return h.hexdigest()


class ConfiguredDispatchAllocator(DispatchAllocatorV5):
    def __init__(self, nodes, edges, config):
        if not isinstance(config, ModelConfig):
            raise TypeError('Expected immutable ModelConfig')
        self._config = config
        super().__init__(nodes, edges, config.rho, config.dispatch_policy, config.max_hops)
        self.input_fingerprint = hashlib.sha256((frame_hash(self.nodes) + frame_hash(edges)).encode()).hexdigest()
        self.source_rank = None
        if config.completion_rule == 'given_order':
            if len(config.source_order) != len(self.ids) or set(config.source_order) != set(self.ids):
                raise ValueError('source_order must be an exact permutation of all source node IDs')
            ranks = {node: rank for rank, node in enumerate(config.source_order)}
            self.source_rank = np.array([ranks[k] for k in self.ids], dtype=int)

    @property
    def config(self):
        return self._config

    def evaluate(self, failed, demand, *, details=False):
        if self.config.completion_rule == 'proportional':
            # Preserve legacy floating-point operations and return values exactly.
            return super().evaluate(failed, demand, details=details)
        metrics, util, _, extra = super().evaluate(failed, demand, details=True)
        demand = np.asarray(demand, float)
        active = np.array([node not in failed for node in self.ids])
        own = np.where(active, np.minimum(demand, self.nominal_capacity), 0.)
        arcs = extra['arcs']
        src = np.array([self.pos[k] for k in arcs.source_id], dtype=int)
        dst = np.array([self.pos[k] for k in arcs.target_id], dtype=int)
        flow = arcs.flow.to_numpy(float)
        correction = float(-np.minimum(flow, 0.).sum())
        if correction > self.config.completion_flow_tolerance:
            raise AssertionError('Negative LP flow exceeds completion tolerance')
        own_done, done = complete_batch(own, src, dst, np.maximum(flow, 0.), self.nominal_capacity,
                                        rule=self.config.completion_rule, source_rank=self.source_rank)
        completed = own_done + np.bincount(src, weights=done, minlength=len(self.ids))
        receiver = own_done + np.bincount(dst, weights=done, minlength=len(self.ids))
        target = np.minimum(extra['load'], self.nominal_capacity)
        if (np.max(abs(receiver - target)) > 1e-6 or np.max(completed-demand) > 1e-6
                or np.min(completed) < -1e-6 or np.max(done-flow, initial=0.) > 1e-6):
            raise AssertionError('Completion attribution violated batch conservation')
        served = float(completed.sum())
        transferred = src != dst
        metrics = dict(metrics, Service_Rate=served / demand.sum(), Served_Load=served,
                       Completion_Flow_Clip_Error=correction,
                       Unserved_Load=float((demand-completed).sum()),
                       Uncompleted_Assigned_Load=float(extra['assigned'].sum()-served),
                       Transferred_Load=float(done[transferred].sum()),
                       Completed_Transfer_Cost=float(np.dot(done, arcs.cost.to_numpy())),
                       Service_Conservation_Error=abs(float(served + (demand-completed).sum()-demand.sum())),
                       Receiver_Conservation_Error=abs(float(receiver.sum()-served)))
        if details:
            extra = dict(extra, served=completed, unserved=demand-completed,
                         completed_receiver=receiver, receiver_completion_fraction=np.divide(receiver, extra['load'],
                         out=np.zeros_like(receiver), where=extra['load'] > 0))
            # A single multiplier cannot reconstruct nonproportional source completion.
            extra.pop('completion_fraction', None)
            extra['completion_rule'] = self.config.completion_rule
            extra['own_completed'] = own_done
            extra['arcs'] = arcs.assign(completed_flow=done)
        return metrics, util, completed, extra if details else None


class ConfiguredDualMechanismRecovery(DualMechanismRecoveryV5):
    def __init__(self, allocator, edges, components, group_names, initial_failed, kind, config=None):
        if not isinstance(allocator, ConfiguredDispatchAllocator):
            raise TypeError('Expected ConfiguredDispatchAllocator')
        config = allocator.config if config is None else config
        if not isinstance(config, ModelConfig) or config.fingerprint != allocator.config.fingerprint:
            raise ValueError('Allocator and recovery must use the same complete configuration')
        self._config = config
        edges = edges.sort_values(['Source_id', 'Target_id']).reset_index(drop=True)
        super().__init__(allocator, edges, components, group_names, initial_failed, kind,
                         config.theta_risk, gamma=config.gamma, overload_enabled=config.overload_enabled,
                         risk_enabled=config.risk_enabled, bonus_amplitude=config.bonus_amplitude,
                         epsilon=config.overload_epsilon)
        self.bonus = np.zeros(len(allocator.ids))
        if kind == 'spatial_cluster':
            self.bonus = config.bonus_amplitude * allocator.nodes['区县'].isin(config.spatial_districts).to_numpy()
        elif kind == 'channel_disruption':
            self.bonus = config.bonus_amplitude * allocator.nodes.family.isin(config.channel_families).to_numpy()
        self.risk_weights = np.array(config.risk_weights)
        self.attributes[2] = np.maximum(np.tile(edges.Composite_transport_cost.to_numpy(float), 2), config.risk_cost_floor)
        p, c, cost, capacity = self.attributes
        w = self.risk_weights
        self.edge_terms = [np.full(len(self.src), w[0]), w[1]*p, w[2]*c,
                           w[3]*np.minimum(cost/config.risk_cost_scale, 1.),
                           w[4]*(1-np.minimum(capacity/config.risk_capacity_scale, 1.))]
        self.edge_exposure = np.sum(self.edge_terms, axis=0)
        h = hashlib.sha256((config.fingerprint + allocator.input_fingerprint + frame_hash(edges)).encode())
        h.update(np.asarray(self.parts, dtype='<f8').tobytes())
        h.update(str(kind).encode())
        self.cache_namespace = h.hexdigest()

    @property
    def config(self):
        return self._config

    def state(self, failed):
        failed = frozenset(failed)
        key = (self.cache_namespace, failed)
        if key not in self.cache:
            self.cache[key] = self.a.evaluate(failed, self.demand)[:3]
        return self.cache[key]

    def pressure(self, failed):
        failed = frozenset(failed)
        key = (self.cache_namespace, failed)
        if key not in self.pressure_cache:
            removed = np.array([node in failed for node in self.a.ids])
            mask = removed[self.dst]
            count = np.bincount(self.src[mask], minlength=len(removed))
            terms = [np.bincount(self.src[mask], weights=t[mask], minlength=len(removed))
                     / np.maximum(self.degree, 1) / self.config.pressure_normalizer for t in self.edge_terms]
            self.pressure_cache[key] = (count, np.sum(terms, axis=0) + self.bonus)
        return self.pressure_cache[key]
