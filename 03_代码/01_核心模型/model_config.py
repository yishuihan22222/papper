"""Immutable, auditable inputs for the configured V5 entry point."""
from dataclasses import dataclass, field, fields
from hashlib import sha256
import json
import math
from pathlib import Path


def parameter(default, unit, kind, source, note=''):
    return field(default=default, metadata=dict(unit=unit, evidence_kind=kind,
                 empirically_calibrated=False, source=source, note=note))


@dataclass(frozen=True)
class ModelConfig:
    theta_risk: float = field(metadata=dict(unit='dimensionless risk score',
        evidence_kind='assumption_or_proxy_design', empirically_calibrated=False,
        source='Required caller input; see theta_source', note='Not an observed failure threshold'))
    rho: float = parameter(1.1, 'capacity multiplier', 'assumption', 'legacy V5 admission limit')
    gamma: float = parameter(1., 'nominal utilization', 'assumption', 'legacy V5 shutdown threshold')
    risk_weights: tuple = parameter((.21, .16, .12, .10, .09), 'score weights', 'proxy_scoring_assumption', 'legacy V5 RISK_WEIGHTS')
    bonus_amplitude: float = parameter(.10, 'risk score', 'assumption', 'legacy V5 scenario bonus')
    pressure_normalizer: float = parameter(.68, 'score weight sum', 'normalization_convention', 'legacy V5 fixed denominator')
    risk_cost_scale: float = parameter(.45, 'transport cost proxy', 'proxy_scale_assumption', 'legacy V5 edge cost score')
    risk_capacity_scale: float = parameter(55., 'edge capacity proxy', 'proxy_scale_assumption', 'legacy V5 edge capacity score')
    risk_cost_floor: float = parameter(.001, 'transport cost proxy', 'numerical_convention', 'legacy V3/V5 cost floor')
    overload_epsilon: float = parameter(1e-7, 'utilization tolerance', 'numerical_convention', 'legacy V5 OVERLOAD_EPSILON')
    completion_flow_tolerance: float = parameter(1e-7, 'proxy flow units', 'numerical_convention', 'configured completion adapter', 'Maximum total tiny negative LP flow corrected to zero; admission/utilization remain unchanged')
    max_hops: int = parameter(2, 'edges', 'reachability_assumption', 'legacy V5 two-hop reachability')
    dispatch_policy: str = parameter('min_overload', 'LP objective convention', 'model_choice', 'legacy V5 lexicographic assignment/excess/cost')
    completion_rule: str = parameter('proportional', 'batch allocation convention', 'model_choice', 'configured V5; historical default proportional')
    source_order: tuple = parameter((), 'ordered source node IDs', 'specified_batch_order', 'Caller supplied; no arrival times inferred', 'Required full source permutation for given_order; not physical FIFO')
    overload_enabled: bool = parameter(True, 'switch', 'mechanism_assumption', 'legacy V5')
    risk_enabled: bool = parameter(True, 'switch', 'mechanism_assumption', 'legacy V5')
    spatial_districts: tuple = parameter(('东西湖区', '新洲区', '黄陂区'), 'district labels', 'scenario_encoding_assumption', 'legacy V5 spatial bonus membership')
    channel_families: tuple = parameter(('airport', 'rail', 'port'), 'proxy family labels', 'scenario_encoding_assumption', 'legacy V5 channel bonus membership')
    theta_source: str = parameter('unspecified_assumption', 'provenance label', 'provenance', 'Caller declaration; does not establish empirical calibration')

    def __post_init__(self):
        for name in ['theta_risk', 'rho', 'gamma', 'bonus_amplitude', 'pressure_normalizer',
                     'risk_cost_scale', 'risk_capacity_scale', 'risk_cost_floor', 'overload_epsilon', 'completion_flow_tolerance']:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Finite numeric parameter required: ' + name)
            object.__setattr__(self, name, float(value))
        for name in ['risk_weights', 'source_order', 'spatial_districts', 'channel_families']:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.theta_risk < 0 or self.rho < 1 or self.gamma < 1 or self.bonus_amplitude < 0 or min(self.overload_epsilon,self.completion_flow_tolerance) < 0:
            raise ValueError('Invalid threshold, multiplier, bonus or tolerance')
        if min(self.pressure_normalizer, self.risk_cost_scale, self.risk_capacity_scale, self.risk_cost_floor) <= 0:
            raise ValueError('Normalizers/scales/floor must be positive')
        if (len(self.risk_weights) != 5 or any(isinstance(w, bool) or not isinstance(w, (int, float))
                or not math.isfinite(w) or w < 0 for w in self.risk_weights)
                or not math.isclose(sum(self.risk_weights), self.pressure_normalizer, abs_tol=1e-12, rel_tol=1e-12)):
            raise ValueError('Five nonnegative weights must sum to the explicit pressure normalizer')
        object.__setattr__(self, 'risk_weights', tuple(float(w) for w in self.risk_weights))
        if type(self.max_hops) is not int or self.max_hops not in range(4):
            raise ValueError('Supported diagnostic hop range: 0..3')
        if self.dispatch_policy not in ['min_overload', 'min_cost']:
            raise ValueError('Unknown dispatch policy')
        if self.completion_rule not in ['proportional', 'local_first', 'given_order']:
            raise ValueError('Unknown completion rule; actual FIFO is not defined')
        if type(self.overload_enabled) is not bool or type(self.risk_enabled) is not bool:
            raise ValueError('Mechanism switches must be booleans')
        for name in ['source_order', 'spatial_districts', 'channel_families']:
            values = getattr(self, name)
            if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
                raise ValueError('Distinct nonempty string labels required: ' + name)
        if (self.completion_rule == 'given_order') != bool(self.source_order):
            raise ValueError('source_order is required only for given_order')
        if not isinstance(self.theta_source, str) or not self.theta_source.strip():
            raise ValueError('theta_source must describe the threshold provenance')

    def audit_rows(self):
        return [dict(parameter=f.name, value=getattr(self, f.name), **dict(f.metadata)) for f in fields(self)]

    def snapshot(self):
        return dict(schema='configured-v5-1', values={f.name: getattr(self, f.name) for f in fields(self)},
                    parameter_audit=self.audit_rows(),
                    semantics='Static batch model; no physical arrival timestamps or empirical calibration inferred')

    def canonical_json(self):
        return json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

    @property
    def fingerprint(self):
        return sha256(self.canonical_json().encode('utf-8')).hexdigest()

    def write_snapshot(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.canonical_json() + '\n'
        if path.exists() and path.read_text(encoding='utf-8') != data:
            raise FileExistsError('Refusing to replace a different configuration snapshot: ' + str(path))
        path.write_text(data, encoding='utf-8')

    @classmethod
    def from_snapshot(cls, snapshot):
        if isinstance(snapshot, (str, Path)):
            snapshot = json.loads(Path(snapshot).read_text(encoding='utf-8'))
        if snapshot.get('schema') != 'configured-v5-1':
            raise ValueError('Unsupported configuration schema')
        result = cls(**snapshot['values'])
        # Metadata is part of the audit identity; do not silently trust altered provenance.
        if json.loads(result.canonical_json()) != json.loads(json.dumps(snapshot)):
            raise ValueError('Configuration metadata does not match this schema implementation')
        return result

    @classmethod
    def for_arm(cls, arm, *, theta_risk, **kwargs):
        if arm not in ['A', 'B', 'C']:
            raise ValueError('Unknown policy arm')
        if {'rho', 'dispatch_policy'} & kwargs.keys():
            raise ValueError('Arm defines rho and dispatch_policy; construct ModelConfig directly for custom settings')
        return cls(theta_risk=theta_risk, rho=1. if arm == 'C' else 1.1,
                   dispatch_policy='min_cost' if arm == 'A' else 'min_overload', **kwargs)
