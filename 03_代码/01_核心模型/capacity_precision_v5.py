"""Versioned removal of the second rounding in the synthetic K=1.25L formula.

Existing L, capacity_index, demand and edge attributes are left unchanged.
This is arithmetic consistency of proxy construction, not empirical calibration.
"""
import numpy as np


def formula_precision_nodes(nodes, baseline_nodes):
    """Preserve each designed capacity contrast while removing baseline K rounding.

    The generator constructs K_s = round(1.25L,2) + contrast_s. This version uses
    K'_s = 1.25L + contrast_s, matching by stable node ID. It does not regenerate
    contrasts, normalize totals, refit demand, or alter the shutdown criterion.
    """
    if nodes.node_id.duplicated().any() or baseline_nodes.node_id.duplicated().any():
        raise ValueError('Unique node IDs required')
    if set(nodes.node_id)!=set(baseline_nodes.node_id):raise ValueError('Baseline node identity mismatch')
    base=baseline_nodes.set_index('node_id').loc[nodes.node_id]
    L=nodes.proxy_daily_load.to_numpy(float)
    if not np.array_equal(L,base.proxy_daily_load.to_numpy(float)):raise ValueError('Baseline loads changed')
    K=nodes.overload_threshold.to_numpy(float);K0=base.overload_threshold.to_numpy(float)
    intended=1.25*L
    if not np.isfinite(np.r_[L,K,K0]).all() or min(L.min(),K.min(),K0.min())<=0:
        raise ValueError('Finite positive load and capacity required')
    if not np.allclose(K0,np.round(intended,2),atol=1e-9,rtol=0):
        raise ValueError('Baseline does not match the documented two-decimal K generator')
    correction=intended-K0
    if abs(correction).max()>.005000001:raise AssertionError('Unexpected rounding correction')
    result=nodes.copy();result['overload_threshold']=intended+(K-K0)
    audit=nodes[['node_id','family','proxy_daily_load']].copy()
    audit['stored_capacity']=K;audit['stored_baseline_capacity']=K0
    audit['formula_baseline_capacity']=intended;audit['rounding_correction']=correction
    audit['preserved_capacity_contrast']=K-K0
    audit['formula_precision_capacity']=result.overload_threshold.to_numpy(float)
    return result,audit
