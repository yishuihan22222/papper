"""Work-conserving allocation of an already assigned static batch.

These rules change completion attribution, never admission or arrival times.
local_first serves local demand before proportional completion of transfers.
given_order requires an explicit source permutation and assumes all items ready.
"""
import numpy as np


def complete_batch(own, src, dst, flow, capacity, *, rule='proportional', source_rank=None):
    own, flow, capacity = (np.asarray(x, dtype=float) for x in (own, flow, capacity))
    src, dst = np.asarray(src), np.asarray(dst)
    n = len(capacity)
    if (n == 0 or own.shape != (n,) or capacity.shape != (n,) or flow.ndim != 1
            or src.shape != flow.shape or dst.shape != flow.shape
            or not np.issubdtype(src.dtype, np.integer) or not np.issubdtype(dst.dtype, np.integer)
            or any(not np.isfinite(a).all() or (a < 0).any() for a in [own, flow, capacity])
            or (src < 0).any() or (src >= n).any() or (dst < 0).any() or (dst >= n).any()):
        raise ValueError('Invalid batch dimensions, indices, nonnegative amounts or capacity')
    load = own + np.bincount(dst, weights=flow, minlength=n)
    if rule == 'proportional':
        fraction = np.divide(np.minimum(load, capacity), load, out=np.zeros_like(load), where=load > 0)
        return own * fraction, flow * fraction[dst]
    if rule == 'local_first':
        own_done = np.minimum(own, capacity)
        spare = capacity - own_done
        local = src == dst
        self_load = np.bincount(dst[local], weights=flow[local], minlength=n)
        local_fraction = np.divide(np.minimum(self_load, spare), self_load,
                                   out=np.zeros(n), where=self_load > 0)
        done = np.zeros_like(flow)
        done[local] = flow[local] * local_fraction[dst[local]]
        spare -= np.bincount(dst[local], weights=done[local], minlength=n)
        transferred = np.bincount(dst[~local], weights=flow[~local], minlength=n)
        transfer_fraction = np.divide(np.minimum(transferred, np.maximum(spare, 0.)), transferred,
                                      out=np.zeros(n), where=transferred > 0)
        done[~local] = flow[~local] * transfer_fraction[dst[~local]]
        return own_done, done
    if rule != 'given_order':
        raise ValueError('Unknown batch completion rule')
    ranks = np.asarray(source_rank)
    if (ranks.shape != (n,) or not np.issubdtype(ranks.dtype, np.integer)
            or set(ranks.tolist()) != set(range(n))):
        raise ValueError('given_order needs one distinct source rank per node')
    amount = np.r_[own, flow]
    origin = np.r_[np.arange(n), src]
    receiver = np.r_[np.arange(n), dst]
    # Own admission precedes any self arc when source/receiver are identical.
    order = np.lexsort((np.arange(len(amount)), ranks[origin], receiver))
    ordered_receiver = receiver[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_receiver)) + 1]
    ends = np.r_[starts[1:], len(order)]
    result = np.zeros_like(amount)
    for start, end in zip(starts, ends):
        indices = order[start:end]
        batch = amount[indices]
        before = np.r_[0., np.cumsum(batch[:-1])]
        result[indices] = np.minimum(batch, np.maximum(capacity[ordered_receiver[start]] - before, 0.))
    return result[:n], result[n:]
