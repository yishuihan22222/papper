"""Read-only adapters for V5 stopping diagnostics and frozen-resource sweeps."""
import json
import numpy as np
from closed_loop_recovery_v3 import RecoveryPlanner


def load_priority(planner):
    loads=planner.engines[0].a.nodes.set_index('node_id').proxy_daily_load
    seq=[]
    for node in sorted(planner.pool,key=lambda k:(-float(loads.loc[k]),k)):
        if planner.affordable(seq+[node]):seq.append(node)
    return seq


def stopping_audit(planner,sequence):
    """Evaluate every remaining candidate at the actual stopping prefix."""
    seq=list(sequence);current=planner.score(seq);rows=[]
    before=[e.trajectory(seq)[0] for e in planner.engines]
    remaining=planner.budget+current[1]
    for node in planner.pool:
        if node in seq:continue
        cost=float(planner.cost[planner.pos[node]])
        row=dict(node_id=node,cost=cost,affordable=planner.affordable(seq+[node]))
        if row['affordable']:
            score=planner.score(seq+[node]);same=[];refail=[];deltas=[]
            for j,e in enumerate(planner.engines):
                after,trace,_=e.trajectory(seq+[node]);same.append(after==before[j])
                refail.append(node in after)
                deltas.append(trace[-1]['Service_Rate']-trace[-2]['Service_Rate'])
                row[f'training{j}_same_failed_set']=same[-1]
                row[f'training{j}_refailed']=refail[-1]
                row[f'training{j}_service_change_pp']=100*deltas[-1]
                row[f'training{j}_new_failures']=trace[-1]['new_failures']
            row.update(marginal_auc=score[0]-current[0],all_training_same_state=all(same),
                       all_training_refailed=all(refail),max_service_change_pp=100*max(deltas))
        rows.append(row)
    feasible=[r for r in rows if r['affordable']]
    if len(seq)>=planner.horizon:reason='horizon_reached'
    elif not rows:reason='no_untried_nodes'
    elif not feasible:reason='budget_exhausted'
    elif max(r['marginal_auc'] for r in feasible)<=1e-12:reason='no_positive_marginal_gain'
    else:reason='positive_candidate_remaining'
    summary=dict(stop_reason=reason,actions=len(seq),spent=-current[1],remaining_budget=remaining,
                 untried=len(rows),affordable_candidates=len(feasible),
                 best_marginal_auc=max((r['marginal_auc'] for r in feasible),default=np.nan),
                 positive_candidates=sum(r['marginal_auc']>1e-12 for r in feasible),
                 same_state_candidates=sum(r['all_training_same_state'] for r in feasible),
                 refailed_in_all_training=sum(r['all_training_refailed'] for r in feasible),
                 absorbing_under_affordable_single_repairs=bool(feasible) and all(r['all_training_same_state'] for r in feasible))
    return summary,rows


def selected_pool(planner,sequence,limit=12):
    """Outcome-independent bounded pool: six largest loads + six cheapest."""
    eligible=[k for k in planner.pool if k not in sequence and planner.affordable(list(sequence)+[k])]
    loads=planner.engines[0].a.nodes.set_index('node_id').proxy_daily_load
    by_load=sorted(eligible,key=lambda k:(-float(loads.loc[k]),k))
    by_cost=sorted(eligible,key=lambda k:(float(planner.cost[planner.pos[k]]),k))
    chosen=list(dict.fromkeys(by_load[:limit//2]+by_cost[:limit//2]))
    for node in by_load:
        if len(chosen)>=limit:break
        if node not in chosen:chosen.append(node)
    return sorted(chosen)


def pair_diagnostics(planner,sequence,absorbing,limit=12):
    """Sequential pairs use existing dynamics; batches are separate counterfactuals.

    If all affordable singles return the identical stable failed set in every
    training demand, induction rules out every sequential extension, not just
    the selected pairs. A simultaneous batch changes action timing and is not
    a feasible policy of the original one-node/closure-per-step model.
    """
    seq=list(sequence);pool=selected_pool(planner,seq,limit);current=planner.score(seq)
    records=[]
    if planner.horizon-len(seq)<2:return pool,records
    before=[e.trajectory(seq)[0] for e in planner.engines]
    base=[e.trajectory(seq)[1][-1]['Service_Rate'] for e in planner.engines]
    for i,a in enumerate(pool):
        for b in pool[i+1:]:
            if not planner.affordable(seq+[a,b]):continue
            sequential=[]
            if not absorbing:
                for order in [(a,b),(b,a)]:
                    sequential.append(dict(sequence_json=json.dumps(seq+list(order),ensure_ascii=False),
                                           objective_change_pp=100*(planner.score(seq+list(order))[0]-current[0])))
            # Batch: remove both failed IDs before closure. Report stable service
            # only, not a fictitious fair-comparison AUC for an unmodelled duration.
            service=[];retained=[]
            for j,e in enumerate(planner.engines):
                failed,metric,_=e.closure(before[j]-{a,b})
                service.append(metric['Service_Rate']);retained.append(int(a not in failed)+int(b not in failed))
            records.append(dict(node_a=a,node_b=b,additional_cost=float(planner.cost[planner.pos[a]]+planner.cost[planner.pos[b]]),
                                sequential_analytically_no_gain=bool(absorbing),
                                sequential_best_auc_change_pp=0. if absorbing else max(r['objective_change_pp'] for r in sequential),
                                sequential_orders_json=json.dumps(sequential,ensure_ascii=False),
                                batch_mean_service_change_pp=100*float(np.mean(np.array(service)-base)),
                                batch_min_service_change_pp=100*float(np.min(np.array(service)-base)),
                                batch_training0_service=service[0],batch_training1_service=service[1],
                                batch_training0_retained=retained[0],batch_training1_retained=retained[1],
                                batch_changes_original_action_timing=True))
    return pool,records
