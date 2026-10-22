"""Analytic completion tests, default regression, and bounded case comparison."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import scipy
from model_config import ModelConfig
from completion_rules import complete_batch
from configured_dual_mechanism_v5 import ConfiguredDispatchAllocator, ConfiguredDualMechanismRecovery
from dual_mechanism_recovery_v5 import DispatchAllocatorV5, DualMechanismRecoveryV5

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '项目结果/configured_completion_rules_20261019'
CODE = Path(__file__).parent
NEW = ['model_config.py', 'completion_rules.py', 'configured_dual_mechanism_v5.py', Path(__file__).name]
CHECKS = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(obj, name):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def check(name, passed):
    CHECKS.append(dict(check=name, passed=bool(passed)))
    if not passed:
        save(CHECKS, 'validation_failed.csv')
        raise AssertionError(name)


def rejects(name, fn):
    try:
        fn()
    except (ValueError, TypeError, FrozenInstanceError):
        check(name, True)
    else:
        check(name, False)


def signature(engine):
    return [(r['round'], r['failed_count'], r['new_failures'], r['overload_only'],
             r['risk_only'], r['both'], sorted(r['new_ids'])) for r in engine.cascade_logs[engine.initial]]


def analytical(n, e):
    c = ModelConfig(theta_risk=.2, theta_source='analytic toy assumption')
    check('configuration audit covers all values', len(c.audit_rows()) == len(c.snapshot()['values']))
    check('config JSON roundtrip', ModelConfig.from_snapshot(c.snapshot()) == c)
    rejects('frozen config', lambda: setattr(c, 'rho', 2.))
    for label, updates in [('rho', dict(rho=.99)), ('gamma', dict(gamma=.99)),
                           ('nan', dict(theta_risk=float('nan'))), ('weight sum', dict(risk_weights=(1,0,0,0,0))),
                           ('rule', dict(completion_rule='fifo')), ('missing order', dict(completion_rule='given_order')),
                           ('hidden unused order', dict(source_order=('a',))), ('normalizer', dict(pressure_normalizer=0)),
                           ('hop bool', dict(max_hops=True)), ('mechanism bool', dict(risk_enabled=1))]:
        rejects('invalid config ' + label, lambda u=updates: replace(c, **u))
    altered = json.loads(c.canonical_json())
    altered['parameter_audit'][0]['empirically_calibrated'] = True
    rejects('tampered audit metadata rejected', lambda: ModelConfig.from_snapshot(altered))
    own, src, dst, f, K = np.array([0.,8.]), np.array([0]), np.array([1]), np.array([3.]), np.array([10.,10.])
    toy_records = []
    expected = {'proportional': (80/11,30/11), 'local_first': (8.,2.), 'given_order': (7.,3.)}
    for rule in expected:
        a, b = complete_batch(own, src, dst, f, K, rule=rule, source_rank=np.array([0,1]))
        check('hand calculated 8+3>10 ' + rule, np.allclose([a[1],b[0]],expected[rule],atol=1e-12,rtol=0))
        toy_records.append(dict(rule=rule, local_completed=float(a[1]), imported_completed=float(b[0]), total=float(a.sum()+b.sum())))
    save(toy_records, '01_analytic_completion_example.csv')
    # Independent batch totals, partial final item, zero arcs, self arc and label invariance.
    a, b = complete_batch([8.,0.], np.array([0,1]), np.array([0,0]), [4.,3.], [10.,10.], rule='local_first')
    check('self admission remains local', np.allclose(a,[8,0]) and np.allclose(b,[2,0]))
    for rule in ['proportional', 'local_first', 'given_order']:
        a, b = complete_batch([0.,3.], np.array([],int), np.array([],int), [], [10.,10.], rule=rule, source_rank=np.array([1,0]))
        check('empty transfer and underload ' + rule, np.array_equal(a,[0,3]) and len(b)==0)
    rng = np.random.default_rng(2026101904)
    for ix in range(30):
        count = 6
        i, j = np.nonzero(rng.random((count,count)) < .35)
        amount = rng.uniform(0,15,len(i)); own = rng.uniform(0,20,count); caps = rng.uniform(0,20,count)
        ranks = rng.permutation(count)
        for rule in ['proportional','local_first','given_order']:
            a, b = complete_batch(own,i,j,amount,caps,rule=rule,source_rank=ranks)
            receiver = a + np.bincount(j,weights=b,minlength=count)
            target = np.minimum(caps,own+np.bincount(j,weights=amount,minlength=count))
            check('batch receiver conservation ' + str((ix,rule)), np.allclose(receiver,target,atol=1e-10,rtol=0)
                  and (a>=0).all() and (a<=own+1e-10).all() and (b>=0).all() and (b<=amount+1e-10).all())
            perm = rng.permutation(count); inverse = np.argsort(perm)
            aa, bb = complete_batch(own[perm],inverse[i],inverse[j],amount,caps[perm],rule=rule,source_rank=ranks[perm])
            check('node relabel invariance '+str((ix,rule)),np.allclose(aa[inverse],a,atol=1e-10,rtol=0)
                  and np.allclose(bb,b,atol=1e-10,rtol=0))
    rejects('negative batch rejected', lambda: complete_batch([0.],np.array([0]),np.array([0]),[-1.],[1.]))
    rejects('duplicate source rank rejected', lambda: complete_batch([0.,0.],np.array([],int),np.array([],int),[],[1.,1.],rule='given_order',source_rank=[0,0]))
    nn=pd.concat([n.iloc[[0]].copy() for _ in range(2)],ignore_index=True)
    nn['node_id']=['S','R'];nn['名称']=['S','R'];nn['family']='test';nn['区县']='test'
    nn['proxy_daily_load']=[3.,8.];nn['overload_threshold']=[10.,10.]
    ee=e.iloc[[0]].copy();ee['Source_id']='S';ee['Target_id']='R';ee['Composite_transport_cost']=1.
    ee['Failure_probability_proxy']=0.;ee['Cross_district']=0.;ee['Edge_capacity_proxy']=55.;ee['Repair_cost_proxy']=1.
    config=ModelConfig(theta_risk=2.,risk_enabled=False,overload_enabled=False,theta_source='analytic toy')
    shared, pressure_shared = {}, {}
    engines=[]
    for rule in ['proportional','local_first','given_order']:
        cc=replace(config,completion_rule=rule,source_order=('S','R') if rule=='given_order' else ())
        aa=ConfiguredDispatchAllocator(nn,ee,cc)
        en=ConfiguredDualMechanismRecovery(aa,ee,[[3.,8.]],['toy'],['S'],'random')
        en.cache=shared;en.pressure_cache=pressure_shared;engines.append(en)
        m,_,sv=en.state({'S'});en.pressure({'S'})
        target=expected[rule]
        check('solver completion known outcome '+rule,np.allclose(sv,[target[1],target[0]],atol=1e-10,rtol=0))
    check('shared state cache isolated',len(shared)==3 and len(pressure_shared)==3)
    for en in engines:
        check('shared cache repeat not contaminated '+en.config.completion_rule,
              np.array_equal(en.state({'S'})[2],en.a.evaluate({'S'},en.demand)[2]))
    rejects('allocator order validates IDs',lambda:ConfiguredDispatchAllocator(nn,ee,replace(config,completion_rule='given_order',source_order=('S','X'))))
    rejects('recovery config mismatch',lambda:ConfiguredDualMechanismRecovery(engines[0].a,ee,[[3.,8.]],['toy'],['S'],'random',replace(config,gamma=1.2)))
    # Config values really drive pressure, labels, denominator, and shutdown.
    custom=replace(config,theta_risk=.5,risk_enabled=True,overload_enabled=True,
                   risk_weights=(1.,0.,0.,0.,0.),pressure_normalizer=1.,bonus_amplitude=.3,
                   spatial_districts=('test',),risk_cost_scale=.7,risk_capacity_scale=25.)
    en=ConfiguredDualMechanismRecovery(ConfiguredDispatchAllocator(nn,ee,custom),ee,[[3.,8.]],['toy'],['S'],'spatial_cluster')
    check('custom weights normalizer and bonus applied',np.allclose(en.pressure({'S'})[1],[.3,1.3],atol=1e-12,rtol=0))
    check('mechanism configuration applied',en.classify({'S'},[0,1.1])['both']==frozenset(['R']))
    high=replace(custom,gamma=1.2,bonus_amplitude=0.,theta_risk=2.)
    en2=ConfiguredDualMechanismRecovery(ConfiguredDispatchAllocator(nn,ee,high),ee,[[3.,8.]],['toy'],['S'],'spatial_cluster')
    check('changed gamma and theta applied',not en2.classify({'S'},[0,1.1])['new'])
    cost_cfg=replace(config,risk_enabled=True,risk_cost_scale=2.,risk_capacity_scale=110.,risk_cost_floor=1.5)
    en3=ConfiguredDualMechanismRecovery(ConfiguredDispatchAllocator(nn,ee,cost_cfg),ee,[[3.,8.]],['toy'],['S'],'random')
    expected_pressure=(.21+.10*(1.5/2)+.09*(1-55/110))/.68
    check('score scaling and cost floor applied',abs(en3.pressure({'S'})[1][1]-expected_pressure)<1e-12)


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    protected=list((ROOT/'数据及代码/data').glob('*.csv'))+[p for p in CODE.glob('*.py') if p.name not in NEW]
    protected += [ROOT.parent/'中文论文初稿_全图美化与正文优化V5_20261019.docx']
    prior=ROOT/'项目结果/dispatch_capacity_null_mainline_20261018'
    protected += [prior/'84a_capacity_policy_results.csv',prior/'86_design_v3.json',
                  ROOT/'项目结果/capacity_heterogeneity_design_20261018/82a_assumed_capacity_scenarios.csv']
    before={str(p):sha(p) for p in protected};new_before={name:sha(CODE/name) for name in NEW}
    design_hash=sha(OUT/'00_design.md')
    dump(dict(protected=before,new_sources=new_before,design_sha256=design_hash,
              runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__)),
         'source_manifest.json')
    core=importlib.import_module('83_dispatch_mainline_core')
    n,e,demands,_=core.load_study();analytical(n,e)
    print('analytic/configuration checks passed',len(CHECKS),flush=True)
    d=json.loads((prior/'86_design_v3.json').read_text(encoding='utf-8'))
    thresholds=[d['thresholds'][0],d['thresholds'][-1]]
    old=pd.read_csv(prior/'84a_capacity_policy_results.csv')
    ids=tuple(n.node_id);rules=['proportional','local_first','given_ascending','given_descending']
    allrows=[];deltas=[];policy=[];roundrows=[];state_cache={};pressure_cache={}
    for cap_id in ['baseline','node_a0.20_s00','family_a0.20_s00']:
        nodes=core.capacity_nodes(n,cap_id)
        for case,theta,demand_name in itertools.product(d['cases'],thresholds,d['demands']):
            parts,groups=demands[demand_name];D=parts.sum(axis=0);bundle={}
            tag=dict(capacity_id=cap_id,case_id=case['case_id'],theta=theta,demand=demand_name)
            for arm,rule_label in itertools.product(['A','B','C'],rules):
                rule='given_order' if rule_label.startswith('given_') else rule_label
                order=ids if rule_label=='given_ascending' else tuple(reversed(ids)) if rule_label=='given_descending' else ()
                cfg=ModelConfig.for_arm(arm,theta_risk=theta,completion_rule=rule,source_order=order,
                                       theta_source='86_design_v3 frozen proxy design; not empirical')
                cfg.write_snapshot(OUT/'configs'/f'{cfg.fingerprint}.json')
                alloc=ConfiguredDispatchAllocator(nodes,e,cfg)
                engine=ConfiguredDualMechanismRecovery(alloc,e,parts,groups,case['initial_ids'],case['kind'])
                engine.cache=state_cache;engine.pressure_cache=pressure_cache
                failed,metric,_=engine.closure(engine.initial)
                initial=alloc.evaluate(engine.initial,D,details=True)
                final=alloc.evaluate(failed,D,details=True)
                for label,result in [('initial',initial),('final',final)]:
                    mm,util,sv,extra=result
                    check('mass/capacity '+str((len(allrows),label)),abs(mm['Service_Conservation_Error'])<1e-6
                          and mm['Max_LP_Violation']<1e-5 and abs(sv.sum()-mm['Served_Load'])<1e-6
                          and np.max(extra['completed_receiver']-alloc.nominal_capacity)<1e-6
                          and np.max(sv-D)<1e-6)
                def city_rates(served):
                    fraction=np.divide(served,D,out=np.zeros_like(D),where=D>0)
                    total=parts.sum(axis=1)
                    return np.divide(parts@fraction,total,out=np.ones_like(total),where=total>0)
                rr=dict(engine=engine,failed=failed,metric=metric,initial=initial,final=final,
                        initial_city=city_rates(initial[2]),final_city=city_rates(final[2]))
                bundle[arm,rule_label]=rr
                if rule_label=='proportional':
                    legacy_alloc=DispatchAllocatorV5(nodes,e,cfg.rho,cfg.dispatch_policy,cfg.max_hops)
                    legacy=DualMechanismRecoveryV5(legacy_alloc,e,parts,groups,case['initial_ids'],case['kind'],theta)
                    lf,lm,_=legacy.closure(legacy.initial)
                    check('default exact legacy metrics '+str(len(allrows)),metric==lm and failed==lf
                          and signature(engine)==signature(legacy)
                          and np.array_equal(engine.state(failed)[2],legacy.state(lf)[2]))
                    li=legacy_alloc.evaluate(legacy.initial,D,details=True)
                    check('default exact allocation details '+str(len(allrows)), initial[0]==li[0]
                          and np.array_equal(initial[1],li[1]) and np.array_equal(initial[2],li[2])
                          and initial[3]['arcs'].equals(li[3]['arcs']))
                    ref=old[(old.capacity_id==cap_id)&(old.case_id==case['case_id'])&(old.arm==arm)
                            &(old.demand==demand_name)&np.isclose(old.theta,theta,atol=1e-12,rtol=0)]
                    check('one historical table target '+str(len(allrows)),len(ref)==1)
                    target=ref.iloc[0]
                    check('historical 84 result reproduced '+str(len(allrows)),abs(metric['Service_Rate']-target.Service_Rate)<1e-9
                          and {alloc.pos[k] for k in failed}==set(json.loads(target.final_failed_indices_json)))
                    legacy_alloc.paths.cache_clear()
                else:
                    ref=bundle[arm,'proportional']
                    check('rule total and failure invariance '+str(len(allrows)),failed==ref['failed']
                          and signature(engine)==signature(ref['engine'])
                          and abs(metric['Served_Load']-ref['metric']['Served_Load'])<1e-6)
                    for label in ['initial','final']:
                        current=rr[label];base=ref[label]
                        check('same dispatch and utilization '+str((len(allrows),label)),
                              np.array_equal(current[1],base[1])
                              and np.array_equal(current[3]['load'],base[3]['load'])
                              and current[3]['arcs'][['source_id','target_id','flow','cost']].equals(
                                  base[3]['arcs'][['source_id','target_id','flow','cost']]))
                        check('same receiver completion '+str((len(allrows),label)),
                              np.allclose(current[3]['completed_receiver'],base[3]['completed_receiver'],atol=1e-6,rtol=0))
                        delta=current[2]-base[2]
                        deltas.append(dict(**tag,arm=arm,rule=rule_label,state=label,
                                           total_gain=float(delta.sum()),changed_origins=int((abs(delta)>1e-7).sum()),
                                           max_origin_change=float(abs(delta).max()),
                                           max_city_rate_change_pp=float(abs(rr[label+'_city']-ref[label+'_city']).max()*100),
                                           transferred_load_change=float(current[0]['Transferred_Load']-base[0]['Transferred_Load']),
                                           completed_transfer_cost_change=float(current[0]['Completed_Transfer_Cost']-base[0]['Completed_Transfer_Cost'])))
                allrows.append(dict(**tag,arm=arm,rule=rule_label,config_id=cfg.fingerprint,
                                    cache_namespace=engine.cache_namespace,service=metric['Service_Rate'],
                                    served_load=metric['Served_Load'],final_failed=len(failed),
                                    final_ids_json=json.dumps(sorted(failed),ensure_ascii=False),
                                    initial_source_completed_json=json.dumps(initial[2].tolist()),
                                    final_source_completed_json=json.dumps(final[2].tolist()),
                                    initial_city_rates_json=json.dumps(rr['initial_city'].tolist()),
                                    final_city_rates_json=json.dumps(rr['final_city'].tolist()),
                                    completed_transfer_cost=metric['Completed_Transfer_Cost']))
                for row in engine.cascade_logs[engine.initial]:
                    roundrows.append(dict(**tag,arm=arm,rule=rule_label,round=row['round'],failed=row['failed_count'],
                                          new_failures=row['new_failures'],new_ids_json=json.dumps(sorted(row['new_ids']),ensure_ascii=False),
                                          overload_only=row['overload_only'],risk_only=row['risk_only'],both=row['both'],
                                          served_load=row['Served_Load'],service=row['Service_Rate']))
                alloc.paths.cache_clear()
            for rule in rules:
                C,B=bundle['C',rule],bundle['B',rule]
                gap=C['metric']['Service_Rate']-B['metric']['Service_Rate']
                check('C>=B total '+str((len(allrows),rule)),gap>=-1e-9 and C['failed']<=B['failed'])
                policy.append(dict(**tag,rule=rule,C_minus_B_pp=100*gap,
                                   origins_C_below_B=int((C['final'][2]<B['final'][2]-1e-7).sum()),
                                   cities_C_below_B=int((C['final_city']<B['final_city']-1e-9).sum())))
            # Limit memory; sharing across all rules has already been exercised.
            state_cache.clear();pressure_cache.clear()
        print('completed configured case grid',cap_id,len(allrows),flush=True)
    check('576 closures and 192 C/B pairs',len(allrows)==576 and len(policy)==192)
    save(allrows,'02_configured_runs.csv');save(deltas,'03_rule_attribution_differences.csv')
    save(policy,'04_C_B_comparisons.csv');save(roundrows,'05_round_trajectories.csv')
    dump(dict(node_order=ids,group_order=list(groups),rules=rules,
              source_order_semantics='Specified source IDs ascending/reversed; not observed arrivals'), '06_profile_metadata.json')
    for path,h in before.items():check('original preserved '+path,sha(path)==h)
    for name,h in new_before.items():check('new source stable '+name,sha(CODE/name)==h)
    check('design stable',sha(OUT/'00_design.md')==design_hash)
    diff=pd.DataFrame(deltas);pol=pd.DataFrame(policy)
    result=dict(completed=True,closures=len(allrows),default_legacy_pairs=144,C_B_pairs=len(policy),
                all_total_and_failure_invariants_passed=True,
                attribution_state_comparisons=len(diff),
                comparisons_with_origin_changes=int((diff.changed_origins>0).sum()),
                max_origin_completed_change=float(diff.max_origin_change.max()),
                max_city_rate_change_pp=float(diff.max_city_rate_change_pp.max()),
                C_B_conditions_with_some_origin_C_lower=int((pol.origins_C_below_B>0).sum()),
                checks=len(CHECKS),passed=sum(r['passed'] for r in CHECKS),
                old_sources_and_data_changed=False,actual_fifo_implemented=False,
                elapsed_seconds=time.perf_counter()-started)
    dump(result,'07_findings.json');save(CHECKS,'08_validation.csv')
    body=f'''# 配置对象与完成规则实现结果

2026-10-19。新入口configured_dual_mechanism_v5.py，历史V5入口及旧脚本未改。完成的是配置审计和完成份额稳健性检查，不称新算法。

## 实现

ModelConfig不可变；每个参数含值、单位、来源、证据类别和未实测标定标记。明确集中rho/gamma/theta、五权重、bonus及标签、归一分母、风险截断尺度、epsilon、跳数、机制开关、分配目标和完成规则。配置支持严格校验、JSON快照往返及SHA256标识；未提供theta来源时明确保留为未指定假设，不升级为已标定。

比例规则保持历史默认及原浮点路径。本地优先先完成本地货（包括自环），外来货用剩余能力按比例完成。给定顺序规则要求完整来源ID顺序，所有本批货已就绪；不是真实FIFO，无到达时刻、等待或跨期库存。非比例详情返回逐弧completed_flow和receiver_completion_fraction，不能将一个接收点比例当所有来源的共同完成乘数。

新入口复用旧派入LP而非修改旧物理代码。非比例规则在同一已派入批次上重新分配完成份额，更新来源完成、未完成、已完成转移量和成本。状态/压力缓存含配置与实际数据签名，即便外部共享字典也隔离；缺顺序、未知ID、参数不合法、配置不匹配均拒绝。

## 验证结果

- 576次闭包：三容量×四既定test损伤×两阈值×两需求×A/B/C×四规则（比例、本地优先、给定升/降序）。给定序列为合成对照，不是观测FIFO。
- 144个默认比例条件逐指标、逐来源、逐轮失效轨迹与原V5精确一致，并与旧84表服务/失效集合一致。
- 192个C/B匹配全部满足全网总服务C≥B；各完成规则下派入负荷、利用率、接收点总完成及完整失效轨迹保持一致。
- {len(diff)}个“非默认规则×条件×初始/终态”比较中，{result['comparisons_with_origin_changes']}个存在来源完成量变化；最大单来源完成量变化{result['max_origin_completed_change']:.6f}代理单位，最大城市代理归属服务率变化{result['max_city_rate_change_pp']:.6f}个百分点。完整正负变化见03，不仅报告最大值。
- 终态C/B比较中，{result['C_B_conditions_with_some_origin_C_lower']}格出现某个来源C完成量低于B；这不否定总量命题，不能将总量优势误写成每来源优势。城市来源归属仍按parts代理分摊，不是真实货物流。
- 手算小例、随机批次守恒、节点重排、无效配置、快照、防缓存串用和自定义风险参数生效均通过。校验项是实现检查，不是独立现实样本。

## 可以写进论文的边界

在固定需求、派入与能力，且完成工作保持、流量可拆分、无跨期结转的条件下，各接收点总完成为min(L,K)。当前过载只看派入利用率，风险只看失效集合，因此改变完成份额不会改变总完成或失效闭包。可以用这一条件性说明和实现核对增强原结论的透明性；不能把它包装为此前未知的新机制，也不意味着到达时刻、非等处理时间或动态排队下仍成立。

比例完成不是C≥B总量结论的必要细节，但会影响服务分配。当前尚未实测标定的容量、负荷、风险以及设施相容性问题仍存在。

## 使用

见Code/README_配置与完成规则_20261019.md。历史入口不会自动使用ModelConfig；新增研究从ConfiguredDispatchAllocator/ConfiguredDualMechanismRecovery进入。每次实验保存config_id、配置JSON和输入哈希。00为边界，01手算例，02运行/来源画像，03归属变化，04政策差，05轨迹，06画像顺序，07结论，08校验，configs/配置快照，completion为源码和输出哈希。原数据、旧源码、历史输出和Word正文均未修改。
'''
    (OUT/'配置与完成规则_实现和验证结果_20261019.md').write_text(body,encoding='utf-8')
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json']
    dump(dict(completed=True,new_sources={name:sha(CODE/name) for name in NEW},
              files={str(p.relative_to(OUT)):sha(p) for p in files},findings=result),'completion.json')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
