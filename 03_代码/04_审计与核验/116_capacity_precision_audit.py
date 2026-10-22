"""Frozen comparison of proxy capacity precision; production inputs stay intact."""
import argparse
import importlib
import itertools
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import scipy
from capacity_precision_v5 import formula_precision_nodes
from configured_dual_mechanism_v5 import ConfiguredDispatchAllocator,ConfiguredDualMechanismRecovery
from model_config import ModelConfig

core=importlib.import_module('83_dispatch_mainline_core')
CODE=Path(__file__).parent;ROOT=CODE.parents[1]
OUT=ROOT/'项目结果/capacity_precision_audit_20261020'
GAMMA=ROOT/'项目结果/gamma_demand_boundary_20261020'
PAPER=ROOT/'项目结果/dispatch_capacity_null_mainline_20261018'


def dump(v,p):core.dump(v,p)
def sha(p):return core.sha(p)
def save(rows,name):core.save(rows,OUT/name)


def verify(d):
    lock=json.loads((OUT/'design_lock.json').read_text(encoding='utf-8'))
    assert sha(OUT/'00_design.json')==lock['design']
    assert sha(OUT/'00_design.md')==lock['note']
    for p,h in d['protected'].items():
        if sha(p)!=h:raise RuntimeError('Changed protected input '+p)
    for p,h in d['frozen'].items():
        if sha(OUT/p)!=h:raise RuntimeError('Changed frozen input '+p)


def key(cap,case,theta,gamma,demand,arm):
    return '|'.join([cap,case,format(float(theta),'.12g'),format(float(gamma),'.12g'),demand,arm])


def design():
    if (OUT/'00_design.json').exists():raise FileExistsError('Design already frozen')
    n,e,demands,_=core.load_study()
    g=json.loads((GAMMA/'00_design.json').read_text(encoding='utf-8'))
    p=json.loads((PAPER/'84_design.json').read_text(encoding='utf-8'))
    # Verify the previous packet before using any old result as a comparator.
    previous=json.loads((GAMMA/'delivery_manifest.json').read_text(encoding='utf-8'))
    for f,h in previous['files'].items():assert sha(GAMMA/f)==h,f
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in ['frozen_inputs','shards','configs']:(OUT/folder).mkdir(exist_ok=True)
    n.to_csv(OUT/'frozen_inputs/baseline_nodes.csv',index=False,encoding='utf-8-sig')
    e.to_csv(OUT/'frozen_inputs/edges.csv',index=False,encoding='utf-8-sig')
    payload={};demands=dict(demands)
    for mult in g['external_multipliers']:
        part,names=demands['nominal'];part=np.asarray(part).copy();part[1:]*=mult
        demands[f'external_x{mult:g}']=(part,names)
    for name,(parts,groups) in demands.items():payload[name]=dict(parts=np.asarray(parts).tolist(),groups=list(groups))
    dump(payload,OUT/'frozen_inputs/demands.json')
    capacities=[]
    for cap in p['capacity_ids']:
        nn=core.capacity_nodes(n,cap);corrected,audit=formula_precision_nodes(nn,n)
        nn.to_csv(OUT/f'frozen_inputs/{cap}_stored.csv',index=False,encoding='utf-8-sig')
        corrected.to_csv(OUT/f'frozen_inputs/{cap}_formula.csv',index=False,encoding='utf-8-sig')
        audit['capacity_id']=cap;capacities.extend(audit.to_dict('records'))
    save(capacities,'01_capacity_construction.csv')
    jobs=[];lookup={}
    for theta,gamma,mult,case,arm in itertools.product(g['thresholds'],g['gamma'],g['external_multipliers'],g['cases'],['B','C']):
        demand=f'external_x{mult:g}';identity=key('baseline',case['case_id'],theta,gamma,demand,arm)
        job=dict(job_id=len(jobs),job_key=identity,capacity_id='baseline',case_id=case['case_id'],kind=case['kind'],
                 initial_ids=case['initial_ids'],control=case['control'],theta=theta,gamma=gamma,demand=demand,arm=arm,
                 external_multiplier=mult,old_scope='gamma',paper_member=False,boundary_member=True)
        jobs.append(job);lookup[identity]=job
    aliases={'nominal':'external_x1','external_plus20':'external_x1.2','external_plus150':'external_x2.5'}
    paper_refs=[]
    for cap,case,theta,name,arm in itertools.product(p['capacity_ids'],p['cases'],p['thresholds'],p['demand_names'],['B','C']):
        demand=aliases.get(name,name);identity=key(cap,case['case_id'],theta,1.,demand,arm)
        if identity not in lookup:
            job=dict(job_id=len(jobs),job_key=identity,capacity_id=cap,case_id=case['case_id'],kind=case['kind'],
                 initial_ids=case['initial_ids'],control=False,theta=theta,gamma=1.,demand=demand,arm=arm,
                 external_multiplier={'nominal':1.,'external_plus20':1.2,'external_plus150':2.5}.get(name),
                 old_scope='paper',paper_member=True,boundary_member=False)
            jobs.append(job);lookup[identity]=job
        lookup[identity]['paper_member']=True
        paper_refs.append(dict(job_id=lookup[identity]['job_id'],capacity_id=cap,case_id=case['case_id'],theta=theta,demand=name,arm=arm))
    assert len(jobs)==1920 and len(paper_refs)==1152
    cold_keys=[]
    for cap,arm in itertools.product(p['capacity_ids'],['B','C']):
        cold_keys.append(key(cap,'train_legacy_spatial_cluster',p['thresholds'][0],1.,'external_x1',arm))
        cold_keys.append(key(cap,'test_channel_00',.2,1.,'external_x2.5',arm))
    for arm in ['B','C']:cold_keys.append(key('baseline','no_damage_random',.42,1.,'external_x2',arm))
    for theta,gamma in itertools.product([.2,.42],[1.,1.025]):
        cold_keys.append(key('baseline','test_random_01',theta,gamma,'external_x1.2','B'))
    cold_ids=[lookup[k]['job_id'] for k in cold_keys]
    assert len(set(cold_ids))==18
    protected=list(CODE.glob('*.py'))+list((CODE.parent/'data').glob('*.csv'))
    protected += [GAMMA/'delivery_manifest.json',GAMMA/'01_runs.csv',GAMMA/'02_rounds.csv',PAPER/'84_design.json',
                  PAPER/'84a_capacity_policy_results.csv',PAPER/'84b_cascade_rounds.csv',
                  ROOT/'项目结果/capacity_heterogeneity_design_20261018/82a_assumed_capacity_scenarios.csv',
                  ROOT.parent/'中文论文初稿_全图美化与正文优化V5_20261019.docx',
                  ROOT.parent/'论文初稿配套图表_V5_20261019/初稿源文件/manuscript.md']
    d=dict(date='2026-10-20',schema='capacity-precision-1',jobs=jobs,paper_refs=paper_refs,
       unique_new_closures=1920,boundary_comparisons=840,paper_comparisons=1152,overlap=72,
       cold_job_ids=cold_ids,cold_solves=36,
       capacity_rule='Knew_s = 1.25*stored_L + (Kstored_s - Kstored_baseline)',
       provenance='04_generate_quantified_datasets.py line 144; 82_build_fixed_total_capacity_scenarios.py base+contrast',
       fixed=['existing rounded L and capacity_index','demands and mapping','all edge risk/cost attributes',
              'gamma/rho/overload_epsilon','source-completion convention','LP objective and ordering'],
       purpose='separate second rounding of K from model overload; no new physical tolerance or calibration',
       stop=['No threshold/epsilon adjustment from results','No new gamma/shock/seed or capacity distribution search',
             'Do not promote new capacity as observed truth','Do not label different outcomes numerical solver errors without evidence'],
       tests=['capacity contrasts preserved and three totals equal',
              'stored mode exactly matches old entry and selected original traces',
              'independent pressure/trigger/receiver reconstruction',
              'predeclared cold starts cover default boundary, high demand, and each capacity',
              'paper and boundary old/new outcomes both retained'],
       limitations=['prior explored conditions; no new empirical samples','only K second rounding removed',
                    'label near rounding resolution is at a fixed LP solution, not robust uncertainty certification',
                    'immediate irreversible overload remains an uncalibrated hypothesis'],
       protected={str(f):sha(f) for f in protected},
       frozen={str(f.relative_to(OUT)):sha(f) for f in (OUT/'frozen_inputs').iterdir()},
       runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__))
    dump(d,OUT/'00_design.json')
    note='''# 容量构造精度版本审计（新运行前冻结）

2026-10-20。追溯04生成器先将L定义为两位小数代理，再把1.25L二次舍入成K；82中的异质容量为同一舍入基准加固定设计对比。
本轮只在新版本副本中用Knew_s=1.25×既有L+(Kstored_s−Kstored_baseline)，保留异质对比及各版本相同总量。
既有L、capacity_index、需求映射与边风险不重算；不更改gamma、rho或数值容差，不替换实测容量，不覆盖旧模型。

复算上一轮840条件及原论文三容量×16损伤×四需求×三阈值×B/C的1152条件，72条件重合，合计1920次新闭包。
原版比较由已验证的旧表提供；另抽取旧版和新版条件冷启动。没有新增损伤、需求、gamma或新容量分布，不以出现更好优势为验收。
比较C-B总体与分需求幅度、终态失效集合及首轮/全程变化。即使总能力仅改0.0225，也按输入版本变化报告，不称原模型所有结果均为舍入造成。

逐个过载事件标注固定LP解下的阈值超额及其与gamma×0.005输入舍入量的关系。这个标记不是改变判据，也不是对容量不确定性集合的稳健证书，因为容量改变后LP可以重分配。
区分数据构造精度、LP数值容差与物理退出阈值；没有业务标定前不增加“合理容忍区”来抹掉退出。

停止线：只完成冻结对照和完整性检查，不追加参数搜索；若校验失败先定位；业务相容性仍缺失，本轮不进入新动态机制。
'''
    (OUT/'00_design.md').write_text(note,encoding='utf-8')
    dump(dict(design=sha(OUT/'00_design.json'),note=sha(OUT/'00_design.md')),OUT/'design_lock.json')
    print(json.dumps(dict(design_frozen=True,new_closures=len(jobs),out=str(OUT)),ensure_ascii=False),flush=True)


def run():
    d=json.loads((OUT/'00_design.json').read_text(encoding='utf-8'));verify(d)
    demands=json.loads((OUT/'frozen_inputs/demands.json').read_text(encoding='utf-8'))
    edges=pd.read_csv(OUT/'frozen_inputs/edges.csv')
    nodes={c:pd.read_csv(OUT/f'frozen_inputs/{c}_formula.csv') for c in ['baseline','node_a0.20_s00','family_a0.20_s00']}
    stored={c:pd.read_csv(OUT/f'frozen_inputs/{c}_stored.csv') for c in nodes}
    started=time.perf_counter();allrows=[];alllogs=[];allevents=[];checks=[]
    def check(name,ok):
        checks.append(dict(check=name,passed=bool(ok)))
        if not ok:save(checks,'validation_failed.csv');raise AssertionError(name)
    for block_start in range(0,len(d['jobs']),20):
        shard=OUT/'shards'/f'{block_start//20:03d}.json'
        if shard.exists():
            b=json.loads(shard.read_text(encoding='utf-8'));assert b['design']==sha(OUT/'00_design.json')
            allrows.extend(b['rows']);alllogs.extend(b['logs']);allevents.extend(b['events']);checks.extend(b['checks']);continue
        rows=[];logs=[];events=[];check_start=len(checks)
        for job in d['jobs'][block_start:block_start+20]:
            cap=job['capacity_id'];n=nodes[cap];K=n.overload_threshold.to_numpy(float)
            data=demands[job['demand']];parts=np.asarray(data['parts']);D=parts.sum(axis=0)
            cfg=ModelConfig.for_arm(job['arm'],theta_risk=job['theta'],gamma=job['gamma'],
                   theta_source='frozen proxy threshold; capacity second-rounding audit 2026-10-20')
            cfg.write_snapshot(OUT/'configs'/f'{cfg.fingerprint}.json')
            a=ConfiguredDispatchAllocator(n,edges,cfg)
            en=ConfiguredDualMechanismRecovery(a,edges,parts,data['groups'],job['initial_ids'],job['kind'])
            failed,m,iterations=en.closure(en.initial)
            tag={k:v for k,v in job.items() if k!='initial_ids'}
            rows.append(dict(**tag,**m,config_id=cfg.fingerprint,input_fingerprint=a.input_fingerprint,
                             final_failed=len(failed),initial_count=len(en.initial),iterations=iterations,
                             final_ids_json=json.dumps(sorted(failed),ensure_ascii=False),
                             initial_service=en.state(en.initial)[0]['Service_Rate']))
            current=set(en.initial)
            for step in en.cascade_logs[en.initial]:
                mm,u,s=en.state(current)
                check('physical round '+str((job['job_id'],step['round'])),mm['Max_LP_Violation']<1e-5
                      and mm['Service_Conservation_Error']<1e-6 and mm['Assignment_Conservation_Error']<1e-6
                      and abs(np.minimum(u*K,K).sum()-s.sum())<1e-6)
                record=dict(step);record['new_ids_json']=json.dumps(record.pop('new_ids'),ensure_ascii=False)
                logs.append(dict(job_id=job['job_id'],**record,util_json=json.dumps(u.tolist()),source_completed_json=json.dumps(s.tolist())))
                current.update(step['new_ids'])
            check('terminal set and attribution '+str(job['job_id']),current==set(failed)
                  and m['Overload_Only_Failures']+m['Risk_Only_Failures']+m['Both_Failures']==len(failed)-len(en.initial))
            for ev in en.cascade_events[en.initial]:
                i=a.pos[ev['node_id']];excess=(ev['utilization']-cfg.gamma)*K[i]
                events.append(dict(job_id=job['job_id'],**ev,capacity=K[i],threshold_excess=excess,
                     within_capacity_rounding_resolution=bool(ev['cause']!='risk_only' and excess<=cfg.gamma*.005+cfg.overload_epsilon*K[i]),
                     interpretation='fixed-solution diagnostic; not a robust certificate'))
            if job['arm']=='C':check('C no overload '+str(job['job_id']),m['Overload_Only_Failures']==m['Both_Failures']==0)
            a.paths.cache_clear()
        dump(dict(design=sha(OUT/'00_design.json'),rows=rows,logs=logs,events=events,checks=checks[check_start:]),shard)
        allrows.extend(rows);alllogs.extend(logs);allevents.extend(events)
        if len(allrows)%100==0 or len(allrows)==len(d['jobs']):
            print(json.dumps(dict(closures=len(allrows),planned=len(d['jobs']),elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    check('complete unique grid',len(allrows)==1920 and len({r['job_id'] for r in allrows})==1920)
    save(allrows,'02_new_runs.csv');save(alllogs,'03_new_rounds.csv');save(allevents,'04_new_events.csv');save(checks,'05_run_checks.csv')
    verify(d)
    files=[f for f in OUT.rglob('*') if f.is_file() and f.name!='completion.json']
    dump(dict(completed=True,new_closures=len(allrows),checks=len(checks),passed=sum(c['passed'] for c in checks),
              elapsed_seconds=time.perf_counter()-started,files={str(f.relative_to(OUT)):sha(f) for f in files}),OUT/'completion.json')
    print(json.dumps(dict(completed=True,new_closures=len(allrows),checks=len(checks))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['design','run']);args=parser.parse_args()
    design() if args.phase=='design' else run()
