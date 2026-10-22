"""Frozen, bounded gamma x demand sensitivity of the configured static V5."""
import argparse
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
from configured_dual_mechanism_v5 import ConfiguredDispatchAllocator, ConfiguredDualMechanismRecovery

CODE=Path(__file__).parent
ROOT=CODE.parents[1]
OUT=ROOT/'项目结果/gamma_demand_boundary_20261020'
PRIOR=ROOT/'项目结果/dispatch_capacity_null_mainline_20261018'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value,path):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def save(rows,name):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')


def design():
    if (OUT/'00_design.json').exists():raise FileExistsError('Frozen design already exists')
    core=importlib.import_module('83_dispatch_mainline_core')
    n,e,demands,_=core.load_study()
    previous=json.loads((PRIOR/'86_design_v3.json').read_text(encoding='utf-8'))
    cases=[dict(c,control=False,scope='previously explored synthetic shock') for c in previous['cases']]
    for kind in sorted({c['kind'] for c in cases}):
        cases.append(dict(case_id='no_damage_'+kind,kind=kind,initial_ids=[],control=True,
                          shock_scope='control',scope='same bonus and risk encoding, no initial failure'))
    protected=list(CODE.glob('*.py'))+list((CODE.parent/'data').glob('*.csv'))
    protected += [PRIOR/'86_design_v3.json',PRIOR/'84a_capacity_policy_results.csv',
                  ROOT.parent/'论文初稿配套图表_V5_20261019/初稿源文件/manuscript.md',
                  ROOT.parent/'中文论文初稿_全图美化与正文优化V5_20261019.docx']
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['shards','configs','frozen_inputs']:(OUT/name).mkdir(exist_ok=True)
    n.to_csv(OUT/'frozen_inputs/nodes.csv',index=False,encoding='utf-8-sig')
    e.to_csv(OUT/'frozen_inputs/edges.csv',index=False,encoding='utf-8-sig')
    parts,groups=demands['nominal']
    dump(dict(parts=np.asarray(parts).tolist(),groups=list(groups),node_order=list(n.node_id)),OUT/'frozen_inputs/demand.json')
    d=dict(schema='gamma-demand-boundary-1',date='2026-10-20',capacity_id='baseline',
           gamma=[1.,1.025,1.05,1.1],external_multipliers=[1.,1.2,1.5,2.,2.5],
           expected_total_multipliers=[1.,1.05,1.125,1.25,1.375],external_share=.25,
           thresholds=previous['thresholds'],arms=['B','C'],cases=cases,
           closures=840,shock_closures=480,control_closures=360,policy_pairs=420,
           completion_rule='proportional',max_hops=2,
           purposes=['map C-B magnitude and overload shutdown sensitivity',
                     'keep no-damage collapse visible',
                     'separate structural endpoint gamma>=rho from empirical discovery'],
           stop_rules=['No adaptive gamma/threshold/seed search after seeing outcomes',
                       'Do not add A merely to enlarge an already degenerate comparison',
                       'If a consistency check fails, diagnose before interpretation',
                       'Do not restart stopped feedback-Y or capacity-rank branches'],
           limitations=['synthetic uncalibrated gamma and demand levels',
                        'four previously explored shocks, not a new heldout set',
                        'one capacity and original proxy graph only',
                        'no arrival times, time queues, recovery or observed compatibility',
                        'no generic monotonicity in gamma assumed',
                        'grid rows are not independent empirical samples'],
           comparison_tolerance=dict(flow=1e-6,service=1e-9),
           sources={str(p):sha(p) for p in protected},
           frozen_inputs={str(p.relative_to(OUT)):sha(p) for p in (OUT/'frozen_inputs').iterdir()},
           runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__))
    dump(d,OUT/'00_design.json')
    text='''# 退出阈值与需求压力边界实验（运行前冻结）

2026-10-20。只运行现有静态V5的假设敏感性，不改原代码、数据或正文。

固定原图、基准容量、需求映射、风险权重、场景bonus及比例完成。比较B（rho=1.1）和C（rho=1）。
gamma为1、1.025、1.05、1.1，未实测标定；外部需求倍率1、1.2、1.5、2、2.5，对应总需求倍率1、1.05、1.125、1.25、1.375。
沿用86的三个风险阈值和四个已被多轮查看的8节点损伤，不能改称新留出验证。三类风险场景各设一个无初始损伤对照。
共840闭包，480损伤、360无损伤，420个C/B匹配。一次完成这一有限网格，不按结果追加更细网格、阈值或种子。

预定输出为逐条件C-B差、失效数、拒收与已接收未完成量，以及相对同场景无损伤控制的描述性差中差。
主摘要只对四个损伤等权，风险阈值分开，最高需求单列；无损伤三场景单独列出，不能混入损伤均值。
不计算现实概率或把网格当独立样本作显著性检验。不假设B随gamma单调。

C在gamma>=1的失效闭包应不随gamma变化；gamma>=rho时过载被接纳上限排除，这是预先已知结构控制，预期B/C总服务相同，不算发现。
总量C>=B为现有假设下的一致性检查，并非本轮新颖性验收。任何检查失败先诊断，不删除不利条件。

保存配置、全部逐轮利用率和来源完成、失效集合、输入/源码SHA256；gamma=1既有条件回归84表，并进行独立压力/触发重构及冷启动复算。
不赋予级联轮次小时/天含义，不将提高gamma解释为新增处理能力。额外接收但未完成仍计未服务，无跨期收益。

业务相容性仍未实测核实；本轮结果不能解决246个混合POI是否能相互承接的问题。
'''
    (OUT/'00_design.md').write_text(text,encoding='utf-8')
    dump(dict(design_sha256=sha(OUT/'00_design.json'),note_sha256=sha(OUT/'00_design.md')),OUT/'design_lock.json')
    print(json.dumps(dict(design_frozen=True,closures=d['closures'],out=str(OUT)),ensure_ascii=False),flush=True)


def check_sources(d):
    lock=json.loads((OUT/'design_lock.json').read_text(encoding='utf-8'))
    assert sha(OUT/'00_design.json')==lock['design_sha256']
    assert sha(OUT/'00_design.md')==lock['note_sha256']
    for p,h in d['sources'].items():
        if sha(p)!=h:raise RuntimeError('Changed protected source '+p)
    for p,h in d['frozen_inputs'].items():
        if sha(OUT/p)!=h:raise RuntimeError('Changed frozen input '+p)


def run():
    d=json.loads((OUT/'00_design.json').read_text(encoding='utf-8'));check_sources(d)
    n=pd.read_csv(OUT/'frozen_inputs/nodes.csv');e=pd.read_csv(OUT/'frozen_inputs/edges.csv')
    frozen=json.loads((OUT/'frozen_inputs/demand.json').read_text(encoding='utf-8'))
    base=np.asarray(frozen['parts']);groups=frozen['groups'];assert list(n.node_id)==frozen['node_order']
    old=pd.read_csv(PRIOR/'84a_capacity_policy_results.csv')
    checks=[];rows=[];traces=[];started=time.perf_counter()
    def check(name,ok):
        checks.append(dict(check=name,passed=bool(ok)))
        if not ok:
            save(checks,'validation_failed.csv');raise AssertionError(name)
    check('external proxy share',abs(base[1:].sum()/base.sum()-.25)<1e-10)
    for index,(theta,gamma,external) in enumerate(itertools.product(d['thresholds'],d['gamma'],d['external_multipliers'])):
        path=OUT/'shards'/f'{index:03d}.json'
        if path.exists():
            block=json.loads(path.read_text(encoding='utf-8'))
            check('resumed design identity '+str(index),block['design_sha256']==sha(OUT/'00_design.json'))
            rows.extend(block['rows']);traces.extend(block['traces']);checks.extend(block['checks'])
            continue
        start_check=len(checks);blockrows=[];blocktraces=[]
        parts=base.copy();parts[1:]*=external;D=parts.sum(axis=0)
        check('total demand multiplier '+str(index),abs(D.sum()/base.sum()-(.75+.25*external))<1e-10)
        for case,arm in itertools.product(d['cases'],d['arms']):
            cfg=ModelConfig.for_arm(arm,theta_risk=theta,gamma=gamma,
                                   theta_source='frozen 86 proxy threshold; gamma uncalibrated sensitivity 2026-10-20')
            cfg.write_snapshot(OUT/'configs'/f'{cfg.fingerprint}.json')
            a=ConfiguredDispatchAllocator(n,e,cfg)
            en=ConfiguredDualMechanismRecovery(a,e,parts,groups,case['initial_ids'],case['kind'])
            final,metric,iterations=en.closure(en.initial)
            tag=dict(run_id=len(rows)+len(blockrows),theta=theta,gamma=gamma,external_multiplier=external,
                     total_demand_multiplier=D.sum()/base.sum(),capacity_id=d['capacity_id'],case_id=case['case_id'],
                     control=case['control'],kind=case['kind'],arm=arm,config_id=cfg.fingerprint)
            row=dict(**tag,**metric,final_failed=len(final),initial_count=len(en.initial),cascade_iterations=iterations,
                     final_ids_json=json.dumps(sorted(final),ensure_ascii=False),
                     initial_completed_service=en.state(en.initial)[0]['Service_Rate'],
                     initial_damaged_demand=float(D[[a.pos[k] for k in en.initial]].sum()),
                     all_failed=len(final)==len(n))
            blockrows.append(row)
            current=set(en.initial)
            for step in en.cascade_logs[en.initial]:
                mm,util,served=en.state(current)
                check('round accounting '+str((tag['run_id'],step['round'])),
                      mm['Max_LP_Violation']<1e-5 and abs(mm['Service_Conservation_Error'])<1e-6
                      and abs(mm['Assignment_Conservation_Error'])<1e-6
                      and abs(np.minimum(util*a.nominal_capacity,a.nominal_capacity).sum()-served.sum())<1e-6)
                log=dict(step);log['new_ids_json']=json.dumps(log.pop('new_ids'),ensure_ascii=False)
                blocktraces.append(dict(run_id=tag['run_id'],**log,util_json=json.dumps(util.tolist()),
                                        source_completed_json=json.dumps(served.tolist())))
                current.update(step['new_ids'])
            check('trace final set '+str(tag['run_id']),current==set(final))
            check('failure partition '+str(tag['run_id']),metric['Overload_Only_Failures']+metric['Risk_Only_Failures']+metric['Both_Failures']==len(final)-len(en.initial))
            if gamma>=cfg.rho:
                check('structural no overload '+str(tag['run_id']),metric['Overload_Only_Failures']==metric['Both_Failures']==0)
            if gamma==1 and external in [1.,1.2,2.5] and not case['control']:
                name={1.:'nominal',1.2:'external_plus20',2.5:'external_plus150'}[external]
                ref=old[(old.capacity_id=='baseline')&(old.case_id==case['case_id'])&(old.arm==arm)
                        &(old.demand==name)&np.isclose(old.theta,theta,atol=1e-12,rtol=0)]
                check('historical target unique '+str(tag['run_id']),len(ref)==1)
                target=ref.iloc[0]
                check('historical gamma1 reproduction '+str(tag['run_id']),abs(metric['Service_Rate']-target.Service_Rate)<1e-9
                      and {a.pos[k] for k in final}==set(json.loads(target.final_failed_indices_json)))
            a.paths.cache_clear()
        dump(dict(design_sha256=sha(OUT/'00_design.json'),rows=blockrows,traces=blocktraces,
                  checks=checks[start_check:]),path)
        rows.extend(blockrows);traces.extend(blocktraces)
        print(json.dumps(dict(block=index+1,blocks=60,closures=len(rows),theta=theta,gamma=gamma,external=external,
                              elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    check('full grid complete',len(rows)==d['closures'] and len({r['run_id'] for r in rows})==d['closures'])
    save(rows,'01_runs.csv');save(traces,'02_rounds.csv');save(checks,'03_run_validation.csv')
    check_sources(d)
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json']
    dump(dict(completed=True,closures=len(rows),checks=len(checks),passed=sum(c['passed'] for c in checks),
              elapsed_seconds=time.perf_counter()-started,files={str(p.relative_to(OUT)):sha(p) for p in files}),OUT/'completion.json')
    print(json.dumps(dict(completed=True,closures=len(rows),checks=len(checks)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['design','run'])
    args=parser.parse_args();design() if args.phase=='design' else run()
