"""Run the unchanged full experiment chain; persist status and compare to snapshot."""
from project_paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR, PAPER_DIR, TABLES_DIR, DOCS_DIR, SOURCES_DIR, BACKUP_DIR, LOGS_DIR
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json, os, subprocess, sys, time
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data'
SNAPSHOT=BACKUP_DIR/'真实数据接入前快照_20261013_161225'

def now():return datetime.now().isoformat(timespec='seconds')

def compare(out):
    rows=[]
    for p in sorted(D.glob('*.csv')):
        old=SNAPSHOT/'data'/p.name
        if not old.exists():continue
        same=hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(old.read_bytes()).digest()
        try:
            a=pd.read_csv(old); b=pd.read_csv(p)
            semantic=a.equals(b)
            rows.append(dict(file=p.name,bytes_equal=same,values_equal=semantic,old_rows=len(a),new_rows=len(b),columns_equal=list(a.columns)==list(b.columns)))
        except Exception as exc:
            rows.append(dict(file=p.name,bytes_equal=same,values_equal=False,old_rows='',new_rows='',columns_equal=False,error=str(exc)))
    pd.DataFrame(rows).to_csv(out/'重跑前后逐表比较.csv',index=False,encoding='utf-8-sig')
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    status_path=out/'status.json'
    status=dict(state='running',pid=os.getpid(),started_local=now(),current_stage='starting',policy='confirmed_same_facility_same_year_same_unit_only',core_simulation_input_replacements=0)
    def save():
        temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(status_path)
    save(); config=json.loads((D/'38_real_data_application.json').read_text(encoding='utf-8'))
    assert config['core_simulation_input_replacements']==0
    rc=1
    try:
        with (out/'full_run.log').open('w',encoding='utf-8',buffering=1) as log:
            command=[sys.executable,'-u',str(ROOT/'Code/33_refresh_supplementary_experiments.py'),'--full']
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
            status['child_pid']=child.pid;save()
            for line in child.stdout:
                log.write(line)
                if line.startswith('RUN '):
                    status.update(current_stage=line.strip()[4:],last_progress_local=now());save()
            rc=child.wait()
        if rc:raise RuntimeError(f'Full chain failed: exit code {rc}; see full_run.log')
        status.update(current_stage='comparing_outputs',last_progress_local=now());save()
        rows=compare(out)
        changed=[r['file'] for r in rows if not r['values_equal']]
        # Check the source identity itself rather than merely the fact that a file changed.
        new=pd.read_csv(D/'公开补充数据_关键枢纽与风险指标.csv')
        matched=new[new['指标'].eq('花湖国际机场航空货邮吞吐量') & new['统计时点'].astype(str).eq('2024')]
        assert len(matched)==1 and abs(float(matched.iloc[0]['数值'])-86.5186783)<1e-12
        checks=pd.read_csv(D/'31_implementation_validation.csv')
        assert checks['status'].eq('PASS').all()
        data_note='真实数据接入说明_20261013.md'
        result=f'''# 严格口径真实资料接入后的重跑结果

完成时间：{now()}。完整核心链、10种子原配置/等预算验证、敏感性、独立动态恢复、损伤状态实验、完整性验收和实验图验收均成功完成；机制与结果验证{len(checks)}项通过。

按用户确认的“同设施、同年份、同单位”规则，更新公开统计背景1项（花湖2024航空货邮量86.5186783万吨）；单设施核心模拟输入可确认替换0项。OD、负荷及恢复曲线仍按既有公式生成，模型和算法保持原配置。本轮是严格口径资料更新后的完整复算，不是全部参数实测化，也不是完成外部性能验证。

逐表比较{len(rows)}张既有CSV，值发生变化的表有{len(changed)}张。详见 `重跑前后逐表比较.csv`。计时和诊断日志字段也会随执行变化，不能把这些差异视为模型性能变化。

当前近似等预算结论：

{pd.read_csv(D/'29e_equal_budget_conclusion.csv').to_string(index=False)}

数据来源和保留原因见 `../项目文档/{data_note}`；正式结果位于 `data/`，图位于 `../项目结果/figures/`。原数据完整备份位于 `{SNAPSHOT.as_posix()}`。后续需要替换设施级参数时，仍须取得符合所选口径的值，或者由用户另行选择明确标注的校准假设。
'''
        (out/'重跑结果说明.md').write_text(result,encoding='utf-8')
        status.update(state='complete',finished_local=now(),current_stage='complete',exit_code=0,checks_passed=len(checks),compared_tables=len(rows),changed_value_tables=changed)
        save()
    except Exception as exc:
        status.update(state='failed',finished_local=now(),exit_code=rc or 1,error=f'{type(exc).__name__}: {exc}');save();raise

if __name__=='__main__':main()
