"""Final immutable delivery audit after all frozen and post-hoc outputs exist."""
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

driver=importlib.import_module('122_matched_location_study');OUT=driver.OUT;CODE=driver.CODE


def main():
    d=json.loads((OUT/'00_design.json').read_text(encoding='utf-8'));driver.verify(d)
    for manifest in ['completion.json','delivery_manifest.json']:
        m=json.loads((OUT/manifest).read_text(encoding='utf-8'))
        assert m['completed']
        for p,h in m['files'].items():
            if driver.old.sha(OUT/p)!=h:raise AssertionError('Changed output '+p)
    expected_counts={'02_witness_checks.csv':364,'13_independent_checks.csv':149671,
                     '16_strict_witness_checks.csv':722,'19_general_condition_checks.csv':144}
    for p,count in expected_counts.items():
        checks=pd.read_csv(OUT/p);assert len(checks)==count and checks.passed.all()
    runs=pd.read_csv(OUT/'03_interventions.csv');plans=pd.read_csv(OUT/'04_initial_plans.csv');effects=pd.read_csv(OUT/'07_paired_effects.csv')
    assert len(runs)==10368 and len(plans)==2592
    primary=effects[(effects.contrast=='primary')&(effects.mechanism=='both')]
    assert len(primary)==648 and primary.changed.sum()==1 and (primary[primary.changed].delta_pp<0).all()
    assert primary[primary.domain=='synthetic'].changed.sum()==0
    strict=json.loads((OUT/'17_strict_witness.json').read_text(encoding='utf-8'))
    assert [r['final_served'] for r in strict['main']]==[0.,30.]
    for r in strict['main']:assert r['assigned']==31. and r['served']==30. and r['excess']==1.
    point=plans[(plans.condition_id==37)&plans.plan.isin(['nearest_cost','balanced_util'])].set_index('plan')
    assert np.isclose(point.loc['nearest_cost','initial_peak_util'],1.1)
    assert np.isclose(point.loc['balanced_util','initial_peak_util'],1.055355081828085)
    sources=[CODE/f for f in ['matched_admission_v5.py','122_matched_location_study.py','123_validate_matched_location.py',
        '124_strict_location_witness.py','125_location_conditions_audit.py','126_finalize_location_study.py','README_等接纳量位置反馈_20261020.md']]
    final=dict(completed=True,original_files_unchanged=True,conditions=648,interventions=10368,
        independent_checks=149671,cold_starts=16,primary_changed=1,primary_pairs=648,
        primary_max_abs_delta_pp=float(abs(primary.delta_pp).max()),
        strict_existence_proven=True,general_sufficient_condition='e/Kb <= gamma+epsilon-1 < e/Ka with additional graph/continuation assumptions',
        empirical_validation=False,novelty_established=False,
        files={str(p.relative_to(OUT)):driver.old.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final_delivery_manifest.json'},
        sources={str(p):driver.old.sha(p) for p in sources})
    driver.dump(final,OUT/'final_delivery_manifest.json')
    print(json.dumps({k:v for k,v in final.items() if k not in ['files','sources']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
