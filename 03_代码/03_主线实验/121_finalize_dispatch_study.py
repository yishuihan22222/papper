"""Verify scientific files and final documentation; do not change results."""
import importlib
import json
from pathlib import Path
import pandas as pd

driver=importlib.import_module('119_dispatch_rule_study');OUT=driver.OUT


def main():
    d=json.loads((OUT/'00_design.json').read_text(encoding='utf-8'));driver.verify(d)
    manifest=json.loads((OUT/'final_manifest.json').read_text(encoding='utf-8'))
    for p,h in manifest['files'].items():assert driver.sha(OUT/p)==h,p
    runs=pd.read_csv(OUT/'02_runs.csv');checks=pd.read_csv(OUT/'12_independent_checks.csv')
    assert len(runs)==2424 and len(checks)==31951 and checks.passed.all()
    # Verify the short readout against the saved scientific records.
    selected=runs[(runs.domain=='geographic_proxy')&(~runs.control)]
    expected={(1.,1.,'lexicographic_lp'):75.35548,(1.,1.,'remaining_capacity_share'):75.34589,
              (1.,1.,'nearest_feasible'):75.28584,(1.,1.1,'lexicographic_lp'):42.32339,
              (1.,1.1,'remaining_capacity_share'):41.34764,(1.,1.1,'nearest_feasible'):0.,
              (1.05,1.,'lexicographic_lp'):75.35548,(1.05,1.,'remaining_capacity_share'):75.34589,
              (1.05,1.,'nearest_feasible'):75.28584,(1.05,1.1,'lexicographic_lp'):44.02333,
              (1.05,1.1,'remaining_capacity_share'):42.33537,(1.05,1.1,'nearest_feasible'):0.}
    for (gamma,rho,rule),value in expected.items():
        block=selected[(selected.gamma==gamma)&(selected.rho==rho)&(selected.rule==rule)]
        assert len(block)==24 and abs(100*block.Service_Rate.mean()-value)<.000006
    assert not manifest['quantitative_external_validation'] and not manifest['wuhan_permissions_verified']
    code=Path(__file__).parent
    sources=[code/name for name in ['dispatch_policy_rules.py','119_dispatch_rule_study.py','120_validate_dispatch_rules.py',Path(__file__).name]]
    documentation=code/'README_同层派入规则对照_20261020.md'
    result=dict(completed=True,closures=2424,lp_regressions=168,cold_starts=12,independent_checks=31951,
        external_business_evidence=True,quantitative_external_validation=False,wuhan_permissions_verified=False,
        original_files_unchanged=True,source_sha256={str(p):driver.sha(p) for p in sources},
        documentation_sha256={str(documentation):driver.sha(documentation)},
        files={str(p.relative_to(OUT)):driver.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'})
    driver.dump(result,OUT/'delivery_manifest.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','source_sha256','documentation_sha256']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
