"""Distinguish demand-induced failures from exogenous-shock amplification."""
import json
from pathlib import Path
import pandas as pd
from closed_loop_recovery_v3 import HERE
from submission_strengthening_common import module
core=module('83_dispatch_mainline_core');OUT=core.OUT

def main():
    design=json.loads((OUT/'84_design.json').read_text(encoding='utf-8'));core.check_sources(design)
    n,e,demands,_=core.load_study();case=dict(case_id='no_external_damage',shock_scope='diagnostic_control',kind='random_failure',initial_ids=[])
    # Scenario-specific bonuses differ. Repeat the no-damage control with each
    # kind present in the primary grid, so later risk propagation is matched.
    kinds=sorted({v['kind'] for v in design['cases']});rows=[];checks=[]
    protocol=dict(reason='High external-demand condition may overload nominal capacity without any external damage; matched no-damage diagnostic added before interpretation, not used to tune policies',
                  kinds=kinds,sources=core.source_hashes([Path(__file__),HERE/'83_dispatch_mainline_core.py',OUT/'84_design.json']))
    core.dump(protocol,OUT/'84b_control_design.json')
    for cap in core.CAP_IDS:
        nn=core.capacity_nodes(n,cap)
        for kind in kinds:
            cc=dict(case,kind=kind)
            r,_,v=core.run_grid(nn,e,demands,[cc],design['thresholds'],design['demand_names'],dict(capacity_id=cap,graph_id='original'))
            rows+=r;checks+=v
        print('84b no-damage controls',cap,'complete',flush=True)
    core.save(rows,OUT/'84c_no_damage_results.csv');core.save(checks,OUT/'84b_control_validation.csv')
    core.check_sources(protocol);core.check_sources(design)
    core.dump(dict(completed=True,rows=len(rows),checks=len(checks),passed=sum(v['passed'] for v in checks),source_sha256=core.sha(__file__),
                   nominal_capacity_loads_unchanged=True,diagnostic_only=True),OUT/'84b_control_completion.json')

if __name__=='__main__':main()
