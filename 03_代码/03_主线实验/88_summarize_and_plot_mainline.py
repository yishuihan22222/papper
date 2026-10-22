"""Descriptive paired summaries; full demand strata and bounded null inference."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from closed_loop_recovery_v3 import HERE
from submission_strengthening_common import module
core=module('83_dispatch_mainline_core');OUT=core.OUT;FIG=OUT/'figures'
SKILL=Path('C:/Users/11828/.codex/skills/nature-figure');sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

KEYS=['capacity_id','graph_id','theta','case_id','demand','kind','shock_scope']
METRICS=['Service_Rate','failure_fraction','Unserved_Load','Demand_Load','Undispatched_Load','Uncompleted_Assigned_Load','Transfer_Cost','Overload_Only_Failures','Risk_Only_Failures','Both_Failures','final_failed']
def paired(r):
    result=None
    for arm in ['A','B','C']:
        g=r[r.arm.eq(arm)][KEYS+METRICS].rename(columns={k:arm+'_'+k for k in METRICS})
        result=g if result is None else result.merge(g,on=KEYS,validate='one_to_one')
    result['C_B_pp']=100*(result.C_Service_Rate-result.B_Service_Rate)
    result['B_A_pp']=100*(result.B_Service_Rate-result.A_Service_Rate)
    result['C_B_failed_reduction']=result.B_final_failed-result.C_final_failed
    return result
def counts(v):return dict(conditions=len(v),wins=int(v.gt(1e-8).sum()),ties=int(v.abs().le(1e-8).sum()),losses=int(v.lt(-1e-8).sum()),mean_pp=float(v.mean()),median_pp=float(v.median()),min_pp=float(v.min()),max_pp=float(v.max()))
def run_audit(script,args,output):
    result=subprocess.run([sys.executable,str(SKILL/'scripts'/script),*map(str,args)],capture_output=True,text=True,encoding='utf-8')
    output.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(f'Figure QA failed: {script}; see {output}')

def export(fig,name):
    stem=FIG/name;fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',overlay_svg=str(stem)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    # Keep the exported canvas identical to the measured physical-size layout.
    for ext in ['pdf','svg']:fig.savefig(str(stem)+'.'+ext)
    fig.savefig(str(stem)+'.png',dpi=300);fig.savefig(str(stem)+'.tiff',dpi=600)
    plt.close(fig)
    run_audit('audit_pdf_text.py',[str(stem)+'.pdf','--min-pt','5','--json'],Path(str(stem)+'.text-audit.json'))
    run_audit('audit_figure_collisions.py',[str(stem)+'.pdf','--json-out',str(stem)+'.collision.json','--overlay-pdf',str(stem)+'.collision-overlay.pdf'],Path(str(stem)+'.collision-audit.txt'))

def main():
    for name in ['84_completion.json','84b_control_completion.json','85_completion_v3.json','86_completion_v3.json','87_completion.json']:
        record=json.loads((OUT/name).read_text(encoding='utf-8'))
        if not record['completed'] or ('checks' in record and record['checks']!=record['passed']):raise RuntimeError('Unfinished validation '+name)
    FIG.mkdir(parents=True,exist_ok=True)
    r=pd.read_csv(OUT/'84a_capacity_policy_results.csv');nr=pd.read_csv(OUT/'86a_null_policy_results.csv');ctrl=pd.read_csv(OUT/'84c_no_damage_results.csv')
    p=paired(r);npair=paired(nr);core.save(p,OUT/'88a_capacity_pairs.csv');core.save(npair,OUT/'88b_null_pairs.csv')
    summary=[];demand_summary=[]
    for cap,g in p.groupby('capacity_id'):
        summary.append(dict(capacity_id=cap,contrast='C-B',**counts(g.C_B_pp)))
        summary.append(dict(capacity_id=cap,contrast='B-A',**counts(g.B_A_pp)))
        for demand,sub in g.groupby('demand'):demand_summary.append(dict(capacity_id=cap,demand=demand,**counts(sub.C_B_pp)))
    core.save(summary,OUT/'88c_capacity_summary.csv');core.save(demand_summary,OUT/'88d_demand_strata.csv')
    quantities=r.copy();quantities['unserved_pct']=100*quantities.Unserved_Load/quantities.Demand_Load
    quantities['not_admitted_pct']=100*quantities.Undispatched_Load/quantities.Demand_Load
    quantities['assigned_unfinished_pct']=100*quantities.Uncompleted_Assigned_Load/quantities.Demand_Load
    q=quantities.groupby(['capacity_id','arm']).agg(conditions=('case_id','size'),mean_service=('Service_Rate','mean'),mean_unserved_pct=('unserved_pct','mean'),
           mean_not_admitted_pct=('not_admitted_pct','mean'),mean_assigned_unfinished_pct=('assigned_unfinished_pct','mean'),mean_transfer_cost=('Transfer_Cost','mean'),
           mean_overload_only=('Overload_Only_Failures','mean'),mean_risk_only=('Risk_Only_Failures','mean'),mean_both=('Both_Failures','mean')).reset_index()
    q['mean_service_pct']=100*q.pop('mean_service');core.save(q,OUT/'88e_physical_metrics.csv')
    control_keys=['capacity_id','theta','demand','kind'];control=ctrl.pivot(index=control_keys,columns='arm',values='Service_Rate').reset_index()
    control['control_C_B_pp']=100*(control.C-control.B)
    diagnostic=p.merge(control[control_keys+['control_C_B_pp']],on=control_keys,validate='many_to_one');diagnostic['gain_difference_from_no_damage_pp']=diagnostic.C_B_pp-diagnostic.control_C_B_pp
    core.save(diagnostic,OUT/'88f_no_damage_matched_diagnostic.csv')
    primary_cap='node_a0.20_s00';selection=p.capacity_id.eq(primary_cap)&p.case_id.isin(core.NULL_CASE_IDS)&p.demand.isin(core.NULL_DEMANDS)
    reference=p[selection];assert len(reference)==24
    graph_stats=[]
    for graph,g in npair.groupby('graph_id'):
        assert len(g)==24
        graph_stats.append(dict(graph_id=graph,mean_C_B_pp=g.C_B_pp.mean(),mean_B_A_pp=g.B_A_pp.mean(),
                                nominal_C_B_pp=g.loc[g.demand.eq('nominal'),'C_B_pp'].mean(),
                                high_C_B_pp=g.loc[g.demand.eq('external_plus150'),'C_B_pp'].mean(),
                                nominal_C_failure_pct=100*g.loc[g.demand.eq('nominal'),'C_failure_fraction'].mean(),
                                nominal_B_failure_pct=100*g.loc[g.demand.eq('nominal'),'B_failure_fraction'].mean()))
    gs=pd.DataFrame(graph_stats);gs=gs.merge(pd.read_csv(OUT/'85a_null_graph_summary.csv'),on='graph_id',validate='one_to_one');core.save(gs,OUT/'88g_null_graph_statistics.csv')
    positions=[]
    for label,column,ref in [('primary_all_24','mean_C_B_pp',reference.C_B_pp.mean()),('secondary_nominal_12','nominal_C_B_pp',reference.loc[reference.demand.eq('nominal'),'C_B_pp'].mean())]:
        values=gs[column];sd=float(values.std(ddof=1));positions.append(dict(statistic=label,original=float(ref),null_graphs=50,null_mean=float(values.mean()),null_sd=sd,
          null_q05=float(values.quantile(.05)),null_median=float(values.median()),null_q95=float(values.quantile(.95)),z_descriptive=None if sd<1e-12 else float((ref-values.mean())/sd),
          upper_tail_mc_position=float((1+values.ge(ref-1e-10).sum())/51),
          interpretation='Descriptive location in approximate finite-chain constrained ensemble; no calibrated exact P value or cross-city generalization'))
    core.save(positions,OUT/'88h_null_positions.csv')
    curve=r.groupby(['capacity_id','demand','theta','arm']).agg(mean_failure_fraction=('failure_fraction','mean'),conditions=('case_id','size')).reset_index()
    core.save(curve,FIG/'failure_curve_source.csv')
    contract=dict(backend='python',archetype='quantitative grid',journal='generic research; author-review draft',size_mm=[183,72],minimum_font_pt=6,
       failure_curves=dict(question='Does policy separation persist under fixed-total capacity redistribution?',panels=['Uniform-utilization reference','Node-level heterogeneity','Family-level heterogeneity'],
                           all_observations='All 1728 main evaluations represented across the four demand-specific figures; each curve point averages all 16 fixed shocks, no exclusions',uncertainty='Descriptive grid means; no confidence intervals or independent-disaster interpretation'),
       null_figure=dict(question='Where does original effect magnitude lie in the restricted topology ensemble, and how does demand explain it?',panels=['All fifty per-graph effects with original reference','Nominal vs high-demand paired effects on all fifty graphs'],
                        sample_definition='50 finite-chain constrained null graphs, 24 conditions per graph; distribution spread is not empirical sampling uncertainty'))
    core.dump(contract,FIG/'figure_contract.json')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
                         'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42})
    color={'A':'#777777','B':'#C77A35','C':'#237A8C'};style={'A':'--','B':'-.','C':'-'};marker={'A':'x','B':'s','C':'o'}
    caps=core.CAP_IDS;titles=['Uniform utilization','Node heterogeneity','Type heterogeneity']
    for demand in ['nominal','ezhou_plus20','external_plus20','external_plus150']:
        fig,axes=plt.subplots(1,3,figsize=(183/25.4,72/25.4),sharey=True)
        fig.subplots_adjust(left=.09,right=.985,bottom=.24,top=.74,wspace=.25)
        for i,(ax,cap,title) in enumerate(zip(axes,caps,titles)):
            for arm in ['A','B','C']:
                g=curve[curve.capacity_id.eq(cap)&curve.demand.eq(demand)&curve.arm.eq(arm)].sort_values('theta')
                ax.plot(g.theta,100*g.mean_failure_fraction,color=color[arm],linestyle=style[arm],marker=marker[arm],markersize=3,linewidth=1.1,label='Policy '+arm)
            ax.set_title(title,pad=5);ax.set_ylim(-3,103);ax.set_yticks([0,25,50,75,100]);ax.set_xticks(sorted(curve.theta.unique()),['0.113','0.20','0.42'])
            ax.set_xlabel('Risk threshold');ax.text(-.16,1.17,'abc'[i],transform=ax.transAxes,fontweight='bold',fontsize=8)
        axes[0].set_ylabel('Final failed nodes (%)');handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False)
        export(fig,'policy_failure_'+demand)
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,76/25.4));fig.subplots_adjust(left=.095,right=.985,bottom=.25,top=.74,wspace=.36)
    original=float(reference.C_B_pp.mean());values=gs.mean_C_B_pp.to_numpy();low=min(values.min(),original);high=max(values.max(),original)
    padding=max((high-low)*.08,.05)
    axes[0].hist(values,bins=np.linspace(low-padding,high+padding,10),color='#82AEB6',edgecolor='white',linewidth=.5)
    axes[0].axvline(original,color='#B86F31',linewidth=1.3,linestyle='--');axes[0].set_xlabel('Mean service gain C - B (pp)');axes[0].set_ylabel('Null graphs');axes[0].set_title('Restricted topology ensemble',pad=5)
    offsets=np.linspace(-.1,.1,len(gs))
    for off,row in zip(offsets,gs.itertuples()):axes[1].plot([off,1+off],[row.nominal_C_B_pp,row.high_C_B_pp],color='#A8B7BB',alpha=.4,linewidth=.5)
    axes[1].scatter(offsets,gs.nominal_C_B_pp,s=6,color='#237A8C',alpha=.65);axes[1].scatter(1+offsets,gs.high_C_B_pp,s=6,color='#C77A35',alpha=.65)
    axes[1].set_xticks([0,1],['Nominal','High demand']);axes[1].set_xlim(-.3,1.3);axes[1].set_ylabel('Mean service gain C - B (pp)');axes[1].set_title('Demand dependence',pad=5)
    for i,ax in enumerate(axes):ax.text(-.16,1.17,'ab'[i],transform=ax.transAxes,fontweight='bold',fontsize=8)
    fig.legend([Line2D([0],[0],color='#B86F31',linestyle='--')],['Original proxy graph'],loc='upper center',bbox_to_anchor=(.5,.99),frameon=False)
    export(fig,'constrained_null_effects')
    run_audit('validate_figure.py',[__file__,'--backend','python','--json'],FIG/'plot_source_audit.json')
    core.dump(dict(completed=True,figure_count=5,capacity_pairs=len(p),null_pairs=len(npair),null_positions=positions,source_sha256=core.sha(__file__),
                   figure_qa='Alignment and PDF text/collision audits run; final visual inspection separately required'),OUT/'88_completion.json')
    print('88 mainline summaries and figures complete',flush=True)

if __name__=='__main__':main()
