# 代码索引

来源：`数据及代码-TJJM20260523014641/数据及代码/Code`（原 206 个 Python 文件）+ `outputs/*/`（探针脚本）

**本目录只保留产出当前结论的代码**，共 **90 个 .py + 11 个 .ps1**，1.1 MB。

---

## 1 目录结构

```
03_代码/
├── 01_核心模型/             17 个  模型内核与配置，被其他脚本 import
├── 02_数据构建/             22 个  POI → 网络 → 代理量 → 公开数据接入
├── 03_主线实验/             22 个  A/B/C、零模型、γ、V7 整改
├── 04_审计与核验/           15 个  证书、等价性、精度、平凡性
├── 05_探针与试跑_20261022/   4 组  2026-10-22 新增
└── 06_入口脚本/             11 个  PowerShell 运行入口
```

---

## 2 01_核心模型（被 import，不单独运行）

| 文件 | 职责 | 关键内容 |
|---|---|---|
| `model_config.py` | 全局参数 | ρ、γ、阈值、需求档；**`rho<1` 会被拒绝** |
| `project_paths.py` | 路径映射 | 输入/结果/文档/原件/备份/日志 |
| `dispatch_policy_rules.py` | 三规则实现 | LP / 剩余空间比例 / 最近可行贪心 |
| `dual_mechanism_recovery_v5.py` | V5 双机制内核 | 风险 + 利用率分离，接纳与完成分离 |
| `configured_dual_mechanism_v5.py` | 配置化封装 | 148 套去重配置 |
| `completion_rules.py` | 四种完成规则 | 比例/本地优先/升序/降序 |
| `matched_admission_v5.py` | 等接纳量位置干预 | 固定首次逐来源接纳 |
| `recovery_evaluator.py` | 恢复评价 | AUC、Mean J |
| `capacity_precision_v5.py` | 容量精度版 | `K = 1.25L + 既有异质差` |
| `protect_repair_v6.py` | 保护—修复预算 | `(1−η)×原风险压力` |
| `closed_loop_recovery_v3.py` | V3 闭环 | 城市需求 + 严格容量 + 稳定闭包 + 累计预算 |
| `closed_loop_recovery_v4.py` | V4 修正 | 固定原始度数分母 + 非负逐边项 |
| `model_optimization_v2_core.py` | V2 原型 | 联合严格容量 LP |
| `v4_parameter_audit_core.py` | V4 参数扫描 | 28 配置 × 8 损伤 × 6 阈值 × 2 需求 |
| `recovery_budget_audit_v5.py` | 预算—时域 | 96 条件 |
| `public_data_inputs_v1.py` | 公开数据加载器 | **要求实测参数变化时报错，不静默假标定** |
| `submission_strengthening_common.py` | 补强公共件 | — |

## 3 02_数据构建

| 编号 | 文件 | 职责 |
|---|---|---|
| 00a—00f | `00a_fix_stage1_hard_errors.py` … `00f_compare_before_after.py` | 阶段一硬错误修复、区县 GDP 爬取与应用、数据体检、来源扩展、前后对比 |
| 01a—01c | `01a_fetch_logistics_nodes.py`、`01b_supplementary_crawler.py`、`01c_clean_node_semantics.py` | POI 采集、补充爬取、语义清洗（生成 `node_id`） |
| 02 | `02_build_distance_matrix.py` | 节点对道路查询 |
| 03a—03c | `03a_TimeSpace_Impedance.py`、`03b_compute_gravity.py`、`03c_export_origin_data.py` | 时空阻抗、引力、源数据导出 |
| 04 | `04_build_multilayer_network.py`、`04_generate_quantified_datasets.py`、`04e_structure_partition_analysis.py` | 双层网络、代理量生成、结构分区 |
| 23—24 | `23_rebuild_edges_with_ids.py`、`24_sparsify_network.py` | 边 ID 重建、**稀疏化并按稀疏图重算加权度** |
| 38—39 | `38_apply_confirmed_public_data.py`、`39_real_data_background_rerun.py` | 已确认公开数据应用、真实数据背景重跑 |
| 130—131 | `130_integrate_public_data.py`、`131_validate_public_data_integration.py` | 公开资料接入与验证（30 观测 / 19 锚点 / L、K 替换为 0） |

> **重要**：`04` 生成器只产生初值；`24` 再按稀疏图重算加权度及依赖属性。**只运行 04 不能替代完整当前网络构造链。**
> 早期说明中的 `1/阻抗` 与当前 `24` 脚本的 `1/(1+阻抗)` 也不同。

## 4 03_主线实验

| 编号 | 文件 | 产出 |
|---|---|---|
| 82 | `82_build_fixed_total_capacity_scenarios.py` | 81 套等总 K 容量情景（只运行 3 套） |
| 83 | `83_dispatch_mainline_core.py` | A/B/C 分层 LP 内核 |
| 84 | `84_run_capacity_policy_mainline.py` | 1728 条主评价 |
| 84b | `84b_no_damage_controls.py` | 432 条无损伤控制 |
| 85c | `85c_generate_constrained_nulls_fixed_steps.py` | **正式 v3 采样器**（固定 100×4579 提案） |
| 86c | `86c_run_null_policy_fixed_steps.py` | 3600 条零模型运行 |
| 87 | `87_validate_mainline.py` | 13161 项独立检查 |
| 88 | `88_summarize_and_plot_mainline.py` | 配对、分层、成本、零模型位置 |
| 89 | `89_write_mainline_report.py` | 报告 |
| 114—115 | `114_gamma_demand_boundary.py`、`115_validate_gamma_boundary.py` | **840 闭包 γ 扫描**（含预登记的 γ≥ρ 等价判定） |
| 119—121 | `119_dispatch_rule_study.py`、`120_validate_dispatch_rules.py`、`121_finalize_dispatch_study.py` | **2424 闭包三规则对照** |
| 122—126 | `122_matched_location_study.py`、`123_validate_matched_location.py`、`126_finalize_location_study.py` | 648 条件位置反馈 |
| 132—133 | `132_node_boundary_and_scope_study.py`、`133_finish_node_boundary_study.py` | 前轮 4 点排除与 242 点对照 |
| 136—138 | `136_audit_processing_units_v7.py`、`137_run_processing_unit_rectification_v7.py`、`138_package_processing_unit_v7.py` | **V7 全量 246→122 整改与 6912 轨迹** |

## 5 04_审计与核验

| 编号 | 文件 | 核验对象 | 结果 |
|---|---|---|---|
| 79 | `79_verify_dispatch_certificates.py` | 安全最大流 / 割 / 承接—超额前沿 | 200 图 + 642 前沿点 + 反例，全通过 |
| 80 | `80_audit_dual_mechanism_triviality.py` | 双机制平凡性 | 50 小网络 + 786 终态子集 + 192 条件 |
| 90 | `90_dynamic_buffer_tradeoff_audit.py` | 单设施缓冲 | 16200 配置 / 64800 结果 |
| **91** | `91_buffer_time_expanded_flow_audit.py` | **缓冲与标准流等价性** | 972 配置，**DP 与连续 LP 完全一致** |
| 92b | `92b_queue_stress_coupling_audit.py` | 队列持续压力 | 2592 配置 / 12960 结果 |
| 94—96 | `94_reversible_blocking_prototype.py`、`95_blocking_peak_and_horizon_audit.py`、`96_validate_reversible_blocking.py` | 共享 FIFO 两站原型 | 5240 配置 / 26920 结果 |
| 105 | `105_mainline_claim_closure.py` | 结论—假设—证据表 | C01—C18 |
| 110 | `110_capacity_rank_identifiability.py` | 容量排名可识别性 | 72 条件全并列 |
| 112 | `112_validate_configured_v5.py` | V5 配置校验 | — |
| 116 | `116_capacity_precision_audit.py` | 容量精度 | 1920 闭包 / 36 配对冷解 |
| 128 | `128_full_objective_face_screen.py` | 三目标最优面 | 150 条件 / 600 轨迹 |
| 135 | `135_diagnose_boundary_degree_normalization.py` | 度数归一化诊断 | 恒等式验证 |
| 140 | `140_verify_v7_delivery.py` | V7 交付核验 | 914 项 |

## 6 05_探针与试跑_20261022（最新）

| 子目录 | 文件 | 内容 |
|---|---|---|
| `a_承诺与成本/` | `commitment_probe.py`、`fixed_total_probe.py`、`cost_probe.py` | 798 可行组承诺溢价；144 组打破笛卡尔积；14160 组成本形态 |
| `b_等待分流试跑/` | `pilot.py`、`uncertain_capacity.py` 等 | 第一轮 3888 配置；第二轮 2808 配置 |
| `c_同信息对照/` | `compare.py`、`verify_rule.py` | **五类策略同信息对照，283 项核验** |
| `d_前瞻边界/` | `verify_lookahead_boundary.py`、`audit_boundary_robustness.py` | **f*=H−1 定理的 50 + 104 项独立复核** |

## 7 06_入口脚本

| 脚本 | 作用 |
|---|---|
| `运行修订模型_V7.ps1` | 检查 V7 设计锁 → 调 137；支持 `CheckOnly` |
| `run_dispatch_mainline.ps1` | 三容量 + 正式 v3 零模型 + 无损伤 + 验证 + 图表 |
| `run_dual_mechanism_v5.ps1` | V5 双机制 |
| `run_recovery_budget_v5.ps1` | V5 预算时域 |
| `run_protect_repair_v6b.ps1` | V6b 训练感知保护修复 |
| `run_closed_loop_v3.ps1` | V3 闭环 |
| `run_mechanism_v4.ps1`、`run_v4_parameter_audit.ps1` | V4 修正与参数扫描 |
| `run_revision.ps1` | 调 33 `--full`（**不代表全部 47 分支**） |
| `run_submission_strengthening.ps1` | 投稿前补强 |
| `run_pilot.ps1` | 等待分流试跑 |

---

## 8 已剔除的代码（及原因）

| 剔除 | 原因 |
|---|---|
| `85`、`85b` | 采样器前两版（v1 接受率过低；v2 抽到跳链） |
| `86`、`86b` | 对应 v1/v2 零模型运行 |
| `92`、`92a` | 队列压力初版与部分开发输出（用 `92b`） |
| `117`、`117b` | 精度核验的中间版本 |
| `54_svd_free_cascade.py` vs `54_threshold_sensitivity_v3.py` | 同号两职责，均未纳入（属已停分支） |
| `05`—`09`、`13`—`20`、`21`—`37`、`40`—`78`、`97`、`99`—`104`、`106`—`109`、`111`、`113`、`118`、`124`、`125`、`127`、`129`、`134`、`139` | 属已停分支或仅报告/绘图/打包，不产生当前结论 |
| `figures/`、`qa_tools/` | 图件生成与排版 QA，不改变实验本体 |

> 完整 206 个文件说明见原工作区 `outputs/data_code_results_inventory_20261021/当前代码逐文件说明.csv`。

---

## 9 复现注意

1. **路径硬编码**：部分脚本含本机绝对路径与时间戳 → 本机可复现工程，非跨机器零配置发布
2. **依赖固定**：NumPy 2.3.5、pandas 3.0.1、SciPy 1.18.1、Matplotlib 3.11.2、PyMuPDF 1.28.2
3. **额外包**：`E:/sci23/.runtime_packages`（含 `fitz`，需 `PYTHONPATH` 指过去）
4. **复算会重写派生结果** → 先备份，或改用 `CheckOnly` / `design` 阶段
5. **`rho<1` 会被 `model_config.py` 拒绝**，做 ρ 连续扫描前须重定义接纳动作
