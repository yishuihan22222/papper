# 数据索引

**27 MB / 119 个文件**，只保留当前有效输入与摘要级结果表。

---

## 1 目录结构

```
04_数据/
├── 01_模型输入_V7/          122 节点 / 1624 边（正式输入）
├── 02_基础数据_246点/       城市指标、代理容量、30126 候选边
├── 03_节点审查台账/         246 逐点处置、179 组关系
├── 04_核心结果表/           10 个分支的摘要表
├── 05_数据来源与锚点/       真实数据锚点清单
└── 06_探针结果/             承诺、成本、试跑、同信息、前瞻
```

---

## 2 01_模型输入_V7（正式计算输入）

| 文件 | 行/内容 | 用途 |
|---|---|---|
| `nodes.csv` | 122 行 | `capacity_index`、`proxy_daily_load`、经纬度、`identity_status`、`capacity_status` |
| `edges.csv` | 1624 行 | 诱导图边；`Duration_mins` p50 17.65 / p90 28.86 / **max 63.03 min** |
| `retained_source_demands.json` | — | 保留来源任务（剔除的 124 点任务不分摊） |
| `使用说明.md` | — | 原始说明 |

**图结构**：6 连通分量（107 / 9 / 3 / 1 / 1 / 1），3 个孤立点，平均度 26.62，最大度 56。
**城市分布**：武汉 101、黄冈 8、鄂州 7、孝感 6。
**family 分布**：park 60、distribution 34、rail 19、port 3、airport 2、coldchain 2、bonded 1、other 1。

---

## 3 02_基础数据_246点

| 文件 | 内容 | 关键事实 |
|---|---|---|
| `04a_城市综合指标_含估算.csv` | 8 外围城市 | GDP/道路/货运量/周转量/快递量/收入/标准化/估算标记/需求强度/引力 |
| `04b_武汉节点代理容量.csv` | 246 节点 | `capacity_index`、`proxy_daily_load`、`overload_threshold` |
| **`04c_武汉边代理属性_候选全边.csv`** | **30126 对** | 246 点全候选；`Duration_mins` p50 55.78 / **max 193.25 min** |
| `04c_武汉边代理属性_稀疏15pct.csv` | **4579 边** | 主图；15% 外生引力 + 每点至少 3 条 |
| `04_upper_layer_nodes.csv` / `04_upper_layer_edges.csv` | 9 行 / 8 行 | 城市层 |
| `04_lower_layer_nodes.csv` | 246 行 | 设施层 |
| `04_cross_layer_edges.csv` | **80 行** | 8 城 × 10 条，接收端去重 15 门户 |
| `04_multilayer_network_summary.csv` | — | 双层汇总 |
| `03a_raw_distances.csv` | — | 原始道路查询 |
| `01c_cleaning_summary.csv` | — | 语义清洗汇总 |
| `04e_node_structural_roles.csv` | — | 结构角色 |
| `05_resilience_metrics.csv` | — | 基础韧性表 |

**关键提示**：`04c_候选全边` 的 30126 是 **246 点**的；同一 122 点的全候选对是 **7381**（=122×121/2）。**两者不可混用。**

---

## 4 03_节点审查台账

| 文件 | 内容 |
|---|---|
| `01_all_246_node_decisions.csv` | **全部 246 逐点处置**（175 KB） |
| `02_removed_or_quarantined.csv` | 34 移出 + 87 隔离（89 KB） |
| `03_processing_site_candidates.csv` | 122 处理场地候选（86 KB） |
| `04_parent_child_relations.csv` | 3 组层级代表 |
| `05_all_nearby_or_homonym_pairs.csv` | **179 组近邻/同名关系**（61 KB） |
| `06_function_label_corrections.csv` | 功能标签修订 |
| `07_shock_projection.csv` | 16 组冲击的共同投影 |
| `19_identity_evidence_index_246.csv` | 身份证据索引（222 KB） |

**处置结果**：122 候选 + 34 移出 + 87 隔离 + 3 层级参考 = 246；**严格证实的同一物理单元合并 = 0**。

---

## 5 04_核心结果表（摘要级，10 个分支）

| 子目录 | 关键文件 | 对应结论 |
|---|---|---|
| `a_V7整改/` | `11_policy_pairs.csv`（356 KB）、`12_policy_summary.csv`、`13_matched_task_effects.csv`（729 KB）、`14_resources.csv`、`16_checks.csv`、`15_cold_restarts.csv` | N05—N11 |
| `b_静态主线ABC/` | `84a_capacity_policy_results.csv`（1.7 MB）、`84c_no_damage_results.csv`、`88a_capacity_pairs.csv`、`88c_capacity_summary.csv`、`88d_demand_strata.csv` | S01—S12 |
| `c_零模型/` | `85a_null_graph_summary.csv`、`88b_null_pairs.csv`、`88g_null_graph_statistics.csv`、`88h_null_positions.csv`、`88e_physical_metrics.csv` | Z01—Z06 |
| `d_gamma边界/` | `04_pairs.csv`、`06_shock_summary.csv`、`08_gamma_adjacent_changes.csv`、`11_findings.json`、`14_nonmonotone_events.csv`、`15_非单调结果_确认与边界.md` | G01—G05 |
| `e_三规则/` | `07_summary.csv`、`06_rule_vs_lp.csv`、`09_tie_sensitivity.csv`、`13_findings.json`、`15_本轮验收与论文使用边界_20261020.md` | R01—R06 |
| `f_位置反馈/` | `08_effect_summary.csv`、`18_general_condition_family.csv`、`19_general_condition_checks.csv` | P01—P05 |
| `g_容量精度/` | `10_paper_demand_summary.csv`、`11_boundary_summary.csv`、`07_boundary_old_new.csv` | F03—F04 |
| `h_保护修复/` | `78a_diagnostic_contrasts.csv`、`78b_cold_replay_summary.csv`、`76g_frozen_component_summary.csv` | B05—B07 |
| `i_动态缓冲/` | `94j_stratified_summary.csv`、`94h_examples.csv`、`95b_paired_results.csv` | Y01—Y07 |
| `j_关键节点/` | 13 个 CSV | K01—K08 |

**未纳入**（体积原因）：`84b_cascade_rounds.csv`（16 MB）、`86b_null_cascade_rounds.csv`（35 MB）、`86a_null_policy_results.csv`（3.7 MB）等逐轮大表。摘要已在 `84a` / `86a` / `88*`。

---

## 6 05_数据来源与锚点

| 文件 | 内容 |
|---|---|
| **`真实数据锚点清单.md`** | W01—W12（武汉）、E01—E05（其他地区）、K01—K02（容量参考）；逐条给值、年份、来源性质、适合用途、接入状态 |
| `README_当前有效数据.md` | 当前有效输入说明 |
| `README_数据目录总览.md` | 原数据目录结构 |
| `README_论文定版结果入口_20261015.md` | 定版结果入口 |
| `代理变量依据说明.md` | 每个代理字段的生成方式与证据性质 |
| `公开运营资料核验/` | 公开运营资料可用清单与容量口径核验 |

### 锚点使用三条规则（原文）

1. **地点匹配后，再判断变量是否匹配。** 年度完成吞吐量对应实际业务规模，通常不识别最大能力；贸易额、旅客数和排名不直接代表物流处理能力。
2. **外地数据保留外地身份。** 可用于业务机制、量级或单独外部案例。
3. **分别保存原单位，并解决模型内的单位一致性。** 吨、标箱、列、亿元、件不能在同一共享任务流里直接相加。

---

## 7 06_探针结果

| 来源 | 文件 |
|---|---|
| 承诺探针 | `commitment_grid.csv`、`commitment_summary.json`、`fixed_total_grid.csv` |
| 成本探针 | `cost_probe_grid.csv`、`cost_probe_summary.json` |
| 第一轮试跑 | `grid.csv`、`summary.json`、`case_studies.json`、`structural_checks.json`、`analytic_rule_checks.json`、`dispatch_limit_ablation.csv` |
| 第二轮 | `uncertain_capacity_grid.csv`、`uncertain_capacity_summary.json`、`uncertain_capacity_independent_checks.csv` |
| **同信息对照** | `comparison_grid.csv`、`threshold_grid.csv`、`summary.json`、`formula_verification_summary.json`、`independent_flow_checks.csv`、`individual_scenario_checks.csv`、`representative_paths.json`、`00_design.md` |
| 前瞻边界 | `lookahead_boundary_checks.csv`、`lookahead_boundary_summary.json`、`boundary_robustness_checks.csv`、`boundary_robustness_summary.json`、`audit_delivery_summary.json` |

---

## 8 数据事实速查（写论文时直接引）

| 事实 | 数值 | 位置 |
|---|---|---|
| 246 点全候选边 | 30126 对（理论 30135，缺 9） | `02_基础数据_246点/04c_..._候选全边.csv` |
| 122 点全候选边 | **7381 对** | 由 30126 按 122 节点筛选 |
| 122 点稀疏边 | **1624 边** | `01_模型输入_V7/edges.csv` |
| 246 点主图边 | 4579 边 | `02_基础数据_246点/04c_..._稀疏15pct.csv` |
| 通行时间 p50（246 全候选） | 55.78 min | — |
| 通行时间 p50（122 全候选） | 53.55 min | — |
| 通行时间 p50（122 稀疏） | 17.65 min | — |
| 通行时间 max（246 全候选） | 193.25 min = 3.22 h | — |
| 通行时间 max（122 全候选） | 170.32 min = 2.84 h | — |
| 通行时间 max（122 稀疏） | 63.03 min = 1.05 h | — |
| ≤120 min 占比（122 全候选） | 97.9% | — |
| 名义利用率 | **0.800**（122 节点全部） | `01_模型输入_V7/nodes.csv` |
| 负荷 CV | 0.194 | — |
| 模型最大节点份额 | 1.09%（均匀 0.82%） | — |
| 真实丰泰份额 | 41%（300 万件/日 ÷ 733 万件/天） | 公开报道 |
| 武汉快递业务量 | 26.74 亿件/年（寄递总量）；25.05 亿件（快递） | 2024 公报 |
| 日均 | 733 万件/天 | 26.74÷365（2024 为 366 天，须说明约定） |

---

## 9 未纳入的数据（体积原因，原工作区仍在）

| 位置 | 体积 | 说明 |
|---|---|---|
| `项目结果/` 完整 | 2.2 GB | 47 个分支的逐轮大表、图件、配置快照 |
| `历史存档/pass_data` | 580 MB | 修复前快照 |
| `outputs/` | 898 MB | 数据补强、重跑、图表处理 |
| `数据来源/` 原件 | 83 MB | PDF、网页存档、XLS/XLSX 原表 |
| `数据及代码/data/` 其余 | 约 15 MB | 05—09、13—20 等早期分支派生表 |
