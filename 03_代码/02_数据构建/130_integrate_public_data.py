"""Build an immutable, executable public-data input version; never fabricate K."""
import json
from pathlib import Path
import platform
import shutil

import pandas as pd

from project_paths import DATA_DIR, SOURCES_DIR
from public_data_inputs_v1 import (DEFAULT_DIR, INPUT_FILES, LOAD, CAPACITY, sha,
                                   observation_catalog, context_links, apply_replacements,
                                   load_public_inputs)


def main():
    out = DEFAULT_DIR
    if out.exists():
        raise FileExistsError('Version already exists; validate or create a different version instead: ' + str(out))
    out.mkdir(parents=True)
    inputs = out / 'model_inputs'
    inputs.mkdir()
    protected = [DATA_DIR / name for name in INPUT_FILES]
    protected += [SOURCES_DIR / '真实数据锚点清单.md', Path(__file__),
                  Path(__file__).with_name('public_data_inputs_v1.py')]
    protected += [Path(__file__).with_name(name) for name in
                  ('closed_loop_recovery_v3.py', 'dual_mechanism_recovery_v5.py',
                   'configured_dual_mechanism_v5.py', 'model_config.py')]
    original_hashes = {str(path): sha(path) for path in protected}
    nodes = pd.read_csv(protected[0]).sort_values('node_id').reset_index(drop=True)
    if nodes.node_id.duplicated().any():
        raise ValueError('Unique node IDs required')
    catalog = observation_catalog()
    if len({v['observation_id'] for v in catalog}) != len(catalog):
        raise ValueError('Duplicate observation ID')
    links = context_links(nodes, catalog)
    lookup = {v['observation_id']: v for v in catalog}
    enriched = apply_replacements(nodes, [])
    attached = links.groupby('node_id').observation_id.agg(list).to_dict()
    payload = []
    for node in enriched.node_id:
        payload.append(json.dumps([{
            k: lookup[key][k] for k in ('observation_id', 'entity', 'metric', 'value', 'unit',
                                        'period', 'quantity_class', 'provenance')}
            for key in attached.get(node, [])], ensure_ascii=False, allow_nan=False))
    enriched['public_observations_json'] = payload
    enriched['public_context_count'] = enriched.node_id.map(lambda key: len(attached.get(key, [])))
    enriched['load_evidence_kind'] = 'existing_proxy_unchanged'
    enriched['capacity_evidence_kind'] = 'existing_proxy_unchanged'
    enriched['model_flow_unit'] = 'proxy_service_unit_per_slot'
    # Preserve source numeric text: parsing then serializing long decimals can
    # introduce a second rounding even when L/K are not intentionally changed.
    stored = pd.read_csv(protected[0], dtype=str, keep_default_na=False)
    stored = stored.set_index('node_id').loc[enriched.node_id].reset_index()
    for name in enriched.columns.difference(nodes.columns):
        stored[name] = enriched[name].to_numpy()
    stored.to_csv(inputs / INPUT_FILES[0], index=False, encoding='utf-8-sig')
    for name in INPUT_FILES[1:]:
        shutil.copy2(DATA_DIR / name, inputs / name)
    (out / 'observations.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame([{k: v for k, v in record.items() if k != 'context_patterns'} for record in catalog]).to_csv(
        out / 'observations.csv', index=False, encoding='utf-8-sig')
    links.to_csv(out / 'node_context_links.csv', index=False, encoding='utf-8-sig')
    gate = []
    for obs in catalog:
        ids = links[links.observation_id.eq(obs['observation_id'])].node_id.tolist()
        gate.append(dict(observation_id=obs['observation_id'], entity=obs['entity'],
                         value=obs['value'], unit=obs['unit'], quantity_class=obs['quantity_class'],
                         context_node_count=len(ids), context_node_ids_json=json.dumps(ids, ensure_ascii=False),
                         applied_to_load=False, applied_to_capacity=False,
                         decision='context_or_external_evidence_only', reason=obs['parameter_block_reason']))
    pd.DataFrame(gate).to_csv(out / 'parameter_import_gate.csv', index=False, encoding='utf-8-sig')
    delta = enriched[['node_id', '名称', 'family', LOAD, CAPACITY, 'public_context_count']].copy()
    delta['load_before'] = nodes[LOAD]
    delta['capacity_before'] = nodes[CAPACITY]
    delta['load_delta'] = delta[LOAD] - delta.load_before
    delta['capacity_delta'] = delta[CAPACITY] - delta.capacity_before
    delta.to_csv(out / 'node_parameter_changes.csv', index=False, encoding='utf-8-sig')
    external_cities = pd.read_csv(inputs / INPUT_FILES[2])['城市'].tolist()
    audit = dict(schema='public-input-v1', import_date='2026-10-21',
                 status='context_integrated_numeric_replacement_blocked', nodes=len(nodes),
                 observations=len(catalog), observation_anchors=len({v['anchor_id'] for v in catalog}),
                 context_links=len(links), context_nodes=links.node_id.nunique(),
                 actual_parameter_replacements=0, real_capacity_replacements=0,
                 real_demand_replacements=0, empirical_capacity_calibration=False,
                 model_units='proxy_service_unit_per_slot',
                 external_city_input=external_cities,
                 wuhan_postal_aggregate_is_not_external_city_input='武汉市' not in external_cities,
                 network_requests_this_run=0, protected_inputs=original_hashes,
                 runtime=dict(python=platform.python_version(), pandas=pd.__version__))
    report = f'''# 真实数据接入结果（2026-10-21）

状态：数据输入链路已接通；真实需求/容量替换卡在实体和单位对应条件。

本轮按用户资料登记{len(catalog)}条独立指标，覆盖{audit['observation_anchors']}项资料；没有联网复核。
现用{len(nodes)}个节点中，{audit['context_nodes']}个节点附有相关资料，形成{len(links)}条背景关联。
这不是同等数量的实测参数，也不是已证实的业务互替关系。

## 实际改了什么

- 新版节点文件增加真实资料值、原单位、年份、指标性质及来源状态的JSON字段，读取入口会载入这些字段。
- 花湖全机场量关联到两条货站记录时，保留“全机场不同于货站”的说明；没有重复赋给两个货站。
- 阳逻港数字仅作园区相关资料，不能当园区处理能力；238.1万和187.63万原值均保留，不择一或取平均。
- 东湖综保区800亿元作为2025年货值背景关联整个园区；不回填2024年处理能力或复制给行政大厅。
- 武汉邮政/快递总量留在城市观测记录中，未强塞进外部8城需求输入。
- 全部{len(nodes)}个节点的L与K保持原值，实际需求替换0项、实际容量替换0项。

## 为什么没有强行换参数

花湖资料是全机场，而原节点是国际货站/城市货站；班列数是运营系统量；盛辉、丰泰、天河与金控长贸没有对应的现用独立设施记录。
其余指标也没有提供当前模型所需的同业务、同时段服务单位转换。按类型归一化不能解决这些问题。
当前两条coldchain记录分别为鄂东（红安）农产品智慧物流园、麻城市农产品综合物流园建设项目部，不能使用金控长贸仓容。
每条指标的具体阻碍见parameter_import_gate.csv；逐节点数值前后差见node_parameter_changes.csv。

## 如何运行

```python
from public_data_inputs_v1 import load_public_inputs
bundle = load_public_inputs()
nodes, edges, city_data, cross_layer = bundle.legacy_inputs()
```

这四个输入可传给现有DemandModel、ConfiguredDispatchAllocator及ConfiguredDualMechanismRecovery；原脚本的默认数据路径未改。
要求本次必须有真实参数变化时使用`load_public_inputs(require_real_parameter_change=True)`，它会明确报错，不静默当成已标定版本。
模型仍计算代理服务单位，资料背景字段不参与LP或失效触发，不把它们包装成新的真实数据实验。

## 要让真实数值改变计算，还缺什么

至少需要将一个观测对应到现用具体设施与业务，再提供可用的需求/能力定义及与模型时段、服务单位的换算依据。
报告设计能力可另建明确标注的设计情景，但仍需要对应设施、负荷和业务相容关系；本次不新增虚构节点或跨企业派入边。
这里停在数据缺口，不以情景假设冒充实测替换。

原始数据、旧代码和V5论文未修改。模型接入检查另见运行验证报告。
'''
    (out / '数据接入结果_20261021.md').write_text(report, encoding='utf-8')
    audit['files'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'input_manifest.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    bundle = load_public_inputs(out)
    if len(bundle.nodes) != 246 or (delta[['load_delta', 'capacity_delta']].to_numpy() != 0).any():
        raise AssertionError('Unexpected core parameter mutation')
    for path, expected in original_hashes.items():
        if sha(path) != expected:
            raise AssertionError('Protected input changed: ' + path)
    print(json.dumps({k: audit[k] for k in ('status', 'nodes', 'observations', 'observation_anchors',
                                           'context_links', 'context_nodes', 'actual_parameter_replacements')},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
