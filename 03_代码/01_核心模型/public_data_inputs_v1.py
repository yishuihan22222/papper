"""Typed public observations alongside the V5 proxy inputs.

The catalog uses the user's 2026-10-21 submission without a new web check.
Context links are never treated as physical identity or parameter conversions.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA_DIR, RESULTS_DIR

DEFAULT_DIR = RESULTS_DIR / 'public_data_integration_20261021_v2'
INPUT_FILES = ('04b_武汉节点代理容量.csv', '04c_武汉边代理属性.csv',
               '04a_城市综合指标_含估算.csv', '04_cross_layer_edges.csv')
LOAD = 'proxy_daily_load'
CAPACITY = 'overload_threshold'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_catalog():
    rows = []

    def add(key, anchor, entity, metric, value, unit, period, kind, *, raw='', url='',
            patterns=(), relation='context_not_identity', reason='', prior=False):
        rows.append(dict(observation_id=key, anchor_id=anchor, entity=entity,
                         metric=metric, value=value, unit=unit, period=period,
                         quantity_class=kind, raw_text=raw, source_url=url,
                         provenance='prior_registered_source' if prior else 'user_supplied_no_new_verification',
                         independently_reverified_this_run=False,
                         context_patterns=list(patterns), relation=relation,
                         parameter_block_reason=reason,
                         capacity_replacement_eligible=False, demand_replacement_eligible=False))

    add('W01_cargo', 'W01', '鄂州花湖机场（全机场）', '年度货邮吞吐量', 1025000., 'tonne/year',
        '2024', 'observed_throughput', raw='102.5万吨/年', patterns=('湖北省鄂州花湖机场国际货站', '花湖机场黄冈城市货站'),
        relation='airport_total_not_cargo_station', reason='全机场量不能赋给单货站；实际吞吐量不是最大能力；吨与代理服务单位未映射')
    add('W02_passengers', 'W02', '武汉天河机场', '年度旅客吞吐量', 31406000., 'passenger/year',
        '2024', 'passenger_throughput', raw='3140.6万人次', reason='未找到对应机场节点；旅客量不是货邮处理能力')
    port_reason = '阳逻238.1万与含阳逻的武汉港187.63万统计范围尚未解释；港口吞吐量不能赋给保税园区；非能力'
    add('W03_yangluo', 'W03', '阳逻港', '年度集装箱吞吐量', 2381000., 'TEU/year',
        '2024', 'observed_throughput_scope_conflict', raw='238.1万标箱',
        patterns=('武汉新港空港综合保税区阳逻港园区', '阳逻保税区综合楼', '阳逻综保区冷库'), reason=port_reason)
    add('W04_wuhan_port', 'W04', '武汉港（用户注明含阳逻）', '年度集装箱吞吐量', 1876300., 'TEU/year',
        '2024', 'observed_throughput_scope_conflict', raw='187.63万标箱', reason=port_reason)
    add('W05_trains', 'W05', '中欧班列（武汉）系统', '年度开行量', 1008., 'train/year',
        '2024', 'system_operating_volume', raw='1008列/年', patterns=('吴家山站', '汉欧国际物流园3号库'),
        relation='operating_system_related_site', reason='班列系统量不是单站全部业务；不能给18个铁路节点重复赋值；非能力')
    add('W06_trade', 'W06', '武汉东湖综合保税区', '年度进出口货值', 80000000000., 'CNY/year',
        '2025', 'trade_value', raw='800亿元，首破', patterns=('武汉东湖综合保税区',),
        relation='same_named_complex_context_only', reason='货值不是处理流量；2025不同于2024基期；缺少服务单位换算')
    add('W07_rank', 'W07', '新港空港综保区/经开综保区', '排名跃升', None, 'ordinal_unspecified',
        '2024', 'qualitative_rank', raw='全国排名跃升，未给名次',
        patterns=('武汉经开综合保税区', '武汉空港综合保税区', '武汉新港空港综合保税区阳逻港园区'),
        reason='排名未给绝对处理量；多个园区统计边界不同')
    add('W08_postal', 'W08', '武汉市邮政行业', '年度寄递业务量', 2674000000., 'piece/year',
        '2024', 'city_operating_volume', raw='26.74亿件/年', relation='city_aggregate',
        reason='城市总量不是单设施量；当前跨层城市输入为外部8城；城市业务量不能推容量')
    add('W08_express', 'W08', '武汉市快递行业', '年度快递业务量', 2505000000., 'piece/year',
        '2024', 'city_operating_volume', raw='25.05亿件/年', relation='city_aggregate',
        reason='未分解至设施；不与寄递总量相加；不是能力或独立需求观测')
    add('W09_forecast', 'W09', '武汉盛辉处理中心', '预计日均处理量', 300000., 'piece/day',
        '2021', 'forecast_workload', raw='30万件/天', reason='未找到对应现用节点；预计量不是最大能力；来源URL未提供')
    hubei_url = 'https://m.cnhubei.com/content/2022-12/17/content_15322256.html'
    add('W10_belts', 'W10', '武汉顺丰丰泰产业园', '皮带机数', 37., 'belt', '2022', 'equipment_count',
        raw='37条皮带机', url=hubei_url, reason='现网无同名节点；设备数不能直接换算处理能力')
    add('W10_duration', 'W10', '武汉顺丰丰泰产业园', '处理时长', 120., 'minute', '2022', 'processing_duration',
        raw='2小时完成', url=hubei_url, reason='现网无同名节点；时长未给同时处理数量，不能推算件/日')
    for key, value, raw, kind in [('planned', 500000., '总仓容50万吨', 'planned_inventory_capacity'),
                                  ('built', 174000., '建成17.4万吨', 'built_inventory_capacity')]:
        add('W11_' + key, 'W11', '金控长贸国际冷链物流园', '仓容', value, 'tonne_inventory',
            '2024', kind, raw=raw, reason='现网两条coldchain节点均非该设施；库存上限不是单位时间处理能力')
    jd_patterns = ('京东亚一分拣中心', '亚一三期物流园')
    add('W12_sorting', 'W12', '京东亚洲一号武汉，期次未定', '分拣规模', None, 'piece/period_unspecified',
        '2019/2020', 'approximate_workload_unspecified_period', raw='百万级分拣', patterns=jd_patterns,
        relation='phase_and_metric_ambiguous', reason='百万级缺统计周期、业务口径及期次；不是精确能力')
    add('W12_duration', 'W12', '京东亚洲一号武汉，期次未定', '出仓时长', 20., 'minute',
        '2019/2020', 'processing_duration', raw='20分钟出仓', patterns=jd_patterns,
        relation='phase_and_metric_ambiguous', reason='出仓时长不是吞吐能力；期次和年份未对应')
    hn = 'https://www.chinapost.com.cn/newspaper/epaper/page/1/2022-11-15/02/83091668490708154.pdf'
    outside = '武汉样本外资料；不是当前设施参数；仅作对应地区业务记录'
    add('E01_import', 'E01', '河南洛阳/驻马店/信阳三中心', '进口分流占比', .46, 'fraction',
        '2022-11-01/2022-11-07', 'diversion_share', raw='46%', url=hn, reason=outside)
    add('E01_export', 'E01', '河南邮政直达邮路', '出口分流占比', .3722, 'fraction',
        '2022-11报道', 'diversion_share', raw='37.22%', url=hn, reason=outside)
    add('E01_transfer', 'E01', '郑州至洛阳调整的邮件处理任务', '日均减轻处理量', 100000., 'piece/day',
        '2022-11报道', 'redistributed_workload', raw='10万件/天', url=hn, reason=outside + '；转移任务量不是洛阳容量')
    add('E02_total', 'E02', '驻马店邮件处理中心', '16天累计处理量', 6660000., 'piece/16_day_window',
        '统计起止日未提供', 'cumulative_processed_volume', raw='16天累计666万件', reason=outside + '；原始出处仍缺')
    for key, value, unit, metric, kind, raw in [
        ('backlog', 750000., 'piece_inventory', '积压', 'backlog_inventory', '75万件积压'),
        ('daily', 75000., 'piece/day', '日均处理量', 'observed_throughput', '7.5万件/天'),
        ('recovery', 10., 'day', '清空时间', 'reported_recovery_duration', '10天清空'),
        ('shift', None, 'shift_description', '班次调整', 'staffing_change', '单班至双班')]:
        add('E03_' + key, 'E03', '乌鲁木齐邮区中心', metric, value, unit, '整组事件期间未提供', kind,
            raw=raw, reason=outside + '；整组同一事件关系保持待核状态，不能构造闭合库存账本')
    add('E04_postal', 'E04', '伊犁州邮政行业', '年度寄递业务量', 58226900., 'piece/year',
        '2023', 'region_operating_volume', raw='5822.69万件/年', reason=outside)
    add('E04_express', 'E04', '伊犁州快递行业', '年度快递业务量', 11771200., 'piece/year',
        '2023', 'region_operating_volume', raw='1177.12万件/年', reason=outside)
    add('E05_diversion', 'E05', '伊宁政府发布会覆盖对象', '累计分流量', 95000., 'piece/cumulative_window_unspecified',
        '未提供', 'cumulative_diversion', raw='9.5万件', reason=outside + '；与投递量未闭合同批同期口径')
    add('E05_delivery', 'E05', '伊宁政府发布会覆盖对象', '累计投递量', 87200., 'piece/cumulative_window_unspecified',
        '未提供', 'cumulative_delivery', raw='8.72万件', reason=outside + '；不能直接计算同批完成率')
    add('K01_max', 'K01', '武汉丰泰智能分拣中心', '报告最大处理能力', 3000000., 'piece/day',
        '2022投产前报道', 'reported_design_maximum', raw='300万件/日', prior=True,
        url='https://sw.wuhan.gov.cn/xwdt/mtbd/202202/t20220223_1927097.shtml',
        reason='设计能力参考；现网无同名设施；与代理服务单位和时段未映射，不能当实测有效K')
    add('K02_max', 'K02', '乌鲁木齐新枢纽邮件处理中心', '报告最高日处理能力', 740000., 'bag_or_piece/day',
        '2022-07报道', 'reported_maximum', raw='74万袋（件）/日', prior=True,
        url='https://www.chinapost.com.cn/cn/report/2207/2427-1.htm', reason=outside + '；袋（件）口径不可省略')
    return rows


def context_links(nodes, catalog):
    rows = []
    for obs in catalog:
        for name in obs['context_patterns']:
            for record in nodes[nodes['名称'].eq(name)].to_dict('records'):
                rows.append(dict(observation_id=obs['observation_id'], node_id=record['node_id'],
                                 node_name=record['名称'], original_family=record['family'],
                                 relation=obs['relation'], confirmed_parameter_entity_match=False))
    return pd.DataFrame(rows, columns=['observation_id', 'node_id', 'node_name', 'original_family',
                                       'relation', 'confirmed_parameter_entity_match'])


def apply_replacements(nodes, replacements):
    """Allow only already resolved, dimensionally compatible observations.

    This import produces no eligible replacements. An attempted raw physical
    overwrite fails explicitly rather than silently relabeling its units.
    """
    result = nodes.copy()
    seen = set()
    for row in replacements:
        key = (row.get('node_id'), row.get('parameter'))
        if key in seen:
            raise ValueError('Duplicate parameter replacement')
        seen.add(key)
        if row.get('parameter') not in (LOAD, CAPACITY):
            raise ValueError('Unknown model parameter')
        if row.get('model_unit') != 'proxy_service_unit_per_slot':
            raise ValueError('Physical units cannot overwrite the proxy-service model')
        required = ('entity_match_resolved', 'period_aligned', 'service_compatible',
                    'quantity_definition_aligned', 'unit_conversion_calibrated')
        if not all(row.get(field) is True for field in required):
            raise ValueError('Unresolved identity, period, service, variable or unit conversion')
        if not row.get('source_observation_id') or not row.get('conversion_evidence'):
            raise ValueError('Observation and conversion provenance are required')
        value = row.get('model_value')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError('Finite positive parameter required')
        mask = result.node_id.eq(row['node_id'])
        if mask.sum() != 1:
            raise ValueError('Replacement must identify exactly one existing node')
        result.loc[mask, row['parameter']] = float(value)
    return result


@dataclass
class PublicInputBundle:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    city_data: pd.DataFrame
    cross_layer: pd.DataFrame
    observations: list
    links: pd.DataFrame
    manifest: dict

    def legacy_inputs(self):
        return self.nodes, self.edges, self.city_data, self.cross_layer


def load_public_inputs(folder=DEFAULT_DIR, *, require_real_parameter_change=False):
    folder = Path(folder)
    manifest = json.loads((folder / 'input_manifest.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['files'].items():
        if sha(folder / rel) != expected:
            raise ValueError('Integrated input integrity mismatch: ' + rel)
    if require_real_parameter_change and manifest['actual_parameter_replacements'] == 0:
        raise ValueError('No real L/K replacement passed the entity and unit gate; context-only import')
    frames = [pd.read_csv(folder / 'model_inputs' / name) for name in INPUT_FILES]
    observations = json.loads((folder / 'observations.json').read_text(encoding='utf-8'))
    links = pd.read_csv(folder / 'node_context_links.csv')
    return PublicInputBundle(*frames, observations, links, manifest)
