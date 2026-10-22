# -*- coding: utf-8 -*-
"""
第二步补充（v2）：爬取孝感、鄂州、黄冈三市下辖县（市、区）的地区生产总值

数据源：红黑统计公报库 https://tjgb.hongheiku.com
        （汇集全国省市县国民经济和社会发展统计公报原文，可溯源到官方发布）

策略：
  1) 区县标签页 /tag/<区县名>  →  解析出该区县历年统计公报文章链接
  2) 优先取 2024 年（与论文主数据年份对齐），无则取最近年份
  3) 在公报正文中正则抽取「地区生产总值 / GDP ... X 亿元」
  4) 抓取不到的区县保留 None，写入日志

输出：
  data_fixed/县市区GDP_爬取结果.csv    区县 -> GDP 亿元 + 来源 URL + 公报年份
  data_fixed/县市区GDP_抓取日志.txt
  data_fixed/raw_downloads/<区县>_<年份>公报.txt   原文留档（便于复核）
"""
import os
import re
import time
import ssl
import urllib.request
import urllib.parse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'data_fixed')
RAW = os.path.join(OUT, 'raw_downloads')
os.makedirs(OUT, exist_ok=True)
os.makedirs(RAW, exist_ok=True)

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

HDR = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

LOG = []
SITE = 'https://tjgb.hongheiku.com'
PREF_YEAR = 2024          # 优先年份（与论文主数据年份对齐）

# 目标区县：论文节点表中出现的非武汉区县
TARGETS = {
    '孝感市': ['孝南区', '孝昌县', '大悟县', '应城市', '安陆市', '云梦县', '汉川市'],
    '鄂州市': ['鄂城区', '华容区', '梁子湖区'],
    '黄冈市': ['黄州区', '团风县', '红安县', '浠水县', '蕲春县', '武穴市',
               '英山县', '麻城市'],
}

# 人工兜底：聚合站正文缺失时，从官方渠道取得的权威值（附来源）
# 均为 2024 年数据，单位：亿元
MANUAL = {
    '安陆市': {
        'value': 368.00,
        'year': 2024,
        'source': 'http://www.anlu.gov.cn/u/cms/anlu/202505/14161540nkoh.pdf',
        'note': '安陆市统计局官方 PDF：《安陆市2024年国民经济和社会发展统计公报》',
    },
}


def log(m):
    print(m)
    LOG.append(str(m))


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers=HDR)
    raw = urllib.request.urlopen(req, timeout=timeout, context=CTX).read()
    for enc in ('utf-8', 'gbk', 'gb18030'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', 'ignore')


def html2text(html):
    txt = re.sub(r'<script[\s\S]*?</script>', ' ', html, flags=re.I)
    txt = re.sub(r'<style[\s\S]*?</style>', ' ', txt, flags=re.I)
    txt = re.sub(r'<[^>]+>', ' ', txt)
    txt = txt.replace('&nbsp;', ' ').replace('&amp;', '&')
    txt = txt.replace('&#12290;', '。').replace('&quot;', '"')
    txt = re.sub(r'[ \t\u3000]+', ' ', txt)
    txt = re.sub(r'\n\s*\n+', '\n', txt)
    return txt


def find_articles(html, county):
    """从标签页解析出该区县的统计公报文章 [(年份, 标题, url), ...]

    站点的文章链接形如：
      <a href="https://.../xjtjgb/xj2020/63047.html"
         title="汉川市2024年国民经济和社会发展统计公报-红黑统计公报库">
    标题信息在 title 属性里，而非 <a> 的内层文本。
    """
    out = []
    for m in re.finditer(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>', html):
        tag = m.group(0)
        href = m.group(1)
        tm = re.search(r'title=["\']([^"\']+)["\']', tag)
        title = tm.group(1).strip() if tm else ''
        if '统计公报' not in title:
            continue
        ym = re.search(r'(20\d{2})\s*年', title)
        yr = int(ym.group(1)) if ym else 0
        full = href if href.startswith('http') else urllib.parse.urljoin(SITE, href)
        out.append((yr, title, full))
    # 去重
    seen, uniq = set(), []
    for it in out:
        if it[2] in seen:
            continue
        seen.add(it[2])
        uniq.append(it)
    return uniq


def _clean_num(s):
    """清理数字串：去空格、全角点、脚注标记等，返回 float 或 None"""
    if s is None:
        return None
    s = s.replace(' ', '').replace('\u3000', '').replace('．', '.').replace('，', '')
    try:
        return float(s)
    except ValueError:
        return None


def extract_gdp(text):
    """在公报正文抽取地区生产总值，统一换算为亿元。

    需兼容的多种写法（已实测）：
      1) 全市地区生产总值为 940.05 亿元
      2) 全县实现地区生产总值（GDP）3 61.56 亿元     ← 数字被空格切开
      3) 全县实现当年价生产总值 [5] 2412662 万元     ← 单位万元 + 脚注标记
      4) 全市实现地区生产总值（GDP）\n 368.00 亿元   ← 中间换行
    """
    # 去掉脚注标记 [数字]，避免夹在标签与数字之间
    t = re.sub(r'\[\d{1,2}\]', '', text)
    # 先规整数字内部空格：将 "3 61.56" -> "361.56"（数字-空格-数字的连续片段）
    def _join(m):
        return m.group(0).replace(' ', '')
    t = re.sub(r'\d(?:[ ]\d)+', _join, t)

    label = r'(?:地区生产总值|当年价生产总值|生产总值|GDP)'
    # 亿元优先
    for p in [
        label + r'[^0-9]{0,30}?([0-9]{2,5}(?:\.[0-9]{1,2})?)\s*亿元',
    ]:
        m = re.search(p, t)
        if m:
            v = _clean_num(m.group(1))
            if v is not None and 30 <= v <= 8000:
                return round(v, 2)
    # 万元：需换算 /10000
    m = re.search(label + r'[^0-9]{0,30}?([0-9]{4,8}(?:\.[0-9]{1,2})?)\s*万元', t)
    if m:
        v = _clean_num(m.group(1))
        if v is not None and 300000 <= v <= 80000000:
            return round(v / 10000.0, 2)
    return None


def grab_county(county):
    rec = {
        '城市': '', '区县': county,
        '地区生产总值_亿元': None, '公报年份': '',
        '数据来源': '', '抓取状态': '未获取',
    }
    tag_url = f'{SITE}/tag/{urllib.parse.quote(county)}'
    try:
        html = fetch(tag_url)
    except Exception as e:
        log(f'    [{county}] 标签页失败: {type(e).__name__}')
        return rec

    arts = find_articles(html, county)
    log(f'    [{county}] 标签页发现公报 {len(arts)} 篇')
    if not arts:
        return rec

    # 排序：优先 PREF_YEAR，其次年份倒序
    arts.sort(key=lambda x: (x[0] != PREF_YEAR, -x[0]))

    for yr, title, url in arts[:3]:
        try:
            page = fetch(url)
        except Exception as e:
            log(f'      {yr} 抓取失败 {type(e).__name__}: {url}')
            continue
        text = html2text(page)
        v = extract_gdp(text)
        # 留档
        fn = os.path.join(RAW, f'{county}_{yr}公报.txt')
        try:
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception:
            pass
        if v is not None:
            rec.update({
                '地区生产总值_亿元': v,
                '公报年份': yr,
                '数据来源': url,
                '抓取状态': '成功',
            })
            log(f'      {yr} 命中 GDP={v} 亿元  {url}')
            return rec
        else:
            log(f'      {yr} 未匹配到 GDP  {url}')
        time.sleep(0.5)

    # 兜底：人工权威值（聚合站正文缺失时）
    if county in MANUAL:
        m = MANUAL[county]
        rec.update({
            '地区生产总值_亿元': m['value'],
            '公报年份': m['year'],
            '数据来源': m['source'],
            '抓取状态': '成功(人工核录)',
        })
        log(f'      人工核录 GDP={m["value"]} 亿元  [{m["note"]}]')
    return rec


def main():
    log('=' * 74)
    log('爬取孝感/鄂州/黄冈 下辖区县 GDP （数据源：红黑统计公报库）')
    log('=' * 74)
    rows = []
    for city, counties in TARGETS.items():
        log(f'\n[{city}]')
        for c in counties:
            r = grab_county(c)
            r['城市'] = city
            rows.append(r)
            time.sleep(0.6)

    import pandas as pd
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, '县市区GDP_爬取结果.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')

    ok = int((df['抓取状态'] == '成功').sum())
    log('')
    log('=' * 74)
    log(f'抓取完成：{ok}/{len(df)} 个区县获取到 GDP')
    log(f'输出：{path}')
    log('=' * 74)
    log(df.to_string(index=False))

    with open(os.path.join(OUT, '县市区GDP_抓取日志.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))


if __name__ == '__main__':
    main()
