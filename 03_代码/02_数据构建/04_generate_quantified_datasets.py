import os,re,math
import numpy as np,pandas as pd
B=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));D=os.path.join(B,'data')
F1=os.path.join(D,'03a_raw_impedance.csv');F2=os.path.join(D,'公开补充数据_城市物流指标.csv');F3=os.path.join(D,'武汉都市圈物流节点_语义清洗版.csv') if os.path.exists(os.path.join(D,'武汉都市圈物流节点_语义清洗版.csv')) else os.path.join(D,'武汉物流枢纽节点数据_初稿.csv');F4=os.path.join(D,'武汉物流节点_连线权重集_含ID.csv') if os.path.exists(os.path.join(D,'武汉物流节点_连线权重集_含ID.csv')) else os.path.join(D,'武汉物流节点_连线权重集.csv');F5=os.path.join(D,'武汉各区GDP.csv')
O1=os.path.join(D,'04a_城市综合指标_含估算.csv');O2=os.path.join(D,'04b_武汉节点代理容量.csv');O3=os.path.join(D,'04c_武汉边代理属性_候选全边.csv')

def mm(s):
 s=pd.to_numeric(pd.Series(s),errors='coerce');v=s.dropna()
 if v.empty:return pd.Series([0.0]*len(s),index=s.index)
 a,b=v.min(),v.max()
 return pd.Series([1.0]*len(s),index=s.index) if math.isclose(a,b) else (s-a)/(b-a)

def est(df,col,base='GDP 亿元'):
 df[col]=pd.to_numeric(df[col],errors='coerce');df[base]=pd.to_numeric(df[base],errors='coerce')
 k=df[df[col].notna()&df[base].notna()].copy();m=[]
 if len(k)>=3 and (k[col]>0).all() and (k[base]>0).all():
  x=np.log(k[base].values);y=np.log(k[col].values);b,a=np.polyfit(x,y,1);f=lambda g:float(np.exp(a+b*np.log(float(g))));name='GDP对数回归估算'
 else:
  ratio=float((k[col]/k[base]).median()) if len(k) else 0.0;f=lambda g:float(ratio*float(g));name='GDP比例中位数估算'
 for i,row in df.iterrows():
  if pd.notna(row[col]):m.append('公开数据')
  elif pd.notna(row[base]) and float(row[base])>0: df.at[i,col]=round(f(row[base]),4);m.append(name)
  else:m.append('缺失')
 df[col+'_估算方式']=m
 return df

def road_water_freight(cp,year=2024):
 # 只汇总已确认的公路+水路吨数；全运输方式背景值和港口吞吐量不参与。
 indicators=['公路和水路货物运输量','公路货物运输量','水运货物运输量','水路货物运输量']
 selected=cp[cp['指标'].isin(indicators)&pd.to_numeric(cp['年份'],errors='coerce').eq(year)].copy()
 if not selected['单位'].astype(str).str.strip().eq('万吨').all():
  raise ValueError('城市公路水路货运量必须使用万吨，不能混入周转量或其他单位')
 selected['数值']=pd.to_numeric(selected['数值'],errors='raise')
 if selected.duplicated(['城市','指标']).any():raise ValueError('同城同年货运指标重复，不能重复汇总')
 result={}
 for city,rows in selected.groupby('城市'):
  combined=rows[rows['指标'].eq('公路和水路货物运输量')]
  road=rows[rows['指标'].eq('公路货物运输量')]
  water=rows[rows['指标'].isin(['水运货物运输量','水路货物运输量'])]
  if len(water)>1:raise ValueError(f'{city}水运分项重复')
  if not combined.empty:
   value=float(combined.iloc[0]['数值'])
   if len(road)==1 and len(water)==1 and abs(value-float(road.iloc[0]['数值'])-float(water.iloc[0]['数值']))>0.11:
    raise ValueError(f'{city}货运合计与分项不一致')
   result[city]=value
  elif len(road)==1 and len(water)==1:
   result[city]=float(road.iloc[0]['数值'])+float(water.iloc[0]['数值'])
  else:raise ValueError(f'{city}缺少公路或水路分项，不能将部分值当成合计')
 return result

def fam(t):
 t=str(t)
 # 功能词优先于地名；“新港空港”品牌名不能覆盖园区实际港口功能。
 if '武汉新港空港综合保税区阳逻港园区' in t:return 'port'
 if '跨越速运(阳逻分拨中心)' in t:return 'distribution'
 if re.search(r'机场|空港|航港|花湖',t):return 'airport'
 if re.search(r'港|码头|新港|集装箱',t):return 'port'
 if re.search(r'铁路|火车站|班列|汉欧|武铁',t):return 'rail'
 if re.search(r'综保区|保税',t):return 'bonded'
 if re.search(r'冷链|农产品',t):return 'coldchain'
 if re.search(r'分拨|分拣|转运',t):return 'distribution'
 if re.search(r'物流园|物流基地|货运中心|货运站|仓储|仓库',t):return 'park'
 return 'other'

def tw(x):return {'airport':1,'port':0.95,'rail':0.9,'bonded':0.85,'distribution':0.8,'coldchain':0.75,'park':0.7,'other':0.55}.get(x,0.55)
def hw(f,n):
 n=str(n)
 if f in ['airport','port','rail']:return 1.0
 if f=='bonded':return 0.88
 if re.search(r'京东|中通|韵达|安能|邮政|百世|跨越',n):return 0.85
 if f in ['distribution','coldchain']:return 0.78
 return 0.68 if f=='park' else 0.5

def alias(x):
 x=str(x)
 if '东西湖' in x:return '武汉临空港经济技术开发区(东西湖区)'
 if '汉南' in x:return '武汉经济技术开发区(汉南区)'
 if '东湖高新' in x:return '东湖高新技术开发区'
 return x

def risk(x):
 return {'东西湖区':0.58,'蔡甸区':0.52,'黄陂区':0.62,'新洲区':0.68,'江夏区':0.55,'洪山区':0.47,'汉阳区':0.46,'江岸区':0.44,'硚口区':0.43,'武昌区':0.42,'青山区(武汉化工区)':0.51,'汉南区':0.57,'江汉区':0.41}.get(str(x),0.45)

def syn(a,b):
 if a==b:return 0.70
 p=tuple(sorted([a,b]));d={('airport','bonded'):0.86,('airport','distribution'):0.88,('bonded','distribution'):0.82,('bonded','park'):0.76,('coldchain','distribution'):0.84,('distribution','park'):0.75,('distribution','port'):0.78,('port','rail'):0.90,('park','rail'):0.82}
 return d.get(p,0.35)

ci=pd.read_csv(F1);cp=pd.read_csv(F2)
ci['GDP 亿元']=pd.to_numeric(ci['GDP 亿元'],errors='coerce')
# 论文公式(2)使用标准化后的时间与距离。旧实现直接使用“分钟 + 0.5*公里”，
# 使量纲和城市样本范围共同进入引力分母。这里在当前城市截面内做 min-max，
# 并用 1/N 作为下界，避免最近城市同时取两个最小值时出现零阻抗奇点。
ci['行驶时间_标准化']=mm(ci['行驶时间_mins'])
ci['物理距离_标准化']=mm(ci['物理距离_km'])
_z=ci['行驶时间_标准化']+0.5*ci['物理距离_标准化']
_z_floor=1.0/max(len(ci),1)
ci['综合时空阻抗']=_z.clip(lower=_z_floor).round(6)
fr=road_water_freight(cp)
to,ev,er,rd={},{},{},{}
postal={};parcel_source={};unit_review=[]
for _,r in cp.iterrows():
 c,i,u,v=r['城市'],str(r['指标']),str(r['单位']),float(r['数值'])
 if i=='货物周转量': to[c]=to.get(c,0)+(v/10000.0 if u.strip()=='万吨公里' else v)
 elif i in ['公路货运周转量','水路货运周转量']: to[c]=to.get(c,0)+v/10000
 elif i in ['快递业务量','快递业务总量']:
  if u.strip() not in ['万件','亿件']:
   unit_review.append({'城市':c,'指标':i,'单位':u,'数值':v,'处理':'非件数单位，未混入快递量','来源URL':r.get('来源URL','')});continue
  ev[c]=v/10000 if u.strip()=='万件' else v;parcel_source[c]='公开快递件数'
 elif i=='邮政行业寄递业务量':
  if u.strip() in ['万件','亿件']:postal[c]=v/10000 if u.strip()=='万件' else v
 elif i=='快递业务收入': er[c]=v/10000 if u=='万元' else v
 elif i in ['公路总里程','公路通车里程']: rd[c]=v
for c,v in postal.items():
 if c not in ev:ev[c]=v;parcel_source[c]='公开邮政寄递件数代理（非快递专属）'
ci['货运量_万吨']=ci['城市'].map(fr);ci['货物周转量_亿吨公里']=ci['城市'].map(to);ci['快递业务量_亿件']=ci['城市'].map(ev);ci['快递件数来源口径']=ci['城市'].map(parcel_source).fillna('GDP估算');ci['快递业务收入_亿元']=ci['城市'].map(er);ci['公路里程_公里']=ci['城市'].map(rd)
ci['货运量统计口径']='2024年公路+水路货物运输量（万吨）'
pd.DataFrame(unit_review,columns=['城市','指标','单位','数值','处理','来源URL']).to_csv(os.path.join(D,'04a_city_unit_review.csv'),index=False,encoding='utf-8-sig')
for c in ['货运量_万吨','货物周转量_亿吨公里','快递业务量_亿件','快递业务收入_亿元','公路里程_公里']: ci=est(ci,c)
for c in ['货运量_万吨','货物周转量_亿吨公里','快递业务量_亿件','快递业务收入_亿元']: ci[c+'_标准化']=mm(ci[c])
ci['枢纽加成alpha']=ci['城市'].map({'鄂州市':2.02}).fillna(1.0)
ci['产业匹配度']=ci['城市'].map({'鄂州市':0.92,'孝感市':0.78,'黄冈市':0.72,'黄石市':0.84,'咸宁市':0.68,'仙桃市':0.66,'天门市':0.61,'潜江市':0.64}).fillna(0.65)
ci['需求强度指数']=(0.3*ci['货运量_万吨_标准化']+0.3*ci['货物周转量_亿吨公里_标准化']+0.2*ci['快递业务量_亿件_标准化']+0.2*ci['快递业务收入_亿元_标准化']).round(4)
ci['综合引力指数_升级版']=ci.apply(lambda r:round(r['枢纽加成alpha']*(r['GDP 亿元']**0.4)*((1+r['需求强度指数'])**0.4)*(r['产业匹配度']**0.2)/(r['综合时空阻抗']**1.5)*1000,4),axis=1)
ci.sort_values('综合引力指数_升级版',ascending=False).to_csv(O1,index=False,encoding='utf-8-sig')

n=pd.read_csv(F3);e=pd.read_csv(F4);g=pd.read_csv(F5)
e['Distance_km']=pd.to_numeric(e['Distance_km'],errors='coerce');e['Duration_mins']=pd.to_numeric(e['Duration_mins'],errors='coerce')
g['地区_alias']=g['地区'].apply(alias);g['地区生产总值']=pd.to_numeric(g['地区生产总值'],errors='coerce');gmap=g.set_index('地区_alias')['地区生产总值'].to_dict()
e['edge_strength']=1/(0.6*e['Duration_mins']+0.4*e['Distance_km'])
# 按 node_id 聚合度（同名节点不再被错误合并）；无 ID 列时回退到名称
_sid='Source_id' if 'Source_id' in e.columns else 'Source'
_tid='Target_id' if 'Target_id' in e.columns else 'Target'
deg=pd.concat([e[[_sid,'edge_strength']].rename(columns={_sid:'名称'}),e[[_tid,'edge_strength']].rename(columns={_tid:'名称'})]).groupby('名称')['edge_strength'].sum()
n['family']=(n['名称'].fillna('')+'|'+n['类型'].fillna('')).apply(fam);n['type_weight']=n['family'].apply(tw);n['hub_weight']=n.apply(lambda r:hw(r['family'],r['名称']),axis=1);n['district_gdp']=n['区县'].apply(alias).map(gmap);n['district_gdp_norm']=mm(n['district_gdp'])
# 唯一主键 node_id 需先于 degree 映射生成
n['node_id']=(n['名称'].astype(str).str.strip()+'@@'+n['城市'].astype(str).str.strip()+'@@'+n['区县'].astype(str).str.strip())
_dm=n.duplicated('node_id',keep=False)
if _dm.any():
 _seq=n.groupby('node_id').cumcount()          # 0,1,2...
 _suffix=_seq.map(lambda k:'' if k==0 else '#'+str(k+1))
 n['node_id']=n['node_id']+_suffix.astype(str)
n['weighted_degree_proxy']=n['node_id'].map(deg).fillna(0);n['weighted_degree_norm']=mm(n['weighted_degree_proxy']);n['risk_exposure']=n['区县'].apply(risk)
n['capacity_index']=(0.35*n['type_weight']+0.20*n['district_gdp_norm']+0.25*n['weighted_degree_norm']+0.20*n['hub_weight']).round(4);n['proxy_daily_load']=(50+950*n['capacity_index']).round(2);n['overload_threshold']=(1.25*n['proxy_daily_load']).round(2)
node_cols=['node_id']+['名称','城市','区县','类型','family','district_gdp','district_gdp_norm','weighted_degree_proxy','weighted_degree_norm','type_weight','hub_weight','risk_exposure','capacity_index','proxy_daily_load','overload_threshold']
n[node_cols].to_csv(O2,index=False,encoding='utf-8-sig')

node_for_edge=n[node_cols].copy()
lu_by_name=node_for_edge.sort_values(['名称','capacity_index'],ascending=[True,False]).drop_duplicates(subset=['名称'],keep='first').set_index('名称').to_dict('index')
lu=node_for_edge.set_index('node_id').to_dict('index');dn=mm(e['Distance_km']);tn=mm(e['Duration_mins']);rows=[]
_has_id=('Source_id' in e.columns) and ('Target_id' in e.columns)
for i,r in e.iterrows():
 s,t=(lu.get(r['Source_id']),lu.get(r['Target_id'])) if _has_id else (lu_by_name.get(r['Source']),lu_by_name.get(r['Target']))
 sf=s['family'] if s else 'other';tf=t['family'] if t else 'other';sc=s['capacity_index'] if s else 0.5;tc=t['capacity_index'] if t else 0.5;sr=s['risk_exposure'] if s else 0.45;tr=t['risk_exposure'] if t else 0.45;sd=s['区县'] if s else '';td=t['区县'] if t else '';cd=0 if sd==td else 1
 # 用 0.15*(1-synergy) 表示“协同越高、成本越低”。它与旧式 -0.15*synergy
 # 仅相差常数，不改变边排序，但可避免大量边被粗暴截断到 0.001、进而支配 Ew。
 sy=syn(sf,tf);rk=round(0.5*((sr+tr)/2)+0.3*cd+0.2*tn.iloc[i],4);gi=round(0.6*r['Duration_mins']+0.4*r['Distance_km'],4);cc_raw=0.35*tn.iloc[i]+0.25*dn.iloc[i]+0.25*rk+0.15*(1-sy);cc=round(max(cc_raw,0.01),4);ec=round((math.sqrt(sc*tc)/(1+cc))*100,4);rc=round(r['Distance_km']*(1+0.6*rk+0.2*cd),4);fp=round(min(0.95,max(0.05,rk*(1+0.2*cc))),4)
 rec={'Source':r['Source'],'Target':r['Target'],'Distance_km':r['Distance_km'],'Duration_mins':r['Duration_mins'],'Source_family':sf,'Target_family':tf,'Cross_district':cd,'Functional_synergy':sy,'Risk_exposure_proxy':rk,'Generalized_impedance':gi,'Composite_transport_cost':cc,'Edge_capacity_proxy':ec,'Repair_cost_proxy':rc,'Failure_probability_proxy':fp}
 if _has_id:
  rec['Source_id']=r['Source_id'];rec['Target_id']=r['Target_id']
 rows.append(rec)
_o3_cols=['Source','Target','Distance_km','Duration_mins','Source_family','Target_family','Cross_district','Functional_synergy','Risk_exposure_proxy','Generalized_impedance','Composite_transport_cost','Edge_capacity_proxy','Repair_cost_proxy','Failure_probability_proxy']
if 'Source_id' in pd.DataFrame(rows).columns:_o3_cols=pd.DataFrame(rows).columns.tolist()
pd.DataFrame(rows)[_o3_cols].to_csv(O3,index=False,encoding='utf-8-sig')

print('done')
