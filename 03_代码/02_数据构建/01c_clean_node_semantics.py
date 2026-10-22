import os,re,pandas as pd

KEEP_WORDS=[r'物流园',r'物流基地',r'货运站',r'货运中心',r'分拨',r'分拣',r'转运',r'配送',r'快递',r'仓储',r'仓库',r'保税',r'综保区',r'铁路',r'班列',r'港',r'码头',r'空港',r'机场货运',r'冷链']
DROP_WORDS=[r'停车场',r'出入口',r'收费站',r'酒店',r'咖啡',r'便利店',r'医务室',r'服务中心$',r'党员群众服务中心',r'驿站',r'充电站',r'地铁站',r'公交站',r'厕所',r'餐厅',r'超市',r'办公楼',r'大厦',r'公司综合委员会',r'工会',r'展示中心']
PORTAL_WORDS=[r'空港',r'机场',r'港',r'码头',r'铁路',r'班列',r'综保区',r'保税']


def label_row(name,node_type):
 t=f'{str(name)}|{str(node_type)}'
 if any(re.search(p,t) for p in DROP_WORDS): return 'drop_noise','语义噪声或配套设施'
 if any(re.search(p,t) for p in PORTAL_WORDS): return 'keep_portal','门户/口岸/保税/铁路港节点'
 if any(re.search(p,t) for p in KEEP_WORDS): return 'keep_logistics','物流功能节点'
 return 'review_other','未匹配到明确物流语义'


if __name__=='__main__':
 print('阶段 01c: 物流节点语义清洗与标签化')
 d=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data')
 f_clean=os.path.join(d,'武汉都市圈物流节点_语义清洗版.csv')
 f_sum=os.path.join(d,'01c_cleaning_summary.csv')
 src=os.path.join(d,'武汉都市圈物流节点_完善版.csv')
 if not os.path.exists(src): src=os.path.join(d,'武汉物流枢纽节点数据_初稿.csv')
 df=pd.read_csv(src).copy()
 df=df.dropna(subset=['名称','经度(lng)','纬度(lat)']).copy()
 df[['semantic_label','semantic_reason']]=df.apply(lambda r: pd.Series(label_row(r.get('名称',''),r.get('类型',''))),axis=1)
 df['name_norm']=df['名称'].astype(str).str.strip().str.lower()
 df['coord_key']=df['经度(lng)'].astype(str).str.strip()+'|'+df['纬度(lat)'].astype(str).str.strip()
 before=len(df)
 df=df.sort_values(['semantic_label','名称']).drop_duplicates(subset=['name_norm','coord_key'],keep='first').copy()
 keep=df[df['semantic_label'].isin(['keep_portal','keep_logistics'])].copy()
 review=df[df['semantic_label']=='review_other'].copy()
 final=pd.concat([keep,review],ignore_index=True)
 final=final.drop(columns=['name_norm','coord_key'])
 final.to_csv(f_clean,index=False,encoding='utf-8-sig')
 summary=pd.DataFrame([
  {'item':'source_file','value':os.path.basename(src)},
  {'item':'raw_rows','value':before},
  {'item':'after_exact_dedup','value':len(df)},
  {'item':'keep_portal','value':int((df['semantic_label']=='keep_portal').sum())},
  {'item':'keep_logistics','value':int((df['semantic_label']=='keep_logistics').sum())},
  {'item':'review_other','value':int((df['semantic_label']=='review_other').sum())},
  {'item':'drop_noise','value':int((df['semantic_label']=='drop_noise').sum())},
  {'item':'final_output_rows','value':len(final)}
 ])
 summary.to_csv(f_sum,index=False,encoding='utf-8-sig')
 print(summary.to_string(index=False))
 print(f'\n清洗后节点文件已保存至: {f_clean}')
 print(f'清洗摘要已保存至: {f_sum}')
