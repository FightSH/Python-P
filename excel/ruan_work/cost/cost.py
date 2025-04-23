import pandas as pd
import os


# file_path = input("请输入文件名路径：")
file_path = input("请输入文件名路径：")

# file_path = 'C:\\Users\Shen\Desktop\\202201至202412医疗保险职工缴费明细-725\\202412.xlsx'

df = pd.read_excel(file_path)

print("开始处理文件: ")

df['费款所属期'] = df['费款所属期'].astype(str)
df['证件号码'] = df['证件号码'].astype(str)

grouped_df = df.groupby(['人员姓名', '证件号码', '人员类别','费款所属期','险种'])[['个人实缴金额', '单位实缴金额', '其他缴费金额']].sum().round(2).reset_index()

# print(grouped_df)

pivot_df = pd.pivot_table(
    grouped_df,
    values=['个人实缴金额', '单位实缴金额', '其他缴费金额'],  # 需要聚合的列
    index=['人员姓名', '证件号码','人员类别'],  # 行标签D:\codeenv\upx-3.96-win64
    columns='险种',  # 列标签
    aggfunc='sum',  # 聚合函数
    fill_value=0  # 填充缺失值为 0
)

# 重置列索引
pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
pivot_df = pivot_df.reset_index()

# print("数据透视表:")
# print(pivot_df.columns)
# print(pivot_df)




pivot_df = pivot_df.drop(
    columns=['其他缴费金额_大额医疗费用补助', '其他缴费金额_职工基本医疗保险', '个人实缴金额_公务员医疗补助'])

# print(pivot_df)
new_order = [
    '人员姓名',
    '证件号码',
    '人员类别',
    '个人实缴金额_大额医疗费用补助',
     '单位实缴金额_大额医疗费用补助',
    '个人实缴金额_职工基本医疗保险',
     '单位实缴金额_职工基本医疗保险',
   '单位实缴金额_公务员医疗补助',
     '其他缴费金额_公务员医疗补助'
]
pivot_df = pivot_df[new_order]

# print(pivot_df)

pivot_df['个人'] = pivot_df['个人实缴金额_大额医疗费用补助'] + pivot_df['个人实缴金额_职工基本医疗保险']
pivot_df['单位'] = pivot_df['单位实缴金额_大额医疗费用补助']+pivot_df['单位实缴金额_职工基本医疗保险']
pivot_df['公务员'] = pivot_df['单位实缴金额_公务员医疗补助'] + pivot_df['其他缴费金额_公务员医疗补助']

# print("数据透视表和汇总统计:")
# print(merged_df)
# print(pivot_df)

# pivot_df.to_excel('2.xlsx', index=True)


new_columns = [
    ('人员姓名', ''),
    ('证件号码', ''),
    ('人员类别', ''),
    ('大额医疗费用补助', '个人实缴金额_大额医疗费用补助'),
    ('大额医疗费用补助', '单位实缴金额_大额医疗费用补助'),
    ('职工基本医疗保险', '个人实缴金额_职工基本医疗保险'),
    ('职工基本医疗保险', '单位实缴金额_职工基本医疗保险'),
    ('公务员医疗补助', '单位实缴金额_公务员医疗补助'),
    ('公务员医疗补助', '其他缴费金额_公务员医疗补助'),
    ('合计', '个人'),
    ('合计', '单位'),
    ('合计', '公务员')
]

# print(f"现有列数: {len(pivot_df.columns)}")
# print(pivot_df.columns)
# print(f"新列名数量: {len(new_columns)}")
# print(new_columns)

pivot_df.columns = pd.MultiIndex.from_tuples(new_columns)

# print(pivot_df)
pivot_df.to_excel('month.xlsx', index=True)

input('按任意键退出')



