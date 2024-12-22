import pandas as pd
import os

from torch.utils.hipify.hipify_python import value

file_name = 'total.xlsx'
file_path = os.path.join('E:\MyCode\py\pythonP\python-p\excel\\ruan_work', file_name)
print(f"开始处理文件: {file_name}")
df = pd.read_excel(file_path)
print(df)

df['费款所属期'] = df['费款所属期'].astype(str)
df['证件号码'] = df['证件号码'].astype(str)
# df = df[~df['对应费款所属期'].isin(EXCLUDE_PERIODS)]
df['费款所属期'] = df['费款所属期'].str[:4]

grouped_df = df.groupby(['人员姓名', '证件号码', '人员类别','费款所属期','险种'])[['个人实缴金额', '单位实缴金额', '其他缴费金额']].sum().round(2).reset_index()

print(grouped_df)

grouped_df.to_excel('groupData.xlsx', index=True)




pivot_df = grouped_df.pivot_table(
    grouped_df,
    values=['个人实缴金额', '单位实缴金额','其他缴费金额'],  # 需要聚合的列
    index=['人员姓名', '费款所属期'],  # 行标签
    columns='险种',  # 列标签
    aggfunc='sum',  # 聚合函数
    fill_value=0  # 填充缺失值为 0
)
# 重置列索引
pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]



pivot_df = pivot_df.reset_index()

print("数据透视表:")
print(pivot_df)


# 数据透视部分
