import pandas as pd
import os

file_name = 'groupData.xlsx'
file_path = os.path.join('/excel/ruan_work', file_name)
print(f"开始处理文件: {file_name}")
df = pd.read_excel(file_path)
# print(df)



# 2022,2023,2024分年处理
for year in range(2022,2025):
    print(year)
    # 假设费款所属期在 df 的 '费款所属期' 列中
    yearly_df = df[df['费款所属期'] == year]
    print(f"数据 for 年份 {year}:")
    print(yearly_df)
    summary = yearly_df.groupby(['人员姓名', '证件号码'])[['个人实缴金额', '单位实缴金额', '其他缴费金额']].sum()
    # summary = yearly_df[['个人实缴金额', '单位实缴金额', '其他缴费金额']].sum()
    print(summary)
    pivot_df = pd.pivot_table(
        yearly_df,
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
    # print(pivot_df)

    merged_df = pd.merge(pivot_df, summary, on=['证件号码'], how='outer')

    # for col in ['个人实缴金额', '单位实缴金额', '其他缴费金额']:
    #     pivot_df[f'{col}_汇总'] = summary[col].values

    print("数据透视表和汇总统计:")
    print(merged_df)



    merged_df.to_excel(f'{year}.xlsx', index=True)






# # 数据透视部分
