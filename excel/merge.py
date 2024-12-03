import pandas as pd

# 读取第一个 Excel 文件
df1 = pd.read_excel("E:\Wechat files\\xwechat_files\wxid_q9sic4wz6x8u21_b257\msg\\file\\2024-12\\19-23年数据核对.xlsx")

# 读取第二个 Excel 文件
df2 = pd.read_excel('E:\Wechat files\\xwechat_files\wxid_q9sic4wz6x8u21_b257\msg\\file\\2024-12\\processed.xlsx')

# 按姓名和身份证号合并数据
merged_df = pd.merge(df1, df2, on=['身份证号'], how='outer')

# 将合并后的数据保存到新的 Excel 文件
merged_df.to_excel('merged_file.xlsx', index=False)

print("合并完成，结果已保存到 merged_file.xlsx")