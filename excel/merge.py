import pandas as pd

# 读取第一个 Excel 文件
df1 = pd.read_excel('E:\home\新建文件夹\核对表1.xlsx')

# 读取第二个 Excel 文件
df2 = pd.read_excel('E:\home\新建文件夹\基础数据表.xlsx')

# 按姓名和身份证号合并数据
merged_df = pd.merge(df1, df2, on=['姓名', '身份证号'], how='outer')

# 将合并后的数据保存到新的 Excel 文件
merged_df.to_excel('merged_file.xlsx', index=False)

print("合并完成，结果已保存到 merged_file.xlsx")