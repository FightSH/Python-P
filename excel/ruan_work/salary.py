import pandas as pd
import os

# 指定Excel文件的路径
# file_path = input("请输入文件名路径：")
file_path = 'C:\\Users\Shen\Desktop\\ruanwork\\副本19-23代扣数.xlsx'

# 使用pd.ExcelFile读取Excel文件
xls = pd.ExcelFile(file_path)

# 创建一个空的DataFrame用于汇总数据
combined_df = pd.DataFrame()

# 遍历每个sheet并将其内容读取到DataFrame中
for sheet_name in xls.sheet_names:
    # 读取当前sheet的内容
    print(f"正在读取{sheet_name}内容")
    df = pd.read_excel(xls, sheet_name=sheet_name)
    # 对每个数据项去除空格
    df['人员编号'] = df['人员编号'].fillna(-1).astype(float).astype(int).astype(str)
    df['姓名'] = df['姓名'].astype(str)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    print(df)
    # 将当前sheet的内容追加到combined_df中
    combined_df = pd.concat([combined_df, df], ignore_index=True)





# 打印combined_df的表头内容
print("汇总后的表头内容：", combined_df.columns.tolist())
print(combined_df)


# 让用户输入想要作为的列
count_columns = input("请输入想要统计(即要汇总的)的列名（用英文逗号分隔）比如 姓名,人员编号：").split(',')
# count_columns = ['姓名']
# 让用户输入想要聚合的列
groupBy_columns = input("请输入想要聚合(即要汇总的)的列名（用英文逗号分隔）：").split(',')

# 去除列名中的空格
groupBy_columns = [col.strip() for col in groupBy_columns]

# 检查输入的列名是否存在于DataFrame中
invalid_columns = [col for col in groupBy_columns if col not in combined_df.columns]
if invalid_columns:
    print(f"以下列名不存在于DataFrame中：{invalid_columns}")
else:
    # 进行聚合操作
    aggregated_df = combined_df.groupby(count_columns)[groupBy_columns].sum().reset_index()
    print(aggregated_df)
    # 将合并后的数据保存到新的Excel文件
    aggregated_df.to_excel("salary_sum.xlsx", index=True)

input('按任意键退出......')