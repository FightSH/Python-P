import pandas as pd
import os

folder_path = 'C:\\Users\Shen\Desktop\\202201至202412医疗保险职工缴费明细-725'
output_file = 'E:\MyCode\py\pythonP\python-p\excel\\ruan_work\\total.xlsx'

combined_df = pd.DataFrame()

# 遍历文件夹中的所有Excel文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        file_path = os.path.join(folder_path, file_name)
        # 读取Excel文件
        df = pd.read_excel(file_path)
        # 将数据追加到combined_df中
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        # print(combined_df)

print(combined_df)
# 将合并后的数据保存到新的Excel文件
combined_df.to_excel(output_file, index=True)
print(f"合并完成，结果已保存到 {output_file}")
