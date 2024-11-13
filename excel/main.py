
import os
import pandas as pd

# 常量定义
# FOLDER_PATH = 'E:\\excelFile'
YEARS = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
EXCLUDE_PERIODS = ['202311', '202312']

def initialize_dataframe():
    data = {'姓名': [], '社会保障号': []}
    for year in YEARS:
        data[f'{year}缴费'] = []
    return pd.DataFrame(data)


def process_file(file_path):
    try:
        df = pd.read_excel(file_path, skiprows=1)
        df['费款所属期'] = df['费款所属期'].astype(str)
        df['社会保障号'] = df['社会保障号'].astype(str)
        df = df[~df['费款所属期'].isin(EXCLUDE_PERIODS)]
        df['费款所属期'] = df['费款所属期'].str[:4]
        grouped_df = df.groupby(['姓名', '社会保障号', '费款所属期'])['个人缴费'].sum().reset_index()
        return grouped_df
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {e}")
        return None
def build_new_row(grouped_df):





    new_row = {'姓名': '', '社会保障号': ''}
    for year in YEARS:
        new_row[f'{year}缴费'] = 0
    for row in grouped_df.itertuples():
        new_row['姓名'] = row.姓名
        new_row['社会保障号'] = row.社会保障号
        if row.费款所属期 in YEARS:
            new_row[f'{row.费款所属期}缴费'] = row.个人缴费
    return new_row

def main():
    totalDf = initialize_dataframe()
    print("请确保文件表头中有以下字段：姓名，社会保障号，费款所属期，个人缴费等名称")
    FOLDER_PATH = input("请输入文件夹路径：")

    for file_name in os.listdir(FOLDER_PATH):
        if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            file_path = os.path.join(FOLDER_PATH, file_name)
            print(f"开始处理文件: {file_name}")
            grouped_df = process_file(file_path)

            if grouped_df is not None:
                # 使用 pivot_table 方法将数据透视
                pivot_df = grouped_df.pivot_table(index=['姓名', '社会保障号'], columns='费款所属期', values='个人缴费',
                                                  aggfunc='sum').fillna(0).reset_index()

                # 重命名列
                pivot_df.columns = ['姓名', '社会保障号'] + [f'{year}缴费' for year in pivot_df.columns[2:]]

                # 计算缴费总计
                pivot_df['缴费总计'] = pivot_df.iloc[:, 2:].sum(axis=1)

                print(pivot_df)

                # 将pivot_df 加入到totalDf 中
                totalDf = pd.concat([totalDf, pivot_df], ignore_index=True)


            # 使用 pivot_table 方法将数据透视
            # pivot_df = grouped_df.pivot_table(index=['姓名', '社会保障号'], columns='费款所属期', values='个人缴费',
            #                                   aggfunc='sum').fillna(0).reset_index()
            #
            # # 重命名列
            # pivot_df.columns = ['姓名', '社会保障号'] + [f'{year}缴费' for year in pivot_df.columns[2:]]
            #
            # # 计算缴费总计
            # pivot_df['缴费总计'] = pivot_df.iloc[:, 2:].sum(axis=1)
            #
            # print(pivot_df)

            # 将pivot_df 加入到totalDf 中


    output_file = 'processed.xlsx'
    # print(pivot_df)

    totalDf.to_excel(output_file, index=False)
    print(f"所有文件处理完毕，结果已保存到: {output_file}")

    input('按任意键退出......')

if __name__ == "__main__":
    main()

    # 重新排序列
    # pivot_df = pivot_df[['姓名', '社会保障号'] + [f'{year}缴费' for year in YEARS] + ['缴费总计']]