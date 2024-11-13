import pandas as pd
import openpyxl


def read_and_print_excel(file_path):
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path, engine='openpyxl')

        # 打印每一行
        for index, row in df.iterrows():
            print(row)
    except Exception as e:
        print(f"发生错误: {e}")


# def read_and_aggregate_excel(file_path, columns_to_sum):
#     try:
#         # 读取Excel文件
#         df = pd.read_excel(file_path, engine='openpyxl')
#
#         # 检查指定的列是否存在于DataFrame中
#         missing_columns = [col for col in columns_to_sum if col not in df.columns]
#         if missing_columns:
#             print(f"以下列不存在于Excel文件中: {missing_columns}")
#             return
#
#         # 设置pandas显示选项，不使用科学计数法
#         pd.set_option('display.float_format', '{:.2f}'.format)
#         # 按保函类型和区域类型进行分组，并计算指定列的汇总值
#         # grouped_df = df.groupby(['产品名称','保函类型', '区域名称'])[columns_to_sum].sum()
#
#         grouped_df = df.groupby(['产品名称', '保函类型', '区域名称']).agg({
#             **{col: 'sum' for col in columns_to_sum},
#             '保函编号': 'count'
#         }).rename(columns={'保函编号': '完成订单'})
#
#         # 打印汇总结果
#         print(grouped_df)
#     except Exception as e:
#         print(f"发生错误: {e}")


# %%



def read_and_aggregate_excel(file_path, output_file_path):
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path, engine='openpyxl')

        # 设置pandas显示选项，不使用科学计数法
        pd.set_option('display.float_format', '{:.2f}'.format)

        # 处理产品名称，将线下投标单独分组，其他产品名称分在一个组里
        df['产品名称'] = df['产品名称'].apply(lambda x: '线下投标保函' if x == '线下投标保函' else '标后保函')

        # 按区域名称和营销人员进行分组
        grouped_df = df.groupby(['区域名称', '营销人员', '产品名称']).agg({
            '订单金额': 'sum',
            '保证金金额': 'sum',
            '保函编号': 'count'
        }).rename(columns={'保函编号': '完成订单'}).reset_index()

        # 计算合计数据，即不对产品名称进行分组
        total_grouped_df = df.groupby(['区域名称', '营销人员']).agg({
            '订单金额': 'sum',
            '保证金金额': 'sum',
            '保函编号': 'count'
        }).rename(columns={'保函编号': '保函数量'}).reset_index()

        # 添加合计行
        total_grouped_df['产品名称'] = '合计'

        # 合并分组数据和合计数据
        final_df = pd.concat([grouped_df, total_grouped_df], ignore_index=True)
        # 将结果保存到Excel文件
        final_df.to_excel(output_file_path, index=False, engine='openpyxl')

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 获取用户输入的Excel文件路径
    # C:\Users\Administrator\Desktop\订单列表_2024-11-7.xlsx
    file_path = input("请输入Excel文件的路径: ")
    # 保证金金额,订单金额
    # 获取用户输入的需要求和的列名，以逗号分隔
    # columns_input = input("请输入需要求和的列名，以逗号分隔: ")
    # columns_to_sum = [col.strip() for col in columns_input.split(',')]
    out_file_path = input("请输入处理后的文件的路径: ")
    # 调用函数读取并汇总Excel文件
    read_and_aggregate_excel(file_path, out_file_path)

    # 调用函数读取并打印Excel文件
    # read_and_print_excel(file_path)
