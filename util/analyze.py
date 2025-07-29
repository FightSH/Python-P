import json
from collections import Counter, defaultdict
import os

def load_json_data(file_path):
    """加载JSON数据文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return []
    except json.JSONDecodeError:
        print(f"JSON文件 {file_path} 格式错误")
        return []

def analyze_errors(data):
    """分析错误数据"""
    print("="*50)
    print("错误问题分析报告")
    print("="*50)

    # 统计总体信息
    total_errors = len(data)
    print(f"总错误数量: {total_errors}")

    if total_errors == 0:
        print("没有错误数据需要分析")
        return

    # 统计video_name错误次数
    video_error_count = Counter()
    question_type_error_count = Counter()

    # 按问题类型分组的详细统计
    question_type_details = defaultdict(list)

    for item in data:
        video_name = item.get('video_name', 'unknown')
        question_types = item.get('question_type', [])
        question_content = item.get('question_content', '')
        predicted_answer = item.get('predicted_answer', '')
        true_answer = item.get('true_answer', '')

        # 统计video错误次数
        video_error_count[video_name] += 1

        # 将question_type数组拼接成单个类型
        combined_question_type = ' + '.join(question_types) if question_types else 'Unknown'
        question_type_error_count[combined_question_type] += 1
        question_type_details[combined_question_type].append({
            'video_name': video_name,
            'question': question_content,
            'predicted': predicted_answer,
            'true': true_answer
        })

    # 显示video_name错误统计
    print("\n" + "="*30)
    print("视频错误次数统计 (按错误次数降序)")
    print("="*30)

    most_error_videos = video_error_count.most_common()
    for i, (video_name, count) in enumerate(most_error_videos, 1):
        print(f"{i:2d}. {video_name}: {count} 次错误")

    # 显示question_type错误统计
    print("\n" + "="*30)
    print("问题类型错误次数统计 (按错误次数降序)")
    print("="*30)

    most_error_question_types = question_type_error_count.most_common()
    for i, (q_type, count) in enumerate(most_error_question_types, 1):
        print(f"{i:2d}. {q_type}: {count} 次错误")

    # 显示最多错误的详细信息
    print("\n" + "="*40)
    print("错误最多的视频和问题类型详细信息")
    print("="*40)

    if most_error_videos:
        top_video = most_error_videos[0]
        print(f"\n错误最多的视频: {top_video[0]} ({top_video[1]} 次错误)")

        # 显示该视频的所有错误问题
        video_errors = [item for item in data if item.get('video_name') == top_video[0]]
        for j, error in enumerate(video_errors, 1):
            print(f"  {j}. 问题: {error.get('question_content', '')}")
            print(f"     类型: {', '.join(error.get('question_type', []))}")
            print(f"     预测答案: {error.get('predicted_answer', '')}")
            print(f"     正确答案: {error.get('true_answer', '')}")
            print()

    if most_error_question_types:
        top_question_type = most_error_question_types[0]
        print(f"错误最多的问题类型: {top_question_type[0]} ({top_question_type[1]} 次错误)")

        # 显示该问题类型的错误分布
        type_details = question_type_details[top_question_type[0]]
        video_count_for_type = Counter([detail['video_name'] for detail in type_details])

        print(f"  涉及的视频数量: {len(video_count_for_type)}")
        print("  各视频在此类型下的错误次数:")
        for video, count in video_count_for_type.most_common():
            print(f"    {video}: {count} 次")

    # 问题类型组合分析
    print("\n" + "="*30)
    print("问题类型组合分析")
    print("="*30)

    # 统计不同问题类型组合的出现次数
    type_combinations = Counter()
    for item in data:
        question_types = item.get('question_type', [])
        if question_types:
            combination = ' + '.join(sorted(question_types))
            type_combinations[combination] += 1

    if type_combinations:
        print("问题类型组合错误统计:")
        for combo, count in type_combinations.most_common():
            print(f"  {combo}: {count} 次错误")
    else:
        print("没有问题类型数据")

def main():
    """主函数"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, 'test_errors.json')

    print(f"正在读取文件: {json_file_path}")

    # 加载数据
    data = load_json_data(json_file_path)

    if data:
        # 分析错误
        analyze_errors(data)
    else:
        print("没有数据可供分析")

if __name__ == "__main__":
    main()
