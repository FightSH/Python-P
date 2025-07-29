import json
from collections import Counter
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

def simple_analyze(data):
    """简化的错误分析"""
    print("="*50)
    print("错误问题分析报告")
    print("="*50)
    
    total_errors = len(data)
    print(f"总错误数量: {total_errors}")
    
    if total_errors == 0:
        return
    
    # 统计video_name错误次数
    video_error_count = Counter()
    question_type_error_count = Counter()
    
    for item in data:
        video_name = item.get('video_name', 'unknown')
        question_types = item.get('question_type', [])
        
        video_error_count[video_name] += 1
        
        # 将question_type数组拼接成单个类型
        combined_question_type = ' + '.join(question_types) if question_types else 'Unknown'
        question_type_error_count[combined_question_type] += 1

    # 显示错误最多的前10个视频
    print("\n" + "="*30)
    print("错误最多的前10个视频:")
    print("="*30)
    most_error_videos = video_error_count.most_common(10)
    for i, (video_name, count) in enumerate(most_error_videos, 1):
        print(f"{i:2d}. {video_name}: {count} 次错误")
    
    # 显示所有问题类型错误统计
    print("\n" + "="*30)
    print("问题类型错误次数统计:")
    print("="*30)
    most_error_question_types = question_type_error_count.most_common()
    for i, (q_type, count) in enumerate(most_error_question_types, 1):
        print(f"{i:2d}. {q_type}: {count} 次错误")
    
    # 显示最多错误的详细信息
    if most_error_videos and most_error_question_types:
        print("\n" + "="*40)
        print("总结:")
        print("="*40)
        top_video = most_error_videos[0]
        top_question_type = most_error_question_types[0]
        print(f"错误最多的视频: {top_video[0]} - {top_video[1]}次错误")
        print(f"错误最多的问题类型: {top_question_type[0]} - {top_question_type[1]}次错误")

def main():
    """主函数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, 'test_errors.json')
    
    print(f"正在读取文件: {json_file_path}")
    
    data = load_json_data(json_file_path)
    
    if data:
        simple_analyze(data)
    else:
        print("没有数据可供分析")

if __name__ == "__main__":
    main()
