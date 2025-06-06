# 1. 读取 JSON 文件
import json
import sys

# 命令行参数处理
if len(sys.argv) > 1:
    output_file = sys.argv[1]
else:
    output_file = '大区业务组长.txt'  # 默认输出文件���

input_file = 'test.json'  # 输入文件固定为test.json


def parse_json_with_errors(content):
    """
    修复非标准 JSON 格式（多个对象拼接）
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        content = content.strip()
        if content.endswith(','):
            content = content[:-1]
        content = content.replace('}{', '},{')
        return json.loads(f'[{content}]')


def filter_checked_leaf_nodes(nodes, checked_keys, indent_level=0, result=None):
    """
    递归过滤仅包含被��选的叶子节点（无 children 或 children 为空）
    :param nodes: 当前节点列表
    :param checked_keys: 勾选的 id 集合
    :param indent_level: 当前缩进层级
    :param result: 累积结果列表
    :return: 过滤后的文本行列表
    """
    if result is None:
        result = []

    for node in nodes:
        node_id = node.get('id')
        label = node.get('label', '')
        children = node.get('children', [])
        is_leaf = not children

        if is_leaf:
            if node_id in checked_keys:
                result.append('  ' * indent_level + label)
            continue  # Process next node in the list

        direct_checked_leaf_child_labels = []
        children_for_recursion = []
        if children:
            for child in children:
                child_node_id = child.get('id')
                child_label_text = child.get('label', '')
                child_is_leaf = not child.get('children', [])

                if child_is_leaf and child_node_id in checked_keys:
                    direct_checked_leaf_child_labels.append(child_label_text)
                else:
                    children_for_recursion.append(child)

        lines_from_recursive_calls = []
        if children_for_recursion:
            filter_checked_leaf_nodes(children_for_recursion,
                                      checked_keys,
                                      indent_level + 1,
                                      lines_from_recursive_calls)

        if direct_checked_leaf_child_labels or lines_from_recursive_calls:
            if direct_checked_leaf_child_labels:
                result.append('  ' * indent_level + f"{label}: {', '.join(direct_checked_leaf_child_labels)}")
            else:
                result.append('  ' * indent_level + label)

            if lines_from_recursive_calls:
                result.extend(lines_from_recursive_calls)

    return result


def main():
    # 读取文件内容
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析 JSON 数据（修复格式）
        try:
            data = parse_json_with_errors(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解码错误: {e}")
            exit(1)

        # 提取 checkedKeys 和权限树
        checked_keys = set(data.get('checkedKeys', []))
        tree_data = data.get('menus', [])

        # 过滤仅展示被勾选的叶子节点
        filtered_lines = filter_checked_leaf_nodes(tree_data, checked_keys)

        # 写入 TXT 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(filtered_lines))

        print(f"✅ 仅勾选权限结构已导出至 {output_file}")
    except FileNotFoundError:
        print(f"❌ 找不到输入文件: {input_file}")
        exit(1)
    except IOError as e:
        print(f"❌ 文件操作错误: {e}")
        exit(1)


if __name__ == "__main__":
    main()
