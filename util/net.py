# filepath: e:\\MyCode\\python\\pythonP\\util\\net.py
import urllib.request
import json
import os
import re
import subprocess # Added for calling process_json.py

# --- 请将下面的URL列表中的URL替换为实际的API端点 ---
# "xxxxxx" 应替换为正确的域名。
# "/YOUR_FIRST_ROLE_ID" 和 "/YOUR_SECOND_ROLE_ID" 应替换为实际的角色ID或相应的路径参数。
# 例如:
urls_to_call = [
   "https://actual_domain.com/xxx/xxx/xxx/xxx/1",
   "https://actual_domain.com/xxx/xxx/xxx/xx/2"
]
# 当前使用占位符URL：
# urls_to_call = [
# ]

def fetch_url(api_url):
    """
    发送GET请求到指定的URL并返回响应内容字符串。
    """
    print(f"正在请求: {api_url}")
    try:
        headers = {"Authorization": "Bearer "}
        request_obj = urllib.request.Request(api_url, headers=headers)

        with urllib.request.urlopen(request_obj) as response:
            status_code = response.getcode()
            print(f"来自 {api_url} 的响应状态: {status_code}")

            if status_code == 200:
                content = response.read().decode('utf-8')
                # print(f"来自 {api_url} 的响应内容 (前500字符):")
                # print(content[:500] + ('...' if len(content) > 500 else ''))
                return content
            else:
                print(f"请求失败，状态码: {status_code}")
                # 尝试读取错误响应体
                error_content = response.read().decode('utf-8', errors='ignore')
                print(f"错误响应内容 (前500字符): {error_content[:500] + ('...' if len(error_content) > 500 else '')}")
                return None

    except urllib.error.HTTPError as e:
        print(f"请求 {api_url} 时发生HTTP错误: {e.code} {e.reason}")
        try:
            error_body = e.read().decode('utf-8', errors='ignore')
            print(f"错误详情 (前500字符): {error_body[:500] + ('...' if len(error_body) > 500 else '')}")
        except Exception as read_err:
            print(f"读取HTTPError响应体失败: {read_err}")
        return None
    except urllib.error.URLError as e:
        print(f"请求 {api_url} 时发生URL错误: {e.reason}")
        return None
    except Exception as e:
        print(f"请求 {api_url} 时发生意外错误: {type(e).__name__} - {e}")
        return None

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    if not isinstance(name, str):
        name = str(name)
    name = re.sub(r'[\\/*?:"<>|]',"", name) # Remove invalid characters
    name = name.replace(" ", "_") # Replace spaces with underscores
    if not name: # Handle empty string after sanitization
        name = "untitled"
    return name

# 循环调用定义的URL列表中的每个URL
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_json_filename = "test.json"
    test_json_path = os.path.join(base_dir, test_json_filename)

    output_txt_final_path = None

    if len(urls_to_call) < 2:
        print("错误: `urls_to_call` 列表需要至少两个URL。")
    else:
        role_info_url = urls_to_call[0]
        menu_data_url = urls_to_call[1]

        # 1. 从第一个接口获取 roleName
        print(f"\\n步骤1: 从 {role_info_url} 获取角色名称...")
        role_info_content = fetch_url(role_info_url)

        role_name_for_file = "未命名角色"

        if role_info_content:
            try:
                role_data = json.loads(role_info_content)
                extracted_role_name = role_data.get("data", {}).get("roleName")
                if extracted_role_name:
                    role_name_for_file = extracted_role_name
                    print(f"成功提取角色名称: '{role_name_for_file}'")
                else:
                    print(f"警告: 未能在响应 {role_info_url} 中找到 'data.roleName'。")
                    print(f"将使用默认角色名称 '{role_name_for_file}' 生成文件名。")
            except json.JSONDecodeError:
                print(f"错误: 解析来自 {role_info_url} 的响应为JSON失败。")
                print(f"将使用默认角色名称 '{role_name_for_file}' 生成文件名。")
        else:
            print(f"错误: 未能从 {role_info_url} 获取内容。")
            print(f"将使用默认角色名称 '{role_name_for_file}' 生成文件名。")

        sanitized_role_name = sanitize_filename(role_name_for_file)
        output_txt_filename_only = f"{sanitized_role_name}.txt"
        output_txt_final_path = os.path.join(base_dir, output_txt_filename_only)
        print(f"最终输出的txt文件路径将是: {output_txt_final_path}")

        # 2. 从第二个接口获取JSON并保存为 test.json
        print(f"\\n步骤2: 从 {menu_data_url} 获取菜单数据并保存到 {test_json_path}...")
        menu_data_content = fetch_url(menu_data_url)

        if menu_data_content:
            try:
                json.loads(menu_data_content) # Validate JSON structure before writing
                with open(test_json_path, 'w', encoding='utf-8') as f:
                    f.write(menu_data_content)
                print(f"菜单数据已成功保存到 {test_json_path}")

                # 3. 运行 process_json.py
                process_script_path = os.path.join(base_dir, "process_json.py")
                command_parts = [
                    "python",
                    process_script_path,
                    output_txt_final_path
                ]

                print(f"\\n步骤3: 执行 process_json.py...")
                print(f"命令: {' '.join(command_parts)}")

                try:
                    process_result = subprocess.run(command_parts, capture_output=True, text=True, check=False, cwd=base_dir, encoding='utf-8')
                    print("--- process_json.py STDOUT ---")
                    print(process_result.stdout)
                    if process_result.stderr:
                        print("--- process_json.py STDERR ---")
                        print(process_result.stderr)
                    if process_result.returncode == 0:
                        print(f"process_json.py 执行成功。输出应在: {output_txt_final_path}")
                    else:
                        print(f"process_json.py 执行失败，返回码: {process_result.returncode}")
                except FileNotFoundError:
                    print(f"错误: python 解释器或脚本 {process_script_path} 未找到。")
                except Exception as e:
                    print(f"执行 process_json.py 时发生错误: {e}")

            except json.JSONDecodeError:
                print(f"错误: 来自 {menu_data_url} 的内容不是有效的JSON。无法保存到 {test_json_path}")
                print("process_json.py 将不会被执行。")
            except IOError as e:
                print(f"错误: 写入 {test_json_path} 失败: {e}")
                print("process_json.py 将不会被执行。")
        else:
            print(f"错误: 未能从 {menu_data_url} 获取菜单数据。无法生成 {test_json_path}。")
            print("process_json.py 将不会被执行。")
