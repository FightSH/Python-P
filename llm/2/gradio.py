import os
import json
from typing import List, Dict, Tuple

import openai
import gradio as gr

# TODO: 设置你的 OPENAI API 密钥，这里以阿里云 DashScope API 为例进行演示
OPENAI_API_KEY = "sk-bb43503a64df4dedbe3178d89ff1803a"
# 不设置则默认使用环境变量
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 使用阿里云大模型API
)

# 检查是否正确设置了 API
# 如果一切正常，你将看到 "API 设置成功！！"
try:
    response = client.chat.completions.create(
            model="qwen-turbo",  # 使用阿里云 DashScope 的模型
            messages=[{'role': 'user', 'content': "测试"}],  # 设置一个简单的测试消息
            max_tokens=1,
    )
    print("API 设置成功！！")  # 输出成功信息
except Exception as e:
    print(f"API 可能有问题，请检查：{e}")  # 输出详细的错误信息


prompt = "请你帮我写一下文章的摘要"
article = "填写你的文章内容"
# 拼接成一个prompt
input_text = f"{prompt}\n{article}"

response = client.chat.completions.create(
    model="qwen-turbo",  # 使用通义千问-Turbo大模型
    messages=[{'role': 'user', 'content': input_text}], # 把拼接后的prompt传递给大模型
)

print(response.choices[0].message.content)