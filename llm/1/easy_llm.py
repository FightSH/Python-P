import os
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = 'sk-bb43503a64df4dedbe3178d89ff1803a'


def get_response(messages):
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=messages
    )
    return completion


messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
# 您可以自定义设置对话轮数，当前为3
for i in range(3):
    user_input = input("请输入：")

    # 将用户问题信息添加到messages列表中，这部分等价于之前的单轮对话
    messages.append({'role': 'user', 'content': user_input})
    assistant_output = get_response(messages).choices[0].message.content

    # 将大模型的回复信息添加到messages列表中，这里是历史记录，保存上下文
    messages.append({'role': 'assistant', 'content': assistant_output})
    print(f'用户输入：{user_input}')
    print(f'模型输出：{assistant_output}')
    print('\n')

