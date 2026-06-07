from dotenv import load_dotenv
load_dotenv()  # 这会加载 .env 文件中的变量到环境变量中

import os
# 验证一下变量是否成功加载
print(f"Tracing enabled: {os.environ.get('LANGCHAIN_TRACING_V2')}")

from companion_agent import CompanionAgent

agent = CompanionAgent("./")

print("=== 开始聊天 (输入 'bye' 结束对话并保存记忆) ===\n")

while True:
    user_input = input("你: ").strip()

    if user_input.lower() == 'bye':
        resp = agent.chat(user_input)
        print(f"AI: {resp}")
        break

    # 调用 LLM 获取回复
    try:
        resp = agent.chat(user_input)
        print(f"AI: {resp}")
    except Exception as e:
        print(f"错误: {e}\n")