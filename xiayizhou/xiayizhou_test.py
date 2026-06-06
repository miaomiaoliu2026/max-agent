from companion_agent import CompanionAgent

agent = CompanionAgent("./")

print("=== 开始聊天 (输入 'bye' 结束对话并保存记忆) ===\n")

while True:
    user_input = input("你: ").strip()

    if user_input.lower() == 'bye':
        print("\n正在保存对话记忆...")
        resp = agent.chat(user_input)
        print(f"AI: {resp}")
        break

    # 调用 LLM 获取回复
    try:
        resp = agent.chat(user_input)
        print(f"AI: {resp}")
    except Exception as e:
        print(f"错误: {e}\n")