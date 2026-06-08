import os

from dotenv import load_dotenv

load_dotenv()
print(f"LANGCHAIN_TRACING_V2 {os.environ.get('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_API_KEY: {os.environ.get('LANGCHAIN_API_KEY')}")



from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7
from tools import get_weather

model = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash"
)

tools = [get_weather]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个有用的助手",
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": str(uuid7())}}

while True:
    user_input = input("你: ").strip()
    if user_input == 'bye':
        break

    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    print(f"AI: {result['messages'][-1].content}")
