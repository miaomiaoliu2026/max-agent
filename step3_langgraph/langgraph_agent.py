import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import get_weather, execute_code


class State(TypedDict):
    messages: Annotated[list, add_messages]


tools = [get_weather, execute_code]

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash"
)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def should_retry(state: State) -> str:
    last_message = state["messages"][-1]
    # 检查最后一条工具结果是否包含错误
    if hasattr(last_message, 'content'):
        content = last_message.content
        if "执行失败" in content or "执行超时" in content or "执行异常" in content:
            return "retry"
    return "continue"


tool_node = ToolNode(tools)

# d第四步： 构件图
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.set_entry_point("chatbot")

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph_builder.add_conditional_edges(
    "tools",
    should_retry,
    {
        "retry" : "chatbot",
        "continue": "chatbot"
    }
)

graph = graph_builder.compile()

while True:
    user_input = input("你: ").strip()
    if user_input == 'bye':
        break

    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(f'AI: {result["messages"][-1].content}')
