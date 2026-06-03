from max_agent import MaxAgent, get_weather
from log_config import init_logger
from rag import RAGTool

init_logger()

agent = MaxAgent()
agent.register_function(get_weather)
rag_tool = RAGTool()
rag_tool.load_document("G:\py_workspace\max-agent\docs\唐诗三百首（诗文）+作者：蘅塘退士.txt",  force_reload=True)
agent.register_function(rag_tool.query)

res = agent.run("'嘿嘿嘿嘿嘿，哈哈哈哈哈。'属于哪首诗？全文是什么？")
print(res)