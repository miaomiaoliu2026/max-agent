from max_agent import MaxAgent, get_weather

agent = MaxAgent()
agent.register_function(get_weather)

res = agent.run("帮我查询一下Shenzhen的天气")
print(res)