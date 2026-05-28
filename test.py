from max_agent import MaxAgent, get_weather

agent = MaxAgent()
agent.register_function(get_weather)

res = agent.run("输出第39位斐波那契数列的数字")
print(res)