import inspect
import json
import re

from openai import OpenAI
import os
import pydantic


def extract_json(text: str):
    """
    万能提取 LLM 返回的 JSON：
    1. 带 ```json ... ``` → 正常提取
    2. 不带 ``` → 直接提取
    3. 前后有多余文字 → 自动过滤
    4. 格式混乱 → 抓出 {}
    """
    # 第一步：去掉代码块标记 ```json ... ```
    text = re.sub(r'```json\n', '', text)
    text = re.sub(r'```\n?', '', text)

    # 第二步：提取从 { 到 } 的所有内容（最稳健）
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None

    json_str = match.group(0).strip()

    # 第三步：解析 JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


class MaxAgent:
    def __init__(self):
        self.client: OpenAI = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", )
        self.memory: list = []
        self.max_steps: int = 10
        self.system_prompt: str = ''
        self.function_map = {}

    def get_func_desc(self):
        result = []
        for func_name, func in self.function_map.items():
            result.append({'name': func_name, 'doc': func.__doc__})
        return str(result)

    def run(self, task: str):
        self.system_prompt = f"""
        你是一个乐于助人的AI智能助手，在你遇见复杂的问题时，你会先将这个问题进行拆分，拆分成几个最小可执行单元，
        然后依次执行, 你有{self.get_func_desc()}这些方法可以进行调用。
        """ + """
        你只可以输出以下几种格式的回答，输出均为json格式，只要括号以及括号内的内容（重要），可以直接进行解析的格式：
        1、方法调用，格式为：
        {
        "type": "function",
        "function_name": "你需要调用的方法名称",
        "params": 你调用方法时需要传入的参数，
        }
        2、思考过程，格式为：
        {
        "type": "thinking",
        "steps":[
        "1.第一步要做什么","2.第二步要做什么","3.等等"
        ]
        }
        3、结束对话，格式为：
        {
        "type": "answer",
        "content" "你最终要输出的结果"
        }
        """
        self.memory.append({'role': 'system', 'content': self.system_prompt})
        self.memory.append({'role': 'user', 'content': task})
        print(self.memory)
        resp = self.call_llm(self.memory)
        step = 0
        while True:
            step += 1
            step_exe_result = self.step_execute(resp)
            if step_exe_result == 'continue':
                if step > self.max_steps:
                    result = '当前超过模型最大执行次数，未得出结果，请重试。'
                    break
                resp = self.call_llm(messages=self.memory)
            else:
                result = step_exe_result
                break
        return result

    def step_execute(self, resp: str) -> str:
        print(resp)
        data = extract_json(resp)
        step_type = data["type"]
        if step_type == "function":
            params_ = data["params"]
            print(f"下面我将调用{data['function_name']}方法，来获取{params_}的相关信息")
            function_result = self.execute_func(data["function_name"], params_)
            self.memory.append({"role": "assistant", "content": resp})
            self.memory.append({"role": "user", "content": f"函数调用结果为：{function_result}"})
            print(f"调用方法的结果是：{function_result}")
            return "continue"

        elif step_type == "thinking":
            steps_ = data["steps"]
            self.memory.append({"role": "assistant", "content": resp})
            print(f"下面我将开始按照以下步骤进行执行：{steps_}")
            return "continue"

        elif step_type == "answer":
            print(data["content"])
            return data["content"]

    def execute_func(self, func_name, params):
        if func_name not in self.function_map:
            return '函数不存在'

        func = self.function_map[func_name]
        try:
            result = func(params)
        except Exception as e:
            print(f'函数调用出错: {type(e).__name__} = {str(e)}')
            result = f"调用出错：{type(e).__name__} - {str(e)}"
        return result

    def call_llm(self, messages: list):
        response = self.client.chat.completions.create(model="deepseek-v4-flash",
                                                       messages=messages,
                                                       stream=False,
                                                       reasoning_effort="high",
                                                       extra_body={"thinking": {"type": "enabled"}}
                                                       )
        return response.choices[0].message.content

    def register_function(self, func):
        if callable(func):
            self.function_map[func.__name__] = func




# ... existing code ...

def get_weather(location: str) -> str:
    """
    获取指定城市的天气信息

    Args:
        location: 城市名称，例如 "Beijing", "Shanghai"

    Returns:
        天气信息的字符串描述
    """
    import requests

    # OpenWeather API 配置
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "错误：未设置 OPENWEATHER_API_KEY 环境变量"

    base_url = "https://api.openweathermap.org/data/2.5/weather"

    # 构建请求参数
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric",  # 使用摄氏度
        "lang": "zh_cn"  # 中文返回
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 解析天气数据
        city = data["name"]
        country = data["sys"]["country"]
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        weather_desc = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]

        result = (
            f"{city} ({country}) 的天气情况：\n"
            f"天气：{weather_desc}\n"
            f"温度：{temperature}°C\n"
            f"体感温度：{feels_like}°C\n"
            f"湿度：{humidity}%\n"
            f"风速：{wind_speed} m/s"
        )
        return result

    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return f"错误：找不到城市 '{location}'"
        elif response.status_code == 401:
            return "错误：API Key 无效"
        else:
            return f"错误：HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return "错误：请求超时"
    except requests.exceptions.RequestException as e:
        return f"错误：请求失败 - {str(e)}"
    except KeyError as e:
        return f"错误：解析天气数据失败 - {str(e)}"
