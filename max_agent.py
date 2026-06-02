import inspect
import json
import re

from openai import OpenAI
import os
import pydantic
import subprocess
import logging

logger = logging.getLogger(__name__)


def extract_json(text: str):
    """
    万能提取 LLM 返回的 JSON：
    1. 带 ```json ... ``` → 正常提取
    2. 不带 ``` → 直接提取
    3. 前后有多余文字 → 自动过滤
    4. 格式混乱 → 抓出 {}
    """
    if not text or not text.strip():
        return None

    # 第一步：尝试提取 ```json ... ``` 代码块
    json_block_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
    if json_block_match:
        text = json_block_match.group(1)
    else:
        # 去掉所有 ``` 标记
        text = re.sub(r'```json\n', '', text)
        text = re.sub(r'```\n?', '', text)

    # 第二步：提取从 { 到 } 的所有内容（最稳健）
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        logger.warning(f"⚠️  未找到 JSON 对象,原始内容: {text[:200]}...")
        return None

    json_str = match.group(0).strip()

    # 尝试修复常见的 JSON 错误
    # 1. 移除尾部逗号 (如 [1,2,] -> [1,2])
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    # 2. 确保属性名用双引号
    json_str = re.sub(r"'(\w+)':", r'"\1":', json_str)

    # 第三步：解析 JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️  JSON 解析失败: {e}")
        logger.warning(f"   尝试解析的内容: {json_str[:200]}...")
        return None


class MaxAgent:
    def __init__(self):
        self.client: OpenAI = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", )
        self.memory: list = []
        self.max_steps: int = 10
        self.system_prompt: str = ''
        self.function_map = {}
        self.question_map = {}

    def get_func_desc(self):
        result = []
        for func_name, func in self.function_map.items():
            result.append({'name': func_name, 'doc': func.__doc__})
        return str(result)

    def run(self, task: str):
        logger.info(f"用户提了一个问题：{task}")
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
        4、执行代码，格式为：
        {
        "type": "execute_code",
        "code" "要执行的python代码"
        }
        """
        self.memory.append({'role': 'system', 'content': self.system_prompt})
        self.memory.append({'role': 'user', 'content': task})
        resp = self.call_llm(self.memory)
        step = 0
        while True:
            step += 1
            step_exe_result = self.step_execute(task, resp)
            if step_exe_result == 'continue':
                if step > self.max_steps:
                    result = '当前超过模型最大执行次数，未得出结果，请重试。'
                    break
                resp = self.call_llm(messages=self.memory)
            else:
                result = step_exe_result
                break
        return result

    def supervisor(self, msgs: list[dict]):
        """
        对当前的结果进行决策
        :param msgs:
        :return:
        """
        supervisor_prompt = {
            "passed": "True/False",
            "reason": "返回不合理的原因。passed为True时不需要返回",
            "action": "continue/retry/abort"
        }
        msgs.append({
            'role': "system",
            'content': f'''判断当前节点的处理结果是否正确。
        你必须返回标准 JSON 格式（属性名和字符串值都用双引号）：
        {json.dumps(supervisor_prompt, ensure_ascii=False, indent=2)}
        '''
        })
        checked_result = self.call_llm(msgs)
        data = extract_json(checked_result)
        if data is None:
            return {"passed": "False", "reason": "检查失败", "action": "retry"}
        return data

    def step_execute(self, ori_task: str, resp: str) -> str:
        data = extract_json(resp)

        # 检查 JSON 解析是否成功
        if data is None:
            error_msg = "❌ 无法解析 AI 返回的内容为 JSON 格式"
            logger.error(error_msg)
            self.memory.append({"role": "assistant", "content": resp})
            self.memory.append({"role": "user", "content": f"错误: {error_msg},请重新生成有效的JSON响应"})
            return "continue"

        # 检查 type 字段是否存在
        if "type" not in data:
            error_msg = f"❌ JSON 中缺少 'type' 字段,收到的数据: {data}"
            logger.error(error_msg)
            self.memory.append({"role": "assistant", "content": resp})
            self.memory.append({"role": "user", "content": f"错误: {error_msg},请重新生成有效的JSON响应"})
            return "continue"

        step_type = data["type"]
        if step_type == "function":
            params_ = data["params"]
            logger.info(f"下面我将调用{data['function_name']}方法，来获取{params_}的相关信息")
            function_result = self.execute_func(data["function_name"], params_)

            supervisor_params = [
                {"role": "user", "content": f'原始问题为：{ori_task}'},
                {"role": "system", "content": f"当前这一轮的任务是：{resp}"},
                {"role": "system", "content": f"当前这一论的结果是： {function_result}"}
            ]
            supervisor_result = self.supervisor(supervisor_params)
            if supervisor_result['passed'] != 'True':
                reason = supervisor_result['reason']
                action = supervisor_result['action']
                logger.info(f"supervisor判断当前结果不合理，原因为：{reason}")
                if action == 'continue':
                    self.memory.append({"role": "assistant", "content": resp})
                    self.memory.append({"role": "user", "content": f"函数调用结果为：{function_result}"})
                    return "continue"
                elif action == 'retry':
                    self.memory.append({"role": "assistant", "content": resp})
                    self.memory.append({"role": "user", "content": f"{reason }"})
                else:
                    return "abort"

            else:
                self.memory.append({"role": "assistant", "content": resp})
                self.memory.append({"role": "user", "content": f"函数调用结果为：{function_result}"})
                logger.info(f"调用方法的结果是：{function_result}")
                return "continue"

        elif step_type == "thinking":
            # 检查 steps 字段是否存在
            if "steps" not in data:
                error_msg = f"❌ thinking 类型缺少 'steps' 字段"
                logger.error(error_msg)
                self.memory.append({"role": "assistant", "content": resp})
                self.memory.append({"role": "user", "content": f"错误: {error_msg},请重新生成有效的JSON响应"})
                return "continue"

            steps_ = data["steps"]
            self.memory.append({"role": "assistant", "content": resp})
            logger.info(f"下面我将开始按照以下步骤进行执行：{steps_}")
            return "continue"

        elif step_type == "answer":
            return data["content"]

        elif step_type == 'execute_code':
            # 检查 code 字段是否存在
            if "code" not in data:
                error_msg = f"❌ execute_code 类型缺少 'code' 字段"
                logger.error(error_msg)
                self.memory.append({"role": "assistant", "content": resp})
                self.memory.append({"role": "user", "content": f"错误: {error_msg},请重新生成有效的JSON响应"})
                return "continue"
            logger.info("下面我将通过执行一段代码，来获取结果")
            code = self.execute_code(data['code'])

            supervisor_params = [
                {"role": "user", "content": f'原始问题为：{ori_task}'},
                {"role": "system", "content": f"当前这一轮的任务是：{resp}"},
                {"role": "system", "content": f"当前这一论的结果是： {code}"}
            ]
            supervisor_result = self.supervisor(supervisor_params)
            if supervisor_result['passed'] != 'True':
                reason = supervisor_result['reason']
                action = supervisor_result['action']
                logger.info(f"supervisor判断当前结果不合理，原因为：{reason}")
                if action == 'continue':
                    self.memory.append({"role": "assistant", "content": resp})
                    self.memory.append({"role": "user", "content": f"函数调用结果为：{code}"})
                    return "continue"
                elif action == 'retry':
                    self.memory.append({"role": "assistant", "content": resp})
                    self.memory.append({"role": "user", "content": f"{reason}"})
                else:
                    return "abort"

            self.memory.append({"role": "assistant", "content": f'代码执行结果为{code}'})
            return "continue"

    def execute_func(self, func_name, params):
        if func_name not in self.function_map:
            return '函数不存在'

        func = self.function_map[func_name]
        try:
            result = func(params)
        except Exception as e:
            logger.error(f'函数调用出错: {type(e).__name__} = {str(e)}')
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

    def execute_code(self, code: str):
        # 显示将要执行的代码并询问用户
        logger.info("\n" + "=" * 60)
        logger.info("⚠️  即将执行以下代码:")
        logger.info("=" * 60)
        logger.info(code)
        logger.info("=" * 60)

        # 询问用户是否继续执行
        user_input = input("是否执行此代码? (y/n): ").strip().lower()

        if user_input not in ['y', 'yes']:
            logger.info("❌ 用户取消执行")
            return "用户取消了代码执行"

        logger.info("✅ 开始执行代码...")
        try:
            with open("temp.py", "w", encoding="utf-8") as f:
                f.write(code)
            run = subprocess.run(['python', 'temp.py'], capture_output=True, text=True, timeout=30)

            # 检查是否有错误
            if run.returncode != 0:
                logger.info(f"❌ 代码执行失败,返回码: {run.returncode}")
                if run.stderr:
                    logger.info(f"错误信息: {run.stderr}")
                return f"执行失败:\n{run.stderr}"

            logger.info("✅ 代码执行成功")
            return run.stdout
        except subprocess.TimeoutExpired:
            logger.error("❌ 代码执行超时(超过30秒)")
            return "执行超时:代码运行时间过长"
        except Exception as e:
            logger.error(f"❌ 执行出错: {type(e).__name__} - {str(e)}")
            return f"执行异常: {type(e).__name__} - {str(e)}"


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
