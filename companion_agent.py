import json
import os
import logging
import re

from openai import OpenAI
from rag import RAGTool

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
    # 3. 处理字符串值中的控制字符（换行符、制表符等）
    # 将未转义的控制字符替换为转义形式
    def escape_control_chars(match):
        s = match.group(0)
        # 替换常见的控制字符
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return s
    
    # 匹配双引号内的字符串内容并转义控制字符
    json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_control_chars, json_str)

    # 第三步：解析 JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️  JSON 解析失败: {e}")
        logger.warning(f"   尝试解析的内容: {json_str[:200]}...")
        return None


class CompanionAgent:
    def __init__(self, agent_path: str):
        self.identity_file_path = f'{agent_path}/identity.md'
        self.identity = ''
        self.client: OpenAI = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", )
        self.rag_memory: RAGTool = RAGTool(agent_path)
        self.curr_memory = []
        self.remember_items = []
        self.is_init = False

    def chat(self, user_input: str):
        if self.is_init is False:
            self.init()

        # 检查是否结束对话
        if user_input.lower() == 'bye':
            self.remember_items.append({"role": "system", "content": "你是一个记忆提取助手。请仔细阅读以下对话内容，提取并总结其中值得记忆的重要信息。要求：\n1. 只输出总结的要点，不要输出对话原文\n2. 用简洁的语言归纳关键信息\n3. 不要包含任何role=system的系统信息\n4. 直接输出总结内容，不需要其他格式"})
            non_system_messages = [msg for msg in self.curr_memory if msg.get('role') != 'system']
            self.remember_items.extend(non_system_messages)
            remember_info = self.call_llm(self.remember_items)
            print(f'这些我记住了{remember_info}')
            self.rag_memory.add(remember_info)
            return ""

        # 添加到当前对话记忆
        self.curr_memory.append({"role": "user", "content": user_input})
        # 调用 LLM 获取回复
        resp = self.call_llm(self.curr_memory)
        result_json = extract_json(resp)
        if result_json['need_memory'] == 'True':
            mem = self.rag_memory.query(result_json['content'])
            if len(mem['documents'][0]) != 0:
                # 有记忆，使用记忆内容回复
                memory_content = mem['documents'][0][0]
                logger.info(f"从记忆中获取到信息: {memory_content}")
                # 将记忆内容作为上下文，重新生成回复
                enhanced_prompt = f"根据以下记忆信息回答用户问题：\n记忆内容：{memory_content}\n\n请基于以上记忆给出回复："
                self.curr_memory.append({"role": "user", "content": enhanced_prompt})
                resp = self.call_llm(self.curr_memory)
                result_json = extract_json(resp)
                logger.info(result_json['content'])
            else:
                # 没有记忆，诚实告知
                logger.info("记忆中没有相关信息")
                return "抱歉，我的记忆中没有关于这个的信息。你可以告诉我，我会记住的。"
        else:
            logger.info(result_json['content'])

        # 将 AI 回复加入记忆
        self.curr_memory.append({"role": "assistant", "content": resp})

        return result_json['content']

    def call_llm(self, messages: list):
        response = self.client.chat.completions.create(model="deepseek-v4-flash",
                                                       messages=messages,
                                                       stream=False,
                                                       reasoning_effort="high",
                                                       extra_body={"thinking": {"type": "enabled"}}
                                                       )
        return response.choices[0].message.content

    def init(self):
        self.is_init = True
        result_template = {
            "need_memory": "True/False",
            "content": "如果need_memory是False则返回对话的内容，如果need_memory为True，则返回获取用于查询memory的提示词"
        }
        if os.path.exists(self.identity_file_path):
            with open(self.identity_file_path, "r", encoding="utf-8") as f:
                self.identity = f.read()
                self.curr_memory.append({"role": "system", "content": f"这是你的身份信息，你需要严格遵守这个信息来进行聊天：{self.identity}"})
                self.curr_memory.append({"role": "system", "content": f'''【重要】你必须严格按照以下规则执行：

1. **记忆优先原则**：当用户询问任何关于TA个人信息、历史对话、偏好、经历等问题时，你 MUST 设置 need_memory="True"，从记忆中查询
2. **禁止编造**：如果记忆中没有相关信息，绝对不要编造或猜测，必须诚实告知用户"我没有相关记忆"
3. **判断标准**：以下情况必须查询记忆：
   - 用户提到"我之前说过"、"你还记得吗"、"我的xxx是什么"
   - 用户询问自己的喜好、习惯、经历、背景等个人信息
   - 用户提到之前对话中出现过的人、事、物
   - 任何需要你了解用户历史才能准确回答的问题
4. **只有以下情况可以 need_memory="False"**：
   - 通用知识问答（如天气、数学、常识等）
   - 与用户个人无关的闲聊
   - 明确的即时性问题，不需要历史信息

请严格输出标准JSON格式（属性名和字符串值都用双引号）：{json.dumps(result_template, ensure_ascii=False, indent=2)}'''})
        else:
            with open(self.identity_file_path, 'w', encoding="utf-8") as f:
                f.write("")
