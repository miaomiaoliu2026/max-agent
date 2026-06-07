import logging
import os

from openai import OpenAI

from rag import RAGTool

logger = logging.getLogger(__name__)


class CompanionAgent:
    def __init__(self, agent_path: str):
        self.identity_file_path = f'{agent_path}/identity.md'
        self.identity = ''
        self.client: OpenAI = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", )
        self.rag_memory: RAGTool = RAGTool(agent_path)
        self.curr_memory = []
        self.remember_items = []
        self.is_init = False
        self.search_memory_threshold = 0.4

    def chat(self, user_input: str):
        if self.is_init is False:
            self.init()

        # 检查是否结束对话
        if user_input.lower() == 'bye':
            self.save_important_info_to_memory()
            return ""

        # 添加到当前对话记忆
        self.curr_memory.append({"role": "user", "content": user_input})
        memory = self.search_memory(user_input)
        if len(memory) != 0:
            self.curr_memory.append({"role": "system", "content": f"你对这段信息相关的记忆有这些：{str(memory)}"})

        # 调用 LLM 获取回复
        resp = self.call_llm(self.curr_memory)

        logger.info(resp)
        # 将 AI 回复加入记忆
        self.curr_memory.append({"role": "assistant", "content": resp})

        return resp

    def save_important_info_to_memory(self):
        import json
            
        self.remember_items.append({"role": "system",
                                    "content": "你是一个记忆提取助手。请仔细阅读以下对话内容,提取并总结其中值得记忆的重要信息。要求："
                                               "1. 只输出总结的要点,不要输出对话原文"
                                               "2. 用简洁的语言归纳关键信息"
                                               "3. 不要包含任何role=system的系统信息"
                                               "4. 输出的内容需要是原子性的,格式为JSON数组。一句话只能包含一个信息,不要一句话包含很多信息。"
                                               "5. 如果没有需要记忆的,就返回空数组[]"
                                               "6. 必须输出标准JSON格式,例如: [\"用户喜欢编程\", \"用户住在上海\"]"})
        non_system_messages = [msg for msg in self.curr_memory if msg.get('role') != 'system']
        self.remember_items.append({"role": "user", "content": f"需要提取的对话内容:{non_system_messages}"})
        remember_info = self.call_llm(self.remember_items)
            
        # 解析JSON格式的列表
        try:
            # 兼容单引号格式
            import re
            remember_info_fixed = re.sub(r"'(.*?)'", r'"\1"', remember_info)
            remember_list = json.loads(remember_info_fixed)
                
            if isinstance(remember_list, list) and len(remember_list) > 0:
                print(f'这些我记住了: {remember_list}')
                # 逐个添加原子性记忆项
                for item in remember_list:
                    if item and isinstance(item, str):
                        self.rag_memory.add(item)
            else:
                print('没有需要记忆的内容')
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f'解析记忆列表失败: {e}, 原始内容: {remember_info}')

    def search_memory(self, info):
        query = self.rag_memory.query(info)
        filtered = [
            doc for doc, distance in zip(query['documents'][0], query['distances'][0])

            if distance < self.search_memory_threshold
        ]
        return filtered

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
        if os.path.exists(self.identity_file_path):
            with open(self.identity_file_path, "r", encoding="utf-8") as f:
                self.identity = f.read()
                self.curr_memory.append({"role": "system", "content": f"这是你的身份信息，你需要严格遵守这个信息来进行聊天：{self.identity}"})
        else:
            with open(self.identity_file_path, 'w', encoding="utf-8") as f:
                f.write("")
