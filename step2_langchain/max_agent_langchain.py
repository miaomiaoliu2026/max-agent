import os

from langchain_openai import ChatOpenAI


class MaxAgentLangchain:
    def __init__(self, base_url, model_type):
        self.base_url = base_url
        self.model_type = model_type
        self.llm = ChatOpenAI(model=model_type, api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=base_url)
