from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional

def format_chat_prompt(sys_prompt: str, examples: Optional[List]=[]) -> ChatPromptTemplate:
    """
    Format the prompt template for chat use case with examples.

    Args:
    - examples (list): list of examples.
    - custom_prompt (str): custom system prompt provided by the user.

    Returns:
    - formatted_examples (str): formatted chat prompt with examples.
    """
    pass

class LLMChain:
    def __init__(self, llm):
        self.llm = llm

class ChatChain(LLMChain):
    def __init__(self, llm, examples: Optional[List], sys_prompt):
        super().__init__(llm)
        prompt = format_chat_prompt(sys_prompt, examples)
        self.custom_chain = (
            prompt
            | llm
            | StrOutputParser()
        )

    def get_chain(self):
        return self.custom_chain