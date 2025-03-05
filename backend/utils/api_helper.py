from utils.chain_helper import ChatChain
from utils.embed_helper import EmbeddingClient
from utils.llm_helper import LLMClient
import os
from typing import Any, Dict

global bge_embed
bge_embed = EmbeddingClient().get_embed()


def complete(payload: Dict[str, Any]) -> dict:
    """
    Generate text from input prompt and fetch docs from vectorstore

    Args:
    - payload (dict): input dictionary from API request.

    Returns:
    - generated_text (dict): output text from LLM.
    """
    user_prompt = payload.get('user_prompt')
    system_prompt = payload.get('system_prompt')
    chat_model = payload.get('chat_model')
    max_tokens = payload.get('max_output_tokens')
    temperature = payload.get('temperature')

    # TODO: add text embedding and fetch relevant docs from vectorstore
    llm = LLMClient(chat_model, int(max_tokens)).get_llm()
    chain = ChatChain(llm, system_prompt).get_chain() #TODO: pass the docs as examples to chain

    response = chain.with_config(
            configurable={"llm_temperature": temperature}
        ).invoke(
            {"input": system_prompt + "/n/n" + user_prompt}
    )

    return {"result": response}