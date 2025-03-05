from pydantic import BaseModel
from typing import Optional, List, Literal

# Input schema
class CompletionRequest(BaseModel):
    user_prompt: str
    stream: Optional[bool] = False # optional set to false

class EmbedRequest(BaseModel):
    text: str

class Params(BaseModel):
    system_prompt: Optional[str] = ""
    chat_model: Optional[Literal["llama-3"]] = "llama-3"
    embed_model: Optional[Literal["bge_en"]] = "bge_en"
    max_output_token: Optional[int] = 2000
    temperature: Optional[float] = 0.01
