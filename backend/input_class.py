from pydantic import BaseModel
from typing import Optional, List, Literal
class EmbedRequest(BaseModel):
    text: str
class Params(BaseModel):
    embed_model: Optional[Literal["bge_en"]] = "bge_en"
