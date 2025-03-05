import os
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingClient():
    """
    Instantiate embedding model service
    """
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=f"/models/bge-large-en-v1.5",
            model_kwargs={"device": "auto"},
            encode_kwargs={
                "normalize_embeddings": True,
                "precision": "binary"
            }
        )

    def get_embed(self):
        return self.embeddings