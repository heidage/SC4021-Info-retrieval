from utils.embed_helper import EmbeddingClient
import os
from typing import Any, Dict
import numpy as np

global bge_embed
bge_embed = EmbeddingClient().get_embed()


def get_top_k(query: str, k: int = 5):
    # Get embeddings for query
    embeded_query = bge_embed.embed_query(query)

    # TODO: top k search on index and return documents
    return