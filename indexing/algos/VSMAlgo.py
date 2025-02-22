import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Union, Literal, Dict
sys.path.append("..")
import pandas as pd
import numpy as np
from algo_interface import IAlgo
from utils.load_data import load_parquet, load_csv, load_index
from sklearn.metrics.pairwise import cosine_similarity
from utils.embedder import Embedder

class VSMAlgo(IAlgo):
    def __init__(self, dataset: Literal["stocks"]):
        self.vectorizer = Embedder()
        self.data_set_name = dataset
        base_dir = os.path.dirname(__file__)
        raw_file_path = os.path.join(base_dir, f"../data/{dataset}/raw_data/{dataset}.csv")
        embedding_file_path = os.path.join(base_dir, f"../data/{dataset}/embeddings/{embedding}.parquet")
        self.data = load_csv(raw_file_path)
        self.embeddings = np.vstack(load_parquet(embedding_file_path).values)

        self.mode = "cosine_similarity"
        self.method = self.top_k_cosine_similarity

    def top_k_cosine_similarity(self, query_embed, k: int):
        similarities = cosine_similarity(query_embed, self.embeddings)

        # get top k most similar documents
        try:
            top_k_indices = similarities.argsort()[0][-k:][::-1]
        except IndexError as e:
            print(f"IndexError: {e}. Check if 'similarities' has the expected shape and 'k' is within valid range.")
            raise

        top_k_documens = self.data.iloc[top_k_indices]
        return top_k_documens, top_k_indices
    
    def run(self, query, k: int):
        return self.method(query, k)
    
if __name__ == "__main__":
    vsm = VSMAlgo("stocks")
    print(vsm.top_k_cosine_similarity("apple", 5))