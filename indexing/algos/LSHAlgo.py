import os
import pickle
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Union, Literal, Dict, List, Any

sys.path.append("..")
import pandas as dd
import numpy as np
from algos.algo_interface import IAlgo
from utils.evaluation.algo_types import AlgoType
from utils.load_data import load_parquet, load_csv, load_index
from sklearn.metrics.pairwise import cosine_similarity
from utils.embedder import Embedder


class LSH(IAlgo):
    def __init__(self, dataset: Literal["stocks"], nbits, embedding: str = "bge"):
        self.nbits = nbits
        self.vectorizer = Embedder()

        self.data_set_name = dataset
        base_dir = os.path.dirname(__file__)
        raw_file_path = os.path.join(base_dir, f"../data/{dataset}/raw_data/{dataset}.csv")
        embedding_file_path = os.path.join(base_dir, f"../data/{dataset}/embeddings/{embedding}.parquet")
        lsh_indexing_file_path = os.path.join(base_dir, f"../data/{dataset}/indexing/lsh{nbits}.index")

        self.data = load_csv(raw_file_path)
        self.embeddings = np.vstack(load_parquet(embedding_file_path).values)
        self.lsh_index = load_index(lsh_indexing_file_path)

        self.mode = "lsh_similarity"
        if self.mode == "lsh_similarity":
            self.method = self.top_k_lsh_similarity

    def top_k_lsh_similarity(self, embedded_queries:List[Any], k):
        start_time = time.time_ns()
        distances, indices = self.lsh_index.search(embedded_queries, k)
        end_time = time.time_ns() 
        top_k_documents = self.data.iloc[indices.flatten()]

        duration = end_time - start_time
        return top_k_documents, duration

    def run(self, query, k):
        return self.method(query, k)
    
    def details(self) -> Dict[str, Union[str, int]]:
        return {
            "embedding": "bge",
            "mode": self.mode,
            "nbits": self.nbits
        }

    def name(self) -> AlgoType:
        return AlgoType.LSH

    def data_source(self) -> str:
        return self.data_set_name
    
if __name__ == "__main__":
    lsh = LSH("stocks", 2)
    print(lsh.top_k_lsh_similarity("NVDIA", 5))