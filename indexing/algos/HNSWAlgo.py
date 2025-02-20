import os
import pickle
import sys
import time
import faiss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Union, Literal, Dict, List, Any, Optional

sys.path.append("..")
import pandas as pd
import numpy as np
from algos.algo_interface import IAlgo
from utils.load_data import load_csv, load_parquet
from utils.embedder import Embedder


class HNSWAlgo(IAlgo):
    def __init__(
            self,
            dataset: Literal["stocks"],
            M: int,
            efConstruction: Optional[int] = None,
            efSearch: Optional[int] = None,
            embedding_type: str = "bge",
    ):
        """
        dataset: name of dataset folder
        embedding_type: name of embeddings to use
        M: no. of neighbours during insertion
        efConstruction: optional, number of nearest neighbours to explore during construction, default: 40
        efSearch: optional, nuber of nearest neighbours to explore during search, default: 16
        """
        self.embedding_type = embedding_type
        self.vectorizer = Embedder()
        self.d = 1024 # dimension of embeddings
        self.data_set_name = dataset
        self.efConstruction = efConstruction
        self.efSearch = efSearch

        base_dir = os.path.dirname(__file__)
        raw_file_path = os.path.join(base_dir, f"../../data/{dataset}/raw_data/{dataset}.csv")
        embedding_file_path = os.path.join(base_dir, f"../../data/{dataset}/embeddings/{embedding_type}.parquet")

        # load in raw data
        self.data = load_csv(raw_file_path)

        # load in embeddings
        self.data_store_embeddings = np.vstack(load_parquet(embedding_file_path).values).astype('float32')
        self.M = M

        #fetch index
        self.index = self.__construct_graph(self.efConstruction, self.efSearch)

        self.mode = "hnsw"
        self.method = self.search

    def __construct_graph(self, efConstruction=None, efSearch=None, mL=None):
        """
        efConstruction: no. of nearest neighbours to explore during construction (optional)
        efSearch: no. of nearest neighbours to explore during search (optional)
        mL: normalization factor (optional)

        Returns the created HNSW Index
        """
        index = faiss.IndexHNSWFlat(self.d, self.M)

        if efConstruction is not None:
            index.hnsw.efConstruction = efConstruction

        index.add(self.data_store_embeddings)  # build the index

        # change efSearch after adding the data
        if efSearch is not None:
            index.hnsw.efSearch = efSearch

        print(index.hnsw.entry_point)
        print(f"Index constructed with parameters: efConstruction: {efConstruction}, efSearch: {efSearch}")
        return index
    

    def search(self, embedded_queries: List[Any], k:int):
        """
        embedded_query: single query to search for (embedded)
        k: no. of nearest neighbours to search 

        Returns (list of indices corresponding to nearest neighbours of query, list of data rows corresponding to nearest neighbours of query)
        """

        start_time = time.time()
        D, I = self.index.search(embedded_queries, k)
        end_time = time.time()
        duration = end_time - start_time

        results = self.data.iloc[I.flatten()]
        return results, duration // 1_000_000
    
    def run(self, query, k):
        return self.method(query, k)
    


if __name__ == "__main__":
    algo = HNSWAlgo("stocks", 5)
    results = algo.run("NVDIA", 5)
    print(results)