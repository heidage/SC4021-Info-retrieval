import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import faiss
import numpy as np
import pandas as pd
from utils.load_data import load_parquet, load_csv, load_index

class HNSW:
    def __init__(self, dataset, M: int) -> None:
        """
        dataset: name of dataset folder
        M: no. of neighbors during insertion
        """
        self.dataset = dataset
        self.d = 1024 # dimension of the vectors

        base_dir = os.path.dirname(__file__)
        raw_file_path = os.path.join(base_dir, f"../data/{dataset}/raw_data/{dataset}.csv")
        embedding_file_path = os.path.join(base_dir, f"../data/{dataset}/embeddings/bge.parquet")

        # load raw data
        self.data = load_csv(raw_file_path)

        # load embeddings
        self.data_store_embeddings = np.vstack(load_parquet(embedding_file_path).values).astype('float32')

        self.M = M

        # create index
        self.index = faiss.IndexHNSWFlat(self.d, self.M)

    def construct_graph(self, efConstruction=None, efSearch=None, M=None, mL=None) -> None:
        """
        efConstruction: no. of nearest neighbors to explore during construction (optional)
        efSearch: no. of nearest neighbors to explore during search (optional)
        M: no. of neighbors during insertion (optional)
        mL: normalization factor (optional)

        Returns created HNSW Index and time taken to create index in nanoseconds
        """
        start = time.time_ns()
        if efConstruction is not None:
            self.efConstruction = efConstruction
            self.index.hnsw.efConstruction = efConstruction

        if M is not None and mL is not None:
            self.index.set_default_probas(M, mL)

        self.index.add(self.data_store_embeddings) # build index

        # change efSearch after adding data
        if efSearch is not None:
            self.efSearch = efSearch
            self.index.hnsw.efSearch = efSearch
        end = time.time_ns()
        duration = end - start
        return self.index, duration

    def save_index(self):
        """
        Saves build HNSW index into index file
        """
        base_dir = os.path.dirname(__file__)
        index_file_path = os.path.join(base_dir, f"../data/{self.dataset}/indexing/hnsw_m={self.M}_c={self.efConstruction}_s={self.efSearch}.index")
        faiss.write_index(self.index, index_file_path)
        return
    
def main():
    dataset = "stock_reviews"
    m_vals = [2**i for i in range(4,10)]
    con_vals = [2**i for i in range(10)]
    search_vals = [2**i for i in range(10)]

    index_dict = {
        "m": [],
        "efConstruction": [],
        "efSearch": [],
        "time_taken": []
    }

    for m in m_vals:
        for con in con_vals:
            for search in search_vals:
                hnsw = HNSW(dataset, m)
                hnsw.index, duration = hnsw.construct_graph(efConstruction=con, efSearch=search)
                hnsw.save_index()
                index_dict["m"].append(m)
                index_dict["efConstruction"].append(con)
                index_dict["efSearch"].append(search)
                index_dict["time_taken"].append(duration)

    df = pd.DataFrame.from_dict(index_dict)
    base_dir = os.path.dirname(__file__)
    output_file = os.path.join(base_dir, f"../data/{dataset}/indexing/hnsw_construction.parquet")
    df.to_parquet(output_file)

if __name__ == "__main__":
    main()