import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import faiss
import pandas as pd
import numpy as np
from utils.load_data import load_parquet

def get_input_file_path(dataset: str) -> str:
    base_dir = os.path.dirname(__file__)
    input_embedding_dir = os.path.join(base_dir, f"../data/{dataset}/embeddings/bge.parquet")
    return input_embedding_dir

def get_output_file_path(dataset: str, nbits: int) -> str:
    base_dir = os.path.dirname(__file__)
    output_index_dir = os.path.join(base_dir, f"../data/{dataset}/indexing/lsh_{nbits}.index")
    return output_index_dir

def build_index(embeddings, dims: int, nbits:int) -> faiss.IndexLSH:
    # create and configure the index
    index = faiss.IndexLSH(dims, nbits)
    index.add(embeddings)
    return index

def save_index(index, index_file_path: str):
    faiss.write_index(index, index_file_path)

def main():
    dataset = "stock_reviews"
    # get file paths
    input_embedding_file = get_input_file_path(dataset)

    embeddings = np.vstack(load_parquet(input_embedding_file).values)
    dims = embeddings.shape[1]

    index_dict = {
        "nbits": [],
        "time_taken": []
    }

    for nbits in [2,4,8,16,32,64,128,256,512]:
        # build the index
        start = time.time_ns()
        index = build_index(embeddings, dims, nbits)
        end = time.time_ns()
        time_taken = end - start

        # save index
        output_index_file = get_output_file_path(dataset, nbits)
        save_index(index, output_index_file)
        index_dict["nbits"].append(nbits)
        index_dict["time_taken"].append(time_taken)

    base_dir = os.path.dirname(__file__)
    output_file = os.path.join(base_dir, f"../data/{dataset}/indexing/lsh_construction.parquet")
    df = pd.DataFrame.from_dict(index_dict)
    df.to_parquet(output_file)

if __name__ == "__main__":
    main()