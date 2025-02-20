import pandas as pd
import os
import sys
sys.path.append(os.apth.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.embedder import Embedder
from utils.load_data import load_csv
from alive_progress import alive_bar

# just change input_csv_file, output_csv_file and column_name to embed different dataset

def get_file_path():
    base_dir = os.path.dirname(__file__)
    input_csv_file = os.path.join(base_dir, "../data/stock_sentiments/raw_data/stock_sentiments.csv")
    output_embedding_file = os.path.join(base_dir, "../data/stock_sentiments/embeddings/bge.parquet")
    return input_csv_file, output_embedding_file

def embed_comments(description, embedder, prog_bar: alive_bar):
    prog_bar()
    return embedder.embed(description).squeeze(0).numpy()

def process_embedding(data, column_name, embedder):
    with alive_bar(data.shape[0]) as bar:
        data["embedding"] = data[column_name].map(lambda x: embed_comments(x, embedder, bar))
    
    return data

def save_embeddings(data, output_filepath):
    # extract only embedding column and save
    embeddings_df = pd.DataFrame(data['embedding'].tolist())
    embeddings_df.to_parquet(output_filepath)

def main():
    # get file paths
    input_csv_file, output_embedding_file = get_file_path()

    # Load data
    data = load_csv(input_csv_file)

    # Initialize embedder
    embedder = Embedder()

    # Process embedding
    column_name = "comments"
    data = process_embedding(data, column_name, embedder)

    # Save embeddings
    save_embeddings(data, output_embedding_file)
    return

if __name__ == "__main__":
    main()