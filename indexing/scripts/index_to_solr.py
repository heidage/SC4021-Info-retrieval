import pandas as pd
import numpy as np
import json

# Load the Excel file
df = pd.read_csv("../cleaned_stock_data.csv")

# Just use the first 10 rows
#subset = df.head(10)

# Prepare the Solr-friendly JSON format
docs = []
for _, row in df.iterrows():
    if pd.isna(row['downvotes']):
        row['downvotes'] = 0
    if pd.isna(row['upvotes']):
        row['upvotes'] = 0
        
    doc = {
        "id": row['post_id'],  # Unique ID
        "datetime": row['datetime'],
        "type": row['type'],
        "subreddit": row['subreddit'],
        "title": row['title'],
        "author": row['author'],
        "url": row['url'],
        "upvotes": int(row['upvotes']),
        "downvotes": int(row['downvotes']),
        "upvote_ratio": row['upvote_ratio'],
        "body": row['body'],
        "cleaned_body": row['cleaned_body']
    }
    docs.append(doc)

# Save to JSON
with open("dataset/sample_docs.json", "w") as f:
    json.dump(docs, f, indent=2)

print(f"✅ sample_docs.json created with {len(docs)} documents.")
