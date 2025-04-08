import pandas as pd
import json

# Load the Excel file
df = pd.read_excel("cleaned_stock_data.xlsx")

# Just use the first 10 rows
#subset = df.head(10)

# Prepare the Solr-friendly JSON format
docs = []
for _, row in df.iterrows():
    doc = {
        "id": row['post_id'],  # Unique ID
        "type": row['type'],
        "subreddit": row['subreddit'],
        "title": row['title'],
        "author": row['author'],
        "url": row['url'],
        "score": int(row['score']),
        "body": row['body'],
        "cleaned_body": row['cleaned_body']
    }
    docs.append(doc)

# Save to JSON
with open("sample_docs.json", "w") as f:
    json.dump(docs, f, indent=2)

print(f"✅ sample_docs.json created with {len(docs)} documents.")
