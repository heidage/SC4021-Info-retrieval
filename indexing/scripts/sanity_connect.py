from urllib.request import urlopen
from urllib.parse import urlencode, quote
import json
import time

# Build the query
base_url = "http://localhost:8983/solr/mycollection/select?"

params = {
    "q": 'body:"stock options" AND type:post',
    "wt": "json",        # get results in JSON
    "rows": 10,          # limit results
    "sort": "score desc" # sort by score
}

full_url = base_url + urlencode(params, quote_via=quote)


# ------------------------- Test One Query first -------------------------
start = time.time()
response = urlopen(full_url)
data = json.loads(response.read().decode('utf-8'))
end = time.time()

print("Results:", len(data['response']['docs']))
print("Time taken: {:.3f} seconds".format(end - start))
for doc in data['response']['docs']:
    print(doc)


# ------------------------- Test Multiple Queries -------------------------
query_list = [
    'type:post',
    'type:comment',
    'body:"labor jobs"',
    'subreddit:stocks',
    'score:[30 TO *]'
]

for q in query_list:
    params = {
        "q": q,
        "wt": "json",
        "rows": 5
    }
    full_url = base_url + urlencode(params, quote_via=quote)

    start = time.time()
    response = urlopen(full_url)
    data = json.loads(response.read().decode('utf-8'))
    end = time.time()

    print(f"Query: {q}")
    print(f" - Results: {len(data['response']['docs'])}")
    print(f" - Time: {end - start:.4f}s\n")
