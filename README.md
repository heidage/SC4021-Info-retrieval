# SC4021-Info-retrieval

Building an opinion search engine on stock sentiments (Is it good to buy stocks?)

## Getting started

1. Create models folder in backend folder: `/backend/models`
2. Download [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) into models folder
    - Make sure to install git lfs
    - Run the command `git lfs install` after you have install git lfs and then run `git clone <url>`

### Running the Application (with Automated Indexing)

**1\. Start all services using Docker Compose**

Run the following command on terminal:

```bash
docker-compose up --build
```

This command will:

-   Start the Solr server on port 8983

-   Build and start the backend service on port 8000

-   Automatically index documents into Solr using the `solr-indexer` service, as the `solr-indexer` container waits briefly for Solr to be ready and then automatically sends the documents located in `indexing/dataset/sample_docs.json` to the `mycollection` core in Solr.

✅ Make sure your Docker daemon is running before executing the above command.

You should see `✅ Indexing complete.` in your logs.

---

**2\. Verify Solr Indexing (Optional)**

After running `docker-compose up --build`, you should see log messages like:

`✅ Solr is ready. Indexing now...
...
✅ Indexing complete.`

To manually verify the indexed data:

1\.  Visit: <http://localhost:8983/solr/#/mycollection/query>

2\.  In the query box, type:

    `*:*`

3\.  Click **Execute Query**

4\.  You should see a list of indexed documents in the response panel.
