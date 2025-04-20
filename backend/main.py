import logging

from fastapi import FastAPI, Response, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from response_model import QueryResponse, QueryRequest

from utils.api_helper import get_results, convert_to_query_response

app = FastAPI(title="Backend for stocks opinion analysis")

# add logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

### handler ###
handler = logging.StreamHandler()
logger.addHandler(handler)

### formatter ###
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# this endpoint is to test if server is up and working
@app.get("/ping")
async def ping():
    return Response(content="pong", status_code=200)

@app.post("/api/query", response_model=QueryResponse)
async def query(query_request: QueryRequest = Body(...)) -> QueryResponse:
    query = query_request.query
    platforms = query_request.subreddits
    start_date = query_request.start_date

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Query: {query}")
    try:
        # send query to solr and get relevant matches
        results, keywords = get_results(query, platforms, start_date)

        # convert solr response to query response
        recordCount, subreddits, comments = convert_to_query_response(results)

        return QueryResponse(
            sentiment="positive",
            comments=comments,
            keywords=keywords,
            subreddits=subreddits,
            recordCount=recordCount,
        )
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")