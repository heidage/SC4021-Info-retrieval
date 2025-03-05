import logging
import traceback
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from input_class import CompletionRequest, Params

app = FastAPI(title="Backend for stocks opinion analysis")

# add logging
logger = logging.getLogger(__name__)
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

@app.post("/complete")
async def complete_chat(request: Request, completion_request: CompletionRequest) -> dict:
    '''
    Endpoint to generate answer from prompt using LLM

    Args:
    - request: Incoming request object
    - compeltion_request: CompletionRequest object, input prompt from user

    Returns:
    - result (dict): resultant generated answer
    '''
    if not completion_request.user_prompt:
        raise HTTPException(status_code=400, detail="User needs to ask a question!")
    
    # Check accept header and see if frontend can accept out response type
    accept_header = request.headers.get("accept")
    if accept_header and "application/json" not in accept_header:
        raise HTTPException(status_code=406, detail="Accept header must be application/json")
    
    default_params = Params()
    payload = default_params.dict() | completion_request.dict()
    #Generate response from LLM
    try:
        result = complete()
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    return JSONResponse(result)