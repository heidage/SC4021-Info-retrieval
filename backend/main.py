import logging
import traceback
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from input_class import EmbedRequest
from utils.api_helper import get_top_k

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
