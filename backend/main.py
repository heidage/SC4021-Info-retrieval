import logging
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Reponse, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

@app.post("/complete")
async def complete_chat(request: Request, completion_request: CompletionRequest) -> dict:
    return {"message": "Hello World"}