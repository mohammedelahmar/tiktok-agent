import uvicorn
import logging
import sys
import os
from pathlib import Path

# Add project root to path so we can import legacy web_api and core modules
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import jobs
from utils.logger import setup_logger

setup_logger()

app = FastAPI(title="TikTok Agent API", description="API for extracting viral clips (v2)")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(jobs.router, prefix="/api", tags=["jobs"])

# Serve static files for output/input (Legacy support for now)
import config
app.mount("/files", StaticFiles(directory=config.OUTPUTS_DIR), name="files")
app.mount("/inputs", StaticFiles(directory=config.INPUTS_DIR), name="inputs")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
