import asyncio
import os
import shutil
import uuid
import logging
import time
from typing import Optional, List
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import uvicorn

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from utils.logger import logger, set_log_level
from core.downloader import YouTubeDownloader
from core.file_loader import VideoFileLoader
from core.viral_clip_extractor import ViralClipExtractor
from core.formatter import VideoFormatter
from core.clipper import VideoClipper

# Initialize FastAPI
app = FastAPI(title="TikTok Agent API", description="API for extracting viral clips")

# Configure CORS (allow all for dev, restrict for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
set_log_level(logging.INFO)

# Global executor for CPU-bound tasks
# We limit workers to avoid overloading the system
process_executor = ProcessPoolExecutor(max_workers=2)

# Job storage (In-memory for simplicity, use Redis/DB for production)
jobs = {}

class ProcessingConfig(BaseModel):
    source_type: str  # 'youtube' or 'file'
    source_path: str  # URL or filename
    num_clips: int = 1
    clip_duration: float = 15.0
    min_gap: float = 1.0
    format_method: str = "crop"  # crop, blur, bars
    watermark_enabled: bool = False
    watermark_text: str = ""
    generate_thumbnail: bool = True
    face_detection: str = "mediapipe"
    
class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    created_at: float

# --- Worker Function (Runs in separate process) ---
def process_video_task(job_id: str, cfg: dict):
    """
    Worker function to process video. 
    MUST be top-level to be picklable by ProcessPoolExecutor.
    """
    try:
        # Since we can't easily share 'jobs' dict across processes for progress updates,
        # we will just return the result and handle status in the main process wrapper.
        # For real progress updates, we'd need a shared Manager or Redis.
        # For this MVP, we'll mark as 'processing' and then 'completed'.
        
        # 1. Get Video
        video_path = None
        if cfg['source_type'] == 'youtube':
            downloader = YouTubeDownloader()
            video_path = downloader.download(cfg['source_path'])
        elif cfg['source_type'] == 'file':
            # Assuming file is already in inputs dir (uploaded)
            loader = VideoFileLoader()
            video_path = loader.load(cfg['source_path'])
            
        if not video_path:
            return {"success": False, "error": "Failed to get video"}

        # 2. Extract Clips
        config.FACE_DETECTOR = cfg['face_detection'] # Set global config for this worker
        extractor = ViralClipExtractor()
        
        clips = extractor.extract_multiple_clips(
            video_path,
            num_clips=cfg['num_clips'],
            clip_duration=cfg['clip_duration'],
            min_gap=cfg['min_gap']
        )
        
        if not clips:
             return {"success": False, "error": "No clips found"}
             
        # 3. Format & Post-process
        formatter = VideoFormatter()
        clipper = VideoClipper() # For thumbnails
        
        results = []
        
        watermark_opts = None
        if cfg['watermark_enabled']:
            watermark_opts = {
                'enabled': True,
                'type': 'text',
                'text': cfg['watermark_text'],
                'position': 'bottom-right',
                'opacity': 0.7
            }

        for i, (clip_path, start, end, score) in enumerate(clips):
            # Format
            formatted_path = formatter.format_to_9_16(
                clip_path,
                method=cfg['format_method'],
                watermark_options=watermark_opts
            )
            
            # Thumbnail
            thumb_path = None
            if cfg['generate_thumbnail']:
                 mid_point = (start + end) / 2
                 t_path = str(Path(clip_path).with_suffix('')) + "_thumb.jpg"
                 thumb_path = clipper.generate_thumbnail(video_path, t_path, mid_point)

            if formatted_path:
                results.append({
                    "id": i,
                    "path": formatted_path,
                    "filename": os.path.basename(formatted_path),
                    "thumbnail": os.path.basename(thumb_path) if thumb_path else None,
                    "score": score,
                    "start": start,
                    "end": end
                })
        
        return {"success": True, "clips": results}

    except Exception as e:
        logger.error(f"Process error: {e}")
        return {"success": False, "error": str(e)}

# --- Async Wrapper ---
async def run_processing_job(job_id: str, config: ProcessingConfig):
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["message"] = "Processing started..."
    
    loop = asyncio.get_running_loop()
    
    # Run in process pool
    try:
        # Convert Pydantic model to dict for pickling
        cfg_dict = config.dict()
        
        result = await loop.run_in_executor(
            process_executor, 
            process_video_task, 
            job_id, 
            cfg_dict
        )
        
        if result["success"]:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "Processing complete"
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["result"] = result
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = result.get("error", "Unknown error")
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"System error: {str(e)}"

# --- Endpoints ---

@app.post("/api/process/youtube")
async def start_youtube_job(url: str,
                          num_clips: int = 1,
                          duration: float = 15.0,
                          format_method: str = "crop",
                          watermark: str = None,
                          background_tasks: BackgroundTasks = None):
    
    job_id = str(uuid.uuid4())
    
    job_config = ProcessingConfig(
        source_type="youtube",
        source_path=url,
        num_clips=num_clips,
        clip_duration=duration,
        format_method=format_method,
        watermark_enabled=bool(watermark),
        watermark_text=watermark or "",
        generate_thumbnail=True
    )
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": time.time(),
        "config": job_config.dict()
    }
    
    background_tasks.add_task(run_processing_job, job_id, job_config)
    
    return {"job_id": job_id, "status": "pending"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = config.INPUTS_DIR / filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": filename, "original_name": file.filename}

@app.post("/api/process/file")
async def start_file_job(filename: str,
                       num_clips: int = 1,
                       duration: float = 15.0,
                       format_method: str = "crop",
                       watermark: str = None,
                       background_tasks: BackgroundTasks = None):
                       
    job_id = str(uuid.uuid4())
    
    # Verify file exists
    if not (config.INPUTS_DIR / filename).exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    job_config = ProcessingConfig(
        source_type="file",
        source_path=str(config.INPUTS_DIR / filename),
        num_clips=num_clips,
        clip_duration=duration,
        format_method=format_method,
        watermark_enabled=bool(watermark),
        watermark_text=watermark or "",
        generate_thumbnail=True
    )
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": time.time(),
        "config": job_config.dict()
    }
    
    background_tasks.add_task(run_processing_job, job_id, job_config)
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        message=job.get("message", ""),
        result=job.get("result"),
        created_at=job["created_at"]
    )

# Serve output files
@app.get("/files/{filename}")
async def get_file(filename: str):
    file_path = config.OUTPUTS_DIR / filename
    if not file_path.exists():
         # Check if it's a thumbnail (which might be alongside) - actually outputs are flat in this project structure usually
         # But let's check subdirs if needed. For now assuming flat output dir from main.py logic
         raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# Serve frontend (static files) - This will be enabled once frontend is built
# app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("web_api:app", host="0.0.0.0", port=8000, reload=True)
