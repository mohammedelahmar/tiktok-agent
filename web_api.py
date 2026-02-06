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
import sys

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
from utils.logger import setup_logger, logger

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
setup_logger()

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
    captions_enabled: bool = False
    generate_metadata: bool = False
    face_detection: str = "mediapipe"
    mode: str = "process" # 'process' or 'analyze'

class RenderRequest(BaseModel):
    job_id: str
    clips: List[dict] # List of {'start': float, 'end': float, 'id': int}
    format_method: str = "crop"
    watermark_enabled: bool = False
    watermark_text: str = ""
    generate_thumbnail: bool = True
    captions_enabled: bool = False
    generate_metadata: bool = False

class MetadataRequest(BaseModel):
    job_id: Optional[str] = None
    filename: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    config: Optional[dict] = None
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
            
        # Check Mode
        if cfg.get('mode') == 'analyze':
            # Just find candidates
            extractor = ViralClipExtractor()
            candidates = extractor.find_candidates(
                video_path,
                num_clips=cfg['num_clips'],
                clip_duration=cfg['clip_duration'],
                min_gap=cfg['min_gap']
            )
            
            # Return candidates + video info
            logger.info(f"Analyze finished. Found {len(candidates)} candidates: {candidates}")
            return {
                "success": True, 
                "candidates": [
                    {
                        "id": int(i), 
                        "start": float(s), 
                        "end": float(e), 
                        "score": float(score)
                    } 
                    for i, s, e, score in candidates
                ],
                "video_filename": os.path.basename(video_path),
                "mode": "analyzed"
            }

        # Normal Processing (or Render Step)
        
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

        # Optimization: Init shared resources before loop
        captioner_service = None
        metadata_service = None
        
        if cfg.get('captions_enabled') or cfg.get('generate_metadata'):
             try:
                 from core.captioner import Captioner
                 captioner_service = Captioner()
             except Exception as e:
                 logger.error(f"Failed to init Captioner: {e}")

        if cfg.get('generate_metadata'):
             try:
                 from core.metadata_generator import MetadataGenerator
                 metadata_service = MetadataGenerator()
             except Exception as e:
                 logger.error(f"Failed to init MetadataGenerator: {e}")

        for i, (clip_path, start, end, score) in enumerate(clips):
            # Format
            formatted_path = formatter.format_to_9_16(
                clip_path,
                method=cfg['format_method'],
                watermark_options=watermark_opts
            )
            
            # Captions
            if cfg.get('captions_enabled') and captioner_service:
                 try:
                     formatted_path = captioner_service.process_video(formatted_path)
                 except Exception as e:
                     logger.error(f"Captioning failed for clip {i}: {e}")
            
            # Metadata (Title/Desc/Tags)
            clip_metadata = None
            if cfg.get('generate_metadata') and metadata_service and captioner_service:
                try:
                    # Extract transcript for this specific clip
                    # Note: We use the formatted path (which might have music/watermark? No, usually clean audio)
                    # Actually formatted_path has the audio.
                    clip_transcript = captioner_service.extract_transcript(formatted_path)
                    if clip_transcript:
                        clip_metadata = metadata_service.generate_metadata(clip_transcript)
                except Exception as e:
                    logger.error(f"Metadata generation failed for clip {i}: {e}")

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
                    "end": end,
                    "metadata": clip_metadata
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
            # Check if this was an analysis job
            if result.get("mode") == "analyzed":
                jobs[job_id]["status"] = "analyzed"
                jobs[job_id]["message"] = "Analysis complete. Waiting for user review."
            else:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["message"] = "Processing complete"
            
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["result"] = result
            logger.info(f"Job {job_id} stored result: {result}")

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
                          captions: bool = False,
                          generate_metadata: bool = False,
                          mode: str = "process",
                          background_tasks: BackgroundTasks = None):
    
    job_id = str(uuid.uuid4())
    
    job_config = ProcessingConfig(
        source_type="youtube",
        source_path=url,
        num_clips=num_clips,
        clip_duration=duration,
        format_method=format_method,
        watermark_text=watermark or "",
        generate_thumbnail=True,
        captions_enabled=bool(captions),
        generate_metadata=bool(generate_metadata),
        mode=mode
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
                       captions: bool = False,
                       generate_metadata: bool = False,
                       mode: str = "process",
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
        watermark_text=watermark or "",
        generate_thumbnail=True,
        captions_enabled=bool(captions),
        generate_metadata=bool(generate_metadata),
        mode=mode
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
    if job.get("status") == "analyzed":
        logger.info(f"Returning status for {job_id}. Result keys: {job.get('result', {}).keys()}")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        message=job.get("message", ""),
        result=job.get("result"),
        config=job.get("config"),
        created_at=job["created_at"]
    )

# --- Render Worker ---
def process_render_task(job_id, video_path, clips, cfg):
    """
    Worker for rendering selected clips
    """
    try:

         clipper = VideoClipper()
         formatter = VideoFormatter()
         
         logger.info(f"Starting render task for {len(clips)} clips: {clips}")
         
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
         
         for i, clip_data in enumerate(clips):
             start = clip_data['start']
             end = clip_data['end']
             score = clip_data.get('score', 0.0)
             
             # Extract
             # We use proper output path to ensure uniqueness
             # But here we do manual clipping
             clip_path = clipper.clip(video_path, start, end)
             
             if not clip_path:
                 continue
                 
             # Format
             formatted_path = formatter.format_to_9_16(
                clip_path,
                method=cfg['format_method'],
                watermark_options=watermark_opts
            )
             
             # Captions
             if cfg.get('captions_enabled'):
                 try:
                     from core.captioner import Captioner
                     captioner_instance = Captioner() 
                     formatted_path = captioner_instance.process_video(formatted_path)
                 except Exception as e:
                     logger.error(f"Captioning failed for clip {i}: {e}")
            
             # Metadata (Title/Desc/Tags)
             clip_metadata = None
             if cfg.get('generate_metadata'):
                try:
                    from core.metadata_generator import MetadataGenerator
                    if 'core.captioner' not in sys.modules:
                         from core.captioner import Captioner
                    
                    # Ensure we have a captioner to get transcript
                    captioner_service = Captioner()
                    metadata_service = MetadataGenerator()
                    
                    clip_transcript = captioner_service.extract_transcript(formatted_path)
                    if clip_transcript:
                        clip_metadata = metadata_service.generate_metadata(clip_transcript)
                except Exception as e:
                    logger.error(f"Metadata generation failed for clip {i}: {e}")
                    clip_metadata = {"error": str(e)}

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
                    "start": start,
                    "end": end,
                    "score": score,
                    "metadata": clip_metadata
                })
         
         return {"success": True, "clips": results}
         
    except Exception as e:
        logger.error(f"Render error: {e}")
        return {"success": False, "error": str(e)}

async def run_render_job(job_id, video_path, clips, config):
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["message"] = "Rendering specific clips..."
    
    loop = asyncio.get_running_loop()
    
    try:
        # We need a simplify config for pickling
        cfg_dict = config.dict()
        
        result = await loop.run_in_executor(
            process_executor,
            process_render_task,
            job_id,
            video_path,
            clips,
            cfg_dict
        )
        
        if result["success"]:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "Rendering complete"
            jobs[job_id]["result"] = result
        else:
             jobs[job_id]["status"] = "failed"
             jobs[job_id]["message"] = result.get("error", "Unknown error")
             
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Render error: {str(e)}"

# --- Metadata Worker ---
def process_metadata_task(job_id, video_path):
    """
    Worker for metadata generation (Transcription + LLM)
    """
    try:
        from core.captioner import Captioner
        from core.metadata_generator import MetadataGenerator
        
        # 1. Extract Transcript
        # Check if we already have a transcript file? (Optimization for later)
        # For now, always extract.
        logger.info(f"Metadata Task: Extracting transcript for {video_path}")
        captioner = Captioner() # This might be slow loading model
        transcript = captioner.extract_transcript(video_path)
        
        if not transcript:
            return {"success": False, "error": "Could not extract transcript (empty audio?)"}
            
        # 2. Generate Metadata
        logger.info("Metadata Task: Generating metadata with LLM")
        generator = MetadataGenerator()
        metadata = generator.generate_metadata(transcript)
        
        if "error" in metadata:
             return {"success": False, "error": metadata["error"]}
             
        return {
            "success": True, 
            "metadata": metadata,
            "transcript": transcript
        }
        
    except Exception as e:
        logger.error(f"Metadata task error: {e}")
        return {"success": False, "error": str(e)}

async def run_metadata_job(job_id, video_path):
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["message"] = "Generating AI metadata..."
    
    loop = asyncio.get_running_loop()
    
    try:
        result = await loop.run_in_executor(
            process_executor,
            process_metadata_task,
            job_id,
            video_path
        )
        
        if result["success"]:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "Metadata generation complete"
            jobs[job_id]["result"] = result
        else:
             jobs[job_id]["status"] = "failed"
             jobs[job_id]["message"] = result.get("error", "Unknown error")
             
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"System error: {str(e)}"

@app.post("/api/metadata/generate")
async def generate_metadata(req: MetadataRequest, background_tasks: BackgroundTasks):
    video_path = None
    
    # Resolve video path
    if req.filename:
        video_path = str(config.INPUTS_DIR / req.filename)
    elif req.job_id:
        stored_job = jobs.get(req.job_id)
        if stored_job and stored_job.get("result"):
             # Try to find video filename in result
             res = stored_job["result"]
             if "video_filename" in res:
                 video_path = str(config.INPUTS_DIR / res["video_filename"])
             # Fallback: maybe the job was a render job and 'path' is the output? 
             # But metadata is usually for the source or a specific clip.
             # Let's assume user wants metadata for the Source video of the job.
    
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
        
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": time.time(),
        "config": {"type": "metadata", "source": video_path}
    }
    
    background_tasks.add_task(run_metadata_job, job_id, video_path)
    
    return {"job_id": job_id, "status": "pending"}


@app.post("/api/render")
async def render_clips(req: RenderRequest, background_tasks: BackgroundTasks):
    # 1. Look up original job to get video path
    # But wait, the original job might have finished or been cleared if we restart server.
    # ideally we stored this persistently. 
    # For now, we assume user flow is continuous.
    
    # Actually, we should pass video_path in request or re-derive it?
    # Security risk if we allow arbitrary paths.
    
    # Better: Look up job_id globally.
    stored_job = jobs.get(req.job_id)
    if not stored_job:
        raise HTTPException(status_code=404, detail="Original job not found or expired")
        
    # Get video path from stored result
    # The 'analyzed' result has 'video_filename'
    if not stored_job.get("result") or not stored_job["result"].get("video_filename"):
         raise HTTPException(status_code=400, detail="Job has no video information")
    
    video_filename = stored_job["result"]["video_filename"]
    # We look in Inputs dir
    # Need to handle both YouTube downloads (which go to inputs) and uploaded files
    video_path = str(config.INPUTS_DIR / video_filename)
    
    if not os.path.exists(video_path):
         # Try finding it with glob if youtube dl did something weird (already handled in logic but fallback)
         import glob
         search = glob.glob(str(config.INPUTS_DIR / f"*{video_filename}*"))
         if search:
             video_path = search[0]
         else:
             raise HTTPException(status_code=404, detail="Source video file missing")

    # Create new job or update existing?
    # Let's update existing to keep flow simple for frontend polling
    # jobs[job_id]["status"] = "pending_render"
    
    # Update config args from request
    # stored_job["config"]...
    # Actually create a temporary config object for the worker
    # Create NEW job for rendering to avoid overwriting the analysis job state too early
    render_job_id = str(uuid.uuid4())
    
    jobs[render_job_id] = {
        "job_id": render_job_id,
        "status": "pending",
        "created_at": time.time(),
        "config": {} # Will be filled by worker, or we can copy relevant parts
    }

    render_config = ProcessingConfig(
        source_type="file", 
        source_path=video_path,
        format_method=req.format_method,
        watermark_enabled=req.watermark_enabled,
        watermark_text=req.watermark_text,
        generate_thumbnail=req.generate_thumbnail,
        captions_enabled=req.captions_enabled,
        generate_metadata=req.generate_metadata,
        mode="process",
        num_clips=len(req.clips),
        clip_duration=0 
    )
    
    background_tasks.add_task(run_render_job, render_job_id, video_path, req.clips, render_config)
    
    return {"status": "rendering_started", "job_id": render_job_id}


# Serve output files
@app.get("/files/{filename}")
async def get_file(filename: str):
    file_path = config.OUTPUTS_DIR / filename
    if not file_path.exists():
         # Check if it's a thumbnail (which might be alongside) - actually outputs are flat in this project structure usually
         # But let's check subdirs if needed. For now assuming flat output dir from main.py logic
         raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# Serve Inputs (for preview)
@app.get("/inputs/{filename}")
async def get_input_file(filename: str):
    file_path = config.INPUTS_DIR / filename
    if not file_path.exists():
         raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# Debug endpoint to receive logs from frontend
@app.post("/api/debug")
async def receive_debug_log(payload: dict):
    logger.info(f"FRONTEND DEBUG: {payload}")
    return {"status": "ok"}

# Serve frontend (static files) - This will be enabled once frontend is built
# app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("web_api:app", host="127.0.0.1", port=8000, reload=True)
