from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from typing import List
import uuid
import time
import shutil
import os
import asyncio
from pathlib import Path

# Import from our new package structure
from app.schemas import ProcessingConfig, JobStatus, RenderRequest, MetadataRequest
from app.services.job_manager import job_manager
import config

# Import core logic (assuming these are still in root for now, or we can move them too)
# For now we'll import from root by adding root to sys.path in main.py or relying on Python path
# But typical relative imports might fail if we run from root.
# We will treat 'core' as a top-level package if we run from root.
from web_api import run_processing_job, run_render_job, run_metadata_job

router = APIRouter()

@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/history", response_model=List[JobStatus])
async def get_job_history():
    jobs = job_manager.get_all_jobs()
    # Return as list sorted by creation time desc
    return sorted(jobs.values(), key=lambda x: x.created_at, reverse=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = config.INPUTS_DIR / filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": filename, "original_name": file.filename}

@router.post("/process/youtube")
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
    
    job = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=time.time(),
        config=job_config.dict()
    )
    
    job_manager.add_job(job)
    
    # We need to adapt the run_processing_job to update via job_manager
    # For now, we'll wrap it or trust it updates the global memory dict?
    # NO, we implemented separate persistence.
    # We need to bridge the old run_processing_job with new JobManager.
    # Strategy: Redefine the async wrapper here to use job_manager.
    
    background_tasks.add_task(run_processing_job_wrapper, job_id, job_config)
    
    return {"job_id": job_id, "status": "pending"}

@router.post("/process/file")
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
    
    job = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=time.time(),
        config=job_config.dict()
    )
    
    job_manager.add_job(job)
    
    background_tasks.add_task(run_processing_job_wrapper, job_id, job_config)
    
    return {"job_id": job_id, "status": "pending"}

@router.post("/render")
async def render_clips(req: RenderRequest, background_tasks: BackgroundTasks):
    # Logic similar to web_api.py but using job_manager
    stored_job = job_manager.get_job(req.job_id)
    if not stored_job:
        raise HTTPException(status_code=404, detail="Original job not found")
        
    # Hack to find video path from stored result
    # In Pydantic model this is accessed via attributes
    res = stored_job.result
    # result might be a dict or None
    if not res or not res.get("video_filename"):
          raise HTTPException(status_code=400, detail="Job has no video information")
    
    video_filename = res["video_filename"]
    video_path = str(config.INPUTS_DIR / video_filename)
    
    if not os.path.exists(video_path):
         # Try finding it with glob
         import glob
         search = glob.glob(str(config.INPUTS_DIR / f"*{video_filename}*"))
         if search:
             video_path = search[0]
         else:
             raise HTTPException(status_code=404, detail="Source video file missing")

    render_job_id = str(uuid.uuid4())
    
    job = JobStatus(
        job_id=render_job_id,
        status="pending",
        created_at=time.time(),
        config={} # Will be populated later or we can put render details
    )
    job_manager.add_job(job)

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
    
    background_tasks.add_task(run_render_job_wrapper, render_job_id, video_path, req.clips, render_config)
    
    return {"status": "rendering_started", "job_id": render_job_id}


# --- Wrappers to update JobManager instead of in-memory dict ---

from web_api import process_video_task, process_render_task, process_executor

async def run_processing_job_wrapper(job_id: str, config: ProcessingConfig):
    job_manager.update_job(job_id, {"status": "processing", "message": "Processing started..."})
    
    loop = asyncio.get_running_loop()
    
    try:
        cfg_dict = config.dict()
        result = await loop.run_in_executor(process_executor, process_video_task, job_id, cfg_dict)
        
        if result["success"]:
            if result.get("mode") == "analyzed":
                job_manager.update_job(job_id, {
                    "status": "analyzed", 
                    "message": "Analysis complete",
                    "result": result,
                    "progress": 1.0
                })
            else:
                job_manager.update_job(job_id, {
                    "status": "completed", 
                    "message": "Processing complete",
                    "result": result,
                    "progress": 1.0
                })
        else:
            job_manager.update_job(job_id, {
                "status": "failed", 
                "message": result.get("error", "Unknown error")
            })
            
    except Exception as e:
        job_manager.update_job(job_id, {
            "status": "failed", 
            "message": f"System error: {str(e)}"
        })

async def run_render_job_wrapper(job_id, video_path, clips, config):
    job_manager.update_job(job_id, {"status": "processing", "message": "Rendering clips..."})
    loop = asyncio.get_running_loop()
    try:
        cfg_dict = config.dict()
        result = await loop.run_in_executor(process_executor, process_render_task, job_id, video_path, clips, cfg_dict)
        
        if result["success"]:
            job_manager.update_job(job_id, {
                "status": "completed",
                "message": "Rendering complete",
                "result": result
            })
        else:
            job_manager.update_job(job_id, {
                "status": "failed",
                "message": result.get("error", "Unknown error")
            })
    except Exception as e:
        job_manager.update_job(job_id, {
            "status": "failed",
            "message": f"Render error: {str(e)}"
        })
