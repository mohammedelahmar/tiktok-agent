from typing import Optional, List, Dict, Any
from pydantic import BaseModel

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
    status: str  # pending, processing, completed, failed, analyzed
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    created_at: float
