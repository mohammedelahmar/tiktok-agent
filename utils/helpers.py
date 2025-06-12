import os
import shutil
import uuid
from pathlib import Path
import tempfile

def create_temp_dir():
    """Create a temporary directory and return its path"""
    temp_dir = Path(tempfile.gettempdir()) / f"tiktok_agent_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

def clean_temp_dir(temp_dir):
    """Remove a temporary directory"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def get_video_id_from_url(url):
    """Extract YouTube video ID from URL"""
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    elif "youtube.com" in url:
        # Extract from watch URL format
        if "v=" in url:
            video_id = url.split("v=")[1]
            amp_pos = video_id.find("&")
            if amp_pos != -1:
                return video_id[:amp_pos]
            return video_id
    return None

def sanitize_filename(filename):
    """Sanitize a filename by removing invalid characters"""
    if isinstance(filename, Path):
        filename = str(filename)
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def get_output_path(original_filename, suffix="tiktok", clip_index=None):
    """Generate an output filename based on the original filename
    
    Args:
        original_filename: Original file path or name
        suffix: Suffix to add to the filename
        clip_index: Optional clip index for multi-clip extraction
        
    Returns:
        str: Path to output file
    """
    from pathlib import Path
    import config
    
    original_path = Path(original_filename)
    sanitized_name = sanitize_filename(original_path.stem)
    
    # Add clip index to suffix if provided
    if clip_index is not None:
        suffix = f"{clip_index}_{suffix}"
        
    output_name = f"{sanitized_name}_{suffix}.{config.OUTPUT_FORMAT}"
    # Return as string instead of Path object
    return str(config.OUTPUTS_DIR / output_name)