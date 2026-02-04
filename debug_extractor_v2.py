import sys
import os
import logging
from pathlib import Path

# Setup simple logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tiktok_agent")

sys.path.append(os.path.join(os.getcwd()))

try:
    from core.viral_clip_extractor import ViralClipExtractor
    import config
    
    # Force settings for debugging
    config.FACE_DETECTOR = "opencv"
    config.USE_ENGAGEMENT_MODEL = True
    
    print("Initializing Extractor...")
    extractor = ViralClipExtractor()
    
    video_path = r"c:\Users\PC\Documents\tiktok_agent\inputs\Little_Dark_Age_AMV_Monster_l8SrEka7cJM.mp4"
    if not os.path.exists(video_path):
        # Fallback to the UUID one from logs
        video_path = r"c:\Users\PC\Documents\tiktok_agent\inputs\c0f2e7d9-b3c2-4997-8904-2d39081469fe_Little_Dark_Age_AMV_Monster_l8SrEka7cJM.mp4"
        
    print(f"Testing on video: {video_path}")
    
    candidates = extractor.find_candidates(
        video_path,
        num_clips=1,
        clip_duration=15, # Shorter duration for speed
        min_gap=1.0,
        segment_duration=1.0
    )
    
    print(f"Result candidates: {candidates}")
    
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
