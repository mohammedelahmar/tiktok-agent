import sys
import os
import logging
from pathlib import Path

# Configure logging to file
logging.basicConfig(
    filename='debug_run.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)

logger = logging.getLogger("tiktok_agent")

sys.path.append(os.path.join(os.getcwd()))

try:
    from core.viral_clip_extractor import ViralClipExtractor
    import config
    
    # Force settings
    config.FACE_DETECTOR = "opencv"
    config.USE_ENGAGEMENT_MODEL = True
    
    logger.info("Initializing Extractor...")
    extractor = ViralClipExtractor()
    
    # Use the file we saw in logs
    video_path = r"c:\Users\PC\Documents\tiktok_agent\inputs\dae3ace9-aaac-4f32-8c66-a530a688976a_Little_Dark_Age_AMV_Monster_l8SrEka7cJM.mp4"
    if not os.path.exists(video_path):
        # Fallback to current inputs
        import glob
        files = glob.glob(r"c:\Users\PC\Documents\tiktok_agent\inputs\*.mp4")
        if files:
            video_path = files[0]
        else:
            logger.error("No video file found!")
            sys.exit(1)
            
    logger.info(f"Testing on video: {video_path}")
    
    candidates = extractor.find_candidates(
        video_path,
        num_clips=1,
        clip_duration=15, 
        min_gap=1.0,
        segment_duration=1.0
    )
    
    logger.info(f"FINAL CANDIDATES: {candidates}")
    
    if not candidates:
        logger.error("NO CANDIDATES FOUND!")
    else:
        logger.info("SUCCESS!")

except Exception as e:
    logger.error(f"CRITICAL ERROR: {e}", exc_info=True)
