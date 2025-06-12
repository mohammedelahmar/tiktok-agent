from moviepy.editor import VideoFileClip
import os
from pathlib import Path

from utils.logger import logger
from utils.helpers import get_output_path

class VideoClipper:
    def __init__(self):
        """Initialize the video clipper"""
        pass
    
    def clip(self, video_path, start_time, end_time=None, duration=None, output_path=None):
        """Extract a subclip from a video file
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds (optional if duration is provided)
            duration: Duration of clip in seconds (optional if end_time is provided)
            output_path: Path to save the output clip (optional)
            
        Returns:
            str: Path to the clipped video file
        """
        try:
            if end_time is None and duration is None:
                raise ValueError("Either end_time or duration must be provided")
                
            if end_time is None:
                end_time = start_time + duration
            
            logger.info(f"Extracting clip from {start_time:.2f}s to {end_time:.2f}s")
            
            with VideoFileClip(video_path) as video:
                # Ensure times are within video bounds
                video_duration = video.duration
                start_time = max(0, min(start_time, video_duration - 1))
                end_time = max(start_time + 1, min(end_time, video_duration))
                
                subclip = video.subclip(start_time, end_time)
                
                # Generate output path if not provided
                if output_path is None:
                    output_path = get_output_path(video_path, suffix="clip")
                
                # Convert Path to string if it's a Path object
                if isinstance(output_path, Path):
                    output_path = str(output_path)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Write the subclip
                subclip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile="temp-audio.m4a",
                    remove_temp=True
                )
                
                logger.info(f"Clip extracted and saved to {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error extracting clip: {str(e)}")
            return None