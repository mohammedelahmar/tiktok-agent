from moviepy.editor import VideoFileClip
import os
from pathlib import Path
import uuid

from utils.logger import logger
from utils.helpers import get_output_path

class VideoClipper:
    def __init__(self):
        """Initialize the video clipper"""
        pass
    
    def get_video_duration(self, video_path):
        """Get the duration of a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            float: Duration of the video in seconds
        """
        try:
            with VideoFileClip(video_path) as video:
                return video.duration
        except Exception as e:
            logger.error(f"Error getting video duration: {str(e)}")
            return 0
    
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
        temp_audio_file = None
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
                
                # Generate a unique temp audio filename to prevent conflicts in parallel processing
                temp_audio_file = f"temp-audio-{uuid.uuid4().hex[:8]}.m4a"
                
                # Write the subclip
                subclip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=temp_audio_file,
                    remove_temp=True
                )
                
                logger.info(f"Clip extracted and saved to {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error extracting clip: {str(e)}")
            return None
        finally:
            # Clean up temp file if it exists and wasn't removed
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.remove(temp_audio_file)
                    logger.debug(f"Cleaned up temporary audio file: {temp_audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_audio_file}: {str(e)}")
    
    # Add function to embed metadata

    def embed_metadata(self, video_path, metadata):
        """Embed metadata into a video file using ffmpeg
        
        Args:
            video_path: Path to the video file
            metadata: Dictionary of metadata to embed
            
        Returns:
            str: Path to the output video with metadata
        """
        try:
            import subprocess
            import tempfile
            from pathlib import Path
            
            # Create a temporary file
            temp_output = tempfile.mktemp(suffix=Path(video_path).suffix)
            
            # Build ffmpeg command
            command = ["ffmpeg", "-i", video_path]
            
            # Add metadata arguments
            for key, value in metadata.items():
                command.extend(["-metadata", f"{key}={value}"])
                
            # Add output file and overwrite flag
            command.extend(["-codec", "copy", "-y", temp_output])
            
            # Run ffmpeg
            subprocess.run(command, check=True, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            
            # Replace original file with the metadata-embedded version
            os.replace(temp_output, video_path)
            
            logger.debug(f"Successfully embedded metadata in {video_path}")
            return video_path
        
        except Exception as e:
            logger.error(f"Error embedding metadata: {str(e)}")
            return video_path  # Return original file if embedding fails