import os
from pathlib import Path
from moviepy.editor import VideoFileClip

from utils.logger import logger

class VideoFileLoader:
    def __init__(self):
        """Initialize the video file loader"""
        # Expanded list of supported extensions
        self.supported_extensions = {
            '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv',  # Standard formats
            '.webm', '.m4v', '.3gp', '.ogv', '.mpg', '.mpeg'  # Additional formats
        }
    
    def load(self, file_path):
        """Load and validate a local video file
        
        Args:
            file_path: Path to the video file
            
        Returns:
            str: Absolute path to the video file if valid, None otherwise
        """
        file_path = Path(file_path).absolute()
        
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return None
            
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return None
            
        # Check extension
        if file_path.suffix.lower() not in self.supported_extensions:
            logger.warning(f"File extension {file_path.suffix} not in supported list: {self.supported_extensions}")
            logger.info("Attempting to validate as video anyway...")
        
        # Check if it's a valid video file
        if not self._is_valid_video(str(file_path)):
            logger.error(f"Not a valid video file: {file_path}")
            return None
            
        logger.info(f"Successfully loaded video: {file_path}")
        return str(file_path)
    
    def _is_valid_video(self, file_path):
        """Check if a file is a valid video using only MoviePy
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if valid video, False otherwise
        """
        try:
            # Try to get video info with MoviePy
            with VideoFileClip(file_path) as clip:
                # Check basic video properties
                if clip.duration <= 0:
                    return False
                
                # Try to read a frame to ensure the video is readable
                clip.get_frame(0)
                
                return True
            
        except Exception as e:
            logger.error(f"Error validating video file: {str(e)}")
            return False