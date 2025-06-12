import numpy as np

import config
from utils.logger import logger
from models.engagement_model import EngagementModel
from core.clipper import VideoClipper

class ViralClipExtractor:
    def __init__(self):
        """Initialize the viral clip extractor"""
        self.engagement_model = EngagementModel()
        self.clipper = VideoClipper()
        
        # Try to initialize the model
        if config.USE_ENGAGEMENT_MODEL:
            self.engagement_model.initialize()
    
    def extract_best_clip(self, video_path, clip_duration=None, output_path=None):
        """Extract the most viral clip from a video
        
        Args:
            video_path: Path to the video file
            clip_duration: Duration of clip to extract (defaults to config.DEFAULT_CLIP_DURATION)
            output_path: Path to save the extracted clip (optional)
            
        Returns:
            tuple: (clip_path, start_time, end_time, score)
        """
        if clip_duration is None:
            clip_duration = config.DEFAULT_CLIP_DURATION
            
        logger.info(f"Finding most viral {clip_duration}s clip in {video_path}")
        
        # Score video segments
        segment_scores = self.engagement_model.score_video_segments(
            video_path, 
            segment_duration=1.0  # Score 1-second segments
        )
        
        if not segment_scores:
            logger.error("Failed to score video segments")
            return None, 0, 0, 0
        
        # Find the best starting point for a clip of the requested duration
        best_start, best_score = self._find_best_clip_window(
            segment_scores, 
            clip_duration
        )
        
        best_end = best_start + clip_duration
        
        logger.info(f"Best clip: {best_start:.2f}s to {best_end:.2f}s (score: {best_score:.4f})")
        
        # Extract the best clip
        clip_path = self.clipper.clip(
            video_path, 
            start_time=best_start, 
            end_time=best_end,
            output_path=output_path
        )
        
        return clip_path, best_start, best_end, best_score
    
    def _find_best_clip_window(self, segment_scores, clip_duration):
        """Find the best window of segments for the clip
        
        Args:
            segment_scores: List of (start_time, score) tuples
            clip_duration: Desired clip duration in seconds
            
        Returns:
            tuple: (best_start_time, best_score)
        """
        # Convert to numpy arrays for easier manipulation
        times = np.array([s[0] for s in segment_scores])
        scores = np.array([s[1] for s in segment_scores])
        
        # Sort by time to ensure proper order
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        scores = scores[sort_idx]
        
        # Find segments that would make up a clip of the requested duration
        best_start_idx = 0
        best_score = -1
        
        for i in range(len(times)):
            # Find segments that fall within the window
            end_time = times[i] + clip_duration
            window_mask = (times >= times[i]) & (times < end_time)
            
            # If no segments in the window, skip
            if not np.any(window_mask):
                continue
                
            # Calculate aggregate score for this window
            window_scores = scores[window_mask]
            window_score = np.mean(window_scores)
            
            # Update best window if score is higher
            if window_score > best_score:
                best_start_idx = i
                best_score = window_score
        
        # Return the best starting time and score
        best_start_time = times[best_start_idx]
        return best_start_time, best_score# filepath: c:\Users\PC\Documents\tiktok_agent\core\viral_clip_extractor.py
import numpy as np

import config
from utils.logger import logger
from models.engagement_model import EngagementModel
from core.clipper import VideoClipper

class ViralClipExtractor:
    def __init__(self):
        """Initialize the viral clip extractor"""
        self.engagement_model = EngagementModel()
        self.clipper = VideoClipper()
        
        # Try to initialize the model
        if config.USE_ENGAGEMENT_MODEL:
            self.engagement_model.initialize()
    
    def extract_best_clip(self, video_path, clip_duration=None, output_path=None):
        """Extract the most viral clip from a video
        
        Args:
            video_path: Path to the video file
            clip_duration: Duration of clip to extract (defaults to config.DEFAULT_CLIP_DURATION)
            output_path: Path to save the extracted clip (optional)
            
        Returns:
            tuple: (clip_path, start_time, end_time, score)
        """
        if clip_duration is None:
            clip_duration = config.DEFAULT_CLIP_DURATION
            
        logger.info(f"Finding most viral {clip_duration}s clip in {video_path}")
        
        # Score video segments
        segment_scores = self.engagement_model.score_video_segments(
            video_path, 
            segment_duration=1.0  # Score 1-second segments
        )
        
        if not segment_scores:
            logger.error("Failed to score video segments")
            return None, 0, 0, 0
        
        # Find the best starting point for a clip of the requested duration
        best_start, best_score = self._find_best_clip_window(
            segment_scores, 
            clip_duration
        )
        
        best_end = best_start + clip_duration
        
        logger.info(f"Best clip: {best_start:.2f}s to {best_end:.2f}s (score: {best_score:.4f})")
        
        # Extract the best clip
        clip_path = self.clipper.clip(
            video_path, 
            start_time=best_start, 
            end_time=best_end,
            output_path=output_path
        )
        
        return clip_path, best_start, best_end, best_score
    
    def _find_best_clip_window(self, segment_scores, clip_duration):
        """Find the best window of segments for the clip
        
        Args:
            segment_scores: List of (start_time, score) tuples
            clip_duration: Desired clip duration in seconds
            
        Returns:
            tuple: (best_start_time, best_score)
        """
        # Convert to numpy arrays for easier manipulation
        times = np.array([s[0] for s in segment_scores])
        scores = np.array([s[1] for s in segment_scores])
        
        # Sort by time to ensure proper order
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        scores = scores[sort_idx]
        
        # Find segments that would make up a clip of the requested duration
        best_start_idx = 0
        best_score = -1
        
        for i in range(len(times)):
            # Find segments that fall within the window
            end_time = times[i] + clip_duration
            window_mask = (times >= times[i]) & (times < end_time)
            
            # If no segments in the window, skip
            if not np.any(window_mask):
                continue
                
            # Calculate aggregate score for this window
            window_scores = scores[window_mask]
            window_score = np.mean(window_scores)
            
            # Update best window if score is higher
            if window_score > best_score:
                best_start_idx = i
                best_score = window_score
        
        # Return the best starting time and score
        best_start_time = times[best_start_idx]
        return best_start_time, best_score