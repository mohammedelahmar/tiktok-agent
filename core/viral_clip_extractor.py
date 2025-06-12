import numpy as np

import config
from utils.logger import logger
from models.engagement_model import EngagementModel
from core.clipper import VideoClipper
from utils.helpers import get_output_path

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
    
    def extract_multiple_clips(self, video_path, num_clips=3, clip_duration=None, min_gap=1.0, output_prefix=None):
        """Extract multiple viral clips from a video
        
        Args:
            video_path: Path to the video file
            num_clips: Number of clips to extract
            clip_duration: Duration of each clip in seconds (defaults to config.DEFAULT_CLIP_DURATION)
            min_gap: Minimum gap between clips in seconds
            output_prefix: Prefix for output paths (optional)
            
        Returns:
            list: List of tuples (clip_path, start_time, end_time, score)
        """
        if clip_duration is None:
            clip_duration = config.DEFAULT_CLIP_DURATION
            
        logger.info(f"Finding {num_clips} viral clips of {clip_duration}s each in {video_path}")
        
        # Score video segments
        segment_scores = self.engagement_model.score_video_segments(
            video_path, 
            segment_duration=1.0  # Score 1-second segments
        )
        
        if not segment_scores:
            logger.error("Failed to score video segments")
            return []
        
        # Convert to numpy arrays for easier manipulation
        times = np.array([s[0] for s in segment_scores])
        scores = np.array([s[1] for s in segment_scores])
        
        # Sort by time to ensure proper order
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        scores = scores[sort_idx]
        
        # Extract multiple non-overlapping clips
        result_clips = []
        excluded_ranges = []
        
        for i in range(num_clips):
            # Find best clip avoiding excluded ranges
            best_start, best_score = self._find_best_clip_window_exclude_ranges(
                times, scores, clip_duration, excluded_ranges
            )
            
            # If we couldn't find another good clip, break
            if best_start is None or best_score < 0:
                break
                
            best_end = best_start + clip_duration
            
            # Add this range (plus buffer) to excluded ranges
            buffer_start = max(0, best_start - min_gap)
            buffer_end = best_end + min_gap
            excluded_ranges.append((buffer_start, buffer_end))
            
            logger.info(f"Clip {i+1}: {best_start:.2f}s to {best_end:.2f}s (score: {best_score:.4f})")
            
            # Generate output path with index
            output_path = None
            if output_prefix:
                from pathlib import Path
                
                base_path = Path(output_prefix)
                output_path = str(base_path.parent / f"{base_path.stem}_{i+1}{base_path.suffix}")
            else:
                # Use clip index for automatic naming
                output_path = get_output_path(video_path, suffix="clip", clip_index=i+1)
            
            # Extract the clip
            clip_path = self.clipper.clip(
                video_path, 
                start_time=best_start, 
                end_time=best_end,
                output_path=output_path
            )
            
            if clip_path:
                result_clips.append((clip_path, best_start, best_end, best_score))
        
        logger.info(f"Extracted {len(result_clips)} viral clips")
        return result_clips
    
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
        
        return self._find_best_clip_window_exclude_ranges(times, scores, clip_duration, [])
    
    def _find_best_clip_window_exclude_ranges(self, times, scores, clip_duration, excluded_ranges):
        """Find the best window of segments for the clip, excluding specified ranges
        
        Args:
            times: Numpy array of segment start times
            scores: Numpy array of segment scores
            clip_duration: Desired clip duration in seconds
            excluded_ranges: List of (start_time, end_time) tuples to exclude
            
        Returns:
            tuple: (best_start_time, best_score)
        """
        best_start_idx = None
        best_score = -1
        
        for i in range(len(times)):
            # Skip if this segment is in an excluded range
            current_time = times[i]
            if any(start <= current_time < end for start, end in excluded_ranges):
                continue
            
            # Find segments that fall within the window
            end_time = current_time + clip_duration
            
            # Skip if the end time would fall in an excluded range
            if any(start < end_time <= end for start, end in excluded_ranges):
                continue
                
            # Skip if the window overlaps any excluded range
            if any(max(current_time, start) < min(end_time, end) 
                  for start, end in excluded_ranges):
                continue
            
            window_mask = (times >= current_time) & (times < end_time)
            
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
        if best_start_idx is not None:
            return times[best_start_idx], best_score
        else:
            return None, -1