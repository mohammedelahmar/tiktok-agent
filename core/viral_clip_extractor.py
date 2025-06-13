import numpy as np
import time
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor  # Change to ThreadPoolExecutor
import os

import config
from utils.logger import logger
from models.engagement_model import EngagementModel
from core.clipper import VideoClipper
from utils.helpers import get_output_path

class ViralClipExtractor:
    def __init__(self, engagement_model=None, clipper=None):
        """Initialize the viral clip extractor
        
        Args:
            engagement_model: Optional custom engagement model instance
            clipper: Optional custom video clipper instance
        """
        self.engagement_model = engagement_model or EngagementModel()
        self.clipper = clipper or VideoClipper()
        self._model_initialized = False
    
    def _ensure_model_ready(self):
        """Ensure the engagement model is initialized"""
        if config.USE_ENGAGEMENT_MODEL and not self._model_initialized:
            start_time = time.time()
            self.engagement_model.initialize()
            self._model_initialized = True
            logger.debug(f"Model initialization took {time.time()-start_time:.2f}s")
    
    def extract_best_clip(self, video_path, clip_duration=None, output_path=None, segment_duration=1.0):
        """Extract the most viral clip from a video
        
        Args:
            video_path: Path to the video file
            clip_duration: Duration of clip to extract (defaults to config.DEFAULT_CLIP_DURATION)
            output_path: Path to save the extracted clip (optional)
            segment_duration: Duration of segments to score (default 1.0)
            
        Returns:
            tuple: (clip_path, start_time, end_time, score)
        """
        if clip_duration is None:
            clip_duration = config.DEFAULT_CLIP_DURATION
        
        # Ensure model is ready
        self._ensure_model_ready()
        
        # Check video duration to avoid invalid segments
        video_duration = self.clipper.get_video_duration(video_path)
        if video_duration < clip_duration:
            logger.warning(f"Video shorter than clip duration ({video_duration:.2f}s < {clip_duration:.2f}s). Using full video.")
            clip_duration = video_duration
            
        logger.info(f"Finding most viral {clip_duration:.2f}s clip in {video_path}")
        
        # Score video segments
        start_time = time.time()
        segment_scores = self.engagement_model.score_video_segments(
            video_path, 
            segment_duration=segment_duration
        )
        
        if not segment_scores or len(segment_scores) == 0:
            logger.error("No scorable segments found")
            return None, 0, 0, 0
            
        logger.debug(f"Scored {len(segment_scores)} segments in {time.time()-start_time:.2f}s")
        
        # Find the best starting point for a clip of the requested duration
        best_start, best_score = self._find_best_clip_window(
            segment_scores, 
            clip_duration
        )
        
        if best_start is None or best_score < 0:
            logger.error("Could not find suitable clip window")
            return None, 0, 0, 0
            
        best_end = best_start + clip_duration
        
        logger.info(f"Best clip: {best_start:.2f}s to {best_end:.2f}s (score: {best_score:.4f})")
        
        # Extract the best clip
        clip_path = self.clipper.clip(
            video_path, 
            start_time=best_start, 
            end_time=best_end,
            output_path=output_path
        )
        
        # Verify clip was created successfully
        if not clip_path or not Path(clip_path).exists():
            logger.error(f"Clipping failed for {best_start:.2f}s-{best_end:.2f}s")
            return None, best_start, best_end, best_score
        
        return clip_path, best_start, best_end, best_score
    
    def extract_multiple_clips(self, video_path, num_clips=3, clip_duration=None, 
                               min_gap=1.0, output_prefix=None, segment_duration=1.0,
                               parallel_processing=True):
        """Extract multiple viral clips from a video
        
        Args:
            video_path: Path to the video file
            num_clips: Number of clips to extract
            clip_duration: Duration of each clip in seconds (defaults to config.DEFAULT_CLIP_DURATION)
            min_gap: Minimum gap between clips in seconds
            output_prefix: Prefix for output paths (optional)
            segment_duration: Duration of segments to score (default 1.0)
            parallel_processing: Whether to process clips in parallel (default True)
            
        Returns:
            list: List of tuples (clip_path, start_time, end_time, score)
        """
        if clip_duration is None:
            clip_duration = config.DEFAULT_CLIP_DURATION
        
        # Ensure model is ready
        self._ensure_model_ready()
        
        # Check video duration to avoid invalid segments
        video_duration = self.clipper.get_video_duration(video_path)
        if video_duration < clip_duration:
            logger.warning(f"Video shorter than clip duration ({video_duration:.2f}s < {clip_duration:.2f}s). Using full video.")
            clip_duration = video_duration
            
        # Check if we can extract the requested number of clips
        max_possible_clips = int(video_duration / (clip_duration + min_gap))
        if max_possible_clips < num_clips:
            logger.warning(f"Video too short for {num_clips} non-overlapping clips. Can extract at most {max_possible_clips}.")
            num_clips = max(1, max_possible_clips)
            
        logger.info(f"Finding {num_clips} viral clips of {clip_duration:.2f}s each in {video_path}")
        
        # Score video segments
        start_time = time.time()
        segment_scores = self.engagement_model.score_video_segments(
            video_path, 
            segment_duration=segment_duration
        )
        
        if not segment_scores or len(segment_scores) == 0:
            logger.error("No scorable segments found")
            return []
            
        logger.debug(f"Scored {len(segment_scores)} segments in {time.time()-start_time:.2f}s")
        
        # Convert to numpy arrays for easier manipulation
        times = np.array([s[0] for s in segment_scores])
        scores = np.array([s[1] for s in segment_scores])
        
        # Sort by time to ensure proper order
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        scores = scores[sort_idx]
        
        # Extract multiple non-overlapping clips
        best_segments = []
        excluded_ranges = []
        
        # First find all the best segments
        for i in range(num_clips):
            # Find best clip avoiding excluded ranges
            best_start, best_score = self._find_best_clip_window_efficient(
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
            best_segments.append((i, best_start, best_end, best_score))
        
        # Now process all the clips, possibly in parallel
        result_clips = []
        
        if parallel_processing and len(best_segments) > 1:
            logger.debug(f"Processing {len(best_segments)} clips in parallel")
            
            def process_clip(args):
                idx, start_time, end_time, score = args
                output_path = self._get_output_path(video_path, idx + 1, output_prefix)
                clip_path = self.clipper.clip(
                    video_path, start_time=start_time, end_time=end_time, output_path=output_path
                )
                return (clip_path, start_time, end_time, score) if clip_path and Path(clip_path).exists() else None
            
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(best_segments))) as executor:
                futures = [executor.submit(process_clip, segment) for segment in best_segments]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        result_clips.append(result)
        else:
            # Process sequentially
            for idx, start_time, end_time, score in best_segments:
                output_path = self._get_output_path(video_path, idx + 1, output_prefix)
                
                # Extract the clip
                clip_path = self.clipper.clip(
                    video_path, 
                    start_time=start_time, 
                    end_time=end_time,
                    output_path=output_path
                )
                
                if clip_path and Path(clip_path).exists():
                    result_clips.append((clip_path, start_time, end_time, score))
                else:
                    logger.error(f"Clipping failed for {start_time:.2f}s-{end_time:.2f}s")
        
        logger.info(f"Extracted {len(result_clips)} out of {len(best_segments)} viral clips")
        return result_clips
    
    def _get_output_path(self, video_path, clip_index, output_prefix=None):
        """Get output path for a clip
        
        Args:
            video_path: Original video path
            clip_index: Clip index number
            output_prefix: Optional output prefix
            
        Returns:
            str: Output path for the clip
        """
        if output_prefix:
            output_path = Path(output_prefix)
            return str(output_path.parent / f"{output_path.stem}_{clip_index}{output_path.suffix}")
        else:
            # Use clip index for automatic naming
            return get_output_path(video_path, suffix="clip", clip_index=clip_index)
    
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
        
        return self._find_best_clip_window_efficient(times, scores, clip_duration, [])
    
    def _find_best_clip_window_exclude_ranges(self, times, scores, clip_duration, excluded_ranges):
        """Legacy method - use _find_best_clip_window_efficient instead"""
        logger.warning("Using deprecated _find_best_clip_window_exclude_ranges method")
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
    
    def _find_best_clip_window_efficient(self, times, scores, clip_duration, excluded_ranges):
        """Find the best window of segments for the clip using efficient algorithm
        
        Args:
            times: Numpy array of segment start times
            scores: Numpy array of segment scores
            clip_duration: Desired clip duration in seconds
            excluded_ranges: List of (start_time, end_time) tuples to exclude
            
        Returns:
            tuple: (best_start_time, best_score)
        """
        if len(times) == 0:
            return None, -1
            
        # We assume the times array is already sorted
        start_time = time.time()
        
        # Calculate the segment duration based on the times array
        # Assume uniform segment duration
        if len(times) > 1:
            segment_duration = times[1] - times[0]
        else:
            segment_duration = 1.0  # Default assumption
            
        # Calculate window size in terms of segments
        window_segments = max(1, int(round(clip_duration / segment_duration)))
        
        if window_segments > len(times):
            window_segments = len(times)
            
        # Create exclusion mask
        excluded_mask = np.zeros_like(times, dtype=bool)
        for (start, end) in excluded_ranges:
            excluded_mask |= (times >= start) & (times <= end)
        
        # Use cumulative sum for efficient window calculation
        cumulative_scores = np.cumsum(np.pad(scores, (1, 0), 'constant'))
        
        best_start_idx = -1
        best_score = -1
        
        for i in range(len(times) - window_segments + 1):
            # Skip if any segment in this window is excluded
            if np.any(excluded_mask[i:i+window_segments]):
                continue
                
            # Calculate window score efficiently using cumulative sum
            window_sum = cumulative_scores[i+window_segments] - cumulative_scores[i]
            window_score = window_sum / window_segments
            
            # Update best window if score is higher
            if window_score > best_score:
                best_start_idx = i
                best_score = window_score
        
        logger.debug(f"Best window search completed in {time.time()-start_time:.4f}s")
        
        # Return the best starting time and score
        if best_start_idx >= 0:
            return times[best_start_idx], best_score
        else:
            return None, -1