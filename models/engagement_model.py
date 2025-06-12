import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path

import config
from utils.logger import logger

class EngagementModel:
    def __init__(self, model_path=None):
        """Initialize the engagement model
        
        Args:
            model_path: Path to model weights (defaults to config.MODEL_WEIGHTS_PATH)
        """
        self.model_path = model_path or config.MODEL_WEIGHTS_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.initialized = False
        
    def initialize(self):
        """Load and initialize the model"""
        try:
            if not Path(self.model_path).exists():
                logger.warning(f"Model weights not found at {self.model_path}. Using fallback methods.")
                return False
            
            # Here you would load your actual model
            # This is a placeholder implementation
            self.model = SimpleEngagementModel().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            self.initialized = True
            logger.info(f"Engagement model initialized on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing engagement model: {str(e)}")
            return False
    
    def score_video_segments(self, video_path, segment_duration=1.0):
        """Score video segments for viral potential
        
        Args:
            video_path: Path to video file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            list: List of (start_time, score) tuples
        """
        # If model isn't initialized, use fallback method
        if not self.initialized:
            return self._fallback_scoring(video_path, segment_duration)
        
        try:
            # Process the video and get scores
            # This would use the actual model to generate scores
            # Here's a placeholder implementation
            scores = []
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frames_per_segment = int(segment_duration * fps)
            
            for start_frame in range(0, total_frames, frames_per_segment):
                # Get frames for this segment
                segment_frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for _ in range(min(frames_per_segment, total_frames - start_frame)):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Preprocess frame
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    segment_frames.append(frame)
                
                if segment_frames:
                    # Convert to tensor and get score
                    input_tensor = self._preprocess_frames(segment_frames)
                    with torch.no_grad():
                        score = self.model(input_tensor).item()
                    
                    start_time = start_frame / fps
                    scores.append((start_time, score))
            
            cap.release()
            return scores
            
        except Exception as e:
            logger.error(f"Error scoring video segments: {str(e)}")
            return self._fallback_scoring(video_path, segment_duration)
    
    def _preprocess_frames(self, frames):
        """Preprocess frames for model input"""
        # Convert to numpy array and normalize
        frames_array = np.array(frames) / 255.0
        frames_array = np.transpose(frames_array, (0, 3, 1, 2))  # to NCHW format
        
        # Convert to tensor
        return torch.FloatTensor(frames_array).to(self.device)
    
    def _fallback_scoring(self, video_path, segment_duration=1.0):
        """Fallback method for scoring when the model isn't available
        
        Uses simple heuristics like audio volume and motion detection
        """
        try:
            from moviepy.editor import VideoFileClip
            
            scores = []
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                
                for start_time in np.arange(0, duration - segment_duration, segment_duration):
                    end_time = start_time + segment_duration
                    segment = clip.subclip(start_time, end_time)
                    
                    # Audio energy (volume)
                    audio_score = 0
                    if segment.audio is not None:
                        audio_array = segment.audio.to_soundarray()
                        audio_energy = np.mean(np.abs(audio_array))
                        audio_score = min(1.0, audio_energy * 10)
                    
                    # Simple motion detection
                    motion_score = 0
                    try:
                        # Sample frames at the beginning, middle and end of the segment
                        frame1 = segment.get_frame(0)
                        frame2 = segment.get_frame(segment_duration / 2)
                        frame3 = segment.get_frame(segment_duration - 0.1)
                        
                        # Convert to grayscale
                        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                        gray3 = cv2.cvtColor(frame3, cv2.COLOR_RGB2GRAY)
                        
                        # Calculate differences
                        diff1 = np.abs(gray2 - gray1).mean() / 255
                        diff2 = np.abs(gray3 - gray2).mean() / 255
                        motion_score = (diff1 + diff2) / 2
                    except:
                        # If we can't get frames for some reason, just use audio score
                        motion_score = 0
                    
                    # Combined score
                    combined_score = (audio_score * 0.7) + (motion_score * 0.3)
                    scores.append((start_time, combined_score))
        
            return scores
            
        except Exception as e:
            logger.error(f"Error in fallback scoring: {str(e)}")
            # Return random scores as last resort
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                return [(t, np.random.random()) 
                        for t in np.arange(0, duration - segment_duration, segment_duration)]


class SimpleEngagementModel(nn.Module):
    """Simple placeholder model for engagement prediction"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (B, T, C, H, W)
        batch_size, time_steps, C, H, W = x.shape
        
        # Reshape to process all frames
        x = x.view(-1, C, H, W)
        x = self.features(x)
        x = self.classifier(x)
        
        # Reshape back and average over time
        x = x.view(batch_size, time_steps)
        return x.mean(dim=1)