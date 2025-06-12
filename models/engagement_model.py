import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import os
import tempfile
import librosa
import scipy.signal
import warnings

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
        self._load_face_detector()
        
    def _load_face_detector(self):
        """Load face detection model using OpenCV DNN"""
        try:
            # Path to the pre-trained models directory
            models_dir = Path(config.PROJECT_ROOT) / "models" / "face_detection"
            os.makedirs(models_dir, exist_ok=True)
            
            # File paths for the face detection model
            prototxt_path = models_dir / "deploy.prototxt"
            caffemodel_path = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            
            # Check if the model files exist, otherwise create placeholder warning
            if not prototxt_path.exists() or not caffemodel_path.exists():
                logger.warning("Face detection model files not found. Face detection will be disabled.")
                self.face_detector = None
                return
                
            # Load the DNN model
            self.face_detector = cv2.dnn.readNetFromCaffe(
                str(prototxt_path),
                str(caffemodel_path)
            )
            logger.info("Face detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading face detector: {str(e)}")
            self.face_detector = None

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
    
    def _detect_faces(self, frame, confidence_threshold=0.5):
        """Detect faces in a frame using OpenCV DNN
        
        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Minimum confidence for face detection
            
        Returns:
            int: Number of faces detected
        """
        if self.face_detector is None:
            return 0
            
        try:
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Create a 300x300 blob from the frame
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                1.0, (300, 300), 
                [104, 117, 123], 
                swapRB=False, 
                crop=False
            )
            
            # Pass the blob through the network
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            # Count faces with confidence above threshold
            face_count = 0
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    face_count += 1
                    
            return face_count
        except Exception as e:
            logger.debug(f"Face detection error: {str(e)}")
            return 0

    def _detect_audio_features(self, audio_array, sr, segment_duration):
        """Detect audio features like volume peaks, laughter, and shouting
        
        Args:
            audio_array: Audio samples array
            sr: Sample rate
            segment_duration: Duration of the segment in seconds
            
        Returns:
            dict: Dictionary with audio feature scores
        """
        try:
            result = {
                'volume_peak_score': 0.0,
                'laughter_score': 0.0,
                'shouting_score': 0.0
            }
            
            if audio_array is None or len(audio_array) == 0:
                return result
                
            # Convert stereo to mono if needed
            if audio_array.ndim > 1 and audio_array.shape[1] > 1:
                audio_mono = np.mean(audio_array, axis=1)
            else:
                audio_mono = audio_array.flatten()
            
            # Normalize audio
            audio_mono = audio_mono / (np.max(np.abs(audio_mono)) + 1e-8)
            
            # 1. Volume peaks detection
            # - Calculate RMS energy
            frame_length = int(0.025 * sr)  # 25ms frame
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Use librosa if available, otherwise fallback to manual calculation
            try:
                import librosa
                rms = librosa.feature.rms(
                    y=audio_mono, 
                    frame_length=frame_length,
                    hop_length=hop_length
                )[0]
            except (ImportError, Exception):
                # Manual RMS calculation
                n_frames = 1 + (len(audio_mono) - frame_length) // hop_length
                rms = np.zeros(n_frames)
                for i in range(n_frames):
                    start = i * hop_length
                    end = start + frame_length
                    rms[i] = np.sqrt(np.mean(audio_mono[start:end]**2))
            
            # Calculate peak-to-average ratio
            avg_rms = np.mean(rms)
            if avg_rms > 0:
                peaks = rms[rms > 1.5 * avg_rms]
                peak_ratio = len(peaks) / len(rms) if len(rms) > 0 else 0
                result['volume_peak_score'] = min(1.0, peak_ratio * 5)  # Scale appropriately
            
            # 2. Laughter detection (simplified heuristic)
            # - Laughter often has rhythmic energy patterns
            try:
                # Calculate spectral flux
                spec = np.abs(librosa.stft(audio_mono, n_fft=frame_length, hop_length=hop_length))
                flux = np.sum(np.diff(spec, axis=1)**2, axis=0)
                
                # Laughter often has a rhythmic pattern around 4-8 Hz
                if len(flux) > 1:
                    # Calculate rhythm strength in the laughter range (4-8 Hz)
                    tempo_frame_rate = sr / hop_length
                    rhythm_freqs = np.fft.rfftfreq(len(flux), 1/tempo_frame_rate)
                    rhythm_spec = np.abs(np.fft.rfft(flux))
                    
                    # Check energy in the 4-8 Hz band (typical for laughter)
                    laughter_band = (rhythm_freqs >= 4) & (rhythm_freqs <= 8)
                    if np.any(laughter_band):
                        laughter_energy = np.mean(rhythm_spec[laughter_band])
                        total_energy = np.mean(rhythm_spec)
                        if total_energy > 0:
                            result['laughter_score'] = min(1.0, laughter_energy / total_energy * 2)
            except Exception:
                # Fallback if spectral processing fails
                pass
                
            # 3. Shouting detection
            # - Shouting typically has high energy and specific spectral characteristics
            try:
                # Check for sustained high energy
                high_energy_ratio = np.mean(rms > 0.7 * np.max(rms))
                
                # Calculate spectral centroid (shouting tends to have higher centroid)
                centroid = librosa.feature.spectral_centroid(
                    y=audio_mono, sr=sr, 
                    n_fft=frame_length, 
                    hop_length=hop_length
                )[0]
                
                # Higher centroid typically indicates shouting/excitement
                centroid_normalized = np.mean(centroid) / (sr/2)  # Normalize by Nyquist frequency
                result['shouting_score'] = min(1.0, high_energy_ratio * 0.5 + centroid_normalized * 0.5)
                
            except Exception:
                # Simple fallback based just on volume
                high_vol_ratio = np.mean(np.abs(audio_mono) > 0.6)
                result['shouting_score'] = min(1.0, high_vol_ratio * 2)
                
            return result
            
        except Exception as e:
            logger.debug(f"Audio feature detection error: {str(e)}")
            return {
                'volume_peak_score': 0.0,
                'laughter_score': 0.0,
                'shouting_score': 0.0
            }

    def _fallback_scoring(self, video_path, segment_duration=1.0):
        """Enhanced fallback method for scoring when the model isn't available
        
        Uses audio features (peaks, laughter, shouting) and face detection
        """
        try:
            from moviepy.editor import VideoFileClip
            
            scores = []
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                
                # Pre-sample frames for face detection to avoid re-loading the video
                # Sample a frame every 0.5 seconds
                face_samples = {}
                sampling_rate = 0.5  # seconds
                
                logger.info("Pre-sampling frames for face detection...")
                for t in np.arange(0, duration, sampling_rate):
                    try:
                        frame = clip.get_frame(t)
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        face_count = self._detect_faces(frame_bgr)
                        face_samples[t] = face_count
                    except:
                        continue
                
                logger.info(f"Analyzing segments with enhanced features...")
                for start_time in np.arange(0, duration - segment_duration, segment_duration):
                    end_time = start_time + segment_duration
                    segment = clip.subclip(start_time, end_time)
                    
                    # ---------------- AUDIO ANALYSIS ----------------
                    audio_features = {
                        'volume_peak_score': 0.0,
                        'laughter_score': 0.0,
                        'shouting_score': 0.0
                    }
                    
                    if segment.audio is not None:
                        # Extract audio features
                        audio_array = segment.audio.to_soundarray()
                        sr = segment.audio.fps
                        
                        # Get detailed audio features
                        audio_features = self._detect_audio_features(audio_array, sr, segment_duration)
                        
                        # Basic volume level (legacy code)
                        audio_energy = np.mean(np.abs(audio_array))
                        audio_score = min(1.0, audio_energy * 10) 
                    else:
                        audio_score = 0.0
                    
                    # ---------------- MOTION DETECTION ----------------
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
                    
                    # ---------------- FACE DETECTION ----------------
                    # Use the pre-sampled face data
                    face_times = [t for t in face_samples.keys() 
                                if start_time <= t < end_time]
                    
                    max_faces = 0
                    avg_faces = 0
                    
                    if face_times:
                        face_counts = [face_samples[t] for t in face_times]
                        max_faces = max(face_counts)
                        avg_faces = sum(face_counts) / len(face_counts)
                    
                    # Face score: more faces = higher score, with diminishing returns
                    face_score = min(1.0, (0.5 * avg_faces + 0.5 * max_faces) / 3)
                    
                    # ---------------- COMBINED SCORING ----------------
                    # Base score from audio and motion
                    base_score = (audio_score * 0.6) + (motion_score * 0.4)
                    
                    # Feature boosts
                    audio_boost = (
                        audio_features['volume_peak_score'] * 0.4 +
                        audio_features['laughter_score'] * 0.3 +
                        audio_features['shouting_score'] * 0.3
                    ) * 0.25  # Scale the boost (0-0.25)
                    
                    face_boost = face_score * 0.15  # Scale the boost (0-0.15)
                    
                    # Final combined score
                    combined_score = min(1.0, base_score + audio_boost + face_boost)
                    
                    # Log high-scoring segments with their features for debugging
                    if combined_score > 0.8:
                        logger.debug(f"High score segment at {start_time:.1f}s: {combined_score:.3f} " +
                                   f"(Faces: {max_faces}, Peaks: {audio_features['volume_peak_score']:.2f}, " +
                                   f"Laughter: {audio_features['laughter_score']:.2f}, " +
                                   f"Shouting: {audio_features['shouting_score']:.2f})")
                    
                    scores.append((start_time, combined_score))
        
            return scores
            
        except Exception as e:
            logger.error(f"Error in enhanced fallback scoring: {str(e)}")
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