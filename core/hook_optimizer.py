import librosa
import numpy as np
from utils.logger import logger

class HookOptimizer:
    def __init__(self):
        """
        Initialize the HookOptimizer.
        """
        # High-impact keywords that often start viral shorts
        self.hook_keywords = [
            "wait", "stop", "look", "secret", "crazy", "omg", "wow", 
            "listen", "did you know", "watch this", "warning", "never",
            "mistake", "hack", "trick", "reveal", "finally", "pov"
        ]
    
    def calculate_hook_score(self, video_path, start_time, duration=3.0, transcript_text=None):
        """
        Calculate a composite score (0.0 - 1.0) for the hook (start of the clip).
        
        Args:
            video_path: Path to the video file
            start_time: Start time of the clip in seconds
            duration: Duration to analyze (first 3 seconds usually)
            transcript_text: Text spoken in the first few seconds (optional)
            
        Returns:
            float: Hook score
        """
        audio_score = self._analyze_audio(video_path, start_time, duration)
        text_score = self._analyze_text(transcript_text)
        
        # Weighted average
        # Audio (loudness/energy) is 60%, Text (keywords) is 40%
        # If no text provided, rely 100% on audio
        if transcript_text:
            final_score = (audio_score * 0.6) + (text_score * 0.4)
        else:
            final_score = audio_score
            
        return round(final_score, 3)

    def _analyze_audio(self, video_path, start, duration):
        """
        Analyze audio loudness and energy.
        Returns a score 0.0 - 1.0 based on RMS energy and spectral centroid.
        """
        try:
            # Load just the hook segment
            # sr=None preserves native sampling rate for speed
            y, sr = librosa.load(video_path, sr=None, offset=start, duration=duration)
            
            if len(y) == 0:
                return 0.5 # Neutral score on error/empty
                
            # 1. Loudness (RMS)
            rms = librosa.feature.rms(y=y)[0]
            avg_rms = np.mean(rms)
            
            # Normalize RMS roughly. 
            # Typical speech is around 0.01 to 0.1. 
            # Let's heavily weight "loud/clear" starts.
            # Sigmoid-ish normalization
            loudness_score = min(avg_rms * 20, 1.0) 
            
            # 2. Energy/Brightness (Spectral Centroid)
            # High spectral centroid = "bright", energetic, shouting, impact sounds
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_cent = np.mean(cent)
            
            # Normalize centroid (human voice usually < 3000Hz, higher is more energetic/noise)
            # 4000Hz+ is very bright
            energy_score = min(avg_cent / 4000, 1.0)
            
            # Combined audio score
            return (loudness_score * 0.7) + (energy_score * 0.3)
            
        except Exception as e:
            logger.warning(f"Audio analysis failed for hook at {start}s: {e}")
            return 0.5 # Default to neutral

    def _analyze_text(self, text):
        """
        Check for high-impact keywords in the transcript.
        """
        if not text:
            return 0.0
            
        text_lower = text.lower()
        
        # Check for keywords
        keyword_hits = sum(1 for word in self.hook_keywords if word in text_lower)
        
        # If at least one keyword, give high score.
        # 1 keyword = 0.7, 2+ = 1.0
        if keyword_hits >= 2:
            return 1.0
        elif keyword_hits == 1:
            return 0.7
        
        # Check for question mark (curiosity gap)
        if "?" in text:
            return 0.6
            
        return 0.3 # Baseline score for having text at all
