import os
import math
import ffmpeg
import torch
from faster_whisper import WhisperModel
from utils.logger import logger
from utils.helpers import get_output_path
import config

class Captioner:
    def __init__(self, model_size="base", device=None, compute_type="int8"):
        """
        Initialize the Captioner with Faster-Whisper.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2)
            device: 'cuda' or 'cpu' (auto-detected if None)
            compute_type: Quantization (int8, float16)
        """
        if device is None:
            self.device = "cuda" if config.CUDA_AVAILABLE else "cpu"
        else:
            self.device = device
            
        self.compute_type = compute_type
        # fallback to int8 on CPU if float16 is requested but not supported
        if self.device == "cpu" and compute_type == "float16":
            self.compute_type = "int8"
            
        logger.info(f"Loading Whisper model '{model_size}' on {self.device} ({self.compute_type})...")
        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def process_video(self, video_path, output_path=None, style="highlight"):
        """
        Full pipeline: Transcribe -> Generate .ass -> Burn into video.
        """
        if output_path is None:
            output_path = get_output_path(video_path, suffix="captioned")
            
        logger.info(f"Starting captioning pipeline for: {video_path}")
        
        # 1. Transcribe
        segments, info = self.model.transcribe(video_path, word_timestamps=True)
        
        # 2. Generate ASS content
        ass_content = self._generate_ass_file(segments, style=style)
        
        # 3. Save ASS file temporarily
        ass_filename = video_path + ".ass"
        with open(ass_filename, "w", encoding="utf-8") as f:
            f.write(ass_content)
            
        # 4. Burn subtitles using FFmpeg
        try:
            self._burn_subtitles(video_path, ass_filename, output_path)
            logger.info(f"Captioned video saved to: {output_path}")
        finally:
            # Cleanup
            if os.path.exists(ass_filename):
                os.remove(ass_filename)
                
        return output_path

    def _burn_subtitles(self, video_path, ass_path, output_path):
        """Burn .ass subtitles into video using ffmpeg-python."""
        stream = ffmpeg.input(video_path)
        audio = stream.audio
        
        # Re-encode video with subtitles filter
        # Note: 'ass' filter requires the file path. Windows paths need escaping in some ffmpeg versions, 
        # but ffmpeg-python usually handles it. strict='experimental' might be needed for aac.
        
        # Fix for Windows path escaping in filter argument
        # Simply use basename and ensure we are referring to the file in the CWD context if possible
        # Or, since we know we are running from project root and outputs are in project root subdirs,
        # relative path 'outputs/...' should work fine with forward slashes.
        
        try:
            rel_path = os.path.relpath(ass_path)
            # FORCE forward slashes and NO escaped colons. 
            # FFmpeg on Windows handles "outputs/video.mp4.ass" perfectly fine.
            ass_path_fixed = rel_path.replace('\\', '/')
        except Exception:
            # If relpath fails (different drives), we must use absolute but try to avoid the colon issue
            # by using the 8.3 filename or just standardizing.
            # But for this user case, relpath MUST work.
            # Let's fallback to just a simple replace, NO colon escaping which caused C\\\:/
            ass_path_fixed = ass_path.replace('\\', '/')

        logger.info(f"Burn subtitles using ASS file: {ass_path_fixed}")
        
        stream = stream.filter('ass', ass_path_fixed)
        
        out = ffmpeg.output(stream, audio, output_path, vcodec='libx264', acodec='aac', loglevel='error')
        out.run(overwrite_output=True)

    def _generate_ass_file(self, segments, style="highlight"):
        """
        Convert Whisper segments to ASS format with Karaoke tags.
        """
        # Header
        header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,80,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []
        
        for segment in segments:
            for word in segment.words:
                start = self._format_time(word.start)
                end = self._format_time(word.end)
                
                # Active word highlight (Karaoke style) is complex in ASS with standard tags across a sentence.
                # A simpler "TikTok" style is to just show the current word or small phrase.
                # OR show the whole sentence with the current word highlighted.
                # To highlight just the current word in a sentence requires splitting the sentence into multiple events 
                # or using advanced {\k} tags which are for karaoke timing.
                
                # For "Alex Hormozi" style (one or two words at a time, big and centered):
                # We simply create an event for each word (or small group).
                
                text = word.word.strip()
                
                # Uppercase for impact
                text = text.upper()
                
                # Add some color/animation? 
                # Basic style: Just the word, yellow/white.
                # We used "Default" style above which is White with Black outline.
                # Let's make it Yellow (&H0000FFFF in BGR) or Green? 
                # Let's stick to the Default style defined in header for now (White text).
                
                # To make it pop, we can add a specific color tag {\c&H00FFFF&} (Yellow)
                pop_text = f"{{\\c&H00FFFF&}}{text}"
                
                # Event line
                # Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
                line = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pop_text}"
                events.append(line)

        return header + "\n".join(events)

    def _format_time(self, seconds):
        """Format seconds to H:MM:SS.cs (centiseconds) for ASS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds - int(seconds)) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"

if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        c = Captioner()
        c.process_video(sys.argv[1])
