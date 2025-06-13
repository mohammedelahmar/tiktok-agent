import unittest
import os
import tempfile
from pathlib import Path
from moviepy.editor import VideoFileClip

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.clipper import VideoClipper
import config

class TestVideoClipper(unittest.TestCase):
    def setUp(self):
        self.clipper = VideoClipper()
        # Create a small test video
        self.temp_dir = tempfile.mkdtemp()
        self.test_video = os.path.join(self.temp_dir, "test_video.mp4")
        
        # Create a small blank video for testing
        # Convert Path to string for VideoFileClip
        sample_path = str(Path(config.PROJECT_ROOT) / "tests" / "resources" / "sample_video.mp4")
        clip = VideoFileClip(sample_path)
        clip.write_videofile(self.test_video, codec="libx264", audio_codec="aac")
        
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_get_video_duration(self):
        """Test retrieving video duration"""
        duration = self.clipper.get_video_duration(self.test_video)
        self.assertTrue(duration > 0, "Duration should be positive")
    
    def test_clip_video(self):
        """Test clipping a video"""
        output_path = os.path.join(self.temp_dir, "output_clip.mp4")
        result = self.clipper.clip(
            self.test_video, 
            start_time=0, 
            duration=1.0, 
            output_path=output_path
        )
        
        self.assertIsNotNone(result, "Clipping should return a path")
        self.assertTrue(os.path.exists(result), "Output clip should exist")
        
        # Verify clip duration
        with VideoFileClip(result) as clip:
            self.assertAlmostEqual(clip.duration, 1.0, delta=0.1)

if __name__ == "__main__":
    # Create test resources directory if it doesn't exist
    os.makedirs(Path(config.PROJECT_ROOT) / "tests" / "resources", exist_ok=True)
    unittest.main()