import unittest
import os
import tempfile
import numpy as np
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.engagement_model import EngagementModel
import config

class TestEngagementModel(unittest.TestCase):
    def setUp(self):
        self.model = EngagementModel()
        # Reference to a test video
        self.test_video = str(Path(config.PROJECT_ROOT) / "tests" / "resources" / "sample_video.mp4")
        
    def test_initialization(self):
        """Test model initialization"""
        # Should not raise an exception
        self.model.initialize()
        
    def test_face_detector(self):
        """Test face detector loading"""
        # Face detector should be loaded or None
        self.assertIn(self.model.face_detector, [None, type(self.model.face_detector)])
    
    def test_fallback_scoring(self):
        """Test fallback scoring method"""
        # If a test video exists
        if os.path.exists(self.test_video):
            scores = self.model._fallback_scoring(self.test_video, segment_duration=1.0)
            self.assertIsInstance(scores, list, "Should return a list of scores")
            if scores:  # If video is valid and scores are generated
                self.assertIsInstance(scores[0], tuple, "Scores should be (time, value) tuples")
                self.assertEqual(len(scores[0]), 2, "Score tuples should have 2 elements")
                self.assertIsInstance(scores[0][0], (int, float), "Time should be numeric")
                self.assertIsInstance(scores[0][1], (int, float), "Score should be numeric")

if __name__ == "__main__":
    unittest.main()