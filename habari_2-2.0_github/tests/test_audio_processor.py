
# test_audio_processor.py

import unittest
import numpy as np
from src.habari.core.audio_processor.feature_extraction import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor()

    def test_extract_features(self):
        audio = np.random.randn(16000)
        features = self.processor.extract_features(audio)
        self.assertIn("mfccs", features)
        self.assertIn("spectrogram", features)
