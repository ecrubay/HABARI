
# habari-implementation.py
# Core Audio Processing and AI Model Logic for HABARI 2.0

import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_audio(self, filepath):
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        return audio, sr

    def extract_features(self, audio):
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        return {"mfccs": mfccs, "spectrogram": spectrogram}

class EcoSoundModel:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, audio):
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
