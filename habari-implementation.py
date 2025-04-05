# core/audio_processor/feature_extraction.py
import numpy as np
import librosa
import torch
from typing import Dict, List, Tuple, Optional, Union

class FeatureExtractor:
    """
    Advanced feature extraction for animal bioacoustics
    """
    def __init__(self, config: Dict = None):
        """
        Initialize feature extractor with configuration
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.config = config or {
            'sample_rate': 44100,
            'n_fft': 2048,
            'hop_length': 512,
            'n_mels': 128,
            'f_min': 20,
            'f_max': 20000,
            'delta_order': 2,
            'window_size': 5,
        }
        
    def extract_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectrogram from audio
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Spectrogram as numpy array
        """
        return np.abs(librosa.stft(
            audio, 
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length']
        ))
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram optimized for animal vocalizations
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Mel spectrogram as numpy array
        """
        return librosa.feature.melspectrogram(
            y=audio,
            sr=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            n_mels=self.config['n_mels'],
            fmin=self.config['f_min'],
            fmax=self.config['f_max']
        )
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCCs from audio
        
        Args:
            audio: Audio signal as numpy array
            n_mfcc: Number of MFCCs to extract
            
        Returns:
            MFCCs as numpy array
        """
        melspec = self.extract_mel_spectrogram(audio)
        return librosa.feature.mfcc(
            S=librosa.power_to_db(melspec),
            n_mfcc=n_mfcc
        )
    
    def extract_delta_features(self, features: np.ndarray) -> List[np.ndarray]:
        """
        Extract delta and delta-delta features
        
        Args:
            features: Base features (e.g., MFCCs)
            
        Returns:
            List of features with deltas [features, delta, delta-delta]
        """
        deltas = []
        current = features
        
        for d in range(self.config['delta_order'] + 1):
            if d == 0:
                deltas.append(current)
            else:
                current = librosa.feature.delta(
                    current, 
                    width=self.config['window_size'],
                    order=1
                )
                deltas.append(current)
                
        return deltas
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chromagram features (useful for tonal sounds)
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Chromagram as numpy array
        """
        return librosa.feature.chroma_stft(
            y=audio,
            sr=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length']
        )
    
    def extract_spectral_contrast(self, audio: np.ndarray, n_bands: int = 6) -> np.ndarray:
        """
        Extract spectral contrast (useful for vocalization analysis)
        
        Args:
            audio: Audio signal as numpy array
            n_bands: Number of contrast bands
            
        Returns:
            Spectral contrast as numpy array
        """
        return librosa.feature.spectral_contrast(
            y=audio,
            sr=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            n_bands=n_bands
        )
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all available features
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic spectrograms
        features['spectrogram'] = self.extract_spectrogram(audio)
        features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)
        
        # MFCCs with deltas
        mfccs = self.extract_mfcc(audio)
        delta_features = self.extract_delta_features(mfccs)
        features['mfcc'] = delta_features[0]
        features['delta_mfcc'] = delta_features[1]
        
        if len(delta_features) > 2:
            features['delta2_mfcc'] = delta_features[2]
        
        # Additional features
        features['chroma'] = self.extract_chroma(audio)
        features['spectral_contrast'] = self.extract_spectral_contrast(audio)
        
        # Specialized features
        features['spectral_flatness'] = librosa.feature.spectral_flatness(
            y=audio,
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length']
        )
        
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length']
        )
        
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length']
        )
        
        return features
    
    def convert_to_torch(self, features: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert numpy features to PyTorch tensors
        
        Args:
            features: Dictionary of numpy features
            
        Returns:
            Dictionary of PyTorch tensors
        """
        return {k: torch.from_numpy(v).float() for k, v in features.items()}


# core/models/acoustic_embedding.py
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, List, Tuple, Optional, Union

class BioacousticEmbedder(nn.Module):
    """
    Neural network for embedding animal vocalizations into a latent space
    """
    def __init__(
        self, 
        input_dim: int = 128,
        hidden_dims: List[int] = [256, 512],
        output_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize the embedder model
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Output embedding layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.embedding_network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to create embeddings
        
        Args:
            x: Input features [batch_size, sequence_length, input_dim]
            
        Returns:
            Embeddings [batch_size, output_dim]
        """
        # Average pooling across sequence dimension if needed
        if len(x.shape) == 3:
            x = x.mean(dim=1)
            
        return self.embedding_network(x)


class PretrainedEmbedder(nn.Module):
    """
    Wrapper for using pretrained audio models for embedding animal sounds
    """
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        freeze_base: bool = True,
        output_dim: int = 256
    ):
        """
        Initialize with a pretrained model
        
        Args:
            model_name: Name of pretrained model
            freeze_base: Whether to freeze pretrained weights
            output_dim: Final embedding dimension
        """
        super().__init__()
        
        self.base_model = AutoModel.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        base_output_dim = self.base_model.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(base_output_dim, base_output_dim // 2),
            nn.LayerNorm(base_output_dim // 2),
            nn.GELU(),
            nn.Linear(base_output_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create embeddings using the pretrained model
        
        Args:
            x: Input audio [batch_size, sequence_length]
            
        Returns:
            Embeddings [batch_size, output_dim]
        """
        # Get base model outputs
        outputs = self.base_model(x)
        
        # Use the mean of hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Project to desired dimension
        return self.projection(embeddings)


# species/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch

class SpeciesAnalyzer(ABC):
    """
    Abstract base class for species-specific analysis
    """
    def __init__(self, config: Dict = None):
        """
        Initialize species analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.name = "base"
        self.common_name = "Base Species"
        self.scientific_name = "Speciesus basicus"
        self.frequency_range = (0, 0)  # Hz
        self.vocalization_types = []
        
    @abstractmethod
    def detect_calls(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """
        Detect vocalizations in audio
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            List of detected calls with metadata
        """
        pass
    
    @abstractmethod
    def classify_call(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Classify vocalization type
        
        Args:
            audio: Audio signal of isolated call
            sample_rate: Sample rate of audio
            
        Returns:
            Classification results with confidence
        """
        pass
    
    @abstractmethod
    def interpret_meaning(self, call_features: Dict) -> Dict:
        """
        Interpret the potential meaning of a call
        
        Args:
            call_features: Features extracted from call
            
        Returns:
            Interpretation with confidence scores
        """
        pass
    
    def get_species_info(self) -> Dict:
        """
        Get information about this species
        
        Returns:
            Dictionary of species information
        """
        return {
            "name": self.name,
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "frequency_range": self.frequency_range,
            "vocalization_types": self.vocalization_types
        }


# species/birds/models.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class BirdVocalizationClassifier(nn.Module):
    """
    Neural network for classifying bird vocalizations
    """
    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 10,
        hidden_dims: List[int] = [256, 128]
    ):
        """
        Initialize the classifier
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of vocalization classes
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features
            
        Returns:
            Class logits
        """
        return self.model(x)


# species/birds/vocalization_types.py
from typing import Dict, List

# Dictionary of common bird vocalization types and their characteristics
BIRD_VOCALIZATION_TYPES = {
    "song": {
        "description": "Complex, often melodious vocalizations used primarily in territory defense and mate attraction",
        "duration_range": (0.5, 15.0),  # seconds
        "frequency_range": (1000, 8000),  # Hz
        "temporal_pattern": "Repeated phrases with distinct elements",
        "context": ["breeding_season", "dawn_chorus", "territory_defense"]
    },
    "call": {
        "description": "Short, simple vocalizations used for various social communications",
        "duration_range": (0.05, 0.5),  # seconds
        "frequency_range": (2000, 12000),  # Hz
        "temporal_pattern": "Brief, often repeated at intervals",
        "context": ["contact", "alarm", "flight", "feeding"]
    },
    "alarm_call": {
        "description": "Specific call type indicating danger or predator presence",
        "duration_range": (0.01, 0.3),  # seconds
        "frequency_range": (3000, 9000),  # Hz
        "temporal_pattern": "Rapid, repeated with short intervals",
        "context": ["predator_presence", "danger", "threat"]
    },
    "contact_call": {
        "description": "Call used to maintain contact between individuals or group",
        "duration_range": (0.1, 0.4),  # seconds
        "frequency_range": (2000, 6000),  # Hz
        "temporal_pattern": "Regular intervals, often reciprocated",
        "context": ["flock_movement", "pair_bonding", "parent_offspring"]
    },
    "begging_call": {
        "description": "Call used by juveniles to solicit food from parents",
        "duration_range": (0.2, 1.0),  # seconds
        "frequency_range": (3000, 10000),  # Hz
        "temporal_pattern": "Rapid, insistent repeats",
        "context": ["juvenile", "feeding", "parent_offspring"]
    },
    "flight_call": {
        "description": "Call given during flight, often for group coordination",
        "duration_range": (0.05, 0.2),  # seconds
        "frequency_range": (4000, 12000),  # Hz
        "temporal_pattern": "Brief, can be repeated during flight",
        "context": ["flight", "migration", "flock_coordination"]
    },
    "duet": {
        "description": "Coordinated vocalizations between paired birds",
        "duration_range": (1.0, 10.0),  # seconds
        "frequency_range": (1500, 7000),  # Hz
        "temporal_pattern": "Alternating or overlapping elements between individuals",
        "context": ["pair_bonding", "territory_defense", "breeding"]
    }
}

def get_vocalization_by_characteristics(
    duration: float,
    frequency: float,
    context: str = None
) -> List[Dict]:
    """
    Find potential vocalization types based on acoustic characteristics
    
    Args:
        duration: Duration in seconds
        frequency: Dominant frequency in Hz
        context: Optional contextual information
        
    Returns:
        List of matching vocalization types with match scores
    """
    matches = []
    
    for type_name, type_info in BIRD_VOCALIZATION_TYPES.items():
        # Check duration and frequency ranges
        duration_match = (
            type_info["duration_range"][0] <= duration <= type_info["duration_range"][1]
        )
        frequency_match = (
            type_info["frequency_range"][0] <= frequency <= type_info["frequency_range"][1]
        )
        
        # Check context if provided
        context_match = True
        if context and context not in type_info["context"]:
            context_match = False
            
        # Calculate match score
        if duration_match and frequency_match:
            match_score = 0.7
            if context_match:
                match_score += 0.3
                
            matches.append({
                "type": type_name,
                "score": match_score,
                "description": type_info["description"]
            })
    
    # Sort by score
    return sorted(matches, key=lambda x: x["score"], reverse=True)


# dashboard/visualizations/spectrograms.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import librosa
import librosa.display
from typing import Dict, List, Tuple, Optional, Union
import io
import base64

class SpectrogramVisualizer:
    """
    Advanced spectrogram visualization for bioacoustic analysis
    """
    def __init__(self, config: Dict = None):
        """
        Initialize visualizer with configuration
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {
            'figsize': (10, 6),
            'dpi': 100,
            'cmap': 'viridis',
            'db_range': (-80, 0),
            'y_axis': 'log',
            'x_axis': 'time'
        }
    
    def create_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int,
        title: str = "Spectrogram"
    ) -> Figure:
        """
        Create spectrogram visualization
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config['figsize'], dpi=self.config['dpi'])
        
        S = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio)), 
            ref=np.max
        )
        
        img = librosa.display.specshow(
            S,
            sr=sample_rate,
            y_axis=self.config['y_axis'],
            x_axis=self.config['x_axis'],
            ax=ax,
            cmap=self.config['cmap'],
            vmin=self.config['db_range'][0],
            vmax=self.config['db_range'][1]
        )
        
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title)
        
        return fig
    
    def create_mel_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int,
        n_mels: int = 128,
        title: str = "Mel Spectrogram"
    ) -> Figure:
        """
        Create mel spectrogram visualization
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            n_mels: Number of mel bands
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config['figsize'], dpi=self.config['dpi'])
        
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=sample_rate,
            n_mels=n_mels
        )
        
        S_db = librosa.power_to_db(S, ref=np.max)
        
        img = librosa.display.specshow(
            S_db,
            sr=sample_rate,
            y_axis='mel',
            x_axis=self.config['x_axis'],
            ax=ax,
            cmap=self.config['cmap'],
            vmin=self.config['db_range'][0],
            vmax=self.config['db_range'][1]
        )
        
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title)
        
        return fig
    
    def create_comparison_plot(
        self, 
        audios: List[np.ndarray],
        sample_rate: int,
        titles: List[str] = None
    ) -> Figure:
        """
        Create comparison of multiple spectrograms
        
        Args:
            audios: List of audio signals
            sample_rate: Sample rate of audios
            titles: List of titles for each spectrogram
            
        Returns:
            Matplotlib figure with multiple spectrograms
        """
        if titles is None:
            titles = [f"Spectrogram {i+1}" for i in range(len(audios))]
            
        fig, axes = plt.subplots(
            len(audios), 
            1, 
            figsize=(self.config['figsize'][0], self.config['figsize'][1] * len(audios)),
            dpi=self.config['dpi'],
            sharex=True
        )
        
        if len(audios) == 1:
            axes = [axes]
            
        for i, (audio, ax, title) in enumerate(zip(audios, axes, titles)):
            S = librosa.amplitude_to_db(
                np.abs(librosa.stft(audio)), 
                ref=np.max
            )
            
            img = librosa.display.specshow(
                S,
                sr=sample_rate,
                y_axis=self.config['y_axis'] if i == 0 else None,
                x_axis=self.config['x_axis'] if i == len(audios) - 1 else None,
                ax=ax,
                cmap=self.config['cmap'],
                vmin=self.config['db_range'][0],
                vmax=self.config['db_range'][1]
            )
            
            ax.set_title(title)
            
        fig.tight_layout()
        return fig
    
    def create_annotated_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int,
        annotations: List[Dict],
        title: str = "Annotated Spectrogram"
    ) -> Figure:
        """
        Create spectrogram with event annotations
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio