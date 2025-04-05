annotations: List of dictionaries with fields:
                - start_time: start time in seconds
                - end_time: end time in seconds
                - label: annotation text
                - color: (optional) color for the box
            title: Plot title
            
        Returns:
            Matplotlib figure with annotations
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
        
        # Add annotations
        for annotation in annotations:
            start_time = annotation['start_time']
            end_time = annotation['end_time']
            label = annotation['label']
            color = annotation.get('color', 'red')
            
            # Draw rectangle for the event
            rect = plt.Rectangle(
                (start_time, 0),
                end_time - start_time,
                1.0,  # Full height in normalized coordinates
                fill=False,
                edgecolor=color,
                linewidth=2,
                transform=ax.get_xaxis_transform()  # Transform to coordinate space
            )
            ax.add_patch(rect)
            
            # Add text label
            ax.text(
                (start_time + end_time) / 2,
                0.95,
                label,
                horizontalalignment='center',
                backgroundcolor='white',
                alpha=0.7,
                transform=ax.get_xaxis_transform()
            )
        
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title)
        
        return fig
        
    def figure_to_base64(self, fig: Figure) -> str:
        """
        Convert a matplotlib figure to base64 encoded string
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64 encoded string
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_str


# dashboard/web/app.py
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
from typing import Dict, Any
import json

# Import HABARI modules
from core.audio_processor.feature_extraction import FeatureExtractor
from species.registry import get_species_analyzer
from dashboard.visualizations.spectrograms import SpectrogramVisualizer

app = Flask(__name__)

# Initialize components
feature_extractor = FeatureExtractor()
spectrogram_visualizer = SpectrogramVisualizer()

@app.route('/')
def index():
    """Render the dashboard homepage"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Handle audio file upload and initial analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Save the file temporarily
    temp_path = os.path.join('/tmp', file.filename)
    file.save(temp_path)
    
    try:
        # Load the audio file
        audio, sr = librosa.load(temp_path, sr=None)
        
        # Generate spectrograms
        spec_fig = spectrogram_visualizer.create_spectrogram(audio, sr)
        melspec_fig = spectrogram_visualizer.create_mel_spectrogram(audio, sr)
        
        # Extract features
        features = feature_extractor.extract_all_features(audio)
        
        # Create basic response
        response = {
            'filename': file.filename,
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'spectrograms': {
                'regular': spectrogram_visualizer.figure_to_base64(spec_fig),
                'mel': spectrogram_visualizer.figure_to_base64(melspec_fig)
            },
            'feature_shapes': {k: v.shape for k, v in features.items()}
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/analyze/<species>', methods=['POST'])
def analyze_species(species: str):
    """
    Perform species-specific analysis
    
    Args:
        species: Species identifier string
        
    Returns:
        JSON with analysis results
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    try:
        # Get species analyzer
        analyzer = get_species_analyzer(species)
        if not analyzer:
            return jsonify({'error': f'Species analyzer for {species} not found'}), 404
            
        # Load audio file
        file = request.files['file']
        temp_path = os.path.join('/tmp', file.filename)
        file.save(temp_path)
        
        audio, sr = librosa.load(temp_path, sr=None)
        
        # Detect calls
        calls = analyzer.detect_calls(audio, sr)
        
        # Create visualization with annotations
        annotations = [
            {
                'start_time': call['start_time'],
                'end_time': call['end_time'],
                'label': call['type'],
                'color': 'red' if call['confidence'] > 0.7 else 'orange'
            }
            for call in calls
        ]
        
        fig = spectrogram_visualizer.create_annotated_spectrogram(
            audio, sr, annotations, f"{species.capitalize()} Call Analysis"
        )
        
        # Response
        response = {
            'species': species,
            'species_info': analyzer.get_species_info(),
            'detections': calls,
            'annotated_spectrogram': spectrogram_visualizer.figure_to_base64(fig)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/species')
def list_species():
    """Get list of available species analyzers"""
    from species.registry import get_available_species
    
    species_list = get_available_species()
    return jsonify({'species': species_list})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


# core/models/translation.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

class BioacousticTranslator(nn.Module):
    """
    Neural network for "translating" between different species' vocalizations
    """
    def __init__(
        self,
        input_dim: int = 256,
        latent_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3
    ):
        """
        Initialize the translator model
        
        Args:
            input_dim: Input embedding dimension
            latent_dim: Shared latent space dimension
            output_dim: Output embedding dimension
            num_layers: Number of transformer layers
        """
        super().__init__()
        
        # Encoder for source species
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Transformer for contextual processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Decoder for target species
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, output_dim)
        )
        
    def forward(
        self, 
        source_embedding: torch.Tensor,
        sequence_length: int = 10
    ) -> torch.Tensor:
        """
        Translate source species embedding to target species embedding
        
        Args:
            source_embedding: Source species embedding [batch_size, input_dim]
            sequence_length: Length of sequence to generate
            
        Returns:
            Target species embedding [batch_size, sequence_length, output_dim]
        """
        batch_size = source_embedding.shape[0]
        
        # Encode to latent space
        latent = self.encoder(source_embedding)
        
        # Expand to sequence
        latent_seq = latent.unsqueeze(1).expand(-1, sequence_length, -1)
        
        # Add positional information
        pos = torch.arange(0, sequence_length, device=latent.device).unsqueeze(0).expand(batch_size, -1)
        pos_encoding = self._positional_encoding(pos, latent.shape[-1])
        latent_seq = latent_seq + pos_encoding
        
        # Process with transformer
        context = self.transformer(latent_seq)
        
        # Decode to target space
        target_seq = self.decoder(context)
        
        return target_seq
        
    def _positional_encoding(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding
        
        Args:
            positions: Position indices [batch_size, seq_len]
            dim: Embedding dimension
            
        Returns:
            Positional encoding [batch_size, seq_len, dim]
        """
        batch_size, seq_len = positions.shape
        
        # Create position encoding
        pe = torch.zeros(batch_size, seq_len, dim, device=positions.device)
        
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=positions.device) * 
            (-math.log(10000.0) / dim)
        )
        
        for b in range(batch_size):
            pe[b, :, 0::2] = torch.sin(positions[b].unsqueeze(1) * div_term)
            pe[b, :, 1::2] = torch.cos(positions[b].unsqueeze(1) * div_term)
            
        return pe


# species/registry.py
import importlib
import os
from typing import Dict, List, Optional, Any
import inspect

# Global registry of species analyzers
_SPECIES_REGISTRY = {}

def register_species(name: str, analyzer_class: Any) -> None:
    """
    Register a species analyzer
    
    Args:
        name: Unique identifier for the species
        analyzer_class: Class implementing species analysis
    """
    global _SPECIES_REGISTRY
    _SPECIES_REGISTRY[name] = analyzer_class

def get_species_analyzer(name: str) -> Optional[Any]:
    """
    Get species analyzer by name
    
    Args:
        name: Species identifier
        
    Returns:
        Instance of species analyzer or None if not found
    """
    if name not in _SPECIES_REGISTRY:
        return None
        
    return _SPECIES_REGISTRY[name]()

def get_available_species() -> List[str]:
    """
    Get list of available species analyzers
    
    Returns:
        List of species names
    """
    return list(_SPECIES_REGISTRY.keys())

def auto_discover_species() -> None:
    """
    Automatically discover and register species analyzers
    """
    from species.base import SpeciesAnalyzer
    
    # Get the path to the species directory
    species_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all subdirectories in the species directory
    for item in os.listdir(species_dir):
        item_path = os.path.join(species_dir, item)
        
        # Skip non-directories and special names
        if not os.path.isdir(item_path) or item.startswith('__'):
            continue
            
        # Try to import the module
        try:
            module_name = f"species.{item}.models"
            module = importlib.import_module(module_name)
            
            # Find and register all species analyzer classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, SpeciesAnalyzer) and 
                    obj != SpeciesAnalyzer):
                    
                    # Extract species name from class
                    species_name = item
                    if hasattr(obj, 'name') and obj.name:
                        species_name = obj.name
                        
                    register_species(species_name, obj)
                    print(f"Registered species analyzer: {species_name}")
                    
        except (ImportError, AttributeError) as e:
            print(f"Error discovering species in {item}: {e}")
            
# Auto-discover species when module is imported
auto_discover_species()
