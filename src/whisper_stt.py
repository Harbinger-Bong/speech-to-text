import whisper
import torch
import yaml
from pathlib import Path

class WhisperSTT:
    """
    Speech-to-Text engine using OpenAI Whisper
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize Whisper model with config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['model']
        self.device = model_config['device'] if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading Whisper '{model_config['size']}' model on {self.device}...")
        self.model = whisper.load_model(
            model_config['size'], 
            device=self.device
        )
        print("Model loaded successfully!")
    
    def transcribe_file(self, audio_path, language=None):
        """
        Transcribe an audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional)
        
        Returns:
            dict: Transcription result with text, segments, and metadata
        """
        language = language or self.config['model']['language']
        
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            fp16=self.config['performance']['fp16'] and self.device == 'cuda',
            beam_size=self.config['performance']['beam_size'],
            best_of=self.config['performance']['best_of']
        )
        
        return {
            'text': result['text'],
            'segments': result['segments'],
            'language': result['language']
        }
    
    def transcribe_array(self, audio_array, language=None):
        """
        Transcribe numpy audio array
        
        Args:
            audio_array: numpy array of audio samples
            language: Language code (optional)
        
        Returns:
            dict: Transcription result
        """
        language = language or self.config['model']['language']
        
        result = self.model.transcribe(
            audio_array,
            language=language,
            fp16=self.config['performance']['fp16'] and self.device == 'cuda'
        )
        
        return result

