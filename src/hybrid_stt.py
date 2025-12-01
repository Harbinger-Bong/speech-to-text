import sys
from pathlib import Path
import yaml
import numpy as np

# Import both STT engines
from .whisper_stt import WhisperSTT
from .indic_stt import IndicSTT


class HybridSTT:
    """Hybrid STT combining Whisper (en, ar) and IndicSTT (ml)"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize hybrid STT system"""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("=== Initializing Hybrid STT System ===")
        
        # Initialize Whisper for English & Arabic
        print("\n1. Loading Whisper STT...")
        self.whisper = WhisperSTT(config_path)
        
        # Initialize IndicSTT for Malayalam
        print("\n2. Loading IndicSTT for Malayalam...")
        indic_config = self.config['indic']
        self.indic = IndicSTT(
            model_path=indic_config['model_name'],
            device=indic_config['device']
        )
        
        # Language detection settings
        self.auto_detect = self.config['language_detection']['enabled']
        self.supported_langs = self.config['language_detection']['supported_languages']
        
        self.current_engine = 'whisper'  # Default
        
    def transcribe(self, audio_path=None, audio_array=None, sample_rate=16000):
        """
        Transcribe audio using appropriate engine (auto-detect only)
        
        Args:
            audio_path: Path to audio file
            audio_array: Numpy array of audio
            sample_rate: Audio sample rate
            
        Returns:
            dict with 'text', 'language', 'engine' keys
        """
        # Step 1: Run Whisper (Fast Check)
        if audio_path:
            whisper_result = self.whisper.transcribe_file(audio_path)
        else:
            whisper_result = self.whisper.transcribe_array(audio_array)
        
        detected_lang = whisper_result.get('language', 'en')
        
        # Step 2: If Whisper detects Malayalam, route to IndicSTT
        if detected_lang == 'ml':
            print(f"Detected Malayalam: Routing to IndicSTT...")
            indic_result = self.indic.transcribe(
                audio_path=audio_path,
                audio_array=audio_array,
                sample_rate=sample_rate
            )
            indic_result['engine'] = 'indic'
            self.current_engine = 'indic'
            return indic_result
        
        # Step 3: Otherwise, return Whisper's result
        else:
            print(f"Detected {detected_lang}: Keeping Whisper Output.")
            whisper_result['engine'] = 'whisper'
            self.current_engine = 'whisper'
            return whisper_result
    
    def get_current_engine(self):
        """Get currently active engine"""
        return self.current_engine

