import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import librosa


class IndicSTT:
    """Malayalam speech recognition using Wav2Vec2"""
    
    def __init__(self, model_path=None, device="cuda"):
        """Initialize Malayalam STT model"""
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Use public Malayalam model
        model_name = "gvs/wav2vec2-large-xlsr-malayalam"
        
        print(f"Loading Malayalam STT model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        
        print("Malayalam STT model loaded successfully!")
    
    def transcribe(self, audio_path=None, audio_array=None, sample_rate=16000):
        """
        Transcribe audio to Malayalam text
        
        Args:
            audio_path: Path to audio file
            audio_array: Numpy array of audio data
            sample_rate: Sample rate of audio
            
        Returns:
            dict with 'text' and 'language' keys
        """
        # Load audio
        if audio_path:
            audio, sr = librosa.load(audio_path, sr=16000)
        elif audio_array is not None:
            audio = audio_array
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        else:
            raise ValueError("Either audio_path or audio_array must be provided")
        
        # Process audio
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Move to device
        input_values = inputs.input_values.to(self.device)
        
        # Transcribe
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return {
            'text': transcription,
            'language': 'ml'
        }
    
    def transcribe_stream(self, audio_chunk, sample_rate=16000):
        """Transcribe audio chunk for streaming"""
        return self.transcribe(audio_array=audio_chunk, sample_rate=sample_rate)

