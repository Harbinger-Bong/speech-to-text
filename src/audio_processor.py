import sounddevice as sd
import numpy as np
import queue
import threading

class AudioProcessor:
    """Handle real-time audio input from microphone"""
    
    def __init__(self, sample_rate=16000, chunk_duration=5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def start_recording(self):
        """Start recording from microphone"""
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            dtype=np.float32
        )
        self.stream.start()
        print("Recording started...")
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("Recording stopped.")
    
    def get_audio_chunk(self):
        """Get accumulated audio chunk for transcription"""
        audio_data = []
        target_samples = self.chunk_samples
        
        while len(audio_data) < target_samples and not self.audio_queue.empty():
            audio_data.extend(self.audio_queue.get())
        
        if len(audio_data) > 0:
            return np.array(audio_data).flatten()
        return None

