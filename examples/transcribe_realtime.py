import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.whisper_stt import WhisperSTT
from src.audio_processor import AudioProcessor
import time

def main():
    # Initialize components
    stt = WhisperSTT()
    audio_proc = AudioProcessor(chunk_duration=5)
    
    print("Real-time transcription starting...")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")
    
    audio_proc.start_recording()
    
    try:
        while True:
            # Get audio chunk
            audio_chunk = audio_proc.get_audio_chunk()
            
            if audio_chunk is not None and len(audio_chunk) > 0:
                # Transcribe
                result = stt.transcribe_array(audio_chunk)
                
                # Print result
                if result['text'].strip():
                    print(f"[{time.strftime('%H:%M:%S')}] {result['text']}")
            
            time.sleep(0.5)  # Small delay
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        audio_proc.stop_recording()

if __name__ == "__main__":
    main()

