import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.whisper_stt import WhisperSTT

def main():
    # Initialize STT engine
    stt = WhisperSTT()
    
    # Transcribe a file
    audio_file = "examples/sample_audio/test2.wav"
    
    if not Path(audio_file).exists():
        print(f"Audio file not found: {audio_file}")
        return
    
    print(f"Transcribing {audio_file}...")
    result = stt.transcribe_file(audio_file)
    
    print("\n" + "="*50)
    print("TRANSCRIPTION:")
    print("="*50)
    print(result['text'])
    print("\nDetected language:", result['language'])

if __name__ == "__main__":
    main()

