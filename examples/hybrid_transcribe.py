import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_stt import HybridSTT


def main():
    # Initialize hybrid STT
    print("Initializing STT System...\n")
    stt = HybridSTT()
    
    # Get all test files
    sample_dir = Path("examples/sample_audio")
    audio_files = sorted(sample_dir.glob("*.wav"))
    
    if not audio_files:
        print("   Please add some .wav files to test\n")
        return
    
    # Process each file with auto-detection
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        print("-" * 60)
        
        try:
            # Auto-detect and transcribe
            result = stt.transcribe(str(audio_file))
            
            # Display results
            engine_display = {
                'whisper': '🔵 Whisper',
                'indic': '🟢 IndicSTT (Malayalam)'
            }
            
            print(f"  Engine:   {engine_display.get(result['engine'], result['engine'])}")
            print(f"  Language: {result['language'].upper()}")
            print(f"  Text:     {result['text']}")
            
            if result['engine'] == 'indic':
                print(f"  ℹ️  Auto-switched to IndicSTT (Malayalam detected)")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    print("="*60)
    print("✓ All files processed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

