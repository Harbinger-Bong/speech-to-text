# Hybrid Speech-to-Text

Local, offline speech-to-text system combining OpenAI Whisper with AI4Bharat IndicSTT (Malayalam) for optimal multi-language transcription.

## Features
- Real-time microphone transcription
- Audio file transcription
- GPU-accelerated (CUDA support)
- Fully offline and private
- Configurable model sizes and parameters

## Architecture

1. **Whisper (Default)**: Fast language detection and transcription for English/Arabic
2. **IndicSTT**: Specialized Malayalam model, activated when Malayalam detected
3. **Auto-switching**: Seamless transition between engines

### Prerequisites
- Python 3.8-3.11
- CUDA-capable GPU (optional, recommended for RTX 2050)
- FFmpeg


(For best performance, install a CUDA build of PyTorch separately.)

## Installation
```
 git clone https://github.com/Harbinger-Bong/whisper-stt.git
 cd whisper-stt
 pip install -r requirements.txt
```
## Usage

#### File Transcription
``` 
 python examples/hybrid_transcribe.py
```
Edit `examples/transcribe_file.py` to point to your own file in `examples/sample_audio/`.

#### Real-time GUI
``` 
 python examples/realtime_hybrid.py
```
## Models Used

- **Whisper Small**: English, Arabic detection and transcription
- **Wav2Vec2-Malayalam**: High-accuracy Malayalam transcription (~28% WER)

## Project structure

- `src/` – core STT code
- `examples/` – example scripts and sample audio
- `config/` – configuration file
- `tests/` – test package


## License

MIT License - see LICENSE file for details.

