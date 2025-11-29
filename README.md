# Whisper Speech-to-Text (STT)

Local, offline speech recognition using OpenAI Whisper with GPU acceleration.

## Features
- Real-time microphone transcription
- Audio file transcription
- GPU-accelerated (CUDA support)
- Fully offline and private
- Configurable model sizes and parameters

## Installation
```
 git clone https://github.com/Harbinger-Bong/whisper-stt.git
 cd whisper-stt
 pip install -r requirements.txt
```
### Prerequisites
- Python 3.8-3.11
- CUDA-capable GPU (optional, recommended for RTX 2050)
- FFmpeg


(For best performance, install a CUDA build of PyTorch separately.)

## Usage

### Transcribe an audio file
``` 
 python examples/transcribe_file.py
```
Edit `examples/transcribe_file.py` to point to your own file in `examples/sample_audio/`.

### Real-time microphone transcription
``` 
 python examples/transcribe_realtime.py
```


Speak into your microphone and press Ctrl+C to stop.

## Configuration

Edit `config/config.yaml` to change:

- model size (tiny, base, small, …)
- device (cuda or cpu)
- language (e.g. "en", "ja", or null for auto-detect)

## Project structure

- `src/` – core STT code
- `examples/` – example scripts and sample audio
- `config/` – configuration file
- `tests/` – test package


## License

MIT License - see LICENSE file for details.

