# Whisper Speech-to-Text (STT)

Local, offline speech recognition using OpenAI Whisper with GPU acceleration.

## Features
- 🎤 Real-time microphone transcription
- 📁 Audio file transcription
- 🚀 GPU-accelerated (CUDA support)
- 🔒 Fully offline and private
- ⚙️ Configurable model sizes and parameters

## Installation

### Prerequisites
- Python 3.8-3.11
- CUDA-capable GPU (optional, recommended for RTX 2050)
- FFmpeg

### Setup
1. Clone repository:
git clone <your-repo-url>
cd whisper-stt


2. Install FFmpeg:
   - Linux: `sudo apt install ffmpeg`
   - Windows: Download from ffmpeg.org

3. Install Python dependencies:
 - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 
 Install other requirements
 - pip install -r requirements.txt
