
### Video Generation
- Uses HuggingFace's Stable Diffusion model
- Generates 720p keyframes from prompts
- Creates smooth visual storytelling synchronized with music

### Video Composition
- Combines frame sequences with seamless audio mix
- Maintains perfect audio-video sync
- Outputs professional MP4 file

## Installation

```bash
# Core audio processing
pip install librosa soundfile numpy scipy

# Video creation
pip install moviepy imageio imageio-ffmpeg

# AI video generation
pip install diffusers transformers torch pillow

# Optional: For GPU acceleration (highly recommended)
# NVIDIA GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
