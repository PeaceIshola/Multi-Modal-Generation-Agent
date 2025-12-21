# Multimodal Generation Agent

AI-powered application that generates images and videos from text using Stable Diffusion XL and text-to-video models.

## What Was Done

Converted a Jupyter notebook into a modular Python codebase with:
- Independent modules for text-to-image and text-to-video generation
- Intelligent agent for routing requests between generators
- Web interface using Gradio
- Comprehensive testing framework
- Demo outputs showcasing capabilities

## File Structure

```
Multi-Modal-Generation-Agent/
├── src/
│   ├── __init__.py
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── text_to_image.py       # Stable Diffusion XL image generation
│   │   └── text_to_video.py       # Text-to-video generation
│   ├── agent/
│   │   ├── __init__.py
│   │   └── multimodal_agent.py    # LLM-based request routing
│   └── ui/
│       ├── __init__.py
│       └── web_ui.py              # Gradio web interface
├── tests/
│   ├── __init__.py
│   └── test_individual_modules.py # Testing framework
├── demo/
│   ├── test_image_output.png      # Sample generated image
│   ├── test_video_output.mp4      # Sample generated video
│   └── README.md
├── config.py                       # Hugging Face authentication
├── requirements.txt                # Project dependencies
├── QUICKSTART.md
└── README.md
```

## What to Install

### Requirements
- Python 3.8 or higher
- 8GB+ RAM (16GB+ recommended)
- GPU with 8GB+ VRAM (optional but recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Main packages:**
- `torch` - PyTorch for neural networks
- `transformers` - Hugging Face transformers
- `diffusers` - Stable Diffusion models
- `gradio` - Web interface
- `Pillow`, `numpy`, `opencv-python` - Image/video processing

### Setup Hugging Face Token

**Create a Hugging Face account and token:**

1. Go to https://huggingface.co and sign up (free)
2. Navigate to Settings → Access Tokens (https://huggingface.co/settings/tokens)
3. Click "New token"
4. Name it (e.g., "multimodal-agent")
5. Select "Read" role (sufficient for downloading models)
6. Click "Generate token"
7. Copy the token (starts with `hf_...`)

**Set the token as environment variable:**

```bash
export HF_TOKEN="your_token_here"
```

Or add to `~/.bashrc` or `~/.zshrc` for persistence

⚠️ **First run downloads 5-10 GB of models (takes 5-10 minutes)**

## How to Run

### Option 1: Web Interface (Recommended)

```bash
python -m src.ui.web_ui
```
Open browser to `http://127.0.0.1:7860`

### Option 2: Individual Modules

**Generate Image:**
```bash
python -m src.generators.text_to_image
```

**Generate Video:**
```bash
python -m src.generators.text_to_video
```

### Option 3: Test All Modules

```bash
python -m tests.test_individual_modules --module all
```

Available test options:
- `--module image` - Test image generation only
- `--module video` - Test video generation only
- `--module agent` - Test multimodal agent only
- `--module all` - Test all modules

### Option 4: Use in Code

```python
from config import setup_huggingface_auth
from src.generators.text_to_image import TextToImageGenerator

setup_huggingface_auth()
generator = TextToImageGenerator()
image = generator.generate("Mountain landscape at sunset")
image.save("output.png")
```

## Hardware Performance

- **GPU (8GB+ VRAM)**: 15-30 seconds per image
- **Apple Silicon (M1/M2/M3)**: 20-30 seconds per image
- **CPU only**: 2-3 minutes per image

## Troubleshooting

**Out of memory:** Reduce `num_inference_steps` or `num_frames`  
**Import errors:** Run `pip install -r requirements.txt`  
**Model download slow:** Normal for first run (5-10 GB)  
**CUDA warning on Mac:** Expected, uses MPS instead

## Credits

- Stable Diffusion XL (Stability AI)
- Text-to-Video Model (ModelScope/Alibaba)
- Hugging Face (transformers, diffusers)
- Gradio (web UI framework)
