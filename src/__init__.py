"""
Multi-Modal Generation Agent - Source Package

This package contains all the core modules for the multimodal generation system.
"""

from src.generators.text_to_image import TextToImageGenerator
from src.generators.text_to_video import TextToVideoGenerator
from src.agent.multimodal_agent import MultimodalAgent
from src.ui.web_ui import MultimodalWebUI

__version__ = "1.0.0"

__all__ = [
    "TextToImageGenerator",
    "TextToVideoGenerator",
    "MultimodalAgent",
    "MultimodalWebUI",
]
