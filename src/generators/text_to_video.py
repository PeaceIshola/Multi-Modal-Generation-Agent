"""
Text-to-Video Generation Module

This module provides functionality to generate video clips from text prompts using
a diffusion-based text-to-video model. It can be used independently or as part of 
the multimodal agent.

Usage:
    from text_to_video import TextToVideoGenerator
    
    generator = TextToVideoGenerator()
    frames = generator.generate("Ocean waves at sunset")
    generator.save_video(frames, "output.mp4")
"""

import torch
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image


class TextToVideoGenerator:
    """
    A class to handle text-to-video generation using diffusion models.
    """
    
    def __init__(self, model_id="damo-vilab/text-to-video-ms-1.7b"):
        """
        Initialize the text-to-video generator.
        
        Args:
            model_id (str): Hugging Face model ID for the video diffusion model
        """
        self.model_id = model_id
        self.pipe = None
        self.device = self._get_device()
        
    def _get_device(self):
        """Determine the best available device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self):
        """Load the video diffusion model onto the available device."""
        print(f"Loading text-to-video model: {self.model_id}")
        
        torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype
        )
        
        # Use DPM Solver for faster and more stable sampling
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.device)
        print(f"Model loaded successfully on: {self.device}")
        
    def generate(self, prompt, num_inference_steps=25, num_frames=16, **kwargs):
        """
        Generate video frames from a text prompt.
        
        Args:
            prompt (str): Text description of the video to generate
            num_inference_steps (int): Number of denoising steps (more = higher quality, slower)
            num_frames (int): Number of frames to generate (typical range: 8-24)
            **kwargs: Additional arguments to pass to the pipeline
            
        Returns:
            list: List of numpy arrays representing video frames
        """
        if self.pipe is None:
            self.load_model()
        
        print(f"Generating video ({num_frames} frames): {prompt}")
        frames = self.pipe(
            prompt, 
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            **kwargs
        ).frames[0]
        
        print(f"Generated {len(frames)} frames successfully")
        return frames
    
    def save_video(self, frames, output_path="generated_video.mp4", fps=8):
        """
        Save generated frames as an MP4 video file.
        
        Args:
            frames (list): List of video frames (numpy arrays)
            output_path (str): Path where the video should be saved
            fps (int): Frames per second for the output video
            
        Returns:
            str: Path to the saved video file
        """
        video_path = export_to_video(frames, output_video_path=output_path, fps=fps)
        print(f"Video saved to: {video_path}")
        return video_path
    
    def preview_frame(self, frames, frame_index=0):
        """
        Preview a single frame from the generated video.
        
        Args:
            frames (list): List of video frames
            frame_index (int): Index of the frame to preview
            
        Returns:
            PIL.Image: The selected frame as a PIL Image
        """
        if frame_index >= len(frames):
            frame_index = 0
            
        frame = frames[frame_index]
        frame_image = Image.fromarray((frame * 255).astype(np.uint8))
        return frame_image
    
    def generate_and_save(self, prompt, output_path="generated_video.mp4", 
                         num_inference_steps=25, num_frames=16, fps=8):
        """
        Convenience method to generate and save a video in one call.
        
        Args:
            prompt (str): Text description of the video
            output_path (str): Path where the video should be saved
            num_inference_steps (int): Number of denoising steps
            num_frames (int): Number of frames to generate
            fps (int): Frames per second for the output video
            
        Returns:
            str: Path to the saved video file
        """
        frames = self.generate(prompt, num_inference_steps, num_frames)
        return self.save_video(frames, output_path, fps)


def main():
    """Example usage of the TextToVideoGenerator."""
    from config import setup_huggingface_auth
    
    # Authenticate with Hugging Face
    setup_huggingface_auth()
    
    # Create generator instance
    generator = TextToVideoGenerator()
    
    # Generate a video
    prompt = "A serene beach with waves gently crashing on the shore at sunset"
    frames = generator.generate(prompt, num_inference_steps=25, num_frames=16)
    
    # Preview a frame
    preview = generator.preview_frame(frames, frame_index=0)
    preview.save("preview_frame.png")
    print("Preview frame saved to: preview_frame.png")
    
    # Save the video
    video_path = generator.save_video(frames, "generated_video.mp4")
    print(f"✅ Video saved to: {video_path}")


if __name__ == "__main__":
    main()
