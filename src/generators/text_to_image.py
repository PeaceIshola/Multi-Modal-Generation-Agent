"""
Text-to-Image Generation Module

This module provides functionality to generate images from text prompts using
Stable Diffusion XL. It can be used independently or as part of the multimodal agent.

Usage:
    from text_to_image import TextToImageGenerator
    
    generator = TextToImageGenerator()
    image = generator.generate("A serene mountain landscape at sunset")
    image.save("output.png")
"""

import torch
from diffusers import DiffusionPipeline
from PIL import Image


class TextToImageGenerator:
    """
    A class to handle text-to-image generation using Stable Diffusion XL.
    """
    
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Initialize the text-to-image generator.
        
        Args:
            model_id (str): Hugging Face model ID for the diffusion model
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
        """Load the diffusion model onto the available device."""
        print(f"Loading text-to-image model: {self.model_id}")
        
        torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype
        )
        
        self.pipe = self.pipe.to(self.device)
        print(f"Model loaded successfully on: {self.device}")
        
    def generate(self, prompt, num_inference_steps=25, **kwargs):
        """
        Generate an image from a text prompt.
        
        Args:
            prompt (str): Text description of the image to generate
            num_inference_steps (int): Number of denoising steps (more = higher quality, slower)
            **kwargs: Additional arguments to pass to the pipeline
            
        Returns:
            PIL.Image: Generated image
        """
        if self.pipe is None:
            self.load_model()
        
        print(f"Generating image: {prompt}")
        image = self.pipe(prompt, num_inference_steps=num_inference_steps, **kwargs).images[0]
        return image
    
    def generate_batch(self, prompts, num_inference_steps=25):
        """
        Generate multiple images from multiple prompts.
        
        Args:
            prompts (list): List of text prompts
            num_inference_steps (int): Number of denoising steps
            
        Returns:
            list: List of PIL.Image objects
        """
        images = []
        for prompt in prompts:
            image = self.generate(prompt, num_inference_steps)
            images.append(image)
        return images


def main():
    """Example usage of the TextToImageGenerator."""
    from config import setup_huggingface_auth
    
    # Authenticate with Hugging Face
    setup_huggingface_auth()
    
    # Create generator instance
    generator = TextToImageGenerator()
    
    # Generate an image
    prompt = "A serene mountain landscape at sunset with a lake reflecting the colorful sky"
    image = generator.generate(prompt)
    
    # Save the image
    output_path = "generated_image.png"
    image.save(output_path)
    print(f"✅ Image saved to: {output_path}")
    
    # Display the image (if in an environment that supports it)
    try:
        image.show()
    except:
        print("Open the file to view the generated image.")


if __name__ == "__main__":
    main()
