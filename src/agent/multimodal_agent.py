"""
Multimodal Generation Agent

This module provides an intelligent agent that can classify user requests and route them
to the appropriate generation model (text QA, image generation, or video generation).

Usage:
    from multimodal_agent import MultimodalAgent
    
    agent = MultimodalAgent()
    result = agent.process("Generate an image of a sunset")
"""

import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.generators.text_to_image import TextToImageGenerator
from src.generators.text_to_video import TextToVideoGenerator


class MultimodalAgent:
    """
    An intelligent agent that routes requests to text, image, or video generation.
    """
    
    def __init__(self, llm_model_id="microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize the multimodal agent.
        
        Args:
            llm_model_id (str): Hugging Face model ID for the LLM
        """
        self.llm_model_id = llm_model_id
        self.llm_tokenizer = None
        self.llm_model = None
        self.image_generator = None
        self.video_generator = None
        self.device = self._get_device()
        
    def _get_device(self):
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_llm(self):
        """Load the language model for classification and Q&A."""
        if self.llm_model is not None:
            return
            
        print(f"Loading LLM: {self.llm_model_id}")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_id, 
            trust_remote_code=True
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"LLM loaded successfully on: {self.llm_model.device}")
    
    def load_generators(self, load_image=True, load_video=True):
        """
        Load the image and/or video generators.
        
        Args:
            load_image (bool): Whether to load the image generator
            load_video (bool): Whether to load the video generator
        """
        if load_image and self.image_generator is None:
            self.image_generator = TextToImageGenerator()
            self.image_generator.load_model()
            
        if load_video and self.video_generator is None:
            self.video_generator = TextToVideoGenerator()
            self.video_generator.load_model()
    
    def llm_generate(self, prompt, max_new_tokens=128, temperature=0.7):
        """
        Generate a response using the LLM.
        
        Args:
            prompt (str): Input prompt for the LLM
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated response
        """
        if self.llm_model is None:
            self.load_llm()
        
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        outputs = self.llm_model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            do_sample=True,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text (remove the input prompt)
        response = response[len(prompt):].strip()
        return response
    
    def classify_prompt(self, prompt):
        """
        Classify the user prompt into QA, image, or video generation.
        
        Args:
            prompt (str): User's input prompt
            
        Returns:
            dict: Dictionary with 'type' (qa/image/video) and 'prompt' (improved prompt)
        """
        system_prompt = """Classify the user request into one of these types:
- "qa": general question or conversation
- "image": request to generate a single image
- "video": request to generate a video or animation

Respond with JSON format: {"type": "<qa|image|video>", "prompt": "<improved description for generation>"}

User request: """
        
        full_prompt = system_prompt + prompt + "\n\nClassification:"
        
        try:
            response = self.llm_generate(full_prompt, max_new_tokens=100)
            
            # Try to parse JSON response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "type": result.get("type", "qa"), 
                    "prompt": result.get("prompt", prompt)
                }
        except Exception as e:
            print(f"Classification error: {e}")
        
        # Fallback: check for keywords
        lower_prompt = prompt.lower()
        if any(word in lower_prompt for word in ["image", "picture", "photo", "draw", "paint", "sketch"]):
            return {"type": "image", "prompt": prompt}
        elif any(word in lower_prompt for word in ["video", "clip", "animation", "moving", "motion"]):
            return {"type": "video", "prompt": prompt}
        else:
            return {"type": "qa", "prompt": prompt}
    
    def process(self, user_prompt):
        """
        Process a user request and generate the appropriate output.
        
        Args:
            user_prompt (str): User's input prompt
            
        Returns:
            Depending on the request type:
            - str: Text response for Q&A
            - PIL.Image: Generated image
            - list: Video frames
        """
        # Classify the request
        classification = self.classify_prompt(user_prompt)
        request_type = classification["type"]
        improved_prompt = classification["prompt"]
        
        print(f"Request classified as: {request_type}")
        
        # Route and generate
        if request_type == "image":
            if self.image_generator is None:
                self.load_generators(load_image=True, load_video=False)
            print(f"Generating image: {improved_prompt}")
            return self.image_generator.generate(improved_prompt)
        
        elif request_type == "video":
            if self.video_generator is None:
                self.load_generators(load_image=False, load_video=True)
            print(f"Generating video: {improved_prompt}")
            return self.video_generator.generate(improved_prompt)
        
        else:  # qa
            print(f"Answering question: {user_prompt}")
            # For now, return a simple response
            # You can enhance this by using the LLM for actual Q&A
            return ("I'm a multimodal agent specialized in generating images and videos. "
                   "Please ask me to create an image or video!")
    
    def answer_question(self, question):
        """
        Answer a question using the LLM (without routing).
        
        Args:
            question (str): Question to answer
            
        Returns:
            str: Answer from the LLM
        """
        if self.llm_model is None:
            self.load_llm()
        
        prompt = f"Question: {question}\n\nAnswer:"
        return self.llm_generate(prompt, max_new_tokens=256)


def main():
    """Example usage of the MultimodalAgent."""
    agent = MultimodalAgent()
    
    # Test prompts
    test_prompts = [
        "Generate an image of a cute robot in a garden",
        "Create a video of ocean waves at sunset",
        "What is the capital of France?"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Testing: {prompt}")
        print('='*60)
        
        result = agent.process(prompt)
        
        if isinstance(result, str):
            print(f"Answer: {result}")
        elif hasattr(result, 'save'):
            print("Image generated successfully!")
            result.save(f"test_image.png")
        else:
            print("Video frames generated successfully!")
            agent.video_generator.save_video(result, "test_video.mp4")


if __name__ == "__main__":
    main()
