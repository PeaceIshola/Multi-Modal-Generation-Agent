"""
Test script for individual modules.

This script allows you to test each component independently:
- Text-to-Image generation
- Text-to-Video generation
- Multimodal Agent

Usage:
    python test_individual_modules.py --module image
    python test_individual_modules.py --module video
    python test_individual_modules.py --module agent
    python test_individual_modules.py --module all
"""

import argparse
import sys
from config import setup_huggingface_auth


def test_text_to_image():
    """Test the text-to-image module."""
    print("\n" + "="*60)
    print("Testing Text-to-Image Module")
    print("="*60)
    
    try:
        from src.generators.text_to_image import TextToImageGenerator
        
        # Create generator
        print("\n1. Creating TextToImageGenerator...")
        generator = TextToImageGenerator()
        
        # Generate an image
        print("\n2. Generating image...")
        prompt = "A serene mountain landscape at sunset with a lake"
        image = generator.generate(prompt, num_inference_steps=25)
        
        # Save the image
        output_path = "test_image_output.png"
        image.save(output_path)
        print(f"\n✅ SUCCESS! Image saved to: {output_path}")
        print("   You can open this file to view the generated image.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_to_video():
    """Test the text-to-video module."""
    print("\n" + "="*60)
    print("Testing Text-to-Video Module")
    print("="*60)
    
    try:
        from src.generators.text_to_video import TextToVideoGenerator
        
        # Create generator
        print("\n1. Creating TextToVideoGenerator...")
        generator = TextToVideoGenerator()
        
        # Generate a video
        print("\n2. Generating video (this may take a minute)...")
        prompt = "Ocean waves gently crashing on the shore"
        output_path = "test_video_output.mp4"
        
        video_path = generator.generate_and_save(
            prompt,
            output_path=output_path,
            num_inference_steps=25,
            num_frames=16
        )
        
        print(f"\n✅ SUCCESS! Video saved to: {video_path}")
        print("   You can open this file to view the generated video.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_agent():
    """Test the multimodal agent."""
    print("\n" + "="*60)
    print("Testing Multimodal Agent")
    print("="*60)
    
    try:
        from src.agent.multimodal_agent import MultimodalAgent
        
        # Create agent
        print("\n1. Creating MultimodalAgent...")
        agent = MultimodalAgent()
        
        # Test image generation through agent
        print("\n2. Testing agent with image request...")
        prompt = "Generate an image of a cute robot in a garden"
        result = agent.process(prompt)
        
        if hasattr(result, 'save'):
            output_path = "test_agent_image.png"
            result.save(output_path)
            print(f"✅ Agent generated image: {output_path}")
        else:
            print(f"Result type: {type(result)}")
        
        # Test video generation through agent
        print("\n3. Testing agent with video request...")
        prompt = "Create a short video of clouds in the sky"
        result = agent.process(prompt)
        
        if isinstance(result, list):
            output_path = "test_agent_video.mp4"
            agent.video_generator.save_video(result, output_path)
            print(f"✅ Agent generated video: {output_path}")
        
        print("\n✅ SUCCESS! Agent is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test individual modules")
    parser.add_argument(
        "--module",
        choices=["image", "video", "agent", "all"],
        default="image",
        help="Which module to test (default: image)"
    )
    
    args = parser.parse_args()
    
    # Authenticate with Hugging Face
    print("Authenticating with Hugging Face...")
    if not setup_huggingface_auth():
        print("Authentication failed. Please check your token in config.py")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Starting Module Tests")
    print("="*60)
    print(f"Module to test: {args.module}")
    print("\nNote: First-time execution will download models (~5-10 GB).")
    print("This may take several minutes depending on your connection.")
    print("="*60)
    
    # Run tests based on selection
    results = {}
    
    if args.module == "image" or args.module == "all":
        results["image"] = test_text_to_image()
    
    if args.module == "video" or args.module == "all":
        results["video"] = test_text_to_video()
    
    if args.module == "agent" or args.module == "all":
        results["agent"] = test_multimodal_agent()
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for module, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{module.capitalize()}: {status}")
    print("="*60)


if __name__ == "__main__":
    main()
