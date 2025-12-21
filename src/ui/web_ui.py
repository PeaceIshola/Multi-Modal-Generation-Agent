"""
Interactive Web UI for Multimodal Agent

This module provides a Gradio-based web interface for the multimodal agent,
allowing users to interact with the agent through a browser.

Usage:
    python web_ui.py
    
Then open the provided URL in your browser.
"""

import gradio as gr
from src.agent.multimodal_agent import MultimodalAgent
from diffusers.utils import export_to_video


class MultimodalWebUI:
    """
    A web interface wrapper for the multimodal agent using Gradio.
    """
    
    def __init__(self):
        """Initialize the web UI with a multimodal agent."""
        self.agent = None
        
    def initialize_agent(self):
        """Lazy initialization of the agent (only when first used)."""
        if self.agent is None:
            print("Initializing multimodal agent...")
            self.agent = MultimodalAgent()
            print("Agent initialized!")
    
    def handle_request(self, prompt):
        """
        Handle a user request and return appropriate outputs.
        
        Args:
            prompt (str): User's input prompt
            
        Returns:
            tuple: (text_output, image_output, video_output)
        """
        if not prompt or not prompt.strip():
            return "Please enter a prompt.", None, None
        
        try:
            self.initialize_agent()
            
            # Process the request
            result = self.agent.process(prompt)
            
            # Return the appropriate output based on type
            if isinstance(result, str):
                # Text response
                return result, None, None
            elif hasattr(result, 'save'):
                # Image response
                return "✅ Image generated successfully!", result, None
            else:
                # Video response (list of frames)
                video_path = export_to_video(result, output_video_path="temp_video.mp4")
                return "✅ Video generated successfully!", None, video_path
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            return error_msg, None, None
    
    def create_interface(self):
        """
        Create and configure the Gradio interface.
        
        Returns:
            gr.Blocks: Configured Gradio interface
        """
        with gr.Blocks(title="Multimodal Generation Agent", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🎨 Multimodal Generation Agent
                
                Generate images and videos from text prompts, or ask questions!
                
                **Examples:**
                - "Generate an image of a sunset over mountains"
                - "Create a video of ocean waves"
                - "What is machine learning?"
                """
            )
            
            with gr.Row():
                with gr.Column():
                    inp = gr.Textbox(
                        placeholder="Ask a question or describe what you want to create...",
                        label="Your Prompt",
                        lines=3
                    )
                    btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                    
                    gr.Markdown("### Example Prompts")
                    gr.Examples(
                        examples=[
                            "Generate an image of a cute robot in a garden",
                            "Create a video of a waterfall in a forest",
                            "Draw a picture of a futuristic city at night",
                            "Make a video of clouds moving across the sky",
                        ],
                        inputs=inp
                    )
            
            with gr.Row():
                with gr.Column():
                    out_text = gr.Markdown(label="Response")
                
            with gr.Row():
                with gr.Column():
                    out_img = gr.Image(label="Generated Image", type="pil")
                with gr.Column():
                    out_vid = gr.Video(label="Generated Video")
            
            # Connect the button to the handler
            btn.click(
                fn=self.handle_request,
                inputs=inp,
                outputs=[out_text, out_img, out_vid]
            )
            
            gr.Markdown(
                """
                ---
                **Note:** First-time generation may take a few minutes as models are downloaded and loaded.
                Image generation typically takes 10-30 seconds, video generation takes 30-60 seconds.
                """
            )
        
        return demo
    
    def launch(self, share=False, **kwargs):
        """
        Launch the Gradio interface.
        
        Args:
            share (bool): Whether to create a public shareable link
            **kwargs: Additional arguments to pass to demo.launch()
        """
        demo = self.create_interface()
        demo.launch(share=share, **kwargs)


def main():
    """Launch the web UI."""
    from config import setup_huggingface_auth
    
    print("Starting Multimodal Generation Agent Web UI...")
    print("="*60)
    
    # Authenticate with Hugging Face
    setup_huggingface_auth()
    
    # Launch the UI
    ui = MultimodalWebUI()
    ui.launch(share=False)


if __name__ == "__main__":
    main()
