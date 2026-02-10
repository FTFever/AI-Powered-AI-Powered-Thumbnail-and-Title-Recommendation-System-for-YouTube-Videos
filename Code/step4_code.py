"""
STEP 4: Base Thumbnail Generation
==================================
Generate thumbnails and titles using the trained base model.

TIME REQUIRED: 5 minutes per thumbnail
INPUT: finetuned_thumbnail_model/ (from Step 3)
OUTPUT: Generated thumbnail images and titles

INSTRUCTIONS:
1. Ensure finetuned_thumbnail_model/ exists from Step 3
2. Run: python step4_generate_base.py
3. Enter topics when prompted
4. Review generated thumbnails

NOTE: This uses the BASE model (before RLHF).
      After RLHF training, use step8_generate_final.py instead.
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from pathlib import Path
from datetime import datetime
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# test huggingface access

from huggingface_hub import login

# Replace with your actual token
token = "Place your huggingface token here"  # YOUR TOKEN HERE

try:
    login(token=token)
    print("✓ Login successful!")
    
    # Test it
    from huggingface_hub import whoami
    user_info = whoami()
    print(f"✓ Logged in as: {user_info['name']}")
    
except Exception as e:
    print(f"✗ Login failed: {e}")

# Model paths
# VLM_MODEL_PATH = "finetuned_generator_model"
VLM_MODEL_PATH = "D:\Youtube_Data_Analytics\Data\model"
# SD_MODEL_NAME = "stabilityai/stable-diffusion-2-1-base"
SD_MODEL_NAME = "CompVis/stable-diffusion-v1-4"

# Generation settings
NUM_INFERENCE_STEPS = 30    # Higher = better quality but slower (20-50 recommended)
GUIDANCE_SCALE = 7.5        # How closely to follow prompt (7-9 recommended)
IMAGE_WIDTH = 768           # 16:9 aspect ratio for YouTube
IMAGE_HEIGHT = 432

# Output directory
# OUTPUT_DIR = "generated_thumbnails"
OUTPUT_DIR = "D:\Youtube_Data_Analytics\Data\generated_thumbnails"

# Device
DEVICE = None  # None = auto-detect, or 'cuda'/'cpu'

# =============================================================================
# GENERATOR CLASS
# =============================================================================

class ThumbnailGenerator:
    """Generates viral-optimized thumbnails and titles"""
    
    def __init__(self, vlm_model_path=VLM_MODEL_PATH, 
                 sd_model_name=SD_MODEL_NAME, device=DEVICE):
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n{'='*70}")
        print(f"INITIALIZING THUMBNAIL GENERATOR")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load vision-language model
        print("\n[1/2] Loading vision-language model...")
        print(f"      Path: {vlm_model_path}")
        
        self.vlm_processor = Blip2Processor.from_pretrained(vlm_model_path)
        self.vlm_model = Blip2ForConditionalGeneration.from_pretrained(
            vlm_model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        
        print("      ✓ Vision-language model loaded")
        
        # Load Stable Diffusion
        print("\n[2/2] Loading Stable Diffusion (first time: ~10 min)...")
        print(f"      Model: {sd_model_name}")
        
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        # Use faster scheduler
        self.sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.sd_pipe.scheduler.config
        )
        
        # Memory optimizations
        if self.device == 'cuda':
            self.sd_pipe.enable_attention_slicing()
            try:
                self.sd_pipe.enable_xformers_memory_efficient_attention()
            except:
                pass  # xformers not available
        
        print("      ✓ Stable Diffusion loaded")
        print(f"\n✅ Generator ready!\n")
    
    def generate_title(self, topic, style="exciting"):
        """
        Generate viral YouTube title
        
        Args:
            topic: Main topic/content
            style: 'exciting', 'informative', 'clickbait', 'professional'
        """
        
        style_prompts = {
            'exciting': "Create an exciting, engaging YouTube title with emotional appeal and strong hooks",
            'informative': "Create a clear, informative YouTube title that accurately describes the content",
            'clickbait': "Create an attention-grabbing, curiosity-inducing YouTube title that makes people want to click",
            'professional': "Create a professional, credible YouTube title with authority"
        }
        
        prompt = f"{style_prompts.get(style, style_prompts['exciting'])} about: {topic}"
        
        from PIL import Image
        placeholder_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        inputs = self.vlm_processor(
            images=placeholder_image,
            text=prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        
        with torch.no_grad():
            outputs = self.vlm_model.generate(
                **inputs,
                max_length=80,
                num_beams=5,
                temperature=0.8,
                do_sample=True,
                top_p=0.9
            )
        
        title = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up title
        title = title.strip()
        if len(title) > 100:
            title = title[:97] + "..."
        
        return title
    
    def create_thumbnail_prompt(self, topic):
        """
        Create optimized prompt for Stable Diffusion
        
        Args:
            topic: Video topic/theme
        """
        
        # Base viral thumbnail characteristics
        base_prompt = f"""
        YouTube thumbnail, {topic}, professional photography, 
        high contrast, vibrant colors, cinematic lighting,
        sharp focus, eye-catching composition, dynamic angle,
        studio quality, 8k, highly detailed, trending on artstation
        """
        
        # Clean up whitespace
        prompt = " ".join(base_prompt.split())
        
        # Negative prompt (things to avoid)
        negative_prompt = """
        blurry, low quality, amateur, dark, boring, cluttered,
        watermark, signature, text, letters, words, low contrast,
        oversaturated, distorted, deformed, ugly
        """
        
        negative_prompt = " ".join(negative_prompt.split())
        
        return prompt, negative_prompt
    
    def generate_thumbnail(self, topic, title=None, output_path=None,
                          num_steps=NUM_INFERENCE_STEPS):
        """
        Generate complete thumbnail with title
        
        Args:
            topic: Main topic for thumbnail
            title: Optional custom title (will generate if not provided)
            output_path: Where to save (auto-generated if not provided)
            num_steps: Generation steps (20-50, higher = better quality)
        
        Returns:
            dict with 'image', 'title', 'path'
        """
        
        print(f"\n{'='*70}")
        print(f"GENERATING THUMBNAIL")
        print(f"{'='*70}")
        print(f"Topic: {topic}")
        
        # Generate title if not provided
        if not title:
            print("\nGenerating title...")
            title = self.generate_title(topic, style='exciting')
            print(f"✓ Title: {title}")
        else:
            print(f"Title: {title}")
        
        # Create output path if not provided
        if not output_path:
            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(exist_ok=True, parents=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f"thumbnail_{timestamp}.png"
        
        # Create prompt
        print("\nCreating prompt...")
        prompt, negative_prompt = self.create_thumbnail_prompt(topic)
        print(f"✓ Prompt ready")
        
        # Generate image
        print(f"\nGenerating image ({num_steps} steps)...")
        print("This may take 1-3 minutes...")
        
        try:
            image = self.sd_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=GUIDANCE_SCALE,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT
            ).images[0]
            
            # Save
            image.save(output_path)
            print(f"\n✅ GENERATION COMPLETE!")
            print(f"{'='*70}")
            print(f"Title: {title}")
            print(f"Saved: {output_path}")
            print(f"{'='*70}\n")
            
            return {
                'image': image,
                'title': title,
                'path': str(output_path),
                'topic': topic
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ GPU out of memory!")
                print("Try reducing NUM_INFERENCE_STEPS or use CPU")
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            raise e
    
    def generate_multiple(self, topics, num_per_topic=1):
        """
        Generate multiple thumbnails
        
        Args:
            topics: List of topics
            num_per_topic: Variants per topic
        
        Returns:
            List of generated thumbnails
        """
        
        print(f"\n{'='*70}")
        print(f"BATCH GENERATION")
        print(f"{'='*70}")
        print(f"Topics: {len(topics)}")
        print(f"Variants per topic: {num_per_topic}")
        print(f"Total to generate: {len(topics) * num_per_topic}\n")
        
        results = []
        
        for topic_idx, topic in enumerate(topics, 1):
            print(f"\n[Topic {topic_idx}/{len(topics)}]: {topic}")
            
            for variant in range(num_per_topic):
                if num_per_topic > 1:
                    print(f"\n  Variant {variant+1}/{num_per_topic}")
                
                result = self.generate_thumbnail(
                    topic=topic,
                    num_steps=NUM_INFERENCE_STEPS
                )
                results.append(result)
        
        print(f"\n{'='*70}")
        print(f"✅ BATCH COMPLETE - Generated {len(results)} thumbnails")
        print(f"{'='*70}\n")
        
        return results

# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(generator):
    """Interactive thumbnail generation"""
    
    print(f"\n{'='*70}")
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter topics to generate thumbnails.")
    print("Type 'batch' for batch mode, 'quit' to exit.\n")
    
    while True:
        print("-" * 70)
        user_input = input("\nTopic (or 'batch'/'quit'): ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\nExiting...")
            break
        
        if user_input.lower() == 'batch':
            print("\nBatch mode - enter topics (one per line, empty line to finish):")
            topics = []
            while True:
                topic = input(f"  Topic {len(topics)+1}: ").strip()
                if not topic:
                    break
                topics.append(topic)
            
            if topics:
                generator.generate_multiple(topics, num_per_topic=1)
            else:
                print("No topics entered")
            continue
        
        # Generate single thumbnail
        try:
            result = generator.generate_thumbnail(topic=user_input)
            print(f"\n✓ View your thumbnail: {result['path']}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("STEP 4: Base Thumbnail Generation")
    print("="*70)
    
    # Check if model exists
    if not Path(VLM_MODEL_PATH).exists():
        print(f"\n❌ ERROR: Model not found!")
        print(f"Expected: {VLM_MODEL_PATH}/")
        print(f"\nPlease run Step 3 first:")
        print(f"  python step3_train_model.py")
        return
    
    # Initialize generator
    try:
        generator = ThumbnailGenerator(
            vlm_model_path=VLM_MODEL_PATH,
            sd_model_name=SD_MODEL_NAME,
            device=DEVICE
        )
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Example generations (you can customize this)
    print("Generating example thumbnails...")
    print("(You can modify the topics list in the code)\n")
    
    example_topics = [
        "epic gaming moment first person shooter",
        "tech product review latest smartphone",
        "cooking tutorial chocolate dessert"
    ]
    
    # Ask user what to do
    print("Options:")
    print("  1. Generate examples (pre-defined topics)")
    print("  2. Interactive mode (enter your own topics)")
    print("  3. Exit")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == '1':
        try:
            results = generator.generate_multiple(example_topics, num_per_topic=1)
            print("\n✓ Check the generated_thumbnails/ folder for your images!")
        except Exception as e:
            print(f"\n❌ Generation Error: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == '2':
        interactive_mode(generator)
    
    else:
        print("\nExiting...")
        return
    
    print(f"\n➡️  Next step: Run 'python step5b_feedback_web.py'")
    print("    (Or step5a_feedback_cli.py for command-line feedback)")

if __name__ == "__main__":
    main()
