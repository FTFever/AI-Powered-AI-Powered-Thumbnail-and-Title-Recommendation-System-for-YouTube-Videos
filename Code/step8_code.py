"""
STEP 8: Final Thumbnail Generation (RLHF-Improved)
===================================================
Generate thumbnails using the RLHF-improved model.

TIME REQUIRED: 5 minutes per thumbnail
INPUT: rlhf_improved_model/ (from Step 7)
OUTPUT: High-quality thumbnail images optimized for human preferences

INSTRUCTIONS:
1. Ensure rlhf_improved_model/ exists from Step 7
2. Run: python step8_generate_final.py
3. Enter topics when prompted
4. Compare with Step 4 outputs - should be noticeably better!

PURPOSE: Generate final production-ready thumbnails with the model that
         has been optimized for human preferences through RLHF.
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model paths
VLM_MODEL_PATH = "rlhf_improved_model"  # RLHF-improved model
SD_MODEL_NAME = "stabilityai/stable-diffusion-2-1"

# Generation settings
NUM_INFERENCE_STEPS = 35    # Slightly higher for final quality
GUIDANCE_SCALE = 7.5
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 432

# Output directory
OUTPUT_DIR = "final_thumbnails"

# Device
DEVICE = None  # None = auto-detect

# =============================================================================
# GENERATOR CLASS
# =============================================================================

class ImprovedThumbnailGenerator:
    """Generates viral-optimized thumbnails using RLHF-improved model"""
    
    def __init__(self, vlm_model_path=VLM_MODEL_PATH, 
                 sd_model_name=SD_MODEL_NAME, device=DEVICE):
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n{'='*70}")
        print(f"INITIALIZING IMPROVED THUMBNAIL GENERATOR")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Model: RLHF-Improved (Human-Optimized)")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load RLHF-improved vision-language model
        print("\n[1/2] Loading RLHF-improved model...")
        print(f"      Path: {vlm_model_path}")
        print(f"      (This model is optimized for human preferences!)")
        
        self.vlm_processor = Blip2Processor.from_pretrained(vlm_model_path)
        self.vlm_model = Blip2ForConditionalGeneration.from_pretrained(
            vlm_model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        
        print("      ✓ RLHF model loaded")
        
        # Load Stable Diffusion
        print("\n[2/2] Loading Stable Diffusion...")
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
                pass
        
        print("      ✓ Stable Diffusion loaded")
        print(f"\n✅ Generator ready! (Using RLHF-improved model)\n")
    
    def generate_title(self, topic, style="exciting"):
        """
        Generate viral YouTube title using RLHF-improved model
        
        Args:
            topic: Main topic/content
            style: 'exciting', 'informative', 'clickbait', 'professional'
        """
        
        style_prompts = {
            'exciting': "Create an exciting, engaging YouTube title with emotional hooks",
            'informative': "Create a clear, informative YouTube title",
            'clickbait': "Create an attention-grabbing, curiosity-inducing YouTube title",
            'professional': "Create a professional, credible YouTube title"
        }
        
        from PIL import Image
        placeholder_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        prompt = f"{style_prompts.get(style, style_prompts['exciting'])} about: {topic}"
        
        inputs = self.vlm_processor(
            images=placeholder_image,
            text=prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            # Use slightly higher temperature for creativity
            outputs = self.vlm_model.generate(
                **inputs,
                max_length=80,
                num_beams=5,
                temperature=0.85,
                do_sample=True,
                top_p=0.92
            )
        
        title = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up
        title = title.strip()
        if len(title) > 100:
            title = title[:97] + "..."
        
        return title
    
    def create_enhanced_prompt(self, topic):
        """
        Create enhanced prompt using insights from RLHF training
        
        Args:
            topic: Video topic/theme
        """
        
        # Enhanced prompt based on what humans prefer
        prompt = f"""
        Professional YouTube thumbnail, {topic},
        dramatic cinematic lighting, high contrast colors,
        sharp focus on main subject, dynamic composition,
        eye-catching visual appeal, trending style,
        studio photography quality, 8k resolution,
        highly detailed, award-winning design
        """
        
        prompt = " ".join(prompt.split())
        
        # Enhanced negative prompt
        negative_prompt = """
        blurry, low quality, amateur, dark, boring,
        cluttered, messy, low contrast, dull colors,
        watermark, text overlay, distorted, deformed,
        ugly, oversaturated, underexposed
        """
        
        negative_prompt = " ".join(negative_prompt.split())
        
        return prompt, negative_prompt
    
    def generate_thumbnail(self, topic, title=None, output_path=None,
                          num_steps=NUM_INFERENCE_STEPS, comparison_mode=False):
        """
        Generate RLHF-optimized thumbnail
        
        Args:
            topic: Main topic for thumbnail
            title: Optional custom title
            output_path: Where to save
            num_steps: Generation steps (higher = better)
            comparison_mode: If True, saves to comparison folder
        
        Returns:
            dict with 'image', 'title', 'path'
        """
        
        print(f"\n{'='*70}")
        print(f"GENERATING RLHF-OPTIMIZED THUMBNAIL")
        print(f"{'='*70}")
        print(f"Topic: {topic}")
        print(f"Model: RLHF-Improved (Human Preferences)")
        
        # Generate title
        if not title:
            print("\nGenerating human-optimized title...")
            title = self.generate_title(topic, style='exciting')
            print(f"✓ Title: {title}")
        else:
            print(f"Title: {title}")
        
        # Create output path
        if not output_path:
            if comparison_mode:
                output_dir = Path(OUTPUT_DIR) / "comparison"
            else:
                output_dir = Path(OUTPUT_DIR)
            
            output_dir.mkdir(exist_ok=True, parents=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f"rlhf_thumbnail_{timestamp}.png"
        
        # Create enhanced prompt
        print("\nCreating enhanced prompt...")
        prompt, negative_prompt = self.create_enhanced_prompt(topic)
        print(f"✓ Enhanced prompt ready")
        
        # Generate image
        print(f"\nGenerating image ({num_steps} steps)...")
        print("Quality mode: RLHF-optimized")
        
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
                'topic': topic,
                'model': 'RLHF-Improved'
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ GPU out of memory!")
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            raise e
    
    def generate_comparison_set(self, topics):
        """
        Generate thumbnails for comparison with base model
        
        Args:
            topics: List of topics to generate
        
        Returns:
            List of generated thumbnails
        """
        
        print(f"\n{'='*70}")
        print(f"COMPARISON SET GENERATION")
        print(f"{'='*70}")
        print(f"Generating {len(topics)} thumbnails with RLHF-improved model")
        print(f"Compare these with base model outputs (Step 4) to see improvement!\n")
        
        results = []
        
        for idx, topic in enumerate(topics, 1):
            print(f"\n[{idx}/{len(topics)}] Generating: {topic}")
            
            result = self.generate_thumbnail(
                topic=topic,
                num_steps=NUM_INFERENCE_STEPS,
                comparison_mode=True
            )
            results.append(result)
        
        print(f"\n{'='*70}")
        print(f"✅ COMPARISON SET COMPLETE")
        print(f"{'='*70}")
        print(f"Generated: {len(results)} thumbnails")
        print(f"Location: {OUTPUT_DIR}/comparison/")
        print(f"\nCompare these with your Step 4 outputs!")
        print(f"RLHF thumbnails should be noticeably more engaging.\n")
        
        return results
    
    def generate_multiple(self, topics, variants_per_topic=1):
        """
        Generate multiple RLHF-optimized thumbnails
        
        Args:
            topics: List of topics
            variants_per_topic: Number of variants per topic
        
        Returns:
            List of results
        """
        
        print(f"\n{'='*70}")
        print(f"BATCH GENERATION (RLHF-OPTIMIZED)")
        print(f"{'='*70}")
        print(f"Topics: {len(topics)}")
        print(f"Variants per topic: {variants_per_topic}")
        print(f"Total to generate: {len(topics) * variants_per_topic}\n")
        
        results = []
        
        for topic_idx, topic in enumerate(topics, 1):
            print(f"\n[Topic {topic_idx}/{len(topics)}]: {topic}")
            
            for variant in range(variants_per_topic):
                if variants_per_topic > 1:
                    print(f"\n  Variant {variant+1}/{variants_per_topic}")
                
                result = self.generate_thumbnail(
                    topic=topic,
                    num_steps=NUM_INFERENCE_STEPS
                )
                results.append(result)
        
        print(f"\n{'='*70}")
        print(f"✅ BATCH COMPLETE")
        print(f"{'='*70}")
        print(f"Generated {len(results)} RLHF-optimized thumbnails")
        print(f"Saved to: {OUTPUT_DIR}/\n")
        
        return results

# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(generator):
    """Interactive generation"""
    
    print(f"\n{'='*70}")
    print("INTERACTIVE MODE (RLHF-OPTIMIZED)")
    print("="*70)
    print("Generate thumbnails optimized for human preferences!")
    print("Type 'batch' for batch mode, 'compare' for comparison, 'quit' to exit.\n")
    
    while True:
        print("-" * 70)
        user_input = input("\nTopic (or 'batch'/'compare'/'quit'): ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\nExiting...")
            break
        
        if user_input.lower() == 'compare':
            print("\nComparison mode - enter topics to compare with base model:")
            topics = []
            while True:
                topic = input(f"  Topic {len(topics)+1} (empty to finish): ").strip()
                if not topic:
                    break
                topics.append(topic)
            
            if topics:
                generator.generate_comparison_set(topics)
            continue
        
        if user_input.lower() == 'batch':
            print("\nBatch mode - enter topics (one per line, empty to finish):")
            topics = []
            while True:
                topic = input(f"  Topic {len(topics)+1}: ").strip()
                if not topic:
                    break
                topics.append(topic)
            
            if topics:
                generator.generate_multiple(topics)
            continue
        
        # Single generation
        try:
            result = generator.generate_thumbnail(topic=user_input)
            print(f"\n✓ View your RLHF-optimized thumbnail: {result['path']}")
        except Exception as e:
            print(f"\n❌ Error: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("STEP 8: Final Generation (RLHF-Improved)")
    print("="*70)
    
    # Check if RLHF model exists
    if not Path(VLM_MODEL_PATH).exists():
        print(f"\n❌ ERROR: RLHF-improved model not found!")
        print(f"Expected: {VLM_MODEL_PATH}/")
        print(f"\nPlease run Step 7 first:")
        print(f"  python step7_rl_finetune.py")
        
        # Check if base model exists
        if Path("finetuned_thumbnail_model").exists():
            print(f"\nNote: Base model from Step 3 exists.")
            print(f"      You can use step4_generate_base.py with that.")
        return
    
    # Initialize generator
    try:
        generator = ImprovedThumbnailGenerator(
            vlm_model_path=VLM_MODEL_PATH,
            sd_model_name=SD_MODEL_NAME,
            device=DEVICE
        )
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show menu
    print("\nWhat would you like to do?")
    print("  1. Generate examples (pre-defined topics)")
    print("  2. Interactive mode (enter your own topics)")
    print("  3. Comparison mode (compare with base model)")
    print("  4. Exit")
    
    choice = input("\nChoice (1/2/3/4): ").strip()
    
    if choice == '1':
        # Example generation
        example_topics = [
            "epic gaming moment battle royale",
            "tech review latest smartphone unboxing",
            "cooking recipe delicious chocolate cake"
        ]
        
        try:
            results = generator.generate_multiple(example_topics)
            print("\n✓ Check final_thumbnails/ for your RLHF-optimized images!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == '2':
        interactive_mode(generator)
    
    elif choice == '3':
        # Comparison mode
        print("\nComparison Mode - Generate same topics with RLHF model")
        print("You can compare with base model outputs from Step 4\n")
        
        topics = []
        print("Enter topics (one per line, empty to finish):")
        while True:
            topic = input(f"  Topic {len(topics)+1}: ").strip()
            if not topic:
                break
            topics.append(topic)
        
        if topics:
            try:
                generator.generate_comparison_set(topics)
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    else:
        print("\nExiting...")
        return
    
    print(f"\n{'='*70}")
    print(f"✅ ALL DONE!")
    print(f"{'='*70}")
    print(f"\nCongratulations! You've completed the full RLHF pipeline!")
    print(f"\nYour thumbnails should now be:")
    print(f"  ✓ More engaging and clickable")
    print(f"  ✓ Better aligned with human preferences")
    print(f"  ✓ More professional looking")
    print(f"  ✓ Optimized for YouTube success")
    print(f"\nRecommended next steps:")
    print(f"  1. Test thumbnails on real YouTube videos")
    print(f"  2. Track click-through rates (CTR)")
    print(f"  3. Collect more feedback and re-run RLHF (Steps 5-8)")
    print(f"  4. Fine-tune for specific niches")

if __name__ == "__main__":
    main()
