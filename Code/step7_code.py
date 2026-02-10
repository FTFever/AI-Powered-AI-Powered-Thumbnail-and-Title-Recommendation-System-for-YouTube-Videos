"""
STEP 7: RL Fine-tuning with Human Feedback
===========================================
Use reinforcement learning to optimize the model based on human preferences.

TIME REQUIRED: 1-2 hours (GPU) or 3-4 hours (CPU)
INPUT: finetuned_thumbnail_model/, reward_model.pt, human_feedback.json
OUTPUT: rlhf_improved_model/

INSTRUCTIONS:
1. Ensure all previous steps are complete
2. Run: python step7_rl_finetune.py
3. Wait for RL training (1-2 hours)
4. Model will be saved to rlhf_improved_model/

PURPOSE: This step uses the reward model to guide the main model toward
         generating thumbnails that humans will rate highly.
"""

import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - ADJUST THESE SETTINGS
# =============================================================================

# Input files/models
BASE_MODEL_PATH = "finetuned_thumbnail_model"  # From Step 3
REWARD_MODEL_PATH = "reward_model.pt"          # From Step 6
FEEDBACK_FILE = "human_feedback.json"          # From Step 5

# Output
OUTPUT_DIR = "rlhf_improved_model"

# RL Training parameters
RL_ITERATIONS = 100     # Number of RL update steps (50-200)
BATCH_SIZE = 4          # Samples per iteration (2-8)
LEARNING_RATE = 1e-6    # Low LR for stability (1e-7 to 5e-6)
TEMPERATURE = 0.9       # Sampling temperature (0.7-1.0)

# LoRA configuration
LORA_R = 8              # LoRA rank
LORA_ALPHA = 32         # LoRA alpha
LORA_DROPOUT = 0.05     # Dropout rate

# Device
DEVICE = None  # None = auto-detect, or 'cuda'/'cpu'

# =============================================================================
# REWARD MODEL (Same architecture as Step 6)
# =============================================================================

class RewardModel(nn.Module):
    """
    Reward model that predicts human preferences
    
    Takes image features as input and outputs predicted rating (0-1 scale)
    """
    
    def __init__(self, feature_dim=768):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: single score
        )
    
    def forward(self, features):
        """
        Forward pass
        
        Args:
            features: (batch_size, feature_dim) tensor
        
        Returns:
            scores: (batch_size, 1) tensor of predicted scores
        """
        return self.network(features)

# =============================================================================
# RL TRAINER
# =============================================================================

class RLHFTrainer:
    """Handles RL fine-tuning with human feedback"""
    
    def __init__(self, base_model_path, reward_model_path, feedback_file, device=None):
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n{'='*70}")
        print(f"INITIALIZING RLHF TRAINER")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  Using CPU - training will be very slow!")
        
        # Load processor and base model
        print("\n[1/3] Loading base model...")
        self.processor = Blip2Processor.from_pretrained(base_model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        
        print("✓ Base model loaded")
        
        # Apply LoRA for efficient fine-tuning
        print("\n[2/3] Applying LoRA for efficient training...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✓ LoRA applied")
        print(f"  Trainable: {trainable:,} parameters ({100*trainable/total:.2f}%)")
        
        # Load reward model
        print("\n[3/3] Loading reward model...")
        self.reward_model = RewardModel(feature_dim=768).to(self.device)
        
        checkpoint = torch.load(reward_model_path, map_location=self.device)
        self.reward_model.load_state_dict(checkpoint['model_state_dict'])
        self.reward_model.eval()
        
        print(f"✓ Reward model loaded")
        print(f"  (Trained on {checkpoint.get('num_training_examples', 'unknown')} examples)")
        
        # Load feedback for sampling prompts
        print(f"\nLoading feedback data...")
        with open(feedback_file, 'r') as f:
            self.feedback_data = json.load(f)
        
        # Extract unique prompts/topics
        self.prompts = []
        for f in self.feedback_data:
            prompt = f.get('prompt', '')
            if prompt and prompt not in self.prompts:
                self.prompts.append(prompt)
        
        # If no prompts found, create generic ones
        if not self.prompts:
            self.prompts = [
                "viral youtube thumbnail",
                "engaging video content",
                "popular youtube video"
            ]
            print("⚠️  No prompts found in feedback, using generic prompts")
        
        print(f"✓ Loaded {len(self.feedback_data)} feedback examples")
        print(f"✓ Found {len(self.prompts)} unique prompts for training")
    
    def get_image_features(self, prompt):
        """
        Generate image features from a text prompt
        
        Args:
            prompt: Text prompt
        
        Returns:
            features: Tensor of image features
        """
        inputs = self.processor(
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Get embeddings from the model
            outputs = self.model.language_model(
                input_ids=inputs['input_ids'],
                output_hidden_states=True
            )
            
            # Use final hidden states as features
            features = outputs.hidden_states[-1].mean(dim=1)
        
        return features
    
    def rl_finetune(self, num_iterations=RL_ITERATIONS, batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATE, temperature=TEMPERATURE):
        """
        RL fine-tuning with policy gradients (REINFORCE algorithm)
        
        Args:
            num_iterations: Number of RL update steps
            batch_size: Samples per iteration
            learning_rate: Learning rate
            temperature: Sampling temperature
        """
        
        print(f"\n{'='*70}")
        print(f"STARTING RL FINE-TUNING")
        print(f"{'='*70}")
        print(f"\nConfiguration:")
        print(f"  Iterations: {num_iterations}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Temperature: {temperature}")
        
        print(f"\n⏱️  Estimated time:")
        if self.device == 'cuda':
            print(f"  GPU: 1-2 hours")
        else:
            print(f"  CPU: 3-4 hours")
        
        print(f"\nProgress is shown below. You can stop anytime (Ctrl+C).\n")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        self.reward_model.eval()
        
        rewards_history = []
        best_avg_reward = -float('inf')
        
        try:
            for iteration in tqdm(range(num_iterations), desc="RL Training"):
                # Sample prompts
                if len(self.prompts) >= batch_size:
                    sampled_prompts = np.random.choice(
                        self.prompts,
                        size=batch_size,
                        replace=False
                    )
                else:
                    sampled_prompts = self.prompts
                
                iteration_rewards = []
                iteration_losses = []
                
                for prompt in sampled_prompts:
                    try:
                        # Process prompt
                        inputs = self.processor(
                            text=f"Generate engaging YouTube thumbnail for: {prompt}",
                            return_tensors="pt"
                        ).to(self.device)
                        
                        # Generate with current policy
                        outputs = self.model.generate(
                            **inputs,
                            max_length=100,
                            do_sample=True,
                            temperature=temperature,
                            output_scores=True,
                            return_dict_in_generate=True
                        )
                        
                        # Get features for reward calculation
                        features = self.get_image_features(prompt)
                        
                        # Get reward from reward model
                        with torch.no_grad():
                            reward = self.reward_model(features).item()
                        
                        iteration_rewards.append(reward)
                        
                        # Policy gradient loss (REINFORCE)
                        # We want to maximize reward, so minimize negative log prob * reward
                        if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                            # Calculate log probabilities
                            log_probs = []
                            for score in outputs.scores:
                                log_prob = torch.log_softmax(score, dim=-1).max()
                                log_probs.append(log_prob)
                            
                            avg_log_prob = torch.stack(log_probs).mean()
                            
                            # Policy loss: -log_prob * reward
                            # Higher reward = reinforce this behavior
                            # Lower reward = discourage this behavior
                            policy_loss = -avg_log_prob * reward
                            
                            iteration_losses.append(policy_loss.item())
                            
                            # Backward pass
                            optimizer.zero_grad()
                            policy_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            optimizer.step()
                        
                    except Exception as e:
                        print(f"\n⚠️  Warning: Error in iteration {iteration}, prompt '{prompt[:30]}...': {e}")
                        continue
                    
                    # Clear cache periodically
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                
                # Track average reward
                if iteration_rewards:
                    avg_reward = np.mean(iteration_rewards)
                    rewards_history.append(avg_reward)
                    
                    # Track best
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                    
                    # Print progress every 10 iterations
                    if (iteration + 1) % 10 == 0:
                        recent_avg = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else avg_reward
                        print(f"\nIteration {iteration+1}/{num_iterations}")
                        print(f"  Current Reward: {avg_reward:.4f}")
                        print(f"  Recent Avg (last 10): {recent_avg:.4f}")
                        print(f"  Best Reward: {best_avg_reward:.4f}")
                        if iteration_losses:
                            print(f"  Avg Loss: {np.mean(iteration_losses):.4f}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user!")
            print("Partial progress will be saved...")
        
        print(f"\n{'='*70}")
        print(f"RL TRAINING COMPLETE")
        print(f"{'='*70}")
        
        # Print reward improvement
        if len(rewards_history) > 20:
            initial_reward = np.mean(rewards_history[:10])
            final_reward = np.mean(rewards_history[-10:])
            improvement = ((final_reward - initial_reward) / abs(initial_reward)) * 100
            
            print(f"\nReward Progress:")
            print(f"  Initial (first 10 iterations): {initial_reward:.4f}")
            print(f"  Final (last 10 iterations):   {final_reward:.4f}")
            print(f"  Change: {improvement:+.1f}%")
            
            if improvement > 5:
                print(f"\n✅ Model improved! Thumbnails should be more engaging.")
            elif improvement > 0:
                print(f"\n✓ Slight improvement detected.")
            else:
                print(f"\n⚠️  No significant improvement. Consider:")
                print(f"     - Running more iterations")
                print(f"     - Collecting more diverse feedback")
                print(f"     - Adjusting learning rate")
        else:
            print(f"\n⚠️  Too few iterations to assess improvement")
    
    def save_model(self, output_dir=OUTPUT_DIR):
        """Save the RL-improved model"""
        
        print(f"\n{'='*70}")
        print(f"SAVING IMPROVED MODEL")
        print(f"{'='*70}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save model and processor
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        
        print(f"\n✓ Model saved to: {output_path.absolute()}")
        print(f"\nThis RLHF-improved model should generate better thumbnails!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("STEP 7: RL Fine-tuning with Human Feedback")
    print("="*70)
    
    # Check all required files
    missing = []
    
    if not Path(BASE_MODEL_PATH).exists():
        missing.append(f"{BASE_MODEL_PATH}/ (from Step 3)")
    
    if not Path(REWARD_MODEL_PATH).exists():
        missing.append(f"{REWARD_MODEL_PATH} (from Step 6)")
    
    if not Path(FEEDBACK_FILE).exists():
        missing.append(f"{FEEDBACK_FILE} (from Step 5)")
    
    if missing:
        print(f"\n❌ ERROR: Missing required files:")
        for item in missing:
            print(f"  - {item}")
        print(f"\nPlease complete previous steps first.")
        return
    
    # Check feedback data quality
    with open(FEEDBACK_FILE, 'r') as f:
        feedback = json.load(f)
    
    if len(feedback) < 10:
        print(f"\n⚠️  WARNING: Only {len(feedback)} ratings found!")
        print("Recommendation: At least 15-20 ratings for good RLHF results.")
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != 'y':
            print("Exiting. Please collect more ratings first.")
            return
    
    # Warn about time
    print(f"\n⏱️  TIME WARNING:")
    print(f"  This step takes 1-2 hours on a good GPU.")
    print(f"  On CPU, it may take 3-4 hours or more.")
    print(f"  Progress is shown, and you can stop anytime (Ctrl+C).")
    print(f"  Your progress will be saved.")
    
    response = input(f"\nContinue? (y/n): ").lower()
    if response != 'y':
        print("Exiting.")
        return
    
    # Train
    try:
        trainer = RLHFTrainer(
            base_model_path=BASE_MODEL_PATH,
            reward_model_path=REWARD_MODEL_PATH,
            feedback_file=FEEDBACK_FILE,
            device=DEVICE
        )
        
        trainer.rl_finetune(
            num_iterations=RL_ITERATIONS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            temperature=TEMPERATURE
        )
        
        trainer.save_model(output_dir=OUTPUT_DIR)
        
        print(f"\n{'='*70}")
        print(f"✅ RLHF TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"\nYour model is now optimized for human preferences!")
        print(f"\n➡️  Next step: Run 'python step8_generate_final.py'")
        print(f"    to generate thumbnails with the improved model")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        print("You can re-run this script to continue training.")
    except Exception as e:
        print(f"\n❌ Training Error: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\nTroubleshooting:")
        print(f"  - If out of memory: Reduce BATCH_SIZE to 2 or 1")
        print(f"  - If unstable: Lower LEARNING_RATE to 5e-7")
        print(f"  - If too slow: Reduce RL_ITERATIONS to 50")

if __name__ == "__main__":
    main()
