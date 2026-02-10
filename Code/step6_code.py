"""
STEP 6: Reward Model Training
==============================
Train a neural network to predict human preferences for thumbnails.

TIME REQUIRED: 20-30 minutes
INPUT: human_feedback.json (from Step 5), finetuned_thumbnail_model/
OUTPUT: reward_model.pt

INSTRUCTIONS:
1. Ensure you have human_feedback.json with at least 15 ratings
2. Ensure finetuned_thumbnail_model/ exists from Step 3
3. Run: python step6_train_reward.py
4. Wait for training to complete

PURPOSE: The reward model learns to predict how humans will rate thumbnails,
         which is then used in RL training to improve the main model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files
FEEDBACK_FILE = "human_feedback.json"
VLM_MODEL_PATH = "finetuned_thumbnail_model"

# Output
REWARD_MODEL_OUTPUT = "reward_model.pt"

# Training parameters
EPOCHS = 15
LEARNING_RATE = 1e-4
BATCH_SIZE = 4

# Device
DEVICE = None  # None = auto-detect

# =============================================================================
# REWARD MODEL ARCHITECTURE
# =============================================================================

class RewardModel(nn.Module):
    """
    Neural network that predicts human preference scores from thumbnail features
    
    Input: Image features from vision model (768-dim vector)
    Output: Predicted engagement score (scalar)
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
# DATASET
# =============================================================================

class FeedbackDataset(Dataset):
    """Dataset of human feedback for reward model training"""
    
    def __init__(self, feedback_data, processor, vision_model, device):
        self.data = feedback_data
        self.processor = processor
        self.vision_model = vision_model
        self.device = device
        
        print(f"Loading {len(feedback_data)} feedback examples...")
        
        # Pre-extract all features to speed up training
        self.features = []
        self.scores = []
        
        for item in tqdm(feedback_data, desc="Extracting features"):
            try:
                # Load image
                image = Image.open(item['image_path']).convert('RGB')
                
                # Get image features
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = vision_model.vision_model(**inputs)
                    # Use mean pooling over spatial dimensions
                    features = outputs.last_hidden_state.mean(dim=1).cpu()
                
                self.features.append(features)
                
                # Normalize score to 0-1 range
                score = item['overall_score'] / 10.0
                self.scores.append(score)
                
            except Exception as e:
                print(f"Warning: Could not process {item.get('image_path', 'unknown')}: {e}")
                continue
        
        print(f"✓ Extracted features for {len(self.features)} examples")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx].squeeze(),
            'score': torch.tensor(self.scores[idx], dtype=torch.float32)
        }

# =============================================================================
# TRAINER
# =============================================================================

class RewardModelTrainer:
    """Handles reward model training"""
    
    def __init__(self, feedback_file, vlm_model_path, device=None):
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n{'='*70}")
        print(f"INITIALIZING REWARD MODEL TRAINER")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        # Load feedback data
        print(f"\nLoading feedback from: {feedback_file}")
        with open(feedback_file, 'r') as f:
            self.feedback_data = json.load(f)
        
        print(f"✓ Loaded {len(self.feedback_data)} ratings")
        
        if len(self.feedback_data) < 10:
            print("\n⚠️  Warning: Very few ratings! Recommend at least 15-20 for good results.")
        
        # Load vision model for feature extraction
        print("\nLoading vision model for feature extraction...")
        self.processor = Blip2Processor.from_pretrained(vlm_model_path)
        self.vision_model = Blip2ForConditionalGeneration.from_pretrained(
            vlm_model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        self.vision_model.eval()
        
        print("✓ Vision model loaded")
        
        # Initialize reward model
        print("\nInitializing reward model...")
        self.reward_model = RewardModel(feature_dim=768).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.reward_model.parameters())
        print(f"✓ Reward model initialized ({total_params:,} parameters)")
    
    def train(self, epochs=EPOCHS, learning_rate=LEARNING_RATE, 
              batch_size=BATCH_SIZE, output_path=REWARD_MODEL_OUTPUT):
        """
        Train the reward model
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            output_path: Where to save trained model
        """
        
        print(f"\n{'='*70}")
        print(f"PREPARING TRAINING")
        print(f"{'='*70}\n")
        
        # Create dataset
        dataset = FeedbackDataset(
            self.feedback_data,
            self.processor,
            self.vision_model,
            self.device
        )
        
        if len(dataset) == 0:
            print("\n❌ ERROR: No valid training data!")
            return
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        print(f"\nTraining setup:")
        print(f"  Examples: {len(dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {len(dataloader)}")
        print(f"  Epochs: {epochs}")
        
        # Setup training
        optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}\n")
        
        self.reward_model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                features = batch['features'].to(self.device)
                scores = batch['score'].unsqueeze(1).to(self.device)
                
                # Forward pass
                predictions = self.reward_model(features)
                loss = criterion(predictions, scores)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Print progress
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:2d}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Final Loss: {avg_loss:.4f}")
        print(f"Best Loss: {best_loss:.4f}")
        
        # Save model
        print(f"\nSaving reward model to: {output_path}")
        torch.save({
            'model_state_dict': self.reward_model.state_dict(),
            'feature_dim': 768,
            'final_loss': avg_loss,
            'num_training_examples': len(dataset)
        }, output_path)
        
        print(f"✓ Model saved")
        
        # Test predictions
        self._test_model(dataset)
    
    def _test_model(self, dataset):
        """Test model on training data to verify it learned"""
        print(f"\n{'='*70}")
        print(f"MODEL VERIFICATION")
        print(f"{'='*70}\n")
        
        self.reward_model.eval()
        
        # Sample a few examples
        num_samples = min(5, len(dataset))
        indices = torch.randperm(len(dataset))[:num_samples]
        
        print("Sample predictions vs actual scores:")
        print(f"{'Actual':<10} {'Predicted':<10} {'Difference':<10}")
        print("-" * 30)
        
        with torch.no_grad():
            for idx in indices:
                item = dataset[idx]
                features = item['features'].unsqueeze(0).to(self.device)
                actual = item['score'].item()
                
                prediction = self.reward_model(features).item()
                diff = abs(prediction - actual)
                
                print(f"{actual*10:8.1f}   {prediction*10:8.1f}   {diff*10:8.2f}")
        
        print("\nIf predictions are close to actuals, the model learned well!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("STEP 6: Reward Model Training")
    print("="*70)
    
    # Check if feedback file exists
    if not Path(FEEDBACK_FILE).exists():
        print(f"\n❌ ERROR: Feedback file not found!")
        print(f"Expected: {FEEDBACK_FILE}")
        print(f"\nPlease run Step 5 first:")
        print(f"  python step5b_feedback_web.py")
        print(f"  (or step5a_feedback_cli.py)")
        return
    
    # Check if vision model exists
    if not Path(VLM_MODEL_PATH).exists():
        print(f"\n❌ ERROR: Vision model not found!")
        print(f"Expected: {VLM_MODEL_PATH}/")
        print(f"\nPlease run Step 3 first:")
        print(f"  python step3_train_model.py")
        return
    
    # Check feedback data
    with open(FEEDBACK_FILE, 'r') as f:
        feedback = json.load(f)
    
    if len(feedback) < 5:
        print(f"\n⚠️  WARNING: Only {len(feedback)} ratings found!")
        print("Recommend at least 15-20 ratings for good results.")
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != 'y':
            print("Exiting. Please collect more ratings first.")
            return
    
    # Train reward model
    try:
        trainer = RewardModelTrainer(
            feedback_file=FEEDBACK_FILE,
            vlm_model_path=VLM_MODEL_PATH,
            device=DEVICE
        )
        
        trainer.train(
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            output_path=REWARD_MODEL_OUTPUT
        )
        
        print(f"\n{'='*70}")
        print(f"✅ REWARD MODEL TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"\nThe reward model can now predict how humans will rate thumbnails!")
        print(f"\n➡️  Next step: Run 'python step7_rl_finetune.py'")
        print(f"    (This will take 1-2 hours)")
        
    except Exception as e:
        print(f"\n❌ Training Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
