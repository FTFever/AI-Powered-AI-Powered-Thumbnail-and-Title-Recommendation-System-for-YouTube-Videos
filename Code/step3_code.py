"""
STEP 3: Initial Model Training
===============================
Fine-tune BLIP-2 vision-language model with LoRA for efficient training.

TIME REQUIRED: 2-4 hours (GPU) or 10+ hours (CPU)
INPUT: training_data.json (from Step 2)
OUTPUT: finetuned_thumbnail_model/ directory

INSTRUCTIONS:
1. Ensure training_data.json exists from Step 2
2. Configure settings below if needed
3. Run: python step3_train_model.py
4. Monitor progress (loss should decrease)
5. Wait for completion (be patient!)

HARDWARE NOTES:
- GPU (RTX 3060+): ~2-3 hours
- GPU (GTX 1660): ~4-6 hours
- CPU: 10-15 hours (not recommended)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - ADJUST THESE SETTINGS
# =============================================================================

# Model settings
MODEL_NAME = "Salesforce/blip2-opt-2.7b"  # Lightweight BLIP-2 model

# Training parameters
BATCH_SIZE = 2          # Reduce to 1 if out of memory
EPOCHS = 3              # More epochs = better quality (but slower)
LEARNING_RATE = 2e-5    # Learning rate for training
MAX_SAMPLES = None      # None = use all data, or set number to test quickly

# LoRA configuration (for efficient training)
LORA_R = 8              # LoRA rank (higher = more parameters)
LORA_ALPHA = 32         # LoRA alpha
LORA_DROPOUT = 0.05     # Dropout rate

# Output directory
OUTPUT_DIR = "D:\Youtube_Data_Analytics\Data\model"

# Input files
TRAIN_DATA_FILE = r"D:\Youtube_Data_Analytics\Data\TrainingDataset\training_data.json"
VAL_DATA_FILE = r"D:\Youtube_Data_Analytics\Data\TrainingDataset\training_data_validation.json"

# Device configuration
DEVICE = None  # None = auto-detect, or set to 'cuda' or 'cpu'

# =============================================================================
# DATASET CLASS
# =============================================================================

class ThumbnailDataset(Dataset):
    """Dataset for thumbnail images and titles"""
    
    def __init__(self, data_file, processor, max_samples=None):
        print(f"Loading dataset from: {data_file}")
        
        with open(data_file, 'r',encoding = "utf-8", errors = "ignore") as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:max_samples]
            print(f"  Using {len(self.data)} samples (limited)")
        else:
            print(f"  Total samples: {len(self.data)}")
        
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {item['image_path']}: {e}")
            # Create blank image as fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        # Create training prompt and label based on virality
        if item['is_viral']:
            prompt = "Question: What makes this YouTube thumbnail engaging and viral? Answer:"
            label = "This thumbnail is highly engaging with strong visual appeal, clear composition, vibrant colors, and professional quality that attracts clicks."
        else:
            prompt = "Question: What makes this YouTube thumbnail engaging and viral? Answer:"
            label = "This thumbnail has moderate engagement potential with room for improvement in visual impact, composition, or clarity."
        
        # Process inputs
        try:
            encoding = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            
            # Process labels
            labels = self.processor.tokenizer(
                label,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            
            return {
                'pixel_values': encoding['pixel_values'].squeeze(),
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': labels['input_ids'].squeeze()
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return dummy data to avoid breaking the batch
            return self.__getitem__((idx + 1) % len(self.data))

# =============================================================================
# TRAINER CLASS
# =============================================================================

class ThumbnailModelTrainer:
    """Handles model training with LoRA"""
    
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n{'='*70}")
        print(f"INITIALIZING MODEL")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  Using CPU - training will be slow!")
        
        # Load processor
        print("\nLoading processor...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # Load model
        print("Loading model (this may take several minutes)...")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map="auto" if self.device == 'cuda' else None
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        print("✓ Model loaded")
        
        # Apply LoRA for efficient fine-tuning
        print("\nApplying LoRA (efficient fine-tuning)...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print parameter counts
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        print(f"✓ LoRA applied")
        print(f"\nTrainable parameters: {trainable:,} / {total:,}")
        print(f"Percentage trainable: {100 * trainable / total:.2f}%")
    
    def train(self, train_file=TRAIN_DATA_FILE, val_file=VAL_DATA_FILE,
              output_dir=OUTPUT_DIR, batch_size=BATCH_SIZE, epochs=EPOCHS, 
              learning_rate=LEARNING_RATE, max_samples=MAX_SAMPLES):
        """
        Train the model
        
        Args:
            train_file: Path to training data JSON
            val_file: Path to validation data JSON
            output_dir: Where to save the model
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate
            max_samples: Limit samples for testing (None = use all)
        """
        
        print(f"\n{'='*70}")
        print(f"PREPARING TRAINING")
        print(f"{'='*70}")
        
        # Create datasets
        train_dataset = ThumbnailDataset(train_file, self.processor, max_samples)
        val_dataset = ThumbnailDataset(val_file, self.processor, 
                                       max_samples=max_samples//4 if max_samples else None)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"\nTraining batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        num_training_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        
        # Training loop
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING - {epochs} EPOCHS")
        print(f"{'='*70}\n")
        
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n--- EPOCH {epoch+1}/{epochs} ---")
            
            # Training phase
            total_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move to device
                    pixel_values = batch['pixel_values'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_train_loss += loss.item()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    # Update progress
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    # Clear cache periodically
                    if self.device == 'cuda' and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️  GPU out of memory! Try reducing BATCH_SIZE to 1")
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                        raise e
                    else:
                        print(f"\nError in batch {batch_idx}: {e}")
                        continue
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    pixel_values = batch['pixel_values'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    total_val_loss += outputs.loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Training Loss:   {avg_train_loss:.4f}")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"  ✓ New best model! (val_loss: {best_val_loss:.4f})")
            
            self.model.train()
        
        # Save final model
        print(f"\n{'='*70}")
        print(f"SAVING MODEL")
        print(f"{'='*70}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        
        print(f"\n✅ TRAINING COMPLETE!")
        print(f"Model saved to: {output_path.absolute()}")
        print(f"\nFinal Results:")
        print(f"  Best Validation Loss: {best_val_loss:.4f}")
        print(f"  Total Epochs: {epochs}")
        print(f"\n➡️  Next step: Run 'python step4_generate_base.py'")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main training function"""
    
    print("\n" + "="*70)
    print("STEP 3: Initial Model Training")
    print("="*70)
    
    # Check if training data exists
    if not Path(TRAIN_DATA_FILE).exists():
        print(f"\n❌ ERROR: Training data not found!")
        print(f"Expected: {TRAIN_DATA_FILE}")
        print(f"\nPlease run Step 2 first:")
        print(f"  python step2_label_data.py")
        return
    
    # Initialize trainer
    try:
        print("\nInitializing trainer...")
        trainer = ThumbnailModelTrainer(
            model_name=MODEL_NAME,
            device=DEVICE
        )
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train model
    try:
        trainer.train(
            train_file=TRAIN_DATA_FILE,
            val_file=VAL_DATA_FILE,
            output_dir=OUTPUT_DIR,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            max_samples=MAX_SAMPLES
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Progress has been saved at last checkpoint")
    except Exception as e:
        print(f"\n❌ Training Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
