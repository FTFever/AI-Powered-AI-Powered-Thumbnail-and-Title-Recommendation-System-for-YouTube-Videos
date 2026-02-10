"""
ENHANCED STEP 3: Train on Both Visual Analysis AND Title Generation

This is an OPTIONAL enhancement to Step 3 that adds title generation training.
You can use this instead of the original step3_train_model.py

KEY CHANGES:
- Trains on TWO tasks: visual analysis + title generation
- Alternates between both during training
- Results in better title generation
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
import random
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "Salesforce/blip2-opt-2.7b"
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "finetuned_thumbnail_model_with_titles"
TRAIN_DATA_FILE = "training_data.json"
VAL_DATA_FILE = "training_data_validation.json"
DEVICE = None

# =============================================================================
# ENHANCED DATASET - TRAINS ON BOTH TASKS
# =============================================================================

class EnhancedThumbnailDataset(Dataset):
    """
    Dataset that trains on TWO tasks:
    1. Visual analysis (what makes it viral)
    2. Title generation (create engaging titles)
    """
    
    def __init__(self, data_file, processor, max_samples=None):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        self.processor = processor
        print(f"Dataset size: {len(self.data)} samples")
        print(f"Training on: Visual analysis + Title generation")
    
    def __len__(self):
        # Each sample can be used for BOTH tasks, so 2x the data
        return len(self.data) * 2
    
    def __getitem__(self, idx):
        # Determine which task
        task_type = idx % 2  # 0 = visual analysis, 1 = title generation
        sample_idx = idx // 2
        
        item = self.data[sample_idx]
        
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {item['image_path']}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        if task_type == 0:
            # TASK 1: Visual Analysis
            return self._create_visual_analysis_example(item, image)
        else:
            # TASK 2: Title Generation
            return self._create_title_generation_example(item, image)
    
    def _create_visual_analysis_example(self, item, image):
        """Original task: analyze what makes thumbnail viral"""
        
        if item['is_viral']:
            prompt = "Question: What makes this YouTube thumbnail engaging and viral? Answer:"
            label = "This thumbnail is highly engaging with strong visual appeal, clear composition, vibrant colors, and professional quality."
        else:
            prompt = "Question: What makes this YouTube thumbnail engaging and viral? Answer:"
            label = "This thumbnail has moderate engagement potential with room for improvement in visual impact."
        
        encoding = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        labels = self.processor.tokenizer(
            label,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        )
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }
    
    def _create_title_generation_example(self, item, image):
        """NEW TASK: Generate viral YouTube titles"""
        
        # Extract topic from title (simplified)
        topic = item['title'].lower().replace('|', ' ').replace('-', ' ')[:50]
        
        # Different prompt types for variety
        prompt_templates = [
            "Create an exciting YouTube title for this thumbnail:",
            "Generate a viral, engaging title for this video:",
            "What would be a great title for this YouTube video:",
            "Create a click-worthy title for this content:",
        ]
        
        prompt = random.choice(prompt_templates)
        
        # The actual title is the label
        label = item['title'][:100]  # Limit length
        
        encoding = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        labels = self.processor.tokenizer(
            label,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=80
        )
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

# =============================================================================
# TRAINER (Same as original Step 3)
# =============================================================================

class ThumbnailModelTrainer:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n{'='*70}")
        print(f"INITIALIZING ENHANCED MODEL TRAINER")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Training on: Visual Analysis + Title Generation")
        
        print("\nLoading model...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map="auto" if self.device == 'cuda' else None
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        print("✓ Model loaded")
        
        print("\nApplying LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✓ LoRA applied ({100*trainable/total:.2f}% trainable)")
    
    def train(self, train_file=TRAIN_DATA_FILE, val_file=VAL_DATA_FILE,
              output_dir=OUTPUT_DIR, batch_size=BATCH_SIZE, epochs=EPOCHS,
              learning_rate=LEARNING_RATE):
        
        print(f"\n{'='*70}")
        print(f"PREPARING TRAINING")
        print(f"{'='*70}\n")
        
        # Create enhanced datasets
        train_dataset = EnhancedThumbnailDataset(train_file, self.processor)
        val_dataset = EnhancedThumbnailDataset(val_file, self.processor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\nTraining setup:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Tasks per epoch: Visual analysis + Title generation")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        num_training_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        
        print(f"\n{'='*70}")
        print(f"STARTING ENHANCED TRAINING - {epochs} EPOCHS")
        print(f"{'='*70}\n")
        
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n--- EPOCH {epoch+1}/{epochs} ---")
            
            total_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
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
                    
                    loss = outputs.loss
                    total_train_loss += loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    if self.device == 'cuda' and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️  GPU out of memory! Try reducing BATCH_SIZE")
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                        raise e
                    else:
                        print(f"\nError in batch {batch_idx}: {e}")
                        continue
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
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
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Training Loss:   {avg_train_loss:.4f}")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"  ✓ New best model!")
            
            self.model.train()
        
        # Save
        print(f"\n{'='*70}")
        print(f"SAVING ENHANCED MODEL")
        print(f"{'='*70}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        
        print(f"\n✅ TRAINING COMPLETE!")
        print(f"Model saved to: {output_path.absolute()}")
        print(f"\nThis model is trained on:")
        print(f"  1. Visual thumbnail analysis")
        print(f"  2. Title generation ← NEW!")
        print(f"\n➡️  Use this model in Step 4 for better titles!")

def main():
    if not Path(TRAIN_DATA_FILE).exists():
        print(f"\n❌ ERROR: {TRAIN_DATA_FILE} not found!")
        print(f"Please run Step 2 first.")
        return
    
    trainer = ThumbnailModelTrainer()
    trainer.train(
        train_file=TRAIN_DATA_FILE,
        val_file=VAL_DATA_FILE,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

if __name__ == "__main__":
    main()
