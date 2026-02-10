"""
STEP 2: Data Labeling & Preparation
====================================
Automatically labels videos as viral/non-viral and prepares training dataset.

TIME REQUIRED: 5-10 minutes
INPUT: youtube_data/metadata/*.json (from Step 1)
OUTPUT: training_data.json, training_data_validation.json

INSTRUCTIONS:
1. Make sure you completed Step 1 (data collection)
2. Run: python step2_label_data.py
3. Wait for processing (very fast)
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

# Virality threshold (percentile)
# 0.7 = top 30% are "viral", bottom 70% are "non-viral"
VIRALITY_THRESHOLD = 0.7

# Train/validation split
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# Output files
TRAINING_OUTPUT = r"D:\Youtube_Data_Analytics\Data\TrainingDataset\training_data.json"
VALIDATION_OUTPUT = r"D:\Youtube_Data_Analytics\Data\TrainingDataset\training_data_validation.json"

# TRAINING_OUTPUT = "training_data.json"
# VALIDATION_OUTPUT = "training_data_validation.json"


# =============================================================================
# MAIN CODE
# =============================================================================

class DatasetPreparer:
    """Handles data labeling and preparation"""
    
    def __init__(self, metadata_file):
        print(f"Loading metadata from: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.df = pd.DataFrame(self.data)
        print(f"✓ Loaded {len(self.df)} videos")
    
    def calculate_virality_score(self):
        """
        Calculate virality score combining views and engagement
        
        Formula:
        - View Score: Normalized view count (0-1)
        - Engagement Score: (likes + comments) / views (0-1)
        - Final: 70% view score + 30% engagement score
        """
        print("\nCalculating virality scores...")
        
        # Handle edge cases
        if len(self.df) == 0:
            print("✗ No data to process")
            return self.df
        
        # Normalize view counts (0-1 scale)
        view_min = self.df['view_count'].min()
        view_max = self.df['view_count'].max()
        
        if view_max > view_min:
            self.df['view_score'] = (
                (self.df['view_count'] - view_min) / (view_max - view_min)
            )
        else:
            self.df['view_score'] = 0.5
        
        # Calculate engagement rate
        self.df['engagement_rate'] = (
            (self.df['like_count'] + self.df['comment_count']) / 
            (self.df['view_count'] + 1)  # +1 to avoid division by zero
        )
        
        # Normalize engagement rate
        eng_min = self.df['engagement_rate'].min()
        eng_max = self.df['engagement_rate'].max()
        
        if eng_max > eng_min:
            self.df['engagement_score'] = (
                (self.df['engagement_rate'] - eng_min) / (eng_max - eng_min)
            )
        else:
            self.df['engagement_score'] = 0.5
        
        # Combined virality score (weighted average)
        self.df['virality_score'] = (
            0.7 * self.df['view_score'] + 
            0.3 * self.df['engagement_score']
        )
        
        print("✓ Virality scores calculated")
        return self.df
    
    def create_labels(self, threshold=VIRALITY_THRESHOLD):
        """
        Create binary labels based on virality threshold
        
        Args:
            threshold: Percentile threshold (0-1)
        """
        print(f"\nCreating labels (threshold: {threshold*100:.0f}th percentile)...")
        
        virality_cutoff = self.df['virality_score'].quantile(threshold)
        self.df['is_viral'] = (self.df['virality_score'] >= virality_cutoff).astype(int)
        
        viral_count = self.df['is_viral'].sum()
        non_viral_count = len(self.df) - viral_count
        
        print(f"\n📊 Label Distribution:")
        print(f"  🔥 Viral: {viral_count} ({viral_count/len(self.df)*100:.1f}%)")
        print(f"  📺 Non-viral: {non_viral_count} ({non_viral_count/len(self.df)*100:.1f}%)")
        print(f"  Cutoff score: {virality_cutoff:.3f}")
        
        return self.df
    
    def extract_title_features(self):
        """Extract features from titles that correlate with virality"""
        print("\nExtracting title features...")
        
        # Basic features
        self.df['title_length'] = self.df['title'].str.len()
        self.df['word_count'] = self.df['title'].str.split().str.len()
        
        # Pattern detection
        self.df['has_numbers'] = self.df['title'].str.contains(r'\d+', na=False).astype(int)
        self.df['has_caps'] = self.df['title'].str.contains(r'\b[A-Z]{2,}\b', na=False).astype(int)
        self.df['has_emoji'] = self.df['title'].str.contains(r'[^\w\s]', na=False).astype(int)
        self.df['has_question'] = self.df['title'].str.contains(r'\?', na=False).astype(int)
        self.df['has_exclamation'] = self.df['title'].str.contains(r'!', na=False).astype(int)
        
        # Viral keywords
        viral_keywords = {
            'how_to': 'how to',
            'why': 'why',
            'best': 'best',
            'top': 'top',
            'ultimate': 'ultimate',
            'insane': 'insane',
            'shocking': 'shocking',
            'secret': 'secret',
            'revealed': 'revealed',
            'reaction': 'reaction',
            'beginner':'beginner'
        }
        
        for key, keyword in viral_keywords.items():
            col_name = f'has_{key}'
            self.df[col_name] = (
                self.df['title'].str.lower().str.contains(keyword, na=False).astype(int)
            )
        
        print("✓ Title features extracted")
        return self.df
    
    def prepare_training_data(self, output_file=TRAINING_OUTPUT, train_split=TRAIN_SPLIT):
        """Prepare final training and validation datasets"""
        print("\n" + "="*70)
        print("PREPARING TRAINING DATA")
        print("="*70)
        
        # Calculate scores and labels
        self.calculate_virality_score()
        self.create_labels(threshold=VIRALITY_THRESHOLD)
        self.extract_title_features()
        
        # Filter videos with valid thumbnails
        print("\nValidating thumbnails...")
        self.df = self.df[self.df['local_thumbnail_path'].notna()]
        self.df = self.df[self.df['local_thumbnail_path'].apply(
            lambda x: Path(x).exists() if pd.notna(x) else False
        )]
        
        print(f"✓ Videos with valid thumbnails: {len(self.df)}")
        
        if len(self.df) == 0:
            print("\n❌ ERROR: No valid thumbnails found!")
            print("Make sure Step 1 completed successfully.")
            return []
        
        # Shuffle and split
        shuffled_df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(shuffled_df) * train_split)
        
        train_df = shuffled_df[:split_idx]
        val_df = shuffled_df[split_idx:]
        
        # Create training examples
        def create_examples(df):
            examples = []
            for _, row in df.iterrows():
                example = {
                    'image_path': row['local_thumbnail_path'],
                    'title': row['title'],
                    'is_viral': int(row['is_viral']),
                    'virality_score': float(row['virality_score']),
                    'view_count': int(row['view_count']),
                    'engagement_rate': float(row['engagement_rate']),
                    'category': row['category'],
                    'title_length': int(row['title_length']),
                    'word_count': int(row['word_count']),
                    'has_numbers': int(row['has_numbers']),
                    'has_caps': int(row['has_caps'])
                }
                examples.append(example)
            return examples
        
        training_data = create_examples(train_df)
        validation_data = create_examples(val_df)
        
        # Save datasets
        print("\nSaving datasets...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        val_file = output_file.replace('.json', '_validation.json')
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print(f"\n{'='*70}")
        print("✅ DATA PREPARATION COMPLETE")
        print(f"{'='*70}")
        print(f"\n📁 Files created:")
        print(f"  Training: {output_file} ({len(training_data)} examples)")
        print(f"  Validation: {val_file} ({len(validation_data)} examples)")
        
        print(f"\n📊 Training Dataset Statistics:")
        train_df_stats = pd.DataFrame(training_data)
        viral_count = train_df_stats['is_viral'].sum()
        print(f"  Viral: {viral_count} ({viral_count/len(train_df_stats)*100:.1f}%)")
        print(f"  Non-viral: {len(train_df_stats)-viral_count}")
        
        print(f"\n📈 Average Metrics:")
        viral_data = train_df_stats[train_df_stats['is_viral'] == 1]
        non_viral_data = train_df_stats[train_df_stats['is_viral'] == 0]
        
        if len(viral_data) > 0:
            print(f"  Viral videos:")
            print(f"    Views: {viral_data['view_count'].mean():,.0f}")
            print(f"    Title length: {viral_data['title_length'].mean():.1f} chars")
        
        if len(non_viral_data) > 0:
            print(f"  Non-viral videos:")
            print(f"    Views: {non_viral_data['view_count'].mean():,.0f}")
            print(f"    Title length: {non_viral_data['title_length'].mean():.1f} chars")
        
        print(f"\n➡️  Next step: Run 'python step3_train_model.py'")
        print(f"    (This will take 2-4 hours with GPU)")
        
        return training_data

def find_latest_metadata():
    """Find the most recent metadata file from Step 1"""
    metadata_files = glob.glob("D:\\Youtube_Data_Analytics\\Data\\metadata\\dataset_*.json")
    # metadata_files = "D:\\Youtube_Data_Analytics\\Data\metadata\\dataset_20260102_114007.json"
    
    if not metadata_files:
        print("\n❌ ERROR: No metadata files found!")
        print("\nPlease run Step 1 first:")
        print("  python step1_collect_data.py")
        return None
    
    # Return most recent file
    latest = max(metadata_files, key=lambda x: Path(x).stat().st_mtime)
    return latest

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("STEP 2: Data Labeling & Preparation")
    print("="*70 + "\n")
    
    # Find metadata file
    metadata_file = find_latest_metadata()
    if not metadata_file:
        return
    
    print(f"Using: {metadata_file}\n")
    
    # Prepare data
    try:
        preparer = DatasetPreparer(metadata_file)
        training_data = preparer.prepare_training_data(
            output_file=TRAINING_OUTPUT,
            train_split=TRAIN_SPLIT
        )
        
        if len(training_data) == 0:
            print("\n⚠️  Warning: No training data created")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
