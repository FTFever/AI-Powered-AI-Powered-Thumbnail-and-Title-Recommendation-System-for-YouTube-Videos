"""
STEP 5A: Human Feedback Collection (Command-Line Interface)
============================================================
Command-line interface for rating generated thumbnails.

TIME REQUIRED: 1-2 minutes per thumbnail
INPUT: Generated thumbnail images
OUTPUT: human_feedback.json with all ratings

INSTRUCTIONS:
1. Ensure you have generated thumbnails from Step 4
2. Run: python step5a_feedback_cli.py
3. For each thumbnail, rate 1-10 on 5 dimensions
4. Add optional text feedback
5. Repeat for all thumbnails (minimum 15 recommended)

NOTE: For easier/faster rating, use step5b_feedback_web.py instead!
      This CLI version is for users who prefer terminal interfaces.
"""

import json
from pathlib import Path
from datetime import datetime
import glob
from PIL import Image
import os
import platform

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/output
# THUMBNAILS_DIR = "generated_thumbnails"  # Where your thumbnails are
# FEEDBACK_FILE = "human_feedback.json"    # Where to save ratings

THUMBNAILS_DIR = "D:\Youtube_Data_Analytics\Data\generated_thumbnails"
FEEDBACK_FILE = "D:\Youtube_Data_Analytics\Data\human_feedback\human_feedback.json"

# Display settings
SHOW_IMAGES = True  # Try to display images in terminal (requires compatible terminal)

# =============================================================================
# FEEDBACK COLLECTOR
# =============================================================================

class CLIFeedbackCollector:
    """Handles command-line feedback collection"""
    
    def __init__(self, thumbnails_dir, output_file):
        self.thumbnails_dir = Path(thumbnails_dir)
        self.output_file = output_file
        self.feedback_data = []
        
        # Load thumbnails
        self.thumbnails = self._load_thumbnails()
        
        # Load existing feedback
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                self.feedback_data = json.load(f)
            print(f"Loaded {len(self.feedback_data)} existing ratings")
    
    def _load_thumbnails(self):
        """Load all thumbnail images"""
        thumbnails = []
        image_patterns = ['*.png', '*.jpg', '*.jpeg']
        
        for pattern in image_patterns:
            for img_path in self.thumbnails_dir.glob(pattern):
                # Try to extract title from filename or use placeholder
                title = img_path.stem.replace('_', ' ').replace('-', ' ').title()
                
                thumbnails.append({
                    'path': str(img_path),
                    'title': title,
                    'prompt': 'Generated thumbnail'
                })
        
        return thumbnails
    
    def _display_image(self, image_path):
        """Attempt to display image (works in some terminals)"""
        if not SHOW_IMAGES:
            return
        
        try:
            # Try to open image with default viewer
            if platform.system() == 'Darwin':  # macOS
                os.system(f'open "{image_path}"')
            elif platform.system() == 'Windows':
                os.system(f'start "" "{image_path}"')
            else:  # Linux
                os.system(f'xdg-open "{image_path}" &')
        except Exception as e:
            print(f"Could not display image: {e}")
    
    def _get_rating(self, prompt, min_val=1, max_val=10):
        """Get a numeric rating from user with validation"""
        while True:
            try:
                value = input(f"{prompt} ({min_val}-{max_val}): ").strip()
                
                if not value:
                    print("Please enter a value")
                    continue
                
                rating = int(value)
                
                if min_val <= rating <= max_val:
                    return rating
                else:
                    print(f"Please enter a number between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\nRating interrupted")
                raise
    
    def _print_separator(self, char='=', length=70):
        """Print a separator line"""
        print(char * length)
    
    def _print_header(self, text):
        """Print a formatted header"""
        self._print_separator()
        print(text.center(70))
        self._print_separator()
    
    def collect_feedback(self, thumbnail):
        """
        Collect feedback for a single thumbnail
        
        Args:
            thumbnail: Dict with 'path', 'title', 'prompt'
        
        Returns:
            Dict with ratings, or None if skipped
        """
        print("\n\n")
        self._print_header("THUMBNAIL RATING")
        
        print(f"\nImage Path: {thumbnail['path']}")
        print(f"Title: {thumbnail['title']}")
        print(f"Prompt: {thumbnail.get('prompt', 'N/A')}")
        
        # Try to display the image
        self._display_image(thumbnail['path'])
        
        print("\n" + "-" * 70)
        print("RATING INSTRUCTIONS")
        print("-" * 70)
        print("Rate the thumbnail on each dimension from 1 (worst) to 10 (best)")
        print("Be honest - your ratings directly improve the AI!")
        print("-" * 70 + "\n")
        
        # Collect ratings
        try:
            ratings = {}
            
            # 1. Click-worthiness (35% weight)
            print("1. CLICK-WORTHINESS")
            print("   Would you click on this thumbnail if you saw it on YouTube?")
            print("   Consider: Does it grab attention? Create curiosity?")
            ratings['click_worthiness'] = self._get_rating("   Rating")
            
            print()
            
            # 2. Visual Appeal (25% weight)
            print("2. VISUAL APPEAL")
            print("   Does it look professional and high-quality?")
            print("   Consider: Colors, composition, lighting, clarity")
            ratings['visual_appeal'] = self._get_rating("   Rating")
            
            print()
            
            # 3. Title-Image Match (15% weight)
            print("3. TITLE-IMAGE MATCH")
            print("   Does the image accurately represent the title?")
            print("   Consider: Relevance, coherence, expectations")
            ratings['title_match'] = self._get_rating("   Rating")
            
            print()
            
            # 4. Clarity (15% weight)
            print("4. CLARITY")
            print("   Is the main subject clear and easy to understand?")
            print("   Consider: Focus, simplicity, visual hierarchy")
            ratings['clarity'] = self._get_rating("   Rating")
            
            print()
            
            # 5. Emotional Impact (10% weight)
            print("5. EMOTIONAL IMPACT")
            print("   Does it evoke interest, excitement, or curiosity?")
            print("   Consider: Engagement, memorability, emotional response")
            ratings['emotional_impact'] = self._get_rating("   Rating")
            
            # Calculate overall score (weighted average)
            overall_score = (
                ratings['click_worthiness'] * 0.35 +
                ratings['visual_appeal'] * 0.25 +
                ratings['title_match'] * 0.15 +
                ratings['clarity'] * 0.15 +
                ratings['emotional_impact'] * 0.10
            )
            
            # Optional text feedback
            print("\n" + "-" * 70)
            text_feedback = input("6. ADDITIONAL COMMENTS (optional, press Enter to skip): ").strip()
            
            # Summary
            print("\n" + "=" * 70)
            print(f"OVERALL SCORE: {overall_score:.1f}/10")
            print("=" * 70)
            
            print("\nRating breakdown:")
            print(f"  Click-worthiness: {ratings['click_worthiness']}/10 (35% weight)")
            print(f"  Visual Appeal:    {ratings['visual_appeal']}/10 (25% weight)")
            print(f"  Title Match:      {ratings['title_match']}/10 (15% weight)")
            print(f"  Clarity:          {ratings['clarity']}/10 (15% weight)")
            print(f"  Emotional Impact: {ratings['emotional_impact']}/10 (10% weight)")
            
            # Confirm
            confirm = input("\nSave this rating? (y/n): ").lower().strip()
            if confirm != 'y':
                print("Rating discarded")
                return None
            
            # Save feedback
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'image_path': thumbnail['path'],
                'title': thumbnail['title'],
                'prompt': thumbnail.get('prompt', ''),
                'ratings': ratings,
                'overall_score': overall_score,
                'text_feedback': text_feedback
            }
            
            self.feedback_data.append(feedback_entry)
            self._save_feedback()
            
            print("\n✓ Rating saved successfully!")
            
            return feedback_entry
            
        except KeyboardInterrupt:
            print("\n\nRating cancelled")
            return None
    
    def _save_feedback(self):
        """Save feedback to JSON file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
    
    def collect_batch(self, start_idx=0):
        """
        Collect feedback for multiple thumbnails
        
        Args:
            start_idx: Index to start from (for resuming)
        """
        total = len(self.thumbnails)
        
        if total == 0:
            print("No thumbnails found to rate!")
            return
        
        print("\n" + "=" * 70)
        print(f"BATCH RATING MODE")
        print("=" * 70)
        print(f"Total thumbnails: {total}")
        print(f"Already rated: {len(self.feedback_data)}")
        print(f"Starting from: #{start_idx + 1}")
        print(f"\nTips:")
        print("  - Be consistent in your ratings")
        print("  - Take breaks if needed")
        print("  - Press Ctrl+C to stop (progress is auto-saved)")
        print("=" * 70)
        
        for idx in range(start_idx, total):
            thumbnail = self.thumbnails[idx]
            
            print(f"\n\n{'='*70}")
            print(f"THUMBNAIL {idx + 1} of {total}")
            print(f"{'='*70}")
            
            try:
                result = self.collect_feedback(thumbnail)
                
                if result:
                    print(f"\n✓ Progress: {len(self.feedback_data)}/{total} thumbnails rated")
                
                # Ask to continue
                if idx < total - 1:
                    print("\n" + "-" * 70)
                    continue_choice = input("Continue to next thumbnail? (y/n/s=skip): ").lower().strip()
                    
                    if continue_choice == 'n':
                        print("\nStopping batch rating")
                        break
                    elif continue_choice == 's':
                        print("Skipping to next thumbnail")
                        continue
                
            except KeyboardInterrupt:
                print("\n\nBatch rating interrupted")
                break
        
        # Final summary
        self._print_statistics()
    
    def _print_statistics(self):
        """Print statistics on collected feedback"""
        print("\n\n" + "=" * 70)
        print("FEEDBACK COLLECTION SUMMARY")
        print("=" * 70)
        
        if not self.feedback_data:
            print("\nNo feedback collected yet")
            return
        
        import numpy as np
        
        # Overall statistics
        scores = [f['overall_score'] for f in self.feedback_data]
        
        print(f"\nTotal Ratings: {len(self.feedback_data)}")
        print(f"Average Overall Score: {np.mean(scores):.2f}/10")
        print(f"Std Deviation: {np.std(scores):.2f}")
        print(f"Min Score: {np.min(scores):.2f}/10")
        print(f"Max Score: {np.max(scores):.2f}/10")
        
        # Dimension averages
        print(f"\nAverage by Dimension:")
        dimensions = ['click_worthiness', 'visual_appeal', 'title_match', 'clarity', 'emotional_impact']
        dim_names = ['Click-worthiness', 'Visual Appeal', 'Title Match', 'Clarity', 'Emotional Impact']
        
        for dim, name in zip(dimensions, dim_names):
            avg = np.mean([f['ratings'][dim] for f in self.feedback_data])
            print(f"  {name:20s}: {avg:.2f}/10")
        
        # Score distribution
        print(f"\nScore Distribution:")
        excellent = sum(1 for s in scores if s >= 8.0)
        good = sum(1 for s in scores if 6.0 <= s < 8.0)
        average = sum(1 for s in scores if 4.0 <= s < 6.0)
        poor = sum(1 for s in scores if s < 4.0)
        
        print(f"  Excellent (8-10): {excellent} ({excellent/len(scores)*100:.1f}%)")
        print(f"  Good (6-8):       {good} ({good/len(scores)*100:.1f}%)")
        print(f"  Average (4-6):    {average} ({average/len(scores)*100:.1f}%)")
        print(f"  Poor (0-4):       {poor} ({poor/len(scores)*100:.1f}%)")
        
        print(f"\nFeedback saved to: {self.output_file}")
        
        # Recommendation
        if len(self.feedback_data) < 15:
            print(f"\n⚠️  Recommendation: Collect at least 15 ratings for good RLHF results")
            print(f"   You have {len(self.feedback_data)}, need {15 - len(self.feedback_data)} more")
        elif len(self.feedback_data) < 30:
            print(f"\n✓ Good: You have {len(self.feedback_data)} ratings")
            print(f"  Recommendation: 30+ ratings for best results")
        else:
            print(f"\n✅ Excellent: You have {len(self.feedback_data)} ratings!")
            print(f"   This should give good RLHF results")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("STEP 5A: Human Feedback Collection (CLI)")
    print("="*70)
    
    # Check if thumbnails exist
    thumbnails_path = Path(THUMBNAILS_DIR)
    if not thumbnails_path.exists():
        print(f"\n❌ ERROR: Thumbnail directory not found!")
        print(f"Expected: {THUMBNAILS_DIR}/")
        print(f"\nPlease run Step 4 first:")
        print(f"  python step4_generate_base.py")
        return
    
    # Initialize collector
    collector = CLIFeedbackCollector(THUMBNAILS_DIR, FEEDBACK_FILE)
    
    if len(collector.thumbnails) == 0:
        print(f"\n❌ ERROR: No thumbnails found in {THUMBNAILS_DIR}/")
        print(f"\nPlease generate some thumbnails first (Step 4)")
        return
    
    print(f"\n✓ Found {len(collector.thumbnails)} thumbnails to rate")
    
    # Check existing ratings
    if collector.feedback_data:
        print(f"✓ Loaded {len(collector.feedback_data)} existing ratings")
    
    # Menu
    print("\n" + "-" * 70)
    print("OPTIONS:")
    print("-" * 70)
    print("1. Rate all thumbnails (batch mode)")
    print("2. Rate specific thumbnail")
    print("3. Resume from where I left off")
    print("4. View statistics")
    print("5. Exit")
    print("-" * 70)
    
    choice = input("\nChoice (1-5): ").strip()
    
    try:
        if choice == '1':
            # Batch mode
            collector.collect_batch(start_idx=0)
        
        elif choice == '2':
            # Single thumbnail
            print(f"\nAvailable thumbnails:")
            for idx, thumb in enumerate(collector.thumbnails):
                print(f"  {idx + 1}. {thumb['title']}")
            
            idx = int(input(f"\nSelect thumbnail (1-{len(collector.thumbnails)}): ")) - 1
            
            if 0 <= idx < len(collector.thumbnails):
                collector.collect_feedback(collector.thumbnails[idx])
            else:
                print("Invalid selection")
        
        elif choice == '3':
            # Resume
            start_idx = len(collector.feedback_data)
            if start_idx >= len(collector.thumbnails):
                print("\n✓ All thumbnails already rated!")
                collector._print_statistics()
            else:
                print(f"\nResuming from thumbnail #{start_idx + 1}")
                collector.collect_batch(start_idx=start_idx)
        
        elif choice == '4':
            # Statistics
            collector._print_statistics()
        
        else:
            print("\nExiting...")
            return
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final message
    print("\n" + "="*70)
    print("✓ Feedback collection session complete")
    print("="*70)
    
    if len(collector.feedback_data) >= 15:
        print(f"\n✅ Great! You have {len(collector.feedback_data)} ratings")
        print(f"\n➡️  Next step: Run 'python step6_train_reward.py'")
    else:
        print(f"\n⚠️  You have {len(collector.feedback_data)} ratings")
        print(f"   Recommendation: Collect at least 15 for good RLHF results")
        print(f"\n   Run this script again to add more ratings")

if __name__ == "__main__":
    main()
