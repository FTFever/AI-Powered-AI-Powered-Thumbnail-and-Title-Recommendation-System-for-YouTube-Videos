"""
STEP 1: YouTube Data Collection
================================
Collects viral video data including thumbnails, titles, views, and engagement metrics.

TIME REQUIRED: 2-3 hours for 300 videos
COST: Free (uses YouTube API free tier)
OUTPUT: youtube_data/ folder with thumbnails and metadata

INSTRUCTIONS:
1. Get YouTube API key from console.cloud.google.com
2. Replace YOUR_API_KEY_HERE below with your actual key
3. Run: python step1_collect_data.py
4. Wait for completion (progress shown in terminal)
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime
import time
import pandas as pd

# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

# ⚠️ REQUIRED: Replace with your YouTube API key


BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR/"YOUTUBE_API_KEY.txt",header = None)
YOUTUBE_API_KEY = df.iloc[0,1]
print(f"API Key: {api_key}")


# Categories to scrape (you can add/remove)
CATEGORIES = [
    "food recipe"
]

# 'trending tech',
#     'viral gaming', 
#     'comedy shorts',
#     'how to tutorial',
#     'product review',
#     'food recipe',
#     'fitness workout',
#     'travel vlog',
#     'tech unboxing',
#     "pets & animals",
#     "people & blogs",
#     "videoblogging"
# Videos to collect per category
VIDEOS_PER_CATEGORY = 1  # Adjust based on needs (max 50 per request)

# Output directory
OUTPUT_DIR = Base_DIR/"Data"

# =============================================================================
# MAIN CODE - NO NEED TO EDIT BELOW
# =============================================================================

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"

class YouTubeDataCollector:
    """Handles YouTube API interactions and data collection"""
    
    def __init__(self, api_key, output_dir=OUTPUT_DIR):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "thumbnails").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        print(f"✓ Output directory: {self.output_dir.absolute()}")
    
    def search_viral_videos(self, query="", max_results=50, order="viewCount"):
        """
        Search for videos using YouTube API
        
        Args:
            query: Search term/category
            max_results: Number of results (max 50)
            order: 'viewCount', 'rating', 'relevance', 'date'
        
        Returns:
            JSON response with video data
        """
        url = f"{YOUTUBE_API_URL}/search"
        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'order': order,
            'maxResults': min(max_results, 50),
            'key': self.api_key,
            'videoDuration': 'medium',
            'relevanceLanguage': 'en'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  ✗ API Error: {e}")
            return None
    
    def get_video_statistics(self, video_ids):
        """
        Get detailed statistics for video IDs
        
        Args:
            video_ids: List of YouTube video IDs
        
        Returns:
            JSON with detailed video statistics
        """
        url = f"{YOUTUBE_API_URL}/videos"
        params = {
            'part': 'statistics,snippet,contentDetails',
            'id': ','.join(video_ids),
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Stats Error: {e}")
            return None
    
    def download_thumbnail(self, video_id, thumbnail_url):
        """
        Download thumbnail image
        
        Args:
            video_id: YouTube video ID
            thumbnail_url: URL of thumbnail image
        
        Returns:
            Local file path if successful, None otherwise
        """
        try:
            response = requests.get(thumbnail_url, timeout=10)
            response.raise_for_status()
            
            filepath = self.output_dir / "thumbnails" / f"{video_id}.jpg"
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return str(filepath)
        except Exception as e:
            print(f"  ✗ Download failed for {video_id}: {e}")
            return None
    
    def collect_data(self, categories=None, videos_per_category=30):
        """
        Main data collection function
        
        Args:
            categories: List of search terms/categories
            videos_per_category: Number of videos per category
        
        Returns:
            List of collected video data
        """
        if categories is None:
            categories = CATEGORIES
        
        all_data = []
        total_downloaded = 0
        
        print(f"\n{'='*70}")
        print(f"Starting data collection for {len(categories)} categories")
        print(f"Target: {videos_per_category} videos per category")
        print(f"{'='*70}\n")
        
        for cat_idx, category in enumerate(categories, 1):
            print(f"[{cat_idx}/{len(categories)}] Category: '{category}'")
            
            # Search for videos
            search_results = self.search_viral_videos(
                query=category, 
                max_results=videos_per_category
            )
            
            if not search_results or 'items' not in search_results:
                print(f"  ✗ No results found")
                continue
            
            # Extract video IDs
            video_ids = [item['id']['videoId'] for item in search_results['items']]
            print(f"  Found {len(video_ids)} videos")
            
            # Get detailed statistics
            video_stats = self.get_video_statistics(video_ids)
            
            if not video_stats or 'items' not in video_stats:
                print(f"  ✗ Could not fetch statistics")
                continue
            
            # Process each video
            category_count = 0
            for video in video_stats['items']:
                video_id = video['id']
                snippet = video['snippet']
                stats = video['statistics']
                
                # Get best quality thumbnail
                thumbnails = snippet.get('thumbnails', {})
                thumbnail_url = (
                    thumbnails.get('maxres', {}).get('url') or
                    thumbnails.get('high', {}).get('url') or
                    thumbnails.get('medium', {}).get('url') or
                    thumbnails.get('default', {}).get('url')
                )
                
                if not thumbnail_url:
                    continue
                
                # Download thumbnail
                local_path = self.download_thumbnail(video_id, thumbnail_url)
                
                if local_path:
                    # Compile metadata
                    data_entry = {
                        'video_id': video_id,
                        'title': snippet.get('title', ''),
                        'description': snippet.get('description', '')[:500],  # Truncate
                        'channel_title': snippet.get('channelTitle', ''),
                        'published_at': snippet.get('publishedAt', ''),
                        'category': category,
                        'view_count': int(stats.get('viewCount', 0)),
                        "subscriber_count": int(stats.get('subscriberCount',0)),
                        'like_count': int(stats.get('likeCount', 0)),
                        'comment_count': int(stats.get('commentCount', 0)),
                        'thumbnail_url': thumbnail_url,
                        'local_thumbnail_path': local_path,
                        'collected_at': datetime.now().isoformat(),
                        # "ratio": int(stats.get('viewCount', 0))/int(stats.get('subscriberCount',0)),
                        
                    }
                    
                    all_data.append(data_entry)
                    category_count += 1
                    total_downloaded += 1
            
            print(f"  ✓ Downloaded {category_count} thumbnails")
            
            # Rate limiting (be nice to YouTube API)
            time.sleep(1)
        
        # Save metadata to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_file = self.output_dir / "metadata" / f"dataset_{timestamp}.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        

        # Print summary
        print(f"\n{'='*70}")
        print(f"✅ DATA COLLECTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total videos collected: {len(all_data)}")
        print(f"Total thumbnails downloaded: {total_downloaded}")
        print(f"Metadata saved to: {metadata_file}")
        print(f"Thumbnails saved to: {self.output_dir / 'thumbnails'}")
        print(f"\n📊 Quick Stats:")
        
        if all_data:
            views = [d['view_count'] for d in all_data]
            print(f"  Average views: {sum(views)//len(views):,}")
            print(f"  Max views: {max(views):,}")
            print(f"  Min views: {min(views):,}")
        
        print(f"\n➡️  Next step: Run 'python step2_label_data.py'")
        
        return all_data

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("STEP 1: YouTube Viral Thumbnail Data Collection")
    print("="*70)
    
    # Validate API key
    if YOUTUBE_API_KEY == "YOUR_API_KEY_HERE":
        print("\n❌ ERROR: YouTube API key not configured!")
        print("\nPlease:")
        print("1. Get API key from: https://console.cloud.google.com/")
        print("2. Enable YouTube Data API v3")
        print("3. Edit this file and replace YOUR_API_KEY_HERE with your key")
        print("4. Run again")
        return
    
    # Initialize collector
    try:
        collector = YouTubeDataCollector(api_key=YOUTUBE_API_KEY)
    except Exception as e:
        print(f"\n❌ Setup Error: {e}")
        return
    
    # Collect data
    try:
        data = collector.collect_data(
            categories=CATEGORIES,
            videos_per_category=VIDEOS_PER_CATEGORY
        )
        
        if len(data) == 0:
            print("\n⚠️  Warning: No data collected. Check your API key and internet connection.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Collection Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
