"""
QuickDraw Dataset Downloader
Downloads .npy files from Google QuickDraw dataset for specified classes
Author: nughnguyen
"""

import os
import urllib.request
import sys
from src.config import CLASSES

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
DATA_DIR = "data"

def download_class_data(class_name, data_dir=DATA_DIR):
    """Download .npy file for a specific class"""
    # Replace spaces with %20 for URL
    url_class_name = class_name.replace(" ", "%20")
    filename = f"full_numpy_bitmap_{class_name}.npy"
    filepath = os.path.join(data_dir, filename)
    
    # Skip if already exists
    if os.path.exists(filepath):
        print(f"✓ {class_name} already exists, skipping...")
        return True
    
    url = f"{BASE_URL}{url_class_name}.npy"
    
    try:
        print(f"Downloading {class_name}...", end=" ")
        
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading {class_name}... {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\r✓ {class_name} downloaded successfully!")
        return True
    except Exception as e:
        print(f"\r✗ Failed to download {class_name}: {str(e)}")
        return False

def main():
    """Download all classes defined in config.py"""
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Starting download of {len(CLASSES)} classes...")
    print(f"Downloading to: {os.path.abspath(DATA_DIR)}\n")
    
    success_count = 0
    failed_classes = []
    
    for i, class_name in enumerate(CLASSES, 1):
        print(f"[{i}/{len(CLASSES)}] ", end="")
        if download_class_data(class_name):
            success_count += 1
        else:
            failed_classes.append(class_name)
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Successfully downloaded: {success_count}/{len(CLASSES)} classes")
    
    if failed_classes:
        print(f"\nFailed to download {len(failed_classes)} classes:")
        for class_name in failed_classes:
            print(f"  - {class_name}")
    else:
        print("\n✓ All classes downloaded successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
