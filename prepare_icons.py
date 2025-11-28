"""
Helper script to copy category icons from all_images to images folder
Author: nughnguyen
"""

import os
import shutil
from src.config import CLASSES

ALL_IMAGES_DIR = "all_images"
IMAGES_DIR = "images"

def copy_icons():
    """Copy icon images for all classes"""
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    
    copied = 0
    missing = []
    
    for class_name in CLASSES:
        source_file = os.path.join(ALL_IMAGES_DIR, f"{class_name}.png")
        dest_file = os.path.join(IMAGES_DIR, f"{class_name}.png")
        
        # Skip if already exists
        if os.path.exists(dest_file):
            print(f"✓ {class_name} icon already exists")
            continue
        
        # Copy if source exists
        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            print(f"✓ Copied {class_name} icon")
            copied += 1
        else:
            print(f"✗ Missing icon for {class_name}")
            missing.append(class_name)
    
    print(f"\n{'='*60}")
    print(f"Icon preparation complete!")
    print(f"Copied: {copied} icons")
    print(f"Already existed: {len(CLASSES) - copied - len(missing)} icons")
    
    if missing:
        print(f"\nMissing icons for {len(missing)} categories:")
        for class_name in missing:
            print(f"  - {class_name}")
        print("\nNote: GUI will work without these icons, but they won't be displayed.")
    else:
        print("\n✓ All category icons are ready!")
    print(f"{'='*60}")

if __name__ == "__main__":
    copy_icons()
