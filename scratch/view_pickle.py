import os
import pickle
import numpy as np
import sys

# Set encoding for Windows console if needed
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, 'database')
FEATURE_FILE = os.path.join(DB_DIR, 'features.pkl')
IVF_FILE = os.path.join(DB_DIR, 'ivf_index.pkl')

def view_features():
    print("=== READING FEATURES.PKL ===")
    if not os.path.exists(FEATURE_FILE):
        print(f"File not found: {FEATURE_FILE}")
        return

    with open(FEATURE_FILE, 'rb') as f:
        features = pickle.load(f)

    print(f"Total images in features file: {len(features)}")
    
    keys = list(features.keys())
    print("\nExample 3 Image_IDs:")
    for key in keys[:3]:
        vector = features[key]
        print(f"- Image ID: '{key}'")
        print(f"  + Vector length (dimensions): {len(vector)}")
        print(f"  + Data type: {type(vector)}")
        print(f"  + First 10 values of the vector: {vector[:10]}...")
        print("-" * 50)

def view_ivf_index():
    print("\n=== READING IVF_INDEX.PKL (INVERTED INDEX) ===")
    if not os.path.exists(IVF_FILE):
        print(f"File not found: {IVF_FILE}")
        return

    with open(IVF_FILE, 'rb') as f:
        ivf = pickle.load(f)

    print(f"Total clusters: {len(ivf)}")
    for cluster_id, img_ids in list(ivf.items())[:3]:
        print(f"- Cluster {cluster_id}:")
        print(f"  + Number of images in cluster: {len(img_ids)}")
        print(f"  + First 5 images: {img_ids[:5]}...")
        print("-" * 50)

if __name__ == "__main__":
    view_features()
    view_ivf_index()
