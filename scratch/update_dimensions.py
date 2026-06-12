import os
import sqlite3
import numpy as np
import cv2
import sys

# Set encoding for Windows console if needed
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'database', 'fish_cbir.db')

def update_dimensions():
    print("Connecting to database...")
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get records that need update
    cursor.execute("SELECT Image_ID, File_Path FROM Fish_Metadata WHERE Width IS NULL OR Height IS NULL")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"Found {total} records with missing Width/Height.")

    if total == 0:
        print("All records already have dimensions populated!")
        conn.close()
        return

    updated_count = 0
    for idx, (img_id, file_path) in enumerate(rows, 1):
        try:
            if not os.path.exists(file_path):
                # Fallback to local path relative to project if needed
                file_path = os.path.join(BASE_DIR, 'dataset', img_id + '.jpg')
                if not os.path.exists(file_path):
                    continue

            # Read image using numpy to safely handle Unicode path
            img_array = np.fromfile(file_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                h, w = img.shape[:2]
                cursor.execute(
                    "UPDATE Fish_Metadata SET Width = ?, Height = ? WHERE Image_ID = ?",
                    (w, h, img_id)
                )
                updated_count += 1
            else:
                print(f"[WARN] Failed to decode image: {file_path}")
        except Exception as e:
            print(f"[ERR] Error processing {img_id}: {e}")

        if idx % 100 == 0 or idx == total:
            conn.commit()
            print(f"Processed {idx}/{total} - Updated {updated_count} records.")

    conn.close()
    print("Database update completed successfully!")

if __name__ == "__main__":
    update_dimensions()
