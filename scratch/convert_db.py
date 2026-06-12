import os
import sqlite3
import pickle
import sys

# Set encoding for Windows console if needed
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, 'database', 'fish_cbir.db')
FEATURE_FILE = os.path.join(BASE_DIR, 'database', 'features.pkl')

def migrate_database():
    print("=== STARTING DATABASE MIGRATION ===")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        return

    # 1. Connect and read existing metadata
    print("Reading old metadata from SQLite...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT Image_ID, Species_Label, File_Path FROM Fish_Metadata")
        rows = cursor.fetchall()
        print(f"Read {len(rows)} records from Fish_Metadata.")
    except sqlite3.Error as e:
        print(f"Error reading from old database: {e}")
        conn.close()
        return

    # 2. Re-create the database table with the new schema (ID AUTOINCREMENT, no Width/Height)
    print("Re-creating Fish_Metadata table with new schema...")
    cursor.execute("DROP TABLE IF EXISTS Fish_Metadata")
    cursor.execute("""
        CREATE TABLE Fish_Metadata (
            ID            INTEGER PRIMARY KEY AUTOINCREMENT,
            Image_ID      TEXT NOT NULL UNIQUE,
            Species_Label TEXT NOT NULL,
            File_Path     TEXT NOT NULL
        )
    """)
    conn.commit()

    # 3. Re-insert the data and map old Image_ID to new auto-increment integer ID
    print("Migrating records and generating new integer IDs...")
    old_id_to_new_id = {}
    for image_id, species_label, file_path in rows:
        cursor.execute("""
            INSERT INTO Fish_Metadata (Image_ID, Species_Label, File_Path)
            VALUES (?, ?, ?)
        """, (image_id, species_label, file_path))
        
        new_id = cursor.lastrowid
        old_id_to_new_id[image_id] = new_id

    conn.commit()
    conn.close()
    print("SQLite metadata migration completed.")

    # 4. Migrate features.pkl keys from Image_ID (string) to ID (integer)
    if os.path.exists(FEATURE_FILE):
        print("Migrating features.pkl to use integer IDs...")
        with open(FEATURE_FILE, 'rb') as f:
            old_features = pickle.load(f)
        
        new_features = {}
        missing_count = 0
        for old_key, vector in old_features.items():
            if old_key in old_id_to_new_id:
                new_key = old_id_to_new_id[old_key]
                new_features[new_key] = vector
            else:
                missing_count += 1

        print(f"Mapped {len(new_features)} vectors. Missing matches: {missing_count}")
        
        with open(FEATURE_FILE, 'wb') as f:
            pickle.dump(new_features, f)
        print("features.pkl migration completed.")
    else:
        print("Warning: features.pkl not found, skipping feature conversion.")

    # 5. Re-run clustering and regenerate IVF index
    print("Re-building K-Means clusters and Inverted File Index (IVF)...")
    try:
        from core.cluster_indexer import build_clusters
        build_clusters()
        print("K-Means and IVF index successfully regenerated with new integer IDs.")
    except Exception as e:
        print(f"Error regenerating clusters: {e}")

    print("\n=== MIGRATION COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    migrate_database()
