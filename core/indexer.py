"""
indexer.py
==========
Quét thư mục dataset/, trích xuất đặc trưng từng ảnh cá,
lưu metadata vào SQLite và vector vào file pickle.

Dataset structure (flat folder):
  dataset/
    acanthopagrus_berda_acanthopagrus_berda_1_jpg.rf.xxx.jpg
    acanthopagrus_berda_acanthopagrus_berda_2_jpg.rf.xxx.jpg
    ...
Tên loài được suy ra từ tên file: phần trước "_jpg" đầu tiên, bỏ nửa sau trùng lặp.
"""

import os
import sys
import pickle
import time

# Thêm thư mục gốc vào sys.path để import core/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.db_connection import get_connection, init_db
from core.feature_extractor import get_image_features

DATASET_DIR  = os.path.join(BASE_DIR, 'dataset')
DB_DIR       = os.path.join(BASE_DIR, 'database')
FEATURE_FILE = os.path.join(DB_DIR, 'features.pkl')


def parse_species(filename: str) -> str:
    """
    Suy ra tên loài từ tên file.
    Ví dụ: 'acanthopagrus_berda_acanthopagrus_berda_1_jpg.rf.xxx.jpg'
    → 'acanthopagrus_berda'
    Logic: lấy các token trước '_jpg', bỏ nửa sau lặp lại.
    """
    stem = os.path.splitext(filename)[0]          # bỏ .jpg cuối
    # bỏ hash: ...rf.xxxx
    if '.rf.' in stem:
        stem = stem[:stem.index('.rf.')]
    # bỏ số thứ tự cuối và '_jpg'
    parts = stem.split('_')
    # tìm vị trí 'jpg' hoặc số cuối
    end = len(parts)
    for i, p in enumerate(parts):
        if p == 'jpg' or (p.isdigit()):
            end = i
            break
    # parts[:end] = ['acanthopagrus','berda','acanthopagrus','berda']
    half = end // 2
    return '_'.join(parts[:half]) if half > 0 else '_'.join(parts[:end])


def build_index():
    """
    Pipeline đánh chỉ mục:
    1. Khởi tạo CSDL
    2. Quét dataset/
    3. Trích xuất đặc trưng
    4. Lưu metadata → SQLite, vector → pickle
    """
    os.makedirs(DB_DIR, exist_ok=True)
    init_db()
    conn = get_connection()
    if not conn:
        print("[ERR] Khong the ket noi CSDL.")
        return

    cursor = conn.cursor()
    # Xóa dữ liệu cũ để chạy lại không bị lỗi PK
    cursor.execute("DELETE FROM Fish_Metadata")
    conn.commit()

    features_dict = {}
    total_ok  = 0
    total_err = 0
    start = time.time()

    # Lấy danh sách ảnh trong dataset/ (flat folder)
    image_files = [
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    print(f"Tim thay {len(image_files)} anh. Bat dau trich xuat dac trung...")

    for idx, img_file in enumerate(image_files, 1):
        file_path     = os.path.join(DATASET_DIR, img_file)
        image_id      = os.path.splitext(img_file)[0]   # bỏ đuôi .jpg
        species_label = parse_species(img_file)

        vector = get_image_features(file_path)
        if vector is not None:
            features_dict[image_id] = vector
            cursor.execute("""
                INSERT OR REPLACE INTO Fish_Metadata (Image_ID, Species_Label, File_Path)
                VALUES (?, ?, ?)
            """, (image_id, species_label, file_path))
            total_ok += 1
        else:
            total_err += 1

        if idx % 100 == 0:
            conn.commit()
            elapsed = time.time() - start
            print(f"  [{idx}/{len(image_files)}] ok={total_ok}  err={total_err}  t={elapsed:.1f}s")

    conn.commit()
    cursor.close()
    conn.close()

    # Lưu vector ra file pickle
    with open(FEATURE_FILE, 'wb') as f:
        pickle.dump(features_dict, f)

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"[OK] Hoan thanh! {total_ok} anh thanh cong, {total_err} loi.")
    print(f"Thoi gian: {elapsed:.1f}s")
    print(f"Vector luu tai: {FEATURE_FILE}")
    print(f"CSDL tai:      database/fish_cbir.db")


if __name__ == "__main__":
    build_index()
