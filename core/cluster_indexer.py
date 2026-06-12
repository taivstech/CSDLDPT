"""
cluster_indexer.py
==================
Đọc vector từ features.pkl, chạy K-Means gom cụm,
xây dựng Inverted File Index (IVF) để tìm kiếm nhanh hơn.

Quy trình:
  features.pkl → K-Means(k cụm) → IVF Index
  IVF: {cluster_id: [image_id_1, image_id_2, ...]}

Chạy SAU khi đã chạy indexer.py.
"""

import os
import sys
import pickle
import time
import numpy as np
from sklearn.cluster import KMeans

BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR            = os.path.join(BASE_DIR, 'database')
FEATURE_FILE      = os.path.join(DB_DIR, 'features.pkl')
KMEANS_MODEL_FILE = os.path.join(DB_DIR, 'kmeans_model.pkl')
IVF_INDEX_FILE    = os.path.join(DB_DIR, 'ivf_index.pkl')


def build_clusters(k: int = None):
    """
    1. Đọc features.pkl
    2. Huấn luyện K-Means (k cụm, mặc định = sqrt(N/2))
    3. Xây dựng IVF: {cluster_id: [image_id, ...]}
    4. Lưu model + IVF ra file pickle

    Args:
        k: số cụm (None = tự động tính sqrt(N/2))
    """
    if not os.path.exists(FEATURE_FILE):
        print(f"Khong tim thay {FEATURE_FILE}. Hay chay indexer.py truoc.")
        return None, None

    print("Dang tai vector tu features.pkl...")
    with open(FEATURE_FILE, 'rb') as f:
        features_dict = pickle.load(f)

    if not features_dict:
        print("Du lieu vector trong.")
        return None, None

    list_ids     = list(features_dict.keys())
    list_vectors = list(features_dict.values())
    X = np.array(list_vectors, dtype=np.float32)
    n_samples = X.shape[0]
    print(f"Ma tran du lieu: {n_samples} anh x {X.shape[1]} chieu.")

    # Tu dong chon k: sqrt(N/2), toi thieu 5, toi da 50
    if k is None:
        k = max(5, min(50, int(np.sqrt(n_samples / 2))))

    # Dam bao k <= so anh
    k = min(k, n_samples)
    print(f"So cum K = {k}  (tu dong tu N={n_samples})")

    # Huan luyen K-Means
    start = time.time()
    print(f"Huan luyen K-Means (k={k}, max_iter=300)...")
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(X)

    # Xay dung IVF: cluster_id -> danh sach image_id
    ivf_index = {i: [] for i in range(k)}
    for img_id, cid in zip(list_ids, kmeans.labels_):
        ivf_index[int(cid)].append(img_id)

    # Thong ke phan bo
    cluster_sizes = [len(v) for v in ivf_index.values()]
    inertia  = kmeans.inertia_
    elapsed  = time.time() - start
    max_size = max(cluster_sizes)

    print(f"\nPhan bo anh trong {k} cum:")
    print(f"  Min: {min(cluster_sizes)}  Max: {max_size}  "
          f"Avg: {np.mean(cluster_sizes):.1f}  Inertia: {inertia:.2f}")
    print("\n  Cum  | So anh | Bar")
    print("  " + "-"*44)
    for cid in sorted(ivf_index.keys()):
        sz  = len(ivf_index[cid])
        bar = 'X' * max(1, int(sz / max(1, max_size) * 20))
        print(f"  {cid:>3}  | {sz:>6} | {bar}")

    # Luu model va IVF
    with open(KMEANS_MODEL_FILE, 'wb') as f:
        pickle.dump(kmeans, f)
    with open(IVF_INDEX_FILE, 'wb') as f:
        pickle.dump(ivf_index, f)

    print(f"\n== Hoan thanh! Thoi gian: {elapsed:.1f}s")
    print(f"   K-Means model : {KMEANS_MODEL_FILE}")
    print(f"   IVF Index     : {IVF_INDEX_FILE}")
    return kmeans, ivf_index


if __name__ == "__main__":
    build_clusters()
