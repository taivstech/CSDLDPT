"""
searcher.py
===========
Tim kiem anh ca tuong dong su dung K-NN tren khong gian vector.

Chien luoc:
  - Neu co IVF index (da chay cluster_indexer.py):
      Tim k cum gan nhat (nprobe cum) -> quet chi anh trong cum do -> KNN
  - Neu khong co IVF (chi co features.pkl):
      KNN brute-force toan bo CSDL (chinh xac 100%)

Ho tro 3 do do tuong dong:
  1. cosine     — Cosine Similarity (tuong dong huong vector)
  2. euclidean  — Khoang cach Euclid
  3. histogram  — Histogram Intersection (phu hop voi histogram mau)
"""

import os
import sys
import pickle
import numpy as np

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR        = os.path.join(BASE_DIR, 'database')
FEATURE_FILE  = os.path.join(DB_DIR, 'features.pkl')
KMEANS_FILE   = os.path.join(DB_DIR, 'kmeans_model.pkl')
IVF_FILE      = os.path.join(DB_DIR, 'ivf_index.pkl')

sys.path.insert(0, BASE_DIR)
from core.db_connection import get_connection


class FishSearcher:
    """
    He thong tim kiem anh ca theo phuong phap CBIR.

    Attributes:
        features_db  : {image_id: vector_np} — toan bo CSDL vector
        kmeans       : KMeans model (None neu chua co)
        ivf_index    : {cluster_id: [image_id, ...]} (None neu chua co)
        use_ivf      : bool — co dung IVF khong
    """

    def __init__(self, feature_path: str = FEATURE_FILE):
        self.feature_path = feature_path
        self.features_db  = self._load_features()
        self.kmeans, self.ivf_index = self._load_ivf()
        self.use_ivf = (self.kmeans is not None and self.ivf_index is not None)
        if self.use_ivf:
            print(f"[Searcher] Dung IVF Index ({len(self.ivf_index)} cum).")
        else:
            print(f"[Searcher] Dung KNN Brute-force ({len(self.features_db)} anh).")

    # ──────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────

    def _load_features(self) -> dict:
        if not os.path.exists(self.feature_path):
            print(f"[WARN] Khong tim thay {self.feature_path}. Hay chay indexer.py truoc.")
            return {}
        print("Dang tai CSDL vector len bo nho...")
        with open(self.feature_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Da tai {len(data)} vector.")
        return data

    def _load_ivf(self):
        """Tai KMeans model va IVF index neu ton tai."""
        kmeans = ivf = None
        if os.path.exists(KMEANS_FILE) and os.path.exists(IVF_FILE):
            try:
                with open(KMEANS_FILE, 'rb') as f:
                    kmeans = pickle.load(f)
                with open(IVF_FILE, 'rb') as f:
                    ivf = pickle.load(f)
                print(f"Da tai KMeans model + IVF index.")
            except Exception as e:
                print(f"[WARN] Loi tai IVF: {e}. Su dung brute-force.")
                kmeans = ivf = None
        return kmeans, ivf

    # ──────────────────────────────────────────────────
    # Do do khoang cach
    # ──────────────────────────────────────────────────

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """1 - cosine_similarity (nho hon = giong hon)."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-7
        return float(1.0 - np.dot(a, b) / denom)

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
        """Histogram Intersection: 1 - sum(min(a,b)) (nho hon = giong hon)."""
        return float(1.0 - np.sum(np.minimum(a, b)))

    def calculate_distance(self, q: np.ndarray, d: np.ndarray, metric: str = 'cosine') -> float:
        if metric == 'cosine':
            return self._cosine_distance(q, d)
        elif metric == 'euclidean':
            return self._euclidean_distance(q, d)
        elif metric == 'histogram':
            return self._histogram_intersection(q, d)
        else:
            raise ValueError(f"metric khong hop le: '{metric}'. Dung 'cosine'/'euclidean'/'histogram'.")

    # ──────────────────────────────────────────────────
    # Tim kiem
    # ──────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, k: int = 5,
               metric: str = 'cosine', nprobe: int = 3) -> list:
        """
        Tim k anh giong nhat voi query_vector.

        Args:
            query_vector : vector dac trung cua anh query
            k            : so ket qua tra ve
            metric       : 'cosine' | 'euclidean' | 'histogram'
            nprobe       : so cum IVF can quet (chi dung khi use_ivf=True)

        Returns:
            List[dict]: [{image_id, distance, similarity, species, file_path}, ...]
        """
        if not self.features_db:
            return []

        if self.use_ivf:
            candidate_ids = self._ivf_candidates(query_vector, nprobe)
        else:
            candidate_ids = list(self.features_db.keys())

        # Tinh khoang cach voi tat ca ung vien
        distances = [
            (img_id, self.calculate_distance(query_vector, self.features_db[img_id], metric))
            for img_id in candidate_ids
            if img_id in self.features_db
        ]
        distances.sort(key=lambda x: x[1])
        top_k = distances[:k]
        return self._fetch_metadata(top_k, metric)

    def _ivf_candidates(self, query_vector: np.ndarray, nprobe: int) -> list:
        """
        Tim nprobe cum gan nhat voi query, tra ve danh sach image_id trong cac cum do.
        Giai thich IVF:
          1. Bieu dien query thanh vector
          2. Tim nprobe cum K-Means gan nhat (khoang cach Euclid den centroid)
          3. Chi quet anh trong nprobe cum do
          → Giam khoi luong tinh toan, tang toc tim kiem
        """
        qv = query_vector.reshape(1, -1).astype(np.float32)
        distances_to_centroids = np.linalg.norm(
            self.kmeans.cluster_centers_ - qv, axis=1
        )
        top_clusters = np.argsort(distances_to_centroids)[:nprobe]

        candidates = []
        for cid in top_clusters:
            candidates.extend(self.ivf_index.get(int(cid), []))
        return candidates

    def _fetch_metadata(self, top_k: list, metric: str = 'cosine') -> list:
        """Tra cuu metadata tu SQLite theo ID (so nguyen)."""
        conn = get_connection()
        if not conn:
            return [
                {
                    'id': iid, 'image_id': str(iid), 'distance': d,
                    'similarity': round(max(0.0, (1.0 - d) * 100), 2),
                    'species': 'N/A', 'file_path': ''
                }
                for iid, d in top_k
            ]

        top_ids      = [item[0] for item in top_k]
        placeholders = ','.join(['?'] * len(top_ids))
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT ID, Image_ID, Species_Label, File_Path FROM Fish_Metadata "
            f"WHERE ID IN ({placeholders})",
            top_ids
        )
        rows     = cursor.fetchall()
        db_data  = {row[0]: {'image_id': row[1], 'species': row[2], 'path': row[3]} for row in rows}
        cursor.close()
        conn.close()

        results = []
        for db_id, dist in top_k:
            info = db_data.get(db_id, {'image_id': 'Unknown', 'species': 'Unknown', 'path': ''})
            # Tuong dong: cosine -> (1-dist)*100; euclidean/histogram: max(0, 100-dist*50)
            if metric == 'cosine':
                similarity = max(0.0, (1.0 - dist) * 100)
            else:
                similarity = max(0.0, 100.0 - dist * 50.0)
            results.append({
                'id':         db_id,
                'image_id':   info['image_id'],
                'distance':   round(dist, 6),
                'similarity': round(similarity, 2),
                'species':    info['species'],
                'file_path':  info['path'],
            })
        return results

    # ──────────────────────────────────────────────────
    # Tien ich
    # ──────────────────────────────────────────────────

    @property
    def total_images(self) -> int:
        return len(self.features_db)

    @property
    def index_mode(self) -> str:
        if self.use_ivf:
            k = len(self.ivf_index)
            return f"IVF ({k} cum K-Means)"
        return "KNN Brute-force"


# ──────────────────────────────────────────────────
# Test nhanh
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    import glob
    from core.feature_extractor import get_image_features

    searcher     = FishSearcher()
    dataset_dir  = os.path.join(BASE_DIR, 'dataset')
    samples      = glob.glob(os.path.join(dataset_dir, '*.jpg'))[:1]

    if samples:
        print(f"\nTruy van anh: {os.path.basename(samples[0])}")
        qvec = get_image_features(samples[0])
        if qvec is not None:
            results = searcher.search(qvec, k=5, metric='cosine')
            print(f"\nTop 5 ket qua (mode={searcher.index_mode}):")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['similarity']:.1f}%] {r['species']:30s} {r['file_path']}")
    else:
        print("Khong tim thay anh trong dataset/")
