"""
evaluate_cbir.py
================
Danh gia he thong CBIR anh ca bang cac metric chuan:
  - Top-1 Accuracy
  - Precision@1, Precision@3, Precision@5
  - MAP (Mean Average Precision)

Cach chay:
  python evaluate_cbir.py [--samples 200] [--metric cosine]
"""

import sys
import os
import pickle
import numpy as np
import sqlite3
import random
import argparse
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.searcher import FishSearcher

DB_PATH      = os.path.join(BASE_DIR, 'database', 'fish_cbir.db')
FEATURE_FILE = os.path.join(BASE_DIR, 'database', 'features.pkl')


# ──────────────────────────────────────────────────────────────
# 1. Load du lieu
# ──────────────────────────────────────────────────────────────
def load_data():
    print("[1/4] Dang tai du lieu...")
    with open(FEATURE_FILE, 'rb') as f:
        features_dict = pickle.load(f)

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT Image_ID, Species_Label FROM Fish_Metadata")
    rows = c.fetchall()
    conn.close()

    id2species = {row[0]: row[1] for row in rows}
    print(f"      -> {len(features_dict)} vectors, {len(set(id2species.values()))} loai.")
    return features_dict, id2species


# ──────────────────────────────────────────────────────────────
# 2. Tim kiem voi IVF / brute-force
# ──────────────────────────────────────────────────────────────
def search_topk(searcher, query_vec, k=5, metric='cosine'):
    """Tra ve [(image_id, score), ...] khong ke chinh no."""
    nprobe = 5 if searcher.use_ivf else 0
    results = searcher.search(query_vec, k=k+1, metric=metric, nprobe=nprobe)
    return results  # list of dict: {id, species, score, path}


# ──────────────────────────────────────────────────────────────
# 3. Tinh AP cho 1 query
# ──────────────────────────────────────────────────────────────
def average_precision(retrieved_species, query_species, k):
    """
    Tinh Average Precision tai top-K.
    Precision duoc tinh tai moi vi tri co anh dung loai.
    """
    hits        = 0
    sum_prec    = 0.0
    for i, sp in enumerate(retrieved_species[:k], start=1):
        if sp == query_species:
            hits      += 1
            sum_prec  += hits / i
    # So anh cung loai trong DB (tru chinh no) de tinh recall
    return sum_prec / max(1, hits) if hits > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# 4. Vong lap danh gia
# ──────────────────────────────────────────────────────────────
def evaluate(n_samples=200, metric='cosine', seed=42, k_max=5):
    features_dict, id2species = load_data()

    print(f"[2/4] Khoi tao FishSearcher...")
    searcher = FishSearcher()

    # Lay mau ngau nhien (dam bao moi loai deu co mat)
    all_ids = list(features_dict.keys())
    random.seed(seed)
    sample_ids = random.sample(all_ids, min(n_samples, len(all_ids)))
    print(f"[3/4] Danh gia tren {len(sample_ids)} anh query (metric={metric})...\n")

    # Metrics
    results_by_k = {k: [] for k in [1, 3, k_max]}
    ap_list      = []
    top1_correct = 0
    start        = time.time()

    for i, qid in enumerate(sample_ids, 1):
        q_species = id2species.get(qid, '')
        q_vec     = features_dict[qid]

        top_results = search_topk(searcher, q_vec, k=k_max, metric=metric)

        # Loc bo chinh no neu co
        top_results = [r for r in top_results if r['image_id'] != qid][:k_max]
        retrieved_species = [r['species'] for r in top_results]

        # Top-1 accuracy
        if retrieved_species and retrieved_species[0] == q_species:
            top1_correct += 1

        # Precision@K
        for k in [1, 3, k_max]:
            correct_in_k = sum(1 for sp in retrieved_species[:k] if sp == q_species)
            results_by_k[k].append(correct_in_k / k)

        # AP
        ap = average_precision(retrieved_species, q_species, k_max)
        ap_list.append(ap)

        # Progress bar
        if i % 50 == 0 or i == len(sample_ids):
            elapsed = time.time() - start
            print(f"  [{i:>4}/{len(sample_ids)}]  "
                  f"Acc@1={top1_correct/i:.1%}  "
                  f"P@5={np.mean(results_by_k[k_max]):.1%}  "
                  f"MAP={np.mean(ap_list):.1%}  "
                  f"t={elapsed:.1f}s")

    # ──────────────────────────────────────────────────────────────
    # 5. In ket qua tong hop
    # ──────────────────────────────────────────────────────────────
    n = len(sample_ids)
    top1_acc = top1_correct / n
    p1       = np.mean(results_by_k[1])
    p3       = np.mean(results_by_k[3])
    p5       = np.mean(results_by_k[k_max])
    MAP      = np.mean(ap_list)

    print()
    print("=" * 55)
    print("  KET QUA DANH GIA HE THONG CBIR ANH CA")
    print("=" * 55)
    print(f"  So anh query  : {n}")
    print(f"  Do do          : {metric}")
    print(f"  So loai         : {len(set(id2species.values()))}")
    print("-" * 55)
    print(f"  Top-1 Accuracy : {top1_acc:.4f}  ({top1_acc:.1%})")
    print(f"  Precision@1    : {p1:.4f}  ({p1:.1%})")
    print(f"  Precision@3    : {p3:.4f}  ({p3:.1%})")
    print(f"  Precision@5    : {p5:.4f}  ({p5:.1%})")
    print(f"  MAP@5          : {MAP:.4f}  ({MAP:.1%})")
    print("=" * 55)
    print(f"  Thoi gian      : {time.time()-start:.1f}s")
    print()

    # Goi y danh gia
    if MAP >= 0.70:
        print("  [XUAT SAC] He thong dat hieu suat rat cao!")
    elif MAP >= 0.50:
        print("  [TOT]      He thong hoat dong on dinh.")
    elif MAP >= 0.30:
        print("  [TRUNG BINH] Co the tinh chinh trong so dac trung.")
    else:
        print("  [THAP] Nen kiem tra lai bo du lieu hoac trong so.")

    return {
        'top1': top1_acc, 'p1': p1, 'p3': p3, 'p5': p5, 'map': MAP
    }


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Danh gia he thong CBIR anh ca')
    parser.add_argument('--samples', type=int,   default=200,
                        help='So anh query de danh gia (mac dinh: 200)')
    parser.add_argument('--metric',  type=str,   default='cosine',
                        choices=['cosine', 'euclidean', 'histogram'],
                        help='Do do tuong dong (mac dinh: cosine)')
    parser.add_argument('--k',       type=int,   default=5,
                        help='Top-K (mac dinh: 5)')
    args = parser.parse_args()

    evaluate(n_samples=args.samples, metric=args.metric, k_max=args.k)
