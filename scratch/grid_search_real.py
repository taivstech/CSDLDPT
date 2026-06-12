

import sys, os, time, pickle, sqlite3, random
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.feature_extractor import (
    read_image, rgb_to_grayscale, rgb_to_hsv,
    extract_hog_features, extract_hsv_histogram,
    extract_lbp_features, extract_hu_moments, extract_color_moments
)

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR   = os.path.join(BASE_DIR, 'dataset')
DB_PATH       = os.path.join(BASE_DIR, 'database', 'fish_cbir.db')
CACHE_FILE    = os.path.join(BASE_DIR, 'scratch', 'raw_features_cache.pkl')
RESULTS_FILE  = os.path.join(BASE_DIR, 'scratch', 'grid_search_results.txt')

# ─────────────────────────────────────────────────────────────────────
# 1. Trich xuat raw features (chi chay 1 lan, sau do dung cache)
# ─────────────────────────────────────────────────────────────────────

def extract_raw_features_all():
    """
    Trich xuat 5 raw vector (HOG, HSV, LBP, Hu, CM) cho tung anh.
    Luu vao cache de khong phai chay lai lan sau.
    """
    if os.path.exists(CACHE_FILE):
        print(f"[CACHE] Tim thay file cache: {CACHE_FILE}")
        print("[CACHE] Dang tai raw features tu cache...")
        t0 = time.time()
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        print(f"[CACHE] Da tai {len(cache['ids'])} anh in {time.time()-t0:.1f}s\n")
        return cache

    print("=" * 65)
    print("  [BUOC 1/3] TRICH XUAT RAW FEATURES (chi chay 1 lan)")
    print("=" * 65)

    img_files = [
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    total = len(img_files)
    print(f"  Tong so anh: {total}")
    print(f"  Luu cache vao: {CACHE_FILE}\n")

    ids, hog_mat, hsv_mat, lbp_mat, hu_mat, cm_mat = [], [], [], [], [], []
    errors = 0
    t0 = time.time()

    for i, fname in enumerate(img_files, 1):
        fpath = os.path.join(DATASET_DIR, fname)
        try:
            img_rgb  = read_image(fpath)
            img_gray = rgb_to_grayscale(img_rgb)
            img_hsv  = rgb_to_hsv(img_rgb)

            hv = extract_hog_features(img_gray)
            sv = extract_hsv_histogram(img_hsv)
            lv = extract_lbp_features(img_gray)
            uv = extract_hu_moments(img_gray)
            cv = extract_color_moments(img_hsv)

            image_id = os.path.splitext(fname)[0]
            ids.append(image_id)
            hog_mat.append(hv)
            hsv_mat.append(sv)
            lbp_mat.append(lv)
            hu_mat.append(uv)
            cm_mat.append(cv)
        except Exception as e:
            errors += 1
            continue

        if i % 100 == 0 or i == total:
            elapsed = time.time() - t0
            remain  = elapsed / i * (total - i)
            print(f"  [{i:>4}/{total}] Loi: {errors}  "
                  f"Thoi gian: {elapsed:.0f}s  Con lai: ~{remain:.0f}s")

    cache = {
        'ids':     ids,
        'hog_mat': np.array(hog_mat, dtype=np.float32),
        'hsv_mat': np.array(hsv_mat, dtype=np.float32),
        'lbp_mat': np.array(lbp_mat, dtype=np.float32),
        'hu_mat':  np.array(hu_mat,  dtype=np.float32),
        'cm_mat':  np.array(cm_mat,  dtype=np.float32),
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

    total_time = time.time() - t0
    print(f"\n[CACHE] Da luu xong! {len(ids)} anh, {errors} loi, "
          f"thoi gian: {total_time:.1f}s\n")
    return cache


# ─────────────────────────────────────────────────────────────────────
# 2. Lay nhan loai tu DB
# ─────────────────────────────────────────────────────────────────────

def load_labels():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT Image_ID, Species_Label FROM Fish_Metadata")
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


# ─────────────────────────────────────────────────────────────────────
# 3. Fusion nhanh bang ma tran NumPy (rat nhanh tren RAM)
# ─────────────────────────────────────────────────────────────────────

def l2_norm(mat):
    """Chuan hoa L2 tung hang cua ma tran."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-7
    return mat / norms


def fuse_matrix(cache, alpha, beta, gamma, delta, eps):
    """
    Ket hop 5 raw matrices thanh 1 fused matrix voi trong so cho truoc.
    Toan bo la phep tinh NumPy tren RAM → rat nhanh (< 1 giay).
    """
    fused = np.concatenate([
        alpha * l2_norm(cache['hog_mat']),
        beta  * l2_norm(cache['hsv_mat']),
        gamma * l2_norm(cache['lbp_mat']),
        delta * l2_norm(cache['hu_mat']),
        eps   * l2_norm(cache['cm_mat']),
    ], axis=1).astype(np.float32)

    # Chuan hoa L2 lan cuoi
    norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-7
    return fused / norms


# ─────────────────────────────────────────────────────────────────────
# 4. Danh gia MAP@5 tren validation set
# ─────────────────────────────────────────────────────────────────────

def evaluate_weights(fused_mat, ids, id2label, n_samples=200, k=5, seed=42):
    """
    Tinh MAP@5 bang KNN Cosine Similarity tren tap validation.
    Cosine sim = dot(a, b) vi tat ca vector da duoc chuan hoa L2.
    """
    random.seed(seed)
    all_indices = list(range(len(ids)))
    sample_indices = random.sample(all_indices, min(n_samples, len(ids)))

    ap_list = []
    for qi in sample_indices:
        q_vec     = fused_mat[qi]           # (D,)
        q_species = id2label.get(ids[qi], '')

        # Tinh cosine sim voi toan bo CSDL bang phep nhan ma tran
        sims = fused_mat @ q_vec            # (N,) — nhanh!
        sims[qi] = -1.0                     # Loai chinh no

        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        retrieved = [id2label.get(ids[j], '') for j in top_indices]

        # Average Precision
        hits, sum_prec = 0, 0.0
        for rank, sp in enumerate(retrieved, 1):
            if sp == q_species:
                hits += 1
                sum_prec += hits / rank
        ap = sum_prec / hits if hits > 0 else 0.0
        ap_list.append(ap)

    return float(np.mean(ap_list))


# ─────────────────────────────────────────────────────────────────────
# 5. Dinh nghia khong gian tim kiem (Grid)
# ─────────────────────────────────────────────────────────────────────

def build_grid():
    """
    Cac bo trong so can thu nghiem.
    Tong alpha+beta+gamma+delta+eps = 1.0
    Tap trung thu alpha (HOG) tu 0.20 den 0.80, cac trong so con lai chia deu phan con lai.
    """
    grid = []

    # Thu nhieu gia tri alpha (HOG) khac nhau
    configs = [
        # (alpha, beta,  gamma, delta, eps)   # Mo ta
        (0.20,  0.55,  0.10,  0.05,  0.10),  # Mau sac chiem dao
        (0.30,  0.40,  0.15,  0.05,  0.10),  # Can bang mau sac - hinh dang
        (0.40,  0.25,  0.20,  0.05,  0.10),  # Tang hinh dang nhe
        (0.50,  0.20,  0.15,  0.05,  0.10),  # Hinh dang chinh, mau sac phu
        (0.55,  0.15,  0.15,  0.05,  0.10),  # Hinh dang cao hon
        (0.60,  0.15,  0.10,  0.05,  0.10),  # Gan toi uu - a
        (0.60,  0.10,  0.15,  0.05,  0.10),  # Gan toi uu - b
        (0.65,  0.10,  0.10,  0.05,  0.10),  # Bo trong so hien tai
        (0.65,  0.15,  0.08,  0.02,  0.10),  # Bien the 1
        (0.65,  0.10,  0.10,  0.10,  0.05),  # Bien the 2 (tang Hu)
        (0.70,  0.10,  0.08,  0.02,  0.10),  # HOG cao hon
        (0.70,  0.10,  0.05,  0.05,  0.10),  # HOG cao, it ket cau
        (0.75,  0.08,  0.07,  0.05,  0.05),  # HOG rat cao
        (0.80,  0.05,  0.08,  0.02,  0.05),  # Chi hinh dang
        (0.50,  0.10,  0.10,  0.10,  0.20),  # Tang Color Moments
        (0.60,  0.10,  0.10,  0.05,  0.15),  # Tang Color Moments nhe
        (1/3,   1/3,   1/6,   1/12,  1/12),  # Chia deu 3 nhom
    ]

    for c in configs:
        a, b, g, d, e = c
        total = a + b + g + d + e
        if abs(total - 1.0) < 0.01:  # Kiem tra tong = 1
            grid.append(c)
        else:
            # Tu dong chuan hoa neu tong != 1
            grid.append((a/total, b/total, g/total, d/total, e/total))

    return grid


# ─────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 65)
    print("  GRID SEARCH TRONG SO FEATURE FUSION - ANH CA CBIR")
    print("=" * 65)
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Cache   : {CACHE_FILE}")
    print()

    # B1: Trich xuat / tai cache
    cache    = extract_raw_features_all()
    ids      = cache['ids']
    id2label = load_labels()

    # Loc nhung anh co nhan (co trong DB)
    valid_mask = [i for i, iid in enumerate(ids) if iid in id2label]
    ids      = [ids[i] for i in valid_mask]
    for key in ('hog_mat', 'hsv_mat', 'lbp_mat', 'hu_mat', 'cm_mat'):
        cache[key] = cache[key][valid_mask]

    print(f"[INFO] So anh hop le (co trong DB): {len(ids)}")
    print(f"[INFO] So loai: {len(set(id2label.values()))}")
    print()

    # B2: Grid Search
    grid = build_grid()
    N_SAMPLES = 200  # So anh dung de cham diem MAP

    print("=" * 65)
    print(f"  [BUOC 2/3] GRID SEARCH ({len(grid)} to hop trong so, {N_SAMPLES} anh query)")
    print("=" * 65)
    print(f"  {'No':>3} | {'HOG(a)':>8} | {'HSV(b)':>8} | {'LBP(g)':>8} | "
          f"{'Hu(d)':>7} | {'CM(e)':>7} | {'MAP@5':>8} | {'Time':>6}")
    print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+"
          f"-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}")

    results  = []
    t_start  = time.time()

    for i, (a, b, g, d, e) in enumerate(grid, 1):
        t0 = time.time()

        # Fusion tren RAM (< 1s)
        fused = fuse_matrix(cache, a, b, g, d, e)

        # Danh gia MAP@5
        map5 = evaluate_weights(fused, ids, id2label, n_samples=N_SAMPLES)

        elapsed = time.time() - t0
        results.append((a, b, g, d, e, map5))

        marker = '  *** BEST ***' if map5 == max(r[5] for r in results) else ''
        print(f"  {i:>3} | {a:>8.3f} | {b:>8.3f} | {g:>8.3f} | "
              f"{d:>7.3f} | {e:>7.3f} | {map5:>7.2%} | {elapsed:>5.1f}s{marker}")

    # B3: Tong hop ket qua
    results.sort(key=lambda x: x[5], reverse=True)
    best = results[0]

    total_time = time.time() - t_start
    print()
    print("=" * 65)
    print("  [BUOC 3/3] KET QUA GRID SEARCH")
    print("=" * 65)
    print(f"  Tong thoi gian    : {total_time:.1f}s")
    print(f"  Bo trong so TOI UU:")
    print(f"    alpha (HOG)          = {best[0]:.3f}")
    print(f"    beta  (HSV)          = {best[1]:.3f}")
    print(f"    gamma (LBP)          = {best[2]:.3f}")
    print(f"    delta (Hu Moments)   = {best[3]:.3f}")
    print(f"    eps   (Color Moments)= {best[4]:.3f}")
    print(f"    => MAP@5             = {best[5]:.4f}  ({best[5]:.2%})")
    print()
    print("  Top 3 bo trong so tot nhat:")
    for rank, (a, b, g, d, e, m) in enumerate(results[:3], 1):
        print(f"    #{rank}: HOG={a:.2f} HSV={b:.2f} LBP={g:.2f} "
              f"Hu={d:.2f} CM={e:.2f}  →  MAP={m:.2%}")
    print("=" * 65)

    # Luu ket qua ra file
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write("GRID SEARCH KET QUA - FEATURE FUSION TRONG SO\n")
        f.write("=" * 65 + "\n")
        f.write(f"Dataset: {DATASET_DIR}\n")
        f.write(f"So anh: {len(ids)} | So loai: {len(set(id2label.values()))}\n")
        f.write(f"N_SAMPLES (query): {N_SAMPLES} | Metric: MAP@5 (Cosine)\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"  {'No':>3} | {'HOG':>6} | {'HSV':>6} | {'LBP':>6} | "
                f"{'Hu':>6} | {'CM':>6} | {'MAP@5':>8}\n")
        f.write(f"  {'-'*60}\n")
        for i, (a, b, g, d, e, m) in enumerate(sorted(results, key=lambda x: -x[5]), 1):
            f.write(f"  {i:>3} | {a:>6.3f} | {b:>6.3f} | {g:>6.3f} | "
                    f"{d:>6.3f} | {e:>6.3f} | {m:>7.2%}\n")
        f.write("\nBO TRONG SO TOI UU:\n")
        f.write(f"  alpha (HOG)          = {best[0]:.3f}\n")
        f.write(f"  beta  (HSV)          = {best[1]:.3f}\n")
        f.write(f"  gamma (LBP)          = {best[2]:.3f}\n")
        f.write(f"  delta (Hu Moments)   = {best[3]:.3f}\n")
        f.write(f"  eps   (Color Moments)= {best[4]:.3f}\n")
        f.write(f"  MAP@5                = {best[5]:.4f} ({best[5]:.2%})\n")

    print(f"\n  Ket qua da duoc luu vao: {RESULTS_FILE}")
    print()
    return best

if __name__ == '__main__':
    main()
