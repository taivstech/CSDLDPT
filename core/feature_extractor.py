import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from scipy.stats import skew
from typing import Optional

# ────────────────────────────────────────────
# 1. Tiền xử lý
# ────────────────────────────────────────────

def read_image(file_path: str, size: int = 224) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn (hỗ trợ tên file unicode),
    resize về (size x size), trả về ảnh RGB.
    """
    img_array = np.fromfile(file_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {file_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[:2] != (size, size):
        img_rgb = cv2.resize(img_rgb, (size, size))
    return img_rgb


def rgb_to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def rgb_to_hsv(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)


# ────────────────────────────────────────────
# 2. Các hàm trích xuất đặc trưng
# ────────────────────────────────────────────

def extract_hog_features(img_gray: np.ndarray) -> np.ndarray:
    """
    HOG — Histogram of Oriented Gradients.
    Mô tả hình dạng/đường viền cá, bất biến với màu sắc background.
    Output: vector ~6084 chiều (224x224, cell=16, block=2, orient=9).
    """
    vec = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return vec / (np.linalg.norm(vec) + 1e-7)


def extract_hsv_histogram(img_hsv: np.ndarray, bins=(16, 8, 8)) -> np.ndarray:
    """
    HSV Color Histogram — mô tả màu sắc đặc trưng của cá (vây, thân, hoa văn).
    Dùng toàn ảnh (không mask) vì background thường khác màu cá.
    bins=(H:16, S:8, V:8) → 1024 chiều.
    """
    hist = cv2.calcHist(
        [img_hsv],
        [0, 1, 2],
        None,
        [bins[0], bins[1], bins[2]],
        [0, 180, 0, 256, 0, 256]
    )
    vec = hist.flatten()
    cv2.normalize(vec, vec, alpha=1, norm_type=cv2.NORM_L1)
    return vec


def extract_lbp_features(img_gray: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    LBP — Local Binary Pattern.
    Mô tả kết cấu bề mặt (vảy, vân da) của cá, bất biến với chiếu sáng.
    Phân tích dữ liệu thực tế: Laplacian Variance đạt tới 2303, chứng tỏ
    vảy cá rất phức tạp — LBP bắt cực tốt.
    Output: 256 chiều.
    """
    n_points = 8 * radius
    lbp_img = local_binary_pattern(img_gray, n_points, radius, method='default')
    hist, _ = np.histogram(lbp_img.ravel(), bins=256, range=(0, 256))
    vec = hist.astype("float64")
    return vec / (vec.sum() + 1e-7)


def extract_hu_moments(img_gray: np.ndarray) -> np.ndarray:
    """
    Hu Moments — 7 moment hình dạng bất biến với dịch chuyển, phép quay và tỉ lệ.
    Phân tích thực tế: w/h ratio của cá dao động từ 0.31 đến 2.43 (cá bơi nghiêng,
    khoảng cách chụp khác nhau) — Hu Moments xử lý được điều này.
    Output: 7 chiều (log-transform để ổn định giá trị).
    """
    moments = cv2.moments(img_gray)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform để đồng bộ thang đo (Hu moments rất nhỏ, ~1e-7)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    # Chuẩn hóa L2
    return hu_log / (np.linalg.norm(hu_log) + 1e-7)


def extract_color_moments(img_hsv: np.ndarray) -> np.ndarray:
    """
    Color Moments — Mô-men thống kê màu sắc trên không gian HSV.
    Gồm 3 thống kê: Mean, Std, Skewness trên 3 kênh H, S, V → 9 chiều.

    Lý do thêm đặc trưng này:
    - HSV Histogram phân phối tần suất màu (WHAT colors), nhưng bị ảnh hưởng
      bởi ánh sáng dưới nước làm shift histogram.
    - Color Moments nắm thống kê tổng thể (HOW colors distributed), ổn định hơn
      với biến đổi ánh sáng môi trường nước.
    - Phân tích thực tế: thalassoma_lutescens luôn có H_mean ≈ 33-54,
      trong khi caranx_melampygus có H/S/V khác hẳn → phân biệt được.
    """
    moments = []
    for ch in range(3):  # H, S, V
        channel = img_hsv[:, :, ch].astype(np.float64)
        mean = channel.mean()
        std  = channel.std() + 1e-7
        # Skewness: độ lệch phân phối (mẫu phân phối lệch trái hay phải)
        flat = channel.ravel()
        sk   = float(skew(flat))
        if np.isnan(sk):
            sk = 0.0
        moments.extend([mean, std, sk])
    vec = np.array(moments, dtype=np.float64)
    # Chuẩn hóa min-max về [0,1] vì 3 thống kê có thang đo rất khác nhau
    vmin, vmax = vec.min(), vec.max()
    if vmax - vmin > 1e-7:
        vec = (vec - vmin) / (vmax - vmin)
    return vec / (np.linalg.norm(vec) + 1e-7)


# ────────────────────────────────────────────
# 3. Kết hợp đặc trưng (Feature Fusion)
# ────────────────────────────────────────────

def feature_fusion(
    hog_vec:  np.ndarray,
    hsv_vec:  np.ndarray,
    lbp_vec:  np.ndarray,
    hu_vec:   np.ndarray,
    cm_vec:   np.ndarray,
    alpha: float = 0.75,   # HOG — hinh dang (quan trong nhat, grid-search xac nhan)
    beta:  float = 0.08,   # HSV — mau sac histogram
    gamma: float = 0.07,   # LBP — ket cau vay
    delta: float = 0.05,   # Hu  — hinh dang tong the bat bien
    eps:   float = 0.05,   # Color Moments — thong ke mau on dinh
) -> np.ndarray:
    """
    Nối (concatenate) các vector đặc trưng đã nhân trọng số.
    Chuẩn hóa L2 trước và sau khi kết hợp.
    Tổng chiều: HOG(6084) + HSV(1024) + LBP(256) + Hu(7) + CM(9) = 7380 chiều.
    """
    def l2(v):
        return v / (np.linalg.norm(v) + 1e-7)

    combined = np.concatenate([
        alpha * l2(hog_vec),
        beta  * l2(hsv_vec),
        gamma * l2(lbp_vec),
        delta * l2(hu_vec),
        eps   * l2(cm_vec),
    ])
    combined = np.nan_to_num(combined, nan=0.0)
    return combined / (np.linalg.norm(combined) + 1e-7)


# ────────────────────────────────────────────
# 4. Hàm tổng hợp chính
# ────────────────────────────────────────────

def get_image_features(
    file_path: str,
    alpha: float = 0.75,
    beta:  float = 0.08,
    gamma: float = 0.07,
    delta: float = 0.05,
    eps:   float = 0.05,
) -> Optional[np.ndarray]:
    """
    Pipeline đầy đủ: Đọc ảnh → Tiền xử lý → Trích xuất (HOG+HSV+LBP+Hu+CM) → Kết hợp.
    Trả về vector đặc trưng cuối cùng hoặc None nếu có lỗi.
    """
    try:
        img_rgb  = read_image(file_path)
        img_gray = rgb_to_grayscale(img_rgb)
        img_hsv  = rgb_to_hsv(img_rgb)

        hog_vec = extract_hog_features(img_gray)
        hsv_vec = extract_hsv_histogram(img_hsv)
        lbp_vec = extract_lbp_features(img_gray)
        hu_vec  = extract_hu_moments(img_gray)
        cm_vec  = extract_color_moments(img_hsv)

        return feature_fusion(hog_vec, hsv_vec, lbp_vec, hu_vec, cm_vec,
                               alpha, beta, gamma, delta, eps)
    except Exception as e:
        print(f"[WARN] Loi trich xuat dac trung '{file_path}': {e}")
        return None


def get_image_features_visual(file_path: str):
    """
    Trả về cả ảnh trung gian (để hiển thị UI) lẫn vector.
    Returns: (img_rgb, img_gray, img_hsv, hog_img_uint8, lbp_img_uint8,
               hog_vec, hsv_vec, lbp_vec, hu_vec, cm_vec, feature_vector)
    """
    img_rgb  = read_image(file_path)
    img_gray = rgb_to_grayscale(img_rgb)
    img_hsv  = rgb_to_hsv(img_rgb)

    from skimage import exposure
    _, hog_img = hog(
        img_gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), visualize=True
    )
    hog_img_u8 = (exposure.rescale_intensity(hog_img, in_range=(0, 10)) * 255).astype(np.uint8)
    lbp_img = local_binary_pattern(img_gray, 8, 1, method='default').astype(np.uint8)

    hog_vec = extract_hog_features(img_gray)
    hsv_vec = extract_hsv_histogram(img_hsv)
    lbp_vec = extract_lbp_features(img_gray)
    hu_vec  = extract_hu_moments(img_gray)
    cm_vec  = extract_color_moments(img_hsv)
    fvec    = feature_fusion(hog_vec, hsv_vec, lbp_vec, hu_vec, cm_vec)

    return img_rgb, img_gray, img_hsv, hog_img_u8, lbp_img, hog_vec, hsv_vec, lbp_vec, hu_vec, cm_vec, fvec


# ────────────────────────────────────────────
# Test nhanh
# ────────────────────────────────────────────
if __name__ == "__main__":
    import os, glob
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    samples = glob.glob(os.path.join(dataset_dir, '*.jpg'))[:1]
    if samples:
        vec = get_image_features(samples[0])
        print(f"Vector kich thuoc: {vec.shape[0]} chieu")
        print(f"  HOG  ~6084 | HSV ~1024 | LBP ~256 | Hu ~7 | CM ~9")
        print(f"  Tong : {6084+1024+256+7+9} chieu")
    else:
        print("Khong tim thay anh trong dataset/")
