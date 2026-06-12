import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Đường dẫn file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = os.path.join(BASE_DIR, 'database', 'features.pkl')
KMEANS_FILE = os.path.join(BASE_DIR, 'database', 'kmeans_model.pkl')
OUTPUT_IMAGE = os.path.join(BASE_DIR, 'scratch', 'docs_images', '8_tSNE_Visualization.png')

def visualize_tsne():
    print("1. Đang tải dữ liệu vector...")
    with open(FEATURE_FILE, 'rb') as f:
        features_dict = pickle.load(f)
    
    with open(KMEANS_FILE, 'rb') as f:
        kmeans = pickle.load(f)

    # Chuyển đổi sang ma trận numpy
    img_ids = list(features_dict.keys())
    X = np.array(list(features_dict.values()), dtype=np.float32)
    labels = kmeans.labels_

    print(f"Ma trận gốc: {X.shape[0]} ảnh, {X.shape[1]} chiều.")

    # Bước 1: Dùng PCA giảm số chiều xuống 50 để chạy t-SNE nhanh hơn và giảm nhiễu
    print("2. Đang chạy PCA giảm xuống 50 chiều...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)

    # Bước 2: Dùng t-SNE giảm xuống 2 chiều để vẽ đồ thị
    print("3. Đang chạy t-SNE giảm xuống 2 chiều (Quá trình này mất khoảng 10-20 giây)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_pca)

    # Bước 3: Vẽ biểu đồ Scatter
    print("4. Đang vẽ biểu đồ Scatter...")
    plt.figure(figsize=(14, 10))
    
    # Lấy danh sách các cụm (tối đa 35 cụm theo K-Means của hệ thống)
    unique_clusters = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20', len(unique_clusters))

    for cluster_id in unique_clusters:
        # Tìm các điểm thuộc cụm hiện tại
        idx = np.where(labels == cluster_id)
        plt.scatter(
            X_tsne[idx, 0], X_tsne[idx, 1],
            c=[cmap(cluster_id)], 
            label=f'Cụm {cluster_id}',
            alpha=0.7, edgecolors='none', s=40
        )

    plt.title('Trực quan hóa không gian Vector 7380 chiều bằng t-SNE (Phân cụm K-Means)', fontsize=16, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Đưa chú thích (legend) ra ngoài để không che mất đồ thị
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"5. Hoàn thành! Ảnh đã được lưu tại: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    visualize_tsne()
