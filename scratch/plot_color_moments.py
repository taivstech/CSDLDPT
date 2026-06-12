import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

img1_path = r'd:/Projects/CSDLDPT/dataset/thalassoma_lutescens_thalassoma_lutescens_10_jpg.rf.35ab2d99e6184f619ff07978d7974faa.jpg'
img2_path = r'd:/Projects/CSDLDPT/dataset/thalassoma_lutescens_thalassoma_lutescens_12_jpg.rf.9cd9e9fdf5bbd399cd75e8f435736473.jpg'
img3_path = r'd:/Projects/CSDLDPT/dataset/caranx_melampygus_caranx_melampygus_10_jpg.rf.b5696e447292ad1dd43ca96f969c80d9.jpg'

def get_color_moments(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    moments = []
    for i in range(3): # H, S, V
        channel = img_hsv[:,:,i].flatten()
        mean = np.mean(channel)
        std = np.std(channel)
        s = skew(channel)
        moments.extend([mean, std, s])
    return img_rgb, np.array(moments)

img1, mom1 = get_color_moments(img1_path)
img2, mom2 = get_color_moments(img2_path)
img3, mom3 = get_color_moments(img3_path)

# Normalize moments for visualization (min-max across the 3 vectors)
all_moms = np.vstack([mom1, mom2, mom3])
mom_min = np.min(all_moms, axis=0)
mom_max = np.max(all_moms, axis=0)
denom = mom_max - mom_min
denom[denom == 0] = 1 # prevent div by zero
norm1 = (mom1 - mom_min) / denom
norm2 = (mom2 - mom_min) / denom
norm3 = (mom3 - mom_min) / denom

fig = plt.figure(figsize=(14, 8))

# Images
ax1 = plt.subplot(2, 3, 1)
ax1.imshow(img1)
ax1.set_title('Thalassoma lutescens (Mẫu 1)')
ax1.axis('off')

ax2 = plt.subplot(2, 3, 2)
ax2.imshow(img2)
ax2.set_title('Thalassoma lutescens (Mẫu 2)')
ax2.axis('off')

ax3 = plt.subplot(2, 3, 3)
ax3.imshow(img3)
ax3.set_title('Caranx melampygus')
ax3.axis('off')

# Bar chart
ax_bar = plt.subplot(2, 1, 2)
labels = ['H_mean', 'H_std', 'H_skew', 'S_mean', 'S_std', 'S_skew', 'V_mean', 'V_std', 'V_skew']
x = np.arange(len(labels))
w = 0.25

ax_bar.bar(x - w, norm1, w, label='T. lutescens (Mẫu 1)', color='gold')
ax_bar.bar(x, norm2, w, label='T. lutescens (Mẫu 2)', color='darkorange')
ax_bar.bar(x + w, norm3, w, label='Caranx melampygus', color='royalblue')

ax_bar.set_ylabel('Giá trị chuẩn hóa (0-1)')
ax_bar.set_title('So sánh 9 đặc trưng Color Moments giữa các cá thể')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels)
ax_bar.legend()

plt.tight_layout()
plt.savefig(r'd:\Projects\CSDLDPT\scratch\docs_images\9_Color_Moments.png', dpi=300)
