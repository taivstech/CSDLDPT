import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = r'd:/Projects/CSDLDPT/dataset/caranx_melampygus_caranx_melampygus_10_jpg.rf.b5696e447292ad1dd43ca96f969c80d9.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_orig = img.copy()

img_scaled = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
pad_y = (img.shape[0] - img_scaled.shape[0]) // 2
pad_x = (img.shape[1] - img_scaled.shape[1]) // 2
img_scaled_pad = np.zeros_like(img)
img_scaled_pad[pad_y:pad_y+img_scaled.shape[0], pad_x:pad_x+img_scaled.shape[1]] = img_scaled

M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 45, 1.0)
img_rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def get_hu(im):
    m = cv2.moments(im)
    h = cv2.HuMoments(m).flatten()
    return [-1 * np.sign(v) * np.log10(np.abs(v) + 1e-10) for v in h]

hu_orig = get_hu(img_orig)
hu_scaled = get_hu(img_scaled)
hu_rot = get_hu(img_rot)

fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot(2, 3, 1)
ax1.imshow(img_orig, cmap='gray')
ax1.set_title('Ảnh gốc')
ax1.axis('off')

ax2 = plt.subplot(2, 3, 2)
ax2.imshow(img_scaled_pad, cmap='gray')
ax2.set_title('Thu nhỏ (Scale 50%)')
ax2.axis('off')

ax3 = plt.subplot(2, 3, 3)
ax3.imshow(img_rot, cmap='gray')
ax3.set_title('Xoay (Rotate 45°)')
ax3.axis('off')

ax_bar = plt.subplot(2, 1, 2)
labels = [f'h{i+1}' for i in range(7)]
x = np.arange(len(labels))
w = 0.25

ax_bar.bar(x - w, hu_orig, w, label='Ảnh gốc', color='steelblue')
ax_bar.bar(x, hu_scaled, w, label='Thu nhỏ', color='darkorange')
ax_bar.bar(x + w, hu_rot, w, label='Xoay 45°', color='forestgreen')

ax_bar.set_ylabel('Giá trị Hu Moment (Log scale)')
ax_bar.set_title('So sánh 7 đặc trưng Hu Moments (Tính bất biến)')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels)
ax_bar.legend()

plt.tight_layout()
plt.savefig(r'd:\Projects\CSDLDPT\scratch\docs_images\7_Hu_Moments_Invariance.png', dpi=300)
