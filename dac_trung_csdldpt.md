# HƯỚNG DẪN CHI TIẾT VỀ ĐẶC TRƯNG HÌNH ẢNH (CSDLDPT)
*Tài liệu hướng dẫn ôn tập vấn đáp và thực hành cho Đồ án Cơ sở dữ liệu đa phương tiện*

---

## 📌 PHẦN 1: TỔNG QUAN VỀ VECTOR ĐẶC TRƯNG HỖN HỢP (7380 CHIỀU)

Hệ thống sử dụng kỹ thuật **Feature Fusion (Kết hợp đặc trưng)** để tận dụng cả 3 yếu tố cốt lõi của hình ảnh: **Hình dạng (Shape)**, **Màu sắc (Color)**, và **Kết cấu bề mặt (Texture)**. 

Vector đặc trưng cuối cùng đại diện cho mỗi bức ảnh cá có tổng số chiều là **7380 chiều**, được ghép từ 5 loại đặc trưng:

$$\text{Tổng số chiều} = \underbrace{6084}_{\text{HOG}} + \underbrace{1024}_{\text{HSV}} + \underbrace{256}_{\text{LBP}} + \underbrace{7}_{\text{Hu}} + \underbrace{9}_{\text{CM}} = 7380 \text{ chiều.}$$

---

## 📌 PHẦN 2: CHI TIẾT 5 LOẠI ĐẶC TRƯNG & CÁCH TÍNH TOÁN

### 1. Đặc trưng HOG (Histogram of Oriented Gradients) - Biên cạnh & Dáng vẻ (6084 chiều)
*   **Ý nghĩa:** Trích xuất thông tin về **hình dáng biên cạnh toàn cục** của con cá (như biên vây, đầu, đuôi) giúp nhận dạng loài bất kể màu sắc cá hay phông nền thay đổi.
*   **Cách tính toán toán học:**
    1.  **Tính đạo hàm (Gradient):** Sử dụng toán tử Sobel tính đạo hàm theo hướng ngang $I_x$ và dọc $I_y$ tại mỗi pixel. Tính độ lớn $G$ và hướng góc $\theta$ của gradient:
        $$G(x,y) = \sqrt{I_x^2 + I_y^2}, \quad \theta(x,y) = \arctan\left(\frac{I_y}{I_x}\right)$$
    2.  **Chia ô (Cells):** Chia bức ảnh ($224 \times 224$ pixel) thành các ô nhỏ kích thước $16 \times 16$ pixel. Tổng cộng có $14 \times 14 = 196$ cells.
    3.  **Lập Lược đồ góc (Orientation Histogram):** Trong mỗi ô $16 \times 16$, thống kê hướng gradient $\theta$ của các pixel vào 9 nhóm góc (bins) từ $0^\circ$ đến $180^\circ$. Trọng số đóng góp tỉ lệ với độ lớn $G$.
    4.  **Chuẩn hóa khối (Blocks):** Nhóm $2 \times 2$ cells thành 1 khối (block) $\rightarrow$ Có $13 \times 13 = 169$ blocks. Chuẩn hóa L2 trên mỗi block để triệt tiêu ảnh hưởng ánh sáng.
    5.  **Số chiều:** $169 \text{ blocks} \times (4 \text{ cells} \times 9 \text{ bins}) = 6084 \text{ chiều}$.

### 2. Đặc trưng HSV Histogram - Phân phối màu sắc (1024 chiều)
*   **Ý nghĩa:** Thống kê **phân phối màu sắc tổng thể** trên thân cá (ví dụ: cá có bao nhiêu phần trăm màu vàng, xanh, đỏ).
*   **Cách tính toán toán học:**
    1.  **Chuyển đổi hệ màu:** Đổi ảnh từ hệ RGB sang hệ màu HSV gồm: **H** (Hue - Tông màu, $0-179$), **S** (Saturation - Độ bão hòa, $0-255$), **V** (Value - Độ sáng, $0-255$).
    2.  **Chia nhóm (Bins):** Chia không gian màu thành các nhóm: kênh H chia làm 16 bins, kênh S chia làm 8 bins, kênh V chia làm 8 bins.
    3.  **Thống kê & Loại nền:** Duyệt qua các pixel cá (loại bỏ phông nền bằng mặt nạ Mask). Đếm số lượng pixel rơi vào từng bin màu tương ứng.
    4.  **Chuẩn hóa L1:** Chia tổng số đếm để tổng lược đồ bằng $1.0$ (tạo thành phân phối xác suất màu).
    5.  **Số chiều:** $16 \text{ bins (H)} \times 8 \text{ bins (S)} \times 8 \text{ bins (V)} = 1024 \text{ chiều}$.

### 3. Đặc trưng LBP (Local Binary Patterns) - Kết cấu da và vảy (256 chiều)
*   **Ý nghĩa:** Mô tả **kết cấu bề mặt (Texture)** như vảy cá thô ráp hay da mịn, có sọc hay đốm tròn.
*   **Cách tính toán toán học:**
    1.  **So sánh cục bộ:** Với mỗi pixel trung tâm có giá trị xám $g_c$, xét 8 pixel lân cận $g_p$ xung quanh trong bán kính $R=1$.
    2.  **Ngưỡng hóa nhị phân:** 
        *   Nếu $g_p \ge g_c$: Gán giá trị **`1`**
        *   Nếu $g_p < g_c$: Gán giá trị **`0`**
    3.  **Đổi sang số thập phân:** Ghép 8 giá trị nhị phân xung quanh thành mã 8-bit và đổi sang số thập phân ($0-255$):
        $$LBP = \sum_{p=0}^{7} s(g_p - g_c) 2^p \quad \text{với } s(x)=1 \text{ nếu } x \ge 0, \text{ ngược lại } s(x)=0$$
    4.  **Lập Histogram:** Đếm tần suất xuất hiện các mã LBP từ $0-255$ trên vùng da cá.
    5.  **Số chiều:** Biểu đồ histogram có **256 chiều** (ứng với các giá trị từ $0$ đến $255$).

### 4. Đặc trưng Hu Moments - Mô-men hình dạng bất biến (7 chiều)
*   **Ý nghĩa:** Mô tả hình học toàn cục của cá, **không thay đổi** khi cá bơi nghiêng, bơi thẳng, cá ở xa hay ở gần (bất biến với phép dịch chuyển, phép xoay và tỉ lệ).
*   **Cách tính toán toán học:**
    1.  Tính các mô-men thô $m_{pq}$ từ ảnh nhị phân mặt nạ (Mask): $m_{pq} = \sum_{x} \sum_{y} x^p y^q I(x,y)$.
    2.  Tính mô-men trung tâm $\mu_{pq}$ (trừ đi tọa độ trọng tâm vật thể $\bar{x}, \bar{y}$ để bất biến phép dịch chuyển).
    3.  Tính mô-men trung tâm chuẩn hóa $\eta_{pq}$ để bất biến phép thu phóng (scale).
    4.  Kết hợp $\eta_{pq}$ bậc 2 và 3 theo 7 công thức Hu để triệt tiêu ảnh hưởng phép xoay.
    5.  Áp dụng phép biến đổi logarit để ổn định khoảng số: $h_i = -\text{sign}(\phi_i) \cdot \log_{10}(|\phi_i| + 10^{-10})$.
    6.  **Số chiều:** **7 chiều** (tương ứng với 7 mô-men Hu).

### 5. Đặc trưng Color Moments - Thống kê màu sắc ổn định (9 chiều)
*   **Ý nghĩa:** Mô tả phân bố toán học tổng quát của màu sắc trên 3 kênh H, S, V. Khác với HSV Histogram, Color Moments đo phân bố tổng hợp nên rất ổn định khi ánh sáng thay đổi.
*   **Cách tính toán toán học:** Trên mỗi kênh màu (H, S, V), tính 3 đại lượng thống kê:
    1.  **Mean (Mô-men bậc 1):** Màu sắc chủ đạo. $E_i = \frac{1}{N} \sum p_{ij}$
    2.  **Standard Deviation (Mô-men bậc 2):** Độ rực rỡ/phân tán màu. $\sigma_i = \sqrt{\frac{1}{N} \sum (p_{ij} - E_i)^2}$
    3.  **Skewness (Mô-men bậc 3):** Độ lệch đối xứng của phân phối màu. $S_i = \sqrt[3]{\frac{1}{N} \sum (p_{ij} - E_i)^3}$
    4.  **Số chiều:** $3 \text{ kênh} \times 3 \text{ đại lượng} = 9 \text{ chiều}$.

---

## 📌 PHẦN 3: PHƯƠNG PHÁP KẾT HỢP VECTOR ĐẶC TRƯNG (FEATURE FUSION)

Để ghép các đặc trưng này lại mà không để HOG (6084 chiều) đè bẹp các đặc trưng khác, hệ thống thực hiện phép **Weighted Scaling (Trọng số hóa)**:

1.  **Chuẩn hóa L2 riêng lẻ:** Đưa mỗi vector con về cùng độ dài $1.0$.
2.  **Nhân trọng số tối ưu** (Được tìm ra bằng thuật toán quét lưới Grid Search thực nghiệm để đạt MAP@5 cao nhất):
    *   HOG: $\alpha = \mathbf{0.75}$ (Hình dáng quan trọng nhất)
    *   HSV: $\beta = \mathbf{0.08}$
    *   LBP: $\gamma = \mathbf{0.07}$
    *   Hu: $\delta = \mathbf{0.05}$
    *   CM: $\epsilon = \mathbf{0.05}$
3.  **Nối vector và Chuẩn hóa L2 tổng hợp:**
    $$\mathbf{V}_{\text{Fused}} = \text{Normalize\_L2}\big( \left[ 0.75 \cdot \mathbf{v}_{\text{HOG}}, 0.08 \cdot \mathbf{v}_{\text{HSV}}, 0.07 \cdot \mathbf{v}_{\text{LBP}}, 0.05 \cdot \mathbf{v}_{\text{Hu}}, 0.05 \cdot \mathbf{v}_{\text{CM}} \right] \big)$$

---

## 📌 PHẦN 4: HƯỚNG DẪN THỰC HÀNH CÁCH XEM ĐẶC TRƯNG (VẤN ĐÁP)

Khi thầy cô yêu cầu: **"Cho tôi xem các đặc trưng lưu ở đâu / biểu diễn thế nào?"**, bạn có 3 cách trình diễn trực quan:

### Cách 1: Xem trực quan trên giao diện ứng dụng (Tkinter)
*   **Bước 1:** Chạy ứng dụng giao diện bằng cách gõ lệnh `python app.py` trên Terminal.
*   **Bước 2:** Chọn ảnh và nhấn "Tìm kiếm".
*   **Bước 3:** Chỉ vào màn hình giao diện:
    *   **Cột ② (Các bước trích xuất):** Chỉ cho thầy xem ảnh xám biên cạnh HOG (hiện rõ viền cá) và kết cấu LBP (hiện rõ độ nhám da cá).
    *   **Cột ③ (Biểu đồ đặc trưng):** Chỉ cho thầy biểu đồ tần suất dải màu Hue và lược đồ kết cấu LBP Histogram được vẽ tự động bằng thư viện `matplotlib`.
    *   **Cột ① (Thông tin Vector):** Chỉ cho thầy xem các giá trị số thực của vector 7380 chiều được in trực quan dưới góc trái giao diện.

### Cách 2: Xem file `.pkl` (Chỉ mục đặc trưng vật lý) bằng Python
Vì file `.pkl` là nhị phân, bạn chạy đoạn script mình đã chuẩn bị sẵn để in cấu trúc dữ liệu lên Terminal cho thầy xem:
*   **Lệnh chạy:**
    ```bash
    python scratch/view_pickle.py
    ```
*   **Giải thích:**
    *   *Trình bày với thầy:* File [features.pkl](file:///d:/Projects/CSDLDPT/database/features.pkl) lưu cấu trúc dạng từ điển: `Key` là **ID tự tăng kiểu số nguyên** (đã đồng bộ với khóa chính SQLite), và `Value` là **mảng numpy chứa vector 7380 chiều**.
    *   File [ivf_index.pkl](file:///d:/Projects/CSDLDPT/database/ivf_index.pkl) chứa cấu trúc chỉ mục ngược: `Key` là mã số cụm và `Value` là danh sách các ID số nguyên của các bức ảnh nằm trong cụm đó.

### Cách 3: Xem cơ sở dữ liệu quan hệ SQLite
*   **Bước 1:** Mở phần mềm **DB Browser for SQLite** (hoặc SQLite Viewer trong VS Code).
*   **Bước 2:** Mở file CSDL [fish_cbir.db](file:///d:/Projects/CSDLDPT/database/fish_cbir.db).
*   **Bước 3:** Mở bảng **`Fish_Metadata`**.
*   **Giải thích:** Bảng này lưu trữ siêu dữ liệu sạch sẽ gồm: `ID` (Khóa chính số nguyên tự tăng), `Image_ID` (Mã định danh ảnh gốc), `Species_Label` (Nhãn loài cá), và `File_Path` (Đường dẫn để ứng dụng hiển thị ảnh).
