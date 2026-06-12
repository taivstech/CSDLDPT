"""
app.py — He thong CBIR Nhan dang & Tim kiem Anh Ca
=====================================================
Giao dien Tkinter dark-theme:
  Col 1 : Anh dau vao + nut bam + thong tin vector
  Col 2 : Cac buoc trung gian (Xam, HSV, HOG, LBP)
  Col 3 : Bieu do histogram mau sac va ket cau
  Bottom: Top 5 ket qua tuong dong
  Footer: Log trang thai he thong

Trinh tu su dung:
  1. Chon anh ca bang "Chon anh"
  2. Nhan "Tim kiem" -> hien thi Top 5
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import exposure
from skimage.feature import hog, local_binary_pattern

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.searcher import FishSearcher
from core.feature_extractor import (
    get_image_features_visual,
    read_image,
    rgb_to_grayscale,
    rgb_to_hsv,
    extract_color_moments,
)

# ── Mau sac giao dien ──
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BG_CARD   = "#21262d"
BG_BUTTON = "#238636"
BG_BTN2   = "#1f6feb"
BG_BTN3   = "#6e7681"
RED_BTN   = "#da3633"
FG_WHITE  = "#f0f6fc"
FG_GREEN  = "#3fb950"
FG_YELLOW = "#e3b341"
FG_BLUE   = "#58a6ff"
FG_GRAY   = "#8b949e"
ACCENT    = "#388bfd"


class FishCBIRApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("He CSDL CBIR Tim Kiem Anh Ca  |  HOG + HSV + LBP + Hu Moments")
        self.root.geometry("1500x960")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        self.query_image_path = None
        self.img_refs = []          # Giu reference anh, tranh GC
        self.searcher = None        # Khoi tao lazy sau khi co CSDL

        self._setup_ui()
        self._try_load_searcher()   # Thu tai CSDL neu da co san

    # ═══════════════════════════════════════════════════
    # 1. Xay dung giao dien
    # ═══════════════════════════════════════════════════

    def _setup_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg="#010409", height=48)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="  He CSDL Tim Kiem Anh Ca  |  CBIR (HOG + HSV + LBP + Hu)",
            bg="#010409", fg=FG_BLUE,
            font=("Consolas", 13, "bold")
        ).pack(side=tk.LEFT, padx=16, pady=10)

        # Trang thai CSDL (ben phai header)
        self.lbl_db_status = tk.Label(
            header, text="CSDL: Chua san sang",
            bg="#010409", fg=RED_BTN,
            font=("Consolas", 10, "bold")
        )
        self.lbl_db_status.pack(side=tk.RIGHT, padx=20)

        # ── FOOTER: Log ──
        footer = tk.LabelFrame(
            self.root, text=" Log he thong ",
            bg=BG_DARK, fg=FG_GRAY,
            font=("Consolas", 9), bd=1
        )
        footer.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 6))
        self.log_text = scrolledtext.ScrolledText(
            footer, height=4, bg="#0d1117", fg=FG_GREEN,
            font=("Consolas", 8), state=tk.DISABLED,
            insertbackground=FG_GREEN
        )
        self.log_text.pack(fill=tk.X, padx=4, pady=2)

        # ── BOTTOM: Top 5 ──
        bot = tk.LabelFrame(
            self.root,
            text=" ④ Ket qua truy xuat: Top 5 anh tuong dong nhat ",
            bg=BG_PANEL, fg=FG_YELLOW,
            font=("Consolas", 10, "bold"),
            bd=1, relief=tk.GROOVE
        )
        bot.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 4))
        self._build_results(bot)

        # ── Main area ──
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill=tk.BOTH, expand=True, side=tk.TOP, padx=8, pady=4)

        # ── COT 1: Dieu khien ──
        col1 = tk.LabelFrame(
            main, text=" ① Dieu khien & Vector ",
            bg=BG_PANEL, fg=FG_BLUE,
            font=("Consolas", 10, "bold"),
            bd=1, relief=tk.GROOVE
        )
        col1.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
        self._build_col1(col1)

        # ── COT 2: Buoc trung gian ──
        col2 = tk.LabelFrame(
            main, text=" ② Cac buoc trich xuat dac trung ",
            bg=BG_PANEL, fg=FG_BLUE,
            font=("Consolas", 10, "bold"),
            bd=1, relief=tk.GROOVE
        )
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._build_col2(col2)

        # ── COT 3: Histogram ──
        col3 = tk.LabelFrame(
            main, text=" ③ Dac trung Mau sac & Ket cau ",
            bg=BG_PANEL, fg=FG_BLUE,
            font=("Consolas", 10, "bold"),
            bd=1, relief=tk.GROOVE
        )
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.hist_frame = tk.Frame(col3, bg=BG_PANEL)
        self.hist_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _build_col1(self, parent):
        """Cot 1: Nut bam, anh query, metric, thong tin vector."""
        # Nut Chon anh
        tk.Button(
            parent,
            text="  Chon Anh Ca  ",
            command=self._load_image,
            bg=BG_BTN2, fg=FG_WHITE,
            font=("Consolas", 10, "bold"),
            relief=tk.FLAT, cursor="hand2",
            activebackground="#388bfd",
            padx=8, pady=6, width=24
        ).pack(pady=(12, 4), padx=10)

        # Khung hien thi anh query
        query_frame = tk.Frame(parent, bg=BG_CARD, bd=1, relief=tk.SUNKEN)
        query_frame.pack(pady=4, padx=10)
        self.lbl_query = tk.Label(
            query_frame, bg=BG_CARD,
            width=200, height=180,
            text="[ Chua co anh ]",
            fg=FG_GRAY, font=("Consolas", 9)
        )
        self.lbl_query.pack()

        # Nut Tim kiem
        tk.Button(
            parent,
            text="  Tim Kiem Top 5  ",
            command=self._search,
            bg=RED_BTN, fg=FG_WHITE,
            font=("Consolas", 10, "bold"),
            relief=tk.FLAT, cursor="hand2",
            activebackground="#b62324",
            padx=8, pady=6, width=24
        ).pack(pady=4, padx=10)

        # Chon do do
        metric_frame = tk.Frame(parent, bg=BG_PANEL)
        metric_frame.pack(pady=4)
        tk.Label(
            metric_frame, text="Do do tuong dong:",
            bg=BG_PANEL, fg=FG_GRAY, font=("Consolas", 9)
        ).pack(anchor=tk.W, padx=6)
        self.metric_var = tk.StringVar(value="cosine")
        for m, desc in [("cosine", "Cosine (khuyen dung)"),
                         ("euclidean", "Euclidean"),
                         ("histogram", "Histogram Intersection")]:
            tk.Radiobutton(
                metric_frame, text=desc, variable=self.metric_var, value=m,
                bg=BG_PANEL, fg=FG_WHITE, selectcolor=BG_DARK,
                activebackground=BG_PANEL,
                font=("Consolas", 8)
            ).pack(anchor=tk.W, padx=10)

        # Thong tin vector
        tk.Label(
            parent, text="Vector dac trung:",
            bg=BG_PANEL, fg=FG_GRAY, font=("Consolas", 9)
        ).pack(anchor=tk.W, padx=14, pady=(8, 0))
        self.lbl_vec = tk.Label(
            parent,
            text="  (Chua co du lieu)  ",
            bg=BG_DARK, fg=FG_GREEN,
            font=("Consolas", 8),
            justify=tk.LEFT, anchor='nw',
            width=32, height=10,
            relief=tk.RIDGE, wraplength=240
        )
        self.lbl_vec.pack(padx=10, pady=4)

        # Thong tin CSDL
        self.lbl_db_info = tk.Label(
            parent,
            text="  CSDL: Chua khoi tao  ",
            bg=BG_DARK, fg=FG_YELLOW,
            font=("Consolas", 8),
            justify=tk.LEFT, anchor='nw',
            width=32, height=4,
            relief=tk.RIDGE, wraplength=240
        )
        self.lbl_db_info.pack(padx=10, pady=4)

    def _build_col2(self, parent):
        """Cot 2: Luoi 2x3 hien thi buoc trung gian."""
        steps = [
            ("Anh Xam\n(Grayscale)", 0, 0, "gray"),
            ("Khong gian HSV",       0, 1, "color"),
            ("Kenh H (Hue)",         0, 2, "color"),
            ("Dac trung HOG",        1, 0, "gray"),
            ("Ket cau LBP",          1, 1, "gray"),
        ]
        self._step_labels = {}
        for title, row, col, mode in steps:
            frm = tk.Frame(parent, bg=BG_PANEL)
            frm.grid(row=row, column=col, padx=8, pady=6, sticky='n')
            tk.Label(
                frm, text=title,
                bg=BG_PANEL, fg=FG_GRAY,
                font=("Consolas", 8)
            ).pack()
            lbl = tk.Label(frm, bg=BG_DARK, width=140, height=120)
            lbl.pack()
            self._step_labels[title] = lbl

        # Tro tat ca cac label vao bien de tien truy cap
        step_list = list(self._step_labels.values())
        self.lbl_gray = step_list[0]
        self.lbl_hsv  = step_list[1]
        self.lbl_hue  = step_list[2]
        self.lbl_hog  = step_list[3]
        self.lbl_lbp  = step_list[4]

    def _build_results(self, parent):
        """Phan Bottom: 5 khung ket qua."""
        self.result_frames = []
        for i in range(5):
            color = FG_YELLOW if i == 0 else FG_WHITE
            frm = tk.Frame(parent, bg=BG_CARD, bd=1, relief=tk.GROOVE)
            frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=6)

            # Rank badge
            rank_lbl = tk.Label(
                frm,
                text=f"# {i+1}" if i > 0 else "  TOP 1  ",
                bg=BG_CARD if i > 0 else FG_YELLOW,
                fg=BG_DARK if i == 0 else FG_GRAY,
                font=("Consolas", 9, "bold")
            )
            rank_lbl.pack(fill=tk.X, pady=(4, 0))

            lbl_img = tk.Label(frm, bg=BG_DARK)
            lbl_img.pack(pady=4)

            lbl_info = tk.Label(
                frm,
                text=f"Top {i+1}",
                bg=BG_CARD, fg=color,
                font=("Consolas", 8),
                justify=tk.CENTER
            )
            lbl_info.pack(pady=(0, 6))
            self.result_frames.append((lbl_img, lbl_info))

    # ═══════════════════════════════════════════════════
    # 2. Logic chinh
    # ═══════════════════════════════════════════════════

    def _try_load_searcher(self):
        """Thu tai FishSearcher neu features.pkl da ton tai."""
        feat_path = os.path.join(BASE_DIR, 'database', 'features.pkl')
        if os.path.exists(feat_path):
            try:
                self.searcher = FishSearcher()
                n = self.searcher.total_images
                mode = self.searcher.index_mode
                self.lbl_db_status.config(
                    text=f"CSDL: {n} anh  [{mode}]",
                    fg=FG_GREEN
                )
                self.lbl_db_info.config(
                    text=f"  Tong anh: {n}\n  Mode: {mode}\n  San sang!"
                )
                self._log(f"[OK] Da tai CSDL: {n} anh, mode={mode}")
            except Exception as e:
                self._log(f"[ERR] Loi tai CSDL: {e}")


    def _load_image(self):
        """Mo hop thoai chon anh va hien thi."""
        path = filedialog.askopenfilename(
            title="Chon anh ca",
            filetypes=[("Anh", "*.png;*.jpg;*.jpeg"), ("Tat ca", "*.*")]
        )
        if not path:
            return
        self.query_image_path = path
        img = read_image(path)
        self._show_img(self.lbl_query, img, width=200, height=180)
        self.lbl_vec.config(text="  (Dang cho tim kiem...)")
        self._clear_results()
        self._log(f"Da chon anh: {os.path.basename(path)}")

    def _search(self):
        """Pipeline tim kiem: trich xuat -> hien thi buoc trung gian -> Top 5."""
        if not self.query_image_path:
            messagebox.showwarning("Canh bao", "Vui long chon anh truoc!")
            return
        if not self.searcher:
            messagebox.showerror(
                "Loi",
                "CSDL chua san sang.\n"
                "Hay nhan 'Tao CSDL' truoc khi tim kiem."
            )
            return

        self._log(f"Bat dau tim kiem: {os.path.basename(self.query_image_path)}")
        try:
            (img_rgb, img_gray, img_hsv,
             hog_img, lbp_img,
             hog_vec, hsv_vec, lbp_vec, hu_vec, cm_vec, fvec) = get_image_features_visual(self.query_image_path)
        except Exception as e:
            messagebox.showerror("Loi", f"Khong the trich xuat dac trung:\n{e}")
            return

        # Hien thi anh trung gian
        self._show_img(self.lbl_gray, img_gray, width=140, height=120, gray=True)
        self._show_img(self.lbl_hsv,  img_hsv,  width=140, height=120)
        hue_ch     = img_hsv[:, :, 0]
        hue_color  = cv2.applyColorMap(hue_ch, cv2.COLORMAP_HSV)
        hue_rgb    = cv2.cvtColor(hue_color, cv2.COLOR_BGR2RGB)
        self._show_img(self.lbl_hue, hue_rgb, width=140, height=120)
        self._show_img(self.lbl_hog, hog_img, width=140, height=120, gray=True)
        self._show_img(self.lbl_lbp, lbp_img, width=140, height=120, gray=True)

        # Ve histogram
        self._draw_histograms(img_hsv, lbp_img)

        # Hien thi thong tin vector
        n_total = fvec.shape[0]
        hog_off = hog_vec.shape[0]
        hsv_off = hog_off + hsv_vec.shape[0]
        lbp_off = hsv_off + lbp_vec.shape[0]
        hu_off  = lbp_off + 7
        info = (
            f"  Tong chieu  : {n_total}\n"
            f"  HOG  (a=.35): {hog_vec.shape[0]}\n"
            f"  HSV  (b=.25): {hsv_vec.shape[0]}\n"
            f"  LBP  (g=.20): {lbp_vec.shape[0]}\n"
            f"  Hu   (d=.10): 7\n"
            f"  CM   (e=.10): 9\n\n"
            f"  HOG[0]: {fvec[0]:.5f}\n"
            f"  HSV[0]: {fvec[hog_off]:.5f}\n"
            f"  LBP[0]: {fvec[hsv_off]:.5f}\n"
            f"  CM[0] : {fvec[hu_off]:.5f}"
        )
        self.lbl_vec.config(text=info)

        # Tim kiem Top 5
        metric  = self.metric_var.get()
        nprobe  = 5 if self.searcher.use_ivf else 0
        results = self.searcher.search(fvec, k=5, metric=metric, nprobe=nprobe)
        self._display_results(results)
        self._log(f"Hoan thanh! Do do: {metric}  |  Top1: {results[0]['species']} ({results[0]['similarity']:.1f}%)")

    # ═══════════════════════════════════════════════════
    # 3. Hien thi
    # ═══════════════════════════════════════════════════

    def _show_img(self, label: tk.Label, arr: np.ndarray,
                  width: int = 128, height: int = 128, gray: bool = False):
        """Resize numpy array va hien len Label."""
        arr_r = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
        if gray:
            img_pil = Image.fromarray(arr_r, 'L')
        else:
            img_pil = Image.fromarray(arr_r)
        img_tk = ImageTk.PhotoImage(img_pil)
        label.config(image=img_tk, width=width, height=height, text="")
        label.image = img_tk  # Giữ tham chiếu để tránh Garbage Collection

    def _draw_histograms(self, img_hsv: np.ndarray, lbp_img: np.ndarray):
        """Ve Hue histogram va LBP histogram."""
        plt.close('all') # Đóng figure cũ để giải phóng bộ nhớ
        for w in self.hist_frame.winfo_children():
            w.destroy()

        fig, axes = plt.subplots(2, 1, figsize=(4.2, 4.2),
                                 dpi=95, facecolor=BG_PANEL)
        ax1, ax2 = axes
        for ax in axes:
            ax.set_facecolor(BG_DARK)
            ax.tick_params(colors=FG_GRAY, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

        # HSV Hue histogram (kenh H)
        h_hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 180]).flatten()
        h_hist = h_hist / (h_hist.sum() + 1e-7)
        colors_map = [plt.cm.hsv(i / 180) for i in range(180)]
        ax1.bar(range(180), h_hist, color=colors_map, width=1.0)
        ax1.set_title("HSV Hue Histogram — Mau sac ca", color=FG_WHITE, fontsize=8, pad=4)
        ax1.set_xlabel("Gia tri Hue [0..179]", color=FG_GRAY, fontsize=7)

        # LBP histogram (ket cau)
        lbp_flat = lbp_img.ravel().astype(float)
        lbp_hist, _ = np.histogram(lbp_flat, bins=64, range=(0, 256))
        lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-7)
        ax2.bar(range(64), lbp_hist, color=FG_BLUE, width=1.0, alpha=0.85)
        ax2.set_title("LBP Texture Histogram — Ket cau", color=FG_WHITE, fontsize=8, pad=4)
        ax2.set_xlabel("LBP bins", color=FG_GRAY, fontsize=7)

        fig.tight_layout(pad=1.4)
        canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def _display_results(self, results: list):
        """Hien thi Top 5 ket qua len cac khung Bottom."""
        for i, (res, (lbl_img, lbl_info)) in enumerate(zip(results, self.result_frames)):
            path = res.get('file_path', '')
            try:
                arr = read_image(path)
                self._show_img(lbl_img, arr, width=240, height=160)
                species_display = res['species'].replace('_', ' ').title()
                color = FG_YELLOW if i == 0 else FG_WHITE
                font  = ("Consolas", 8, "bold") if i == 0 else ("Consolas", 8)
                text  = (
                    f"{species_display}\n"
                    f"Tuong dong: {res['similarity']:.1f}%\n"
                    f"Khoang cach: {res['distance']:.5f}"
                )
                lbl_info.config(text=text, fg=color, font=font)
            except Exception:
                lbl_info.config(text=f"Loi tai anh", fg=RED_BTN)

    def _clear_results(self):
        """Xoa hien thi cu."""
        # Tạo ảnh trống 1x1 để giữ chế độ hiển thị pixel (không làm hỏng layout)
        blank_img = Image.new('RGB', (1, 1), (0, 0, 0))
        blank_tk = ImageTk.PhotoImage(blank_img)
        
        for lbl in [self.lbl_gray, self.lbl_hsv, self.lbl_hue, self.lbl_hog, self.lbl_lbp]:
            lbl.config(image=blank_tk, text="")
            lbl.image = blank_tk
            
        for lbl_img, lbl_info in self.result_frames:
            lbl_img.config(image=blank_tk, text="")
            lbl_img.image = blank_tk
            lbl_info.config(text="")
            
        for w in self.hist_frame.winfo_children():
            w.destroy()

    def _log(self, msg: str):
        """Ghi log xuong footer ScrolledText."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"> {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        print(msg)


# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = FishCBIRApp(root)
    root.mainloop()
