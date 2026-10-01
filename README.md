# Lab Day 1 — Xây dựng mạng nơ-ron và thí nghiệm huấn luyện

**Track 4 · Ngày 1 · VinUniversity AICB 2026**
Bài học liên quan: *Mạng Nơ-ron và Huấn Luyện* (`day01-mang-no-ron-va-huan-luyen-gioithieu.pdf`).

> Câu hỏi của bài học: *"Một mạng có loss không giảm sau 2 000 bước huấn luyện. Lỗi nằm ở dữ liệu, ở kiến trúc, hay ở vòng lặp huấn luyện?"*
> Sau lab này bạn phải tự trả lời được câu hỏi đó **bằng số liệu do chính bạn đo**.

---

## 1. Bạn sẽ làm gì

Bạn **tự viết toàn bộ code** (repo này **không** có notebook mẫu hay đáp án, chỉ có hướng dẫn, thang điểm và hai file mẫu để nộp bài):

1. **Xây dựng model** mạng nơ-ron (MLP) bằng PyTorch theo shape quy định ở mục 3.
2. **Huấn luyện** model trên một bài toán thật (Forest CoverType) và theo dõi đường cong huấn luyện.
3. **Thử nghiệm** các yếu tố ảnh hưởng đến huấn luyện: hàm mất mát, bộ tối ưu hoá, hyper-parameter, dropout, gradient clipping, mixed precision, cách khởi tạo tham số. **Bạn tự chọn thử nghiệm nào và bao nhiêu thử nghiệm** (xem GUIDE, Part 3).
4. **Tổng hợp**: bảng so sánh `.xlsx`, ảnh biểu đồ của từng thí nghiệm, báo cáo kết luận, toàn bộ code.

| Phần | Nội dung | Gợi ý |
|---|---|---|
| Part 0 | Môi trường, tải và chia dữ liệu | trên lớp |
| Part 1 | Định nghĩa model, kiểm tra shape và "sức khoẻ" ban đầu | trên lớp |
| Part 2 | Pipeline huấn luyện, baseline, đường cong, độ nhiễu | trên lớp |
| Part 3 | Thí nghiệm (menu gợi ý, tự chọn) | ở nhà |
| Part 4 | Bảng xlsx, ảnh, báo cáo, đóng gói code | ở nhà, nộp trước Day 2 |

## 2. Bài toán và dữ liệu

**Forest CoverType** (Blackard & Dean, UCI): dự đoán loại rừng (7 lớp) từ dữ liệu địa hình, cho các ô đất 30m × 30m ở Roosevelt National Forest, Colorado.

- 581 012 mẫu, 54 đặc trưng: 10 đặc trưng số liên tục (độ cao, độ dốc, khoảng cách đến đường, ...), 4 cột one-hot "Wilderness_Area", 40 cột one-hot "Soil_Type".
- Nhãn `1..7` (bạn phải đổi về `0..6`).
- **Mất cân bằng lớp:** lớp 2 chiếm 48,8%, lớp 1 chiếm 36,5%, còn 5 lớp kia gộp lại chưa đến 15%. Vì thế luôn báo cáo **accuracy lẫn macro-F1**, và biết rằng "đoán luôn lớp 2" đã được 48,8% accuracy.
- Tải bằng `sklearn.datasets.fetch_covtype` (khoảng 11 MB, cần bật Internet).

## 3. Kiến trúc mô hình (cố định)

Đầu vào `x`: `(B, 54)` `float32`. Nhãn `y`: `(B,)` `int64` (`0..6`). Đầu ra: **logits `(B, 7)`** (chưa softmax).

| Tên | Dùng ở | Các lớp | Số tham số |
|---|---|---|---|
| `M-base` | **Bắt buộc**: baseline | `54 → 256 → 128 → 7` | 47 879 |
| `M-wide` | Tuỳ chọn: thí nghiệm độ rộng | `54 → 512 → 256 → 7` | 161 287 |
| `M-deep` | Tuỳ chọn: thí nghiệm độ sâu | `54 → 256 → 128 → 64 → 7` | 55 687 |

ReLU ở mọi lớp ẩn, có bias, dropout (nếu dùng) chỉ đặt sau ReLU của lớp ẩn, không BatchNorm hay residual. Chi tiết trong [`GUIDE.md`](GUIDE.md), mục *Quy định kiến trúc mô hình*.

## 4. Chạy ở đâu

Khuyến nghị dùng GPU miễn phí:

- **Google Colab:** *Runtime → Change runtime type → T4 GPU*. Nên mount Google Drive để lưu kết quả (Colab có thể ngắt kết nối giữa chừng).
- **Kaggle Notebook:** *Settings → Accelerator → GPU*, và **bật Internet** trong Settings (nếu không `fetch_covtype` sẽ không tải được).
- CPU vẫn chạy được nhưng chậm hơn nhiều khi thử nhiều cấu hình.

Phần mềm: Python 3.10+, PyTorch ≥ 2.x, scikit-learn, numpy, pandas, matplotlib, openpyxl. Colab/Kaggle đã cài sẵn.

## 5. Luật chơi

**Bạn phải tự viết:** model (class `nn.Module` do bạn định nghĩa, đúng shape ở mục 3), pipeline huấn luyện và đánh giá, ghi log, vẽ biểu đồ, lập bảng.

**Được dùng thoải mái:** mọi thành phần của PyTorch: `nn.Linear`, `nn.ReLU`, `nn.Dropout`, `nn.Sequential`, `nn.init.*`, `nn.CrossEntropyLoss`, `nn.MSELoss`, `torch.optim.*`, autograd, `clip_grad_norm_`, `torch.autocast`, `GradScaler`; cùng `numpy`, `sklearn` (tải/chia dữ liệu, tính metric), `matplotlib`.

**Không được dùng:** mô hình dựng sẵn hoặc pretrained (`torchvision`/`timm`), `sklearn.neural_network.MLPClassifier`, hay copy mạng làm sẵn thay cho model do bạn định nghĩa.

**Công bằng khi so sánh:**
1. Mọi thí nghiệm dùng **cùng phép chia dữ liệu** (seed 42) và **cùng số epoch** (khuyến nghị 20), trừ yếu tố bạn đang thử.
2. **Mỗi lần chỉ đổi một yếu tố** so với baseline (hoặc ghi rõ khi bạn đổi nhiều yếu tố).
3. **Không nhìn tập test để chọn cấu hình.** Chọn bằng tập validation. Test chỉ dùng ở cuối, cho baseline và cấu hình cuối cùng của bạn.
4. Mọi con số trong báo cáo phải truy được về một dòng trong bảng `.xlsx`. Mọi kết luận "A tốt hơn B" phải cân nhắc **độ nhiễu giữa các seed** (chạy baseline vài seed để đo), hoặc nêu rõ đây là hạn chế.

## 6. Sản phẩm phải nộp

Nộp **một thư mục `submission_<MSSV>/`** (thay `<MSSV>` bằng mã số sinh viên của bạn), nén thành `submission_<MSSV>.zip` nếu nộp qua LMS, theo kênh giảng viên thông báo.

### 6.1 Cây thư mục

```
submission_<MSSV>/
├── REPORT.md
├── experiments.xlsx
├── figures/
│   ├── <exp_id>.png            (một ảnh cho MỖI thí nghiệm trong bảng)
│   └── compare_<nhóm>.png      (nên có, mỗi nhóm thí nghiệm một ảnh)
├── results/                    (khuyến nghị, không bắt buộc)
│   └── <exp_id>.json
└── code/                       (TOÀN BỘ code nằm ở đây)
    ├── lab.ipynb
    ├── *.py                    (tuỳ chọn)
    └── requirements.txt        (tuỳ chọn)
```

### 6.2 Chi tiết từng file / thư mục

| Đường dẫn | Bắt buộc? | Nội dung và yêu cầu |
|---|---|---|
| `REPORT.md` | **Bắt buộc** | Báo cáo kết luận, viết theo [`templates/REPORT_TEMPLATE.md`](templates/REPORT_TEMPLATE.md), khoảng 4 trang. Gồm: thiết lập, kiểm tra ban đầu và độ nhiễu, kết quả theo từng chủ đề đã thử (dự đoán, số liệu trỏ về `exp_id`, ảnh, giải thích cơ chế), cấu hình cuối cùng (nếu có), trả lời câu hỏi dẫn dắt, hạn chế. Chèn ảnh bằng đường dẫn tương đối, ví dụ `![](figures/compare_optimizer.png)`. Chỉ viết cho các chủ đề bạn đã thử. |
| `experiments.xlsx` | **Bắt buộc** | Bảng so sánh, tạo từ [`templates/experiment_table_template.xlsx`](templates/experiment_table_template.xlsx). Giữ nguyên 4 sheet `Legend`, `Experiments`, `Seeds`, `Summary` và không đổi tên cột. **Mỗi thí nghiệm đã chạy là một dòng** trong sheet `Experiments`, với `exp_id` duy nhất. Điền đủ cấu hình và kết quả (hoặc ghi lý do thiếu vào `notes`). Cột `test_acc` / `test_macro_f1` chỉ điền cho baseline và cấu hình cuối cùng. Sheet `Seeds` ghi các lần chạy baseline khác seed. Không để ô công thức lỗi. |
| `figures/` | **Bắt buộc** | Thư mục chứa ảnh biểu đồ, định dạng `.png`. |
| `figures/<exp_id>.png` | **Bắt buộc, một ảnh cho mỗi dòng của bảng** | Tên file **trùng đúng** `exp_id` trong bảng (ví dụ `base-s1.png`, `opt-adam-lr1e-3.png`). Mỗi ảnh có ít nhất 3 ô: (1) train loss và val loss theo epoch, (2) val accuracy (nên có thêm macro-F1), (3) `grad_norm` (đo trước khi clip). Có tiêu đề ghi `exp_id` và cấu hình, nhãn trục, chú thích. Là ảnh lưu từ `plt.savefig` hoặc ảnh chụp màn hình TensorBoard/W&B đều được. Tên file ghi vào cột `figure_file` của bảng. Số ảnh phải bằng số dòng của bảng. |
| `figures/compare_<nhóm>.png` | Nên có | Ảnh chồng các đường của nhiều thí nghiệm cùng một nhóm (ví dụ `compare_optimizer.png`, `compare_dropout.png`) để so sánh trực tiếp. Dùng làm bằng chứng trong báo cáo. |
| `results/` và `results/<exp_id>.json` | Khuyến nghị | Lịch sử của từng lần chạy (mỗi epoch: train/val loss, val acc, macro-F1, grad_norm, thời gian; cùng cấu hình `cfg`). Giúp tạo lại bảng và ảnh mà không cần huấn luyện lại. Một file cho mỗi `exp_id`. |
| `code/` | **Bắt buộc** | **Toàn bộ code của bạn nằm trong thư mục này**, không để file code ở nơi khác. |
| `code/lab.ipynb` | **Bắt buộc** | Notebook duy nhất chạy được từ đầu đến cuối (*Restart & Run All*) trên Colab hoặc Kaggle, **giữ nguyên output** của các ô. Gồm theo thứ tự: tải và chia dữ liệu; định nghĩa model (có `assert` số tham số); các phép thử ban đầu (loss bước 0, quá khớp 20 mẫu, gradient chảy); `run_experiment(cfg)`; baseline; các thí nghiệm bạn chọn (mỗi thí nghiệm có dự đoán trước và nhận xét sau); đoạn code tạo bảng và ảnh. Đường dẫn lưu kết quả phải đúng khi chạy từ trong `code/`, tức là ghi vào `../figures/` và `../results/`. Đặt seed cố định. |
| `code/*.py` | Tuỳ chọn | Nếu bạn tách code ra module (ví dụ `model.py`, `train.py`, `utils.py`), đặt tất cả trong `code/` và `lab.ipynb` import từ đó. Không bắt buộc, nhưng nếu có thì notebook vẫn phải chạy được khi chỉ có thư mục nộp. |
| `code/requirements.txt` | Tuỳ chọn | Liệt kê thư viện và phiên bản nếu bạn dùng thư viện ngoài những cái có sẵn trên Colab/Kaggle. |

### 6.3 Không nộp

- Dữ liệu CoverType (tập này tải lại được bằng `fetch_covtype`) và thư mục cache của sklearn.
- File trọng số mô hình (`.pt`, `.pth`, `.ckpt`).
- `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`.
- Bản sao slide, file PDF, hay ảnh nằm ngoài `figures/`.
- Code đặt ngoài thư mục `code/`.

### 6.4 Kiểm tra nhanh trước khi nộp

- [ ] Tên thư mục là `submission_<MSSV>` và có đủ `REPORT.md`, `experiments.xlsx`, `figures/`, `code/lab.ipynb`.
- [ ] Số ảnh `figures/<exp_id>.png` bằng số dòng thí nghiệm trong `experiments.xlsx`, và tên ảnh trùng `exp_id`.
- [ ] Mọi file `.py` và notebook đều nằm trong `code/`.
- [ ] Mở `code/lab.ipynb` trên Colab/Kaggle, chọn *Restart & Run All* không lỗi, output còn nguyên.
- [ ] Mọi con số trong `REPORT.md` tìm lại được trong `experiments.xlsx`.

## 7. Các file trong thư mục này

| File | Dùng để làm gì |
|---|---|
| [`GUIDE.md`](GUIDE.md) | Hướng dẫn từng bước, menu thí nghiệm, checklist tự kiểm tra |
| [`RUBRIC.md`](RUBRIC.md) | Thang điểm chi tiết (đọc trước khi bắt đầu) |
| [`templates/experiment_table_template.xlsx`](templates/experiment_table_template.xlsx) | Bảng so sánh có sẵn tên cột và công thức |
| [`templates/REPORT_TEMPLATE.md`](templates/REPORT_TEMPLATE.md) | Khung báo cáo |

## 8. Lấy bài lab về

```bash
git pull          # nếu đã clone repo của khoá học
```

Trên Colab, bạn có thể `git clone` repo về rồi mở/viết notebook của riêng bạn trong thư mục `submission_<MSSV>/`.

## 9. Tài liệu tham khảo

Slide Day 1 (đặc biệt Chương 3, 4, 5); Karpathy, *A Recipe for Training Neural Networks*; CS231n, *Neural Networks Part 1–3*; Kingma & Ba (2015) *Adam*; Loshchilov & Hutter (2019) *AdamW*; He et al. (2015) *khởi tạo He*; Srivastava et al. (2014) *Dropout*; Micikevicius et al. (2018) *Mixed Precision Training*.

Nếu kẹt, hãy bắt đầu với **mục "Chẩn đoán" ở cuối GUIDE**.
