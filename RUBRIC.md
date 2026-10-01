# RUBRIC — Lab Day 1 (100 điểm)

Điểm đánh giá **chất lượng thí nghiệm và lập luận**, không đánh giá accuracy cao hay thấp. Một mô hình 70% mà được kiểm chứng và giải thích tốt vẫn hơn một mô hình 90% không ai tin được.

**Số lượng thí nghiệm không bị ép.** Bạn tự chọn chạy bao nhiêu thí nghiệm. Điểm phần thí nghiệm dựa vào *độ phủ chủ đề* (tối đa 7 chủ đề) và *chất lượng thiết kế*, không dựa vào việc chạy đủ một danh sách cố định.

| # | Phần | Điểm |
|---|---|---|
| 1 | Dữ liệu, pipeline và kiểm tra ban đầu | 10 |
| 2 | Model và baseline | 12 |
| 3 | Độ phủ chủ đề thí nghiệm | 14 |
| 4 | Chất lượng thiết kế thí nghiệm | 10 |
| 5 | Báo cáo kết luận | 25 |
| 6 | Bảng `experiments.xlsx` và ảnh biểu đồ | 20 |
| 7 | Chất lượng code và khả năng tái lập | 9 |
| | **Tổng** | **100** |

---

## 1. Dữ liệu, pipeline và kiểm tra ban đầu (10)

| Tiêu chí | Điểm |
|---|---|
| Chia dữ liệu đúng (seed 42, stratified 70/15/15), chuẩn hoá chỉ bằng thống kê train, nhãn `0..6` | 3 |
| `run_experiment(cfg)` ghi đủ: train/val loss (train đo ở eval), val acc, macro-F1, `grad_norm` **trước clip**, loss bước 0, thời gian, bộ nhớ, cờ diverged; cấu hình tách khỏi logic | 4 |
| Hai phép thử sức khoẻ: loss bước 0 ≈ ln 7 (có in và nhận xét); quá khớp được 20 mẫu (có đường cong) | 3 |

## 2. Model và baseline (12)

| Tiêu chí | Điểm |
|---|---|
| Model **tự định nghĩa** đúng shape quy định (`M-base`: logits `(B,7)`, **47 879 tham số**, có `assert`), dropout đặt đúng chỗ (sau ReLU lớp ẩn), softmax không nằm trong model | 4 |
| Khởi tạo đúng với cái bạn ghi trong bảng (không để mặc định `nn.Linear` mà ghi là He); có kiểm tra gradient chảy tới mọi tham số | 2 |
| Baseline đúng cấu hình, đường cong train/val bình thường, vượt mốc "đoán đa số" (48,8%), có mô tả hình dạng đường cong | 3 |
| Đo độ nhiễu seed (≥ 2 seed baseline, báo cáo trung bình ± độ lệch chuẩn); nếu chỉ 1 seed thì phải nêu rõ là hạn chế (tối đa 1/3 điểm mục này) | 3 |

## 3. Độ phủ chủ đề thí nghiệm (14)

7 chủ đề, **mỗi chủ đề tối đa 2 điểm**: hàm mất mát · bộ tối ưu hoá · hyper-parameter · dropout · gradient clipping · mixed precision · khởi tạo tham số.

Một chủ đề được tính đủ 2 điểm khi: có ≥ 1 thí nghiệm chạy xong và so với baseline, có dòng trong bảng, có ảnh riêng, có dự đoán trước và đối chiếu sau. Thí nghiệm chạy nhưng thiếu ảnh/bảng/đối chiếu: 1 điểm. Không có thí nghiệm: 0.

Bạn không cần phủ hết 7 chủ đề để đạt điểm cao ở các phần khác, nhưng mỗi chủ đề bỏ qua là mất 2 điểm ở phần này.

## 4. Chất lượng thiết kế thí nghiệm (10)

Chấm trên các thí nghiệm bạn đã chạy:

| Tiêu chí | Điểm |
|---|---|
| **Công bằng:** mỗi lần chỉ đổi một yếu tố; cùng split, cùng số epoch, cùng seed (hoặc ghi rõ) | 3 |
| **Bộ tối ưu:** nếu so sánh bộ tối ưu, mỗi bộ được thử ≥ 2 lr và so ở lr tốt nhất của nó (nếu không làm chủ đề này, điểm này tính theo mức độ cẩn thận của các so sánh khác) | 2 |
| **Chọn tham số có lý do:** `c` của clipping chọn dựa trên `grad_norm`; kiểm tra clipping ở lr cao; `q` của dropout có liên hệ với mức quá khớp; so loss CE/MSE bằng metric chứ không bằng loss | 2 |
| **Dự đoán trước, đối chiếu sau** cho các thí nghiệm | 2 |
| **Không dùng tập test để chọn cấu hình** (xem thêm mục trừ điểm) | 1 |

## 5. Báo cáo kết luận (25)

| Tiêu chí | Điểm |
|---|---|
| Mọi kết luận có số liệu (trỏ về `exp_id`), ảnh và **giải thích cơ chế** (vì sao xảy ra: gradient, phương sai, số bước cập nhật, ...) | 10 |
| Cân nhắc độ nhiễu seed khi nói "A tốt hơn B"; chênh lệch nhỏ hơn nhiễu được nêu là chưa kết luận được | 5 |
| Trả lời các câu hỏi dẫn dắt trong mẫu cho những chủ đề đã thử, trong đó có câu hỏi quay lại bài học (3 phép kiểm tra đầu tiên khi loss không giảm) | 5 |
| Trung thực: nêu hạn chế, kết quả "không như dự đoán" và giải thích; tách rõ điều đã đo với điều chỉ phỏng đoán | 5 |

## 6. Bảng `experiments.xlsx` và ảnh biểu đồ (20)

| Tiêu chí | Điểm |
|---|---|
| Bảng đầy đủ cột, mỗi thí nghiệm một dòng, giá trị khớp với notebook và báo cáo, không ô công thức lỗi | 8 |
| **Mỗi thí nghiệm có ảnh riêng** `figures/<exp_id>.png` (train/val loss, val acc, grad_norm), đặt đúng tên, đọc được (trục, chú thích, tiêu đề) | 8 |
| Ảnh chồng so sánh theo nhóm (`compare_<nhóm>.png`) và sheet Summary có nhận xét | 4 |

## 7. Chất lượng code và tái lập (9)

| Tiêu chí | Điểm |
|---|---|
| Notebook chạy lại được từ đầu đến cuối (Restart & Run All), seed cố định, có output | 5 |
| Code có cấu trúc (một `run_experiment`, model tách riêng), chú thích chỗ khó | 2 |
| Thư mục nộp đúng cấu trúc README mục 6 (có `results/` là điểm cộng nhỏ, không bắt buộc) | 2 |

---

## Trừ điểm

| Vi phạm | Trừ |
|---|---|
| Không tự định nghĩa model (dùng mô hình dựng sẵn/pretrained/`MLPClassifier`) | Phần 2 = 0 |
| Dùng tập test để chọn cấu hình/lr/epoch | −10 |
| Kết luận so sánh nhưng không nhắc đến nhiễu seed hay hạn chế | −1 đến −3 mỗi chỗ |
| Số trong báo cáo không khớp bảng/ảnh | −1 đến −5 |
| Shape hoặc số tham số model sai quy định mà không giải thích | −2 |
| Nộp trễ | Theo quy định của khoá học |

## Bậc xếp loại gợi ý

| Điểm | Mức |
|---|---|
| ≥ 90 | Xuất sắc: so sánh công bằng, giải thích được cơ chế, phủ rộng chủ đề |
| 75–89 | Tốt: thí nghiệm đủ ý, kết luận có số liệu, còn vài chỗ thiếu cơ chế/nhiễu |
| 60–74 | Đạt: chạy được, nhưng so sánh chưa công bằng hoặc kết luận chung chung |
| < 60 | Chưa đạt: thiếu phần lớn thí nghiệm/bảng/ảnh hoặc model sai quy định |

## Checklist tự chấm trước khi nộp

- [ ] Model đúng shape, `assert` 47 879 tham số
- [ ] Loss bước 0 ≈ 1,946 và quá khớp được 20 mẫu
- [ ] Mỗi thí nghiệm: 1 dòng bảng + 1 ảnh + dự đoán trước + đối chiếu sau
- [ ] Số ảnh = số dòng trong bảng
- [ ] Test chỉ ở baseline và cấu hình cuối cùng
- [ ] Báo cáo: mỗi câu "A hơn B" có số, ảnh, cơ chế và nhắc nhiễu seed
- [ ] Notebook chạy lại từ đầu không lỗi
