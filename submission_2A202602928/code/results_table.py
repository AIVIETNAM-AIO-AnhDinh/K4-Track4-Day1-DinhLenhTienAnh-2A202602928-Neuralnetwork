"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
import math
from copy import copy
from pathlib import Path

FORMULA_COLS = {"step0_gap_vs_lnC", "gap_val_minus_train", "delta_val_f1_vs_base", "beyond_noise"}
SUMMARY_COLS = ("step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB")
# giá trị hiển thị theo danh sách chọn (data validation) của sheet Experiments
LOSS_NAMES = {"ce": "CE", "mse": "MSE"}
OPT_NAMES = {"sgd": "SGD", "sgd_momentum": "SGD+momentum", "adam": "Adam", "adamw": "AdamW"}


def _jsonable(obj):
    """tuple -> list, NaN/inf (lần chạy phân kỳ) -> None, để file là JSON chuẩn."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    exp_id = result["cfg"]["exp_id"]
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{exp_id}.json"
    payload = {k: result[k] for k in ("cfg", "history", "summary")}   # best_state không ghi (trọng số)
    path.write_text(json.dumps(_jsonable(payload), indent=1, ensure_ascii=False, allow_nan=False))
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    return sorted((json.loads(p.read_text()) for p in Path(results_dir).glob("*.json")),
                  key=lambda r: r["cfg"]["exp_id"])


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    cfg, s = result["cfg"], result["summary"]
    exp_id = cfg["exp_id"]
    row = dict(
        exp_id=exp_id, group=cfg["group"], description=cfg["description"],
        loss=LOSS_NAMES[cfg["loss"]], optimizer=OPT_NAMES[cfg["optimizer"]],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch=cfg["batch"], epochs=cfg["epochs"],
        hidden="-".join(str(h) for h in cfg["hidden"]), dropout=cfg["dropout"],
        clip_norm="none" if cfg["clip_norm"] is None else cfg["clip_norm"],
        precision=cfg["precision"], init=cfg["init"], seed=cfg["seed"],
        **{k: s.get(k) for k in SUMMARY_COLS},
        diverged="Y" if s["diverged"] else "N",
        eval_acc=eval_scores["accuracy"] if eval_scores else None,
        eval_macro_f1=eval_scores["macro_f1"] if eval_scores else None,
        figure_file=f"figures/{exp_id}.png", notes=notes,
    )
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str,
               seed_ids: list[str] | None = None, summary_notes: dict | None = None) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    seed_ids     : exp_id các lần chạy baseline khác seed -> cột A của sheet Seeds (tối đa 5 ô A2:A6)
    summary_notes: {group: nhận xét} -> cột "nhận xét ngắn" của sheet Summary
    """
    import openpyxl

    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]
    header = [c.value for c in ws[1]]
    capacity = ws.max_row - 1                               # mẫu có sẵn công thức cho 60 dòng
    assert len(rows) <= capacity, f"{len(rows)} dòng > {capacity} dòng có công thức trong mẫu"
    for i, row in enumerate(rows):
        r = i + 2
        unknown = set(row) - set(header)
        assert not unknown, f"khoá không có trong tiêu đề: {unknown}"
        for col, name in enumerate(header, start=1):
            if name in FORMULA_COLS:
                continue
            cell = ws.cell(r, col)
            cell.value = row.get(name)
            if r == 2:                                      # dòng 2 của mẫu tô chữ xanh (giá trị mẫu) -> đưa về kiểu thường
                cell.font = copy(ws.cell(3, col).font)

    if seed_ids is not None:
        ws_seeds = wb["Seeds"]
        assert len(seed_ids) <= 5, "sheet Seeds chỉ có 5 ô (A2:A6)"
        for r in range(2, 7):
            ws_seeds.cell(r, 1).value = seed_ids[r - 2] if r - 2 < len(seed_ids) else None

    if summary_notes:
        ws_sum = wb["Summary"]
        note_col = [c.value for c in ws_sum[1]].index("nhận xét ngắn (bạn viết)") + 1
        for r in range(2, ws_sum.max_row + 1):
            group = ws_sum.cell(r, 1).value
            if group in summary_notes:
                ws_sum.cell(r, note_col).value = summary_notes[group]
    wb.save(out_path)
