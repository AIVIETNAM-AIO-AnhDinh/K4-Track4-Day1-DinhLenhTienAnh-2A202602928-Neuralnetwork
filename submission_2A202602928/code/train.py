"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.1,                    # chọn bằng VAL ở Part 2 (quét lr-sgdm-0.003 ... 0.3, val macro-F1 cao nhất), không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
    scheduler=None,            # None | "cosine" (ghi vào notes nếu dùng)
)

N_CLASSES = 7


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # seed CPU, và cả CUDA/MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    zeros = np.zeros_like(tp)
    prec = np.divide(tp, tp + fp, out=zeros.copy(), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=zeros.copy(), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=zeros.copy(), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    total = torch.zeros((), device=X.device)
    cm = torch.zeros(N_CLASSES * N_CLASSES, dtype=torch.int64, device=X.device)
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        total += compute_loss(logits, yb, loss_name, reduction="sum")
        # ô (thật, đoán) của ma trận nhầm lẫn đánh số phẳng thật*7 + đoán
        cm += torch.bincount(yb * N_CLASSES + logits.argmax(dim=1), minlength=N_CLASSES * N_CLASSES)
    cm = cm.reshape(N_CLASSES, N_CLASSES).cpu().numpy()
    return dict(loss=total.item() / len(X), acc=float(np.trace(cm) / cm.sum()),
                macro_f1=macro_f1_from_confusion(cm))


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    # Loss luôn tính ở FP32 (kể cả khi forward chạy trong autocast) để tránh tràn số ở FP16.
    logits = logits.float()
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        # Giống nn.MSELoss(): (logit - one_hot)^2 lấy trung bình trên cả 7 lớp và trên lô, không có hệ số 1/2.
        # Ở đây: trung bình theo 7 lớp cho từng mẫu, rồi mean/sum theo mẫu (để evaluate cộng dồn được).
        target = F.one_hot(y, logits.shape[1]).float()
        per_sample = F.mse_loss(logits, target, reduction="none").mean(dim=1)
        return per_sample.mean() if reduction == "mean" else per_sample.sum()
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


def _sync(device: torch.device) -> None:
    """Chờ GPU chạy xong trước khi đo thời gian (CUDA/MPS chạy bất đồng bộ)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _peak_mem_mb(device: torch.device):
    """Bộ nhớ GPU cực đại (MB) kể từ đầu lần chạy; chỉ có trên CUDA (CPU/MPS trả về None)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return None


def run_experiment(cfg: dict, data: dict, verbose: bool = False) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "grad_norm_max": [...], "clip_frac": [...],
                     "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    assert cfg["lr"] is not None, "chưa chọn cfg['lr'] (chọn bằng val)"
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    hidden = tuple(cfg["hidden"])

    # ---- 0. model, optimizer, (scaler), seed
    set_seed(cfg["seed"])
    model = MLP(hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    n_params = count_params(model)
    if hidden in EXPECTED_PARAMS:          # kiến trúc ngoài bảng quy định (nếu có) thì không có số để so
        assert n_params == EXPECTED_PARAMS[hidden], f"{hidden}: {n_params} != {EXPECTED_PARAMS[hidden]}"
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], total_steps=cfg["epochs"] * steps_per_epoch)
    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg["precision"]]
    scaler = torch.amp.GradScaler(device.type) if cfg["precision"] == "fp16" else None
    gen = torch.Generator().manual_seed(cfg["seed"])   # thứ tự lô chỉ phụ thuộc seed
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- 1. loss bước 0, trước bước cập nhật đầu tiên (kỳ vọng ≈ ln 7)
    step0_loss = evaluate(model, X_val, y_val, cfg["loss"])["loss"]

    keys = ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
            "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")
    hist = {k: [] for k in keys}
    best_val_loss, best_epoch, best_state = math.inf, None, None
    diverged = False

    # ---- 2. vòng huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        _sync(device)
        t0 = time.perf_counter()
        norms = []
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xb)                       # autocast chỉ bọc forward + loss
                loss = compute_loss(logits, yb, cfg["loss"])
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)   # luôn unscale: grad_norm phải đo ở thang thật, và trước khi clip
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt
            if not math.isfinite(loss.item()):
                diverged = True
                break
            if scaler is not None:
                scaler.step(optimizer)       # tự bỏ qua bước nếu gradient có inf/NaN (tràn FP16)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            norms.append(gn)
        _sync(device)
        epoch_time = time.perf_counter() - t0          # chỉ tính phần huấn luyện, không tính đánh giá

        finite = [g for g in norms if math.isfinite(g)]   # FP16: bước tràn số (gn = inf) bị scaler bỏ qua
        hist["epoch"].append(epoch)
        hist["grad_norm"].append(float(np.mean(finite)) if finite else math.nan)
        hist["grad_norm_max"].append(float(np.max(finite)) if finite else math.nan)
        hist["clip_frac"].append(float(np.mean([g > cfg["clip_norm"] for g in finite]))
                                 if finite and cfg["clip_norm"] is not None else 0.0)
        hist["epoch_time_s"].append(epoch_time)
        if diverged:                                    # ghi lại epoch dở dang rồi dừng, không để notebook treo
            for k in ("train_loss", "val_loss", "val_acc", "val_macro_f1"):
                hist[k].append(math.nan)
            break

        # cuối epoch, chế độ eval(): train loss trên TOÀN BỘ train để so được với val loss
        tr = evaluate(model, X_tr, y_tr, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        hist["train_loss"].append(tr["loss"])
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"])
        if not math.isfinite(va["loss"]):
            diverged = True
            break
        if va["loss"] < best_val_loss:                  # dừng sớm theo val loss
            best_val_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  ep {epoch:2d}  train {tr['loss']:.4f}  val {va['loss']:.4f}  acc {va['acc']:.4f}  "
                  f"f1 {va['macro_f1']:.4f}  |g| {hist['grad_norm'][-1]:.3f}  {epoch_time:.1f}s")

    # ---- 3. tóm tắt tại best_epoch
    b = best_epoch - 1 if best_epoch is not None else None
    summary = dict(
        step0_loss=step0_loss,
        best_val_loss=best_val_loss if b is not None else None,
        best_epoch=best_epoch,
        final_train_loss=hist["train_loss"][-1] if hist["train_loss"] else None,
        final_val_loss=hist["val_loss"][-1] if hist["val_loss"] else None,
        val_acc=hist["val_acc"][b] if b is not None else None,
        val_macro_f1=hist["val_macro_f1"][b] if b is not None else None,
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])),
        peak_mem_MB=_peak_mem_mb(device),
        diverged=diverged,
        n_params=n_params,
    )
    print(f"[{cfg['exp_id']}] step0 {step0_loss:.4f} | best ep {best_epoch} val_loss "
          f"{summary['best_val_loss'] if b is not None else float('nan'):.4f} acc "
          f"{summary['val_acc'] if b is not None else float('nan'):.4f} f1 "
          f"{summary['val_macro_f1'] if b is not None else float('nan'):.4f} | "
          f"{summary['time_per_epoch_s']:.2f}s/epoch" + (" | DIVERGED" if diverged else ""))
    return {"cfg": copy.deepcopy(cfg), "history": hist, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id, preds = np.asarray(row_id, dtype=np.int64), np.asarray(preds, dtype=np.int64)
    assert row_id.shape == preds.shape and len(np.unique(row_id)) == len(row_id), "row_id phải duy nhất, cùng độ dài với preds"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1, "pred phải nằm trong 0..6"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savetxt(path, np.column_stack([row_id, preds]), fmt="%d", delimiter=",", header="row_id,pred", comments="")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               repo_root: str | None = None, out_json: str | None = None) -> dict | None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    Nếu truyền repo_root và out_json: tự chạy scripts/evaluate.py (từ repo_root) và trả về nội dung eval_result.json.
    """
    device = data["X_eval"].device
    model = MLP(tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])          # trọng số của epoch có val loss thấp nhất
    preds = predict(model, data["X_eval"])                 # eval mode (không dropout), FP32
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    if repo_root is None or out_json is None:
        return None
    out = subprocess.run([sys.executable, "scripts/evaluate.py", "--pred", os.path.abspath(pred_path),
                          "--out", os.path.abspath(out_json)],
                         cwd=repo_root, capture_output=True, text=True)
    print(out.stdout + out.stderr)
    if out.returncode != 0:
        raise RuntimeError("scripts/evaluate.py báo lỗi, xem thông báo ở trên")
    with open(out_json) as f:
        return json.load(f)
