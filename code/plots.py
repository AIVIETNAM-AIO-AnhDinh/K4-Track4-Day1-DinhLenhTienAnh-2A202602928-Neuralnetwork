"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullFormatter


def _cfg_label(cfg: dict) -> str:
    """Một dòng mô tả cấu hình chính, dùng làm tiêu đề phụ."""
    hidden = "-".join(str(h) for h in cfg["hidden"])
    clip = cfg["clip_norm"] if cfg["clip_norm"] is not None else "none"
    return (f"{cfg['optimizer']} lr={cfg['lr']:g} wd={cfg['weight_decay']:g} | loss={cfg['loss']} | "
            f"batch={cfg['batch']} epochs={cfg['epochs']} | {hidden} drop={cfg['dropout']:g} init={cfg['init']} | "
            f"clip={clip} {cfg['precision']} | seed={cfg['seed']}")


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label="train loss (eval mode)")
    ax.plot(ep, h["val_loss"], "s-", ms=3, label="val loss")
    ax.plot([0], [s["step0_loss"]], "k*", ms=9, label=f"val loss bước 0 = {s['step0_loss']:.3f}")
    ax.set(title="Loss", xlabel="epoch", ylabel=f"loss ({cfg['loss']}, thang log)")
    ax.set_yscale("log")    # để loss bước 0 (≈ 2) không ép các đường cong (≈ 0.2-0.5) xuống đáy
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, label="val accuracy")
    ax.plot(ep, h["val_macro_f1"], "s-", ms=3, label="val macro-F1")
    ax.set(title="Chỉ số trên val", xlabel="epoch", ylabel="giá trị")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, label="trung bình mỗi epoch")
    ax.plot(ep, h["grad_norm_max"], "^--", ms=3, alpha=0.7, label="lớn nhất mỗi epoch")
    if cfg["clip_norm"] is not None:
        ax.axhline(cfg["clip_norm"], color="red", ls=":", label=f"ngưỡng clip c = {cfg['clip_norm']:g}")
    ax.set(title="‖g‖ toàn cục (đo TRƯỚC khi clip)", xlabel="epoch", ylabel="grad norm")
    gmax = np.array(h["grad_norm_max"], dtype=float)
    if np.isfinite(gmax).any() and np.nanmax(gmax) > 20 * np.nanmedian(np.array(h["grad_norm"], dtype=float)):
        ax.set_yscale("log")                          # có gai rất lớn -> thang log để vẫn thấy mức thường
        ax.set_ylabel("grad norm (thang log)")

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if s["best_epoch"] is not None:
            ax.axvline(s["best_epoch"], color="gray", ls="--", lw=1, label=f"best epoch = {s['best_epoch']}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    status = " — DIVERGED" if s["diverged"] else ""
    fig.suptitle(f"{cfg['exp_id']}{status}\n{_cfg_label(cfg)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str | list[str], path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    `metric` có thể là một danh sách (ví dụ ["val_loss", "val_macro_f1"]) -> mỗi chỉ số một ô, cạnh nhau.
    """
    labels = {"train_loss": "train loss (eval mode)", "val_loss": "val loss", "val_acc": "val accuracy",
              "val_macro_f1": "val macro-F1", "grad_norm": "grad norm trung bình (trước clip)"}
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.6), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for r in results:
            ax.plot(r["history"]["epoch"], r["history"][m], "o-", ms=3, label=r["cfg"]["exp_id"])
        ax.set(title=labels.get(m, m), xlabel="epoch", ylabel=labels.get(m, m))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
