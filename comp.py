import torch
from pathlib import Path
import csv
import ast
from matplotlib import pyplot as plt


def load_populations_csv(path, device="cpu"):
    path = Path(path)
    out = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue

            alphas = ast.literal_eval(row[0])
            mask   = ast.literal_eval(row[1])
            fit = float(row[2])
            fit_wmask = float(row[3])

            a = torch.tensor(alphas, dtype=torch.float32, device=device)
            m = torch.tensor(mask,   dtype=torch.float32, device=device)

            genome = a.clamp_min(1e-12)
            mask_t = (m > 0.5).float()

            out.append((genome, mask_t, fit, fit_wmask))
    return out


def plot_weights_mask_scatter(
    gen1: torch.Tensor, m1: torch.Tensor,
    gen2: torch.Tensor, m2: torch.Tensor,
    title: str = "Weights (Y) + mask activity"
):
    gen1 = gen1.detach().cpu().reshape(-1)
    gen2 = gen2.detach().cpu().reshape(-1)
    m1b = (m1.detach().cpu().reshape(-1) > 0.5)
    m2b = (m2.detach().cpu().reshape(-1) > 0.5)

    if gen1.numel() != gen2.numel() or gen1.numel() != m1b.numel() or gen2.numel() != m2b.numel():
        raise ValueError("Rozmiary genów/masek muszą się zgadzać.")

    n = gen1.numel()
    x = torch.arange(n, dtype=torch.float32)

    y1 = gen1.clamp_min(1e-30)
    y2 = gen2.clamp_min(1e-30)

    x1 = x
    x2 = x

    idx1_on  = torch.where(m1b)[0]
    idx1_off = torch.where(~m1b)[0]
    idx2_on  = torch.where(m2b)[0]
    idx2_off = torch.where(~m2b)[0]

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)

    ax.scatter(x1[idx1_on],  y1[idx1_on],  marker="o", label="genome1 active",   color="green")
    ax.scatter(x1[idx1_off], y1[idx1_off], marker="o", label="genome1 inactive", color="red")

    ax.scatter(x2[idx2_on],  y2[idx2_on],  marker="x", label="genome2 active",   color="blue")
    ax.scatter(x2[idx2_off], y2[idx2_off], marker="x", label="genome2 inactive", color="orange")

    for i in range(n):
        ax.plot(
            [float(x1[i]), float(x2[i])],
            [float(y1[i]), float(y2[i])],
            linewidth=0.8,
            alpha=0.5,
            color="black",
        )

    for i in range(n):
        dy = abs(float(y2[i] - y1[i]))


        if abs(y1[i]) < 1e-12:
            rel_pct = float("nan")
            rel_txt = "n/a"
        else:
            rel_pct = y2[i] / y1[i]
            rel_txt = f"{rel_pct:.2f}"

        y_top = float(max(y1[i], y2[i]))

        ax.annotate(
            f"{dy:.2f}\n{rel_txt}",
            xy=(float(x[i]), y_top),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            color="black",
            alpha=0.85,
            clip_on=False,
        )

    diff = (m1b ^ m2b)
    diff_idx = torch.where(diff)[0]
    for i in diff_idx.tolist():
        ax.axvline(x=float(i), linewidth=0.6, alpha=0.2)

    ax.set_xlabel("gene index")
    ax.set_ylabel("value")
    ax.set_title(f"{title} | diffs: {diff.sum().item()}/{n}")
    ax.legend(ncol=2, fontsize=9)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.15 * (ymax - ymin))

    fig.savefig("compare.png", dpi=200)


rows = load_populations_csv("tmp.txt", device="cpu")

(gen1, m1, fit1, fitw1) = rows[0]
(gen2, m2, fit2, fitw2) = rows[1]

print(gen1[0]/gen2[0])

plot_weights_mask_scatter(gen1, m1, gen2, m2)