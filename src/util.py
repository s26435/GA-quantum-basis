from datetime import datetime
from typing import Union, Tuple, List
from .globals import log_handle, BLOCKS

import ast
import csv
import math
from pathlib import Path
from typing import Optional

import torch
from matplotlib import pyplot as plt

import random
import shutil

def lg(text: str, log_level: int = 1):
    """
    Logs *text* into stdout and into out.log file.

    :param text: Text to log
    :type text: str
    :param log_level: Level of logging
    :type log_level: int
    """
    msg = "[" + str(datetime.now()) + "] " + text
    if log_level > 0:
        print(msg)
        with open(log_handle, "a") as file:
            file.write(msg + "\n")


def clear_molcas_work(gen_idx: int, work_root: str | Path = "workspace"):
    work_root = Path(work_root).resolve()
    gen_dir = work_root / f"gen_{gen_idx:04d}"

    if not gen_dir.exists():
        raise FileNotFoundError(f"Generation directory does not exist: {gen_dir}")

    molcas_dirs = list(gen_dir.glob("*/molcas_work"))

    for molcas_work in molcas_dirs:
        if not molcas_work.is_dir():
            continue

        lg(f"removing directory: {molcas_work}")
        shutil.rmtree(molcas_work)


def random_variation_ordered(seq: List[float]) -> List[float]:
    """
    Generates ordered ranomized variation of given seqence

    :param seq: template sequence
    :type seq: List[float]
    :return: variation of sorted seq
    :rtype: List[float]
    """
    seq = list(seq)
    n = len(seq)
    sorted_seq = sorted(seq, reverse=True)
    new_sorted = [None] * n
    orig_max = sorted_seq[0]
    orig_second = sorted_seq[1] if n > 1 else orig_max
    lo = orig_second
    hi = orig_max * 1.1
    if hi < lo:
        hi = lo

    new_sorted[0] = random.uniform(lo, hi)
    for i in range(1, n):
        higher = new_sorted[i - 1]
        orig_val = sorted_seq[i]
        if i == n - 1:
            lo = orig_val * 0.9
            hi = min(orig_val * 1.1, higher)
        else:
            orig_lower_neighbor = sorted_seq[i + 1]
            lo = orig_lower_neighbor
            hi = higher
        if hi < lo:
            hi = lo

        new_sorted[i] = random.uniform(lo, hi)
    return new_sorted


def make_initial_population_from_seed(
    seed_alphas: List[float], pop_size: int, device: Union[str, torch.device] = "cpu", include_orginal: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates initial population for GA

    :param seed_alphas: Seed
    :type seed_alphas: List[float]
    :param pop_size: size of given and returned population
    :type pop_size: int
    :param device: device of target tensors
    :type device: Union[str, torch.device]
    :return: tuple of exponents and masks
    :rtype: Tuple[Tensor, Tensor]
    """
    assert len(seed_alphas) == sum(BLOCKS), f"{len(seed_alphas)} is not {sum(BLOCKS)}"

    pop = []
    diff = 0
    if include_orginal:
        diff = 1
        pop.append(seed_alphas)

    off = 0
    for _ in range(pop_size - diff):
        indiv = []
        off = 0
        for b in BLOCKS:
            block = seed_alphas[off : off + b]
            new_block = random_variation_ordered(block)
            indiv.extend(new_block)
            off += b
        pop.append(indiv)

    alphas = torch.tensor(pop, dtype=torch.float32, device=device)

    genome = torch.log(alphas.clamp_min(1e-12))
    return genome, torch.ones_like(genome)


def sanitize_block_with_mask(
    a_block: torch.Tensor,
    m_block: torch.Tensor,
    lo: float,
    hi: float,
    min_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Makes sure that genome is valid and not extends given range

    :param a_block: genome tensor
    :type a_block: torch.Tensor
    :param m_block: mask tensor
    :type m_block: torch.Tensor
    :param lo: low value of clamp
    :type lo: float
    :param hi: high value of clamp
    :type hi: float
    :param min_ratio: maximum diffrence between exponents
    :type min_ratio: float
    :return: tuple of sanitized genome and mask
    :rtype: Tuple[Tensor, Tensor]
    """

    m = m_block > 0.5

    out_a = torch.zeros_like(a_block)
    out_m = torch.zeros_like(m_block, dtype=torch.float32)

    if m.sum().item() == 0:
        j = torch.argmax(a_block)
        m = torch.zeros_like(m, dtype=torch.bool)
        m[j] = True

    vals = a_block[m]
    vals = torch.nan_to_num(vals, nan=hi, posinf=hi, neginf=lo)
    vals = vals.clamp(lo, hi)
    vals, _ = torch.sort(vals, descending=True)

    for k in range(vals.numel() - 1):
        vals[k + 1] = torch.minimum(vals[k + 1], vals[k] / min_ratio)

    vals = vals.clamp(lo, hi)

    k = vals.numel()
    out_a[:k] = vals
    out_m[:k] = 1.0
    return out_a, out_m


def sanitize_blocks(
    a: torch.Tensor,
    mask: torch.Tensor,
    blocks=BLOCKS,
    lo: float = 1e-2,
    hi: float = 1e2,
    min_ratio: float = 1.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sanitizes blocks - aranges exponents in order

    :param a: Description
    :type a: torch.Tensor
    :param mask: Description
    :type mask: torch.Tensor
    :param blocks: Description
    :param lo: low value of clamp
    :type lo: float
    :param hi: high value of clamp
    :type hi: float
    :param min_ratio: maximum diffrence between exponents
    :type min_ratio: float
    :return: tuple of sanitized genome and mask
    :rtype: Tuple[Tensor, Tensor]
    """

    assert a.numel() == mask.numel(), (
        f"a and mask must have same length: {a.numel()} vs {mask.numel()}"
    )

    parts_a = []
    parts_m = []
    off = 0
    for n in blocks:
        a_block = a[off : off + n]
        m_block = mask[off : off + n]
        sa, sm = sanitize_block_with_mask(
            a_block, m_block, lo=lo, hi=hi, min_ratio=min_ratio
        )
        parts_a.append(sa)
        parts_m.append(sm)
        off += n

    return torch.cat(parts_a, dim=0), torch.cat(parts_m, dim=0)


def load_populations_csv(path: Union[str, Path], device: str = "cpu"):
    path = Path(path)
    out = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        for row_idx, row in enumerate(reader):
            if not row:
                continue

            try:
                alphas = ast.literal_eval(row[0])
                mask = ast.literal_eval(row[1])
                fit = float(row[2])
                fit_wmask = float(row[3])
            except Exception as e:
                raise ValueError(
                    f"Could not parse row {row_idx} in {path}: {row}"
                ) from e

            a = torch.tensor(alphas, dtype=torch.float32, device=device)
            m = torch.tensor(mask, dtype=torch.float32, device=device)

            genome = a.clamp_min(1e-30)
            mask_t = (m > 0.5).float()

            out.append((genome, mask_t, fit, fit_wmask))

    return out


def get_best_individual(rows, use_fit_wmask: bool = True):
    if not rows:
        raise ValueError("Population CSV is empty.")

    best_idx = None
    best_value = float("inf")

    for idx, (_, _, fit, fit_wmask) in enumerate(rows):
        value = fit_wmask if use_fit_wmask else fit

        if math.isfinite(value) and value < best_value:
            best_value = value
            best_idx = idx

    if best_idx is None:
        raise ValueError("No valid finite fitness value found in population CSV.")

    genome, mask, fit, fit_wmask = rows[best_idx]

    return best_idx, genome, mask, fit, fit_wmask


def plot_weights_mask_scatter(
    gen1: torch.Tensor,
    m1: torch.Tensor,
    gen2: torch.Tensor,
    m2: torch.Tensor,
    title: str = "Starting seed vs best population individual",
    output_path: Union[str, Path] = "compare.png",
):
    output_path = Path(output_path)

    gen1 = gen1.detach().cpu().reshape(-1)
    gen2 = gen2.detach().cpu().reshape(-1)

    m1b = (m1.detach().cpu().reshape(-1) > 0.5)
    m2b = (m2.detach().cpu().reshape(-1) > 0.5)

    if (
        gen1.numel() != gen2.numel()
        or gen1.numel() != m1b.numel()
        or gen2.numel() != m2b.numel()
    ):
        raise ValueError(
            "Rozmiary genów/masek muszą się zgadzać. "
            f"Got: gen1={gen1.numel()}, gen2={gen2.numel()}, "
            f"m1={m1b.numel()}, m2={m2b.numel()}"
        )

    n = gen1.numel()
    x = torch.arange(n, dtype=torch.float32)

    y1 = gen1.clamp_min(1e-30)
    y2 = gen2.clamp_min(1e-30)

    idx1_on = torch.where(m1b)[0]
    idx1_off = torch.where(~m1b)[0]
    idx2_on = torch.where(m2b)[0]
    idx2_off = torch.where(~m2b)[0]

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)

    ax.scatter(
        x[idx1_on],
        y1[idx1_on],
        marker="o",
        label="starting seed active",
        color="green",
    )
    ax.scatter(
        x[idx1_off],
        y1[idx1_off],
        marker="o",
        label="starting seed inactive",
        color="red",
    )

    ax.scatter(
        x[idx2_on],
        y2[idx2_on],
        marker="x",
        label="best individual active",
        color="blue",
    )
    ax.scatter(
        x[idx2_off],
        y2[idx2_off],
        marker="x",
        label="best individual inactive",
        color="orange",
    )

    for i in range(n):
        ax.plot(
            [float(x[i]), float(x[i])],
            [float(y1[i]), float(y2[i])],
            linewidth=0.8,
            alpha=0.5,
            color="black",
        )

    for i in range(n):
        abs_diff = abs(float(y2[i] - y1[i]))

        if abs(float(y1[i])) < 1e-30:
            ratio_txt = "n/a"
        else:
            ratio = float(y2[i] / y1[i])
            ratio_txt = f"{ratio:.5f}"

        y_top = float(max(y1[i], y2[i]))

        ax.annotate(
            f"{abs_diff:.2e}\n{ratio_txt}",
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

    diff_mask = m1b ^ m2b
    diff_idx = torch.where(diff_mask)[0]

    for i in diff_idx.tolist():
        ax.axvline(x=float(i), linewidth=0.6, alpha=0.2)

    ax.set_xlabel("gene index")
    ax.set_ylabel("alpha value")
    ax.set_title(f"{title} | mask diffs: {int(diff_mask.sum().item())}/{n}")
    ax.legend(ncol=2, fontsize=9)
    ax.set_yscale("log")
    ax.grid(True)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.25)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def compare_starting_seed_with_best_population(
    starting_seed,
    populations_csv_path: Union[str, Path],
    starting_mask: Optional[Union[list, torch.Tensor]] = None,
    output_path: Union[str, Path] = "compare.png",
    device: str = "cpu",
    use_fit_wmask: bool = True,
):
    rows = load_populations_csv(populations_csv_path, device=device)

    best_idx, best_genome, best_mask, best_fit, best_fit_wmask = get_best_individual(
        rows,
        use_fit_wmask=use_fit_wmask,
    )

    seed_genome = torch.tensor(
        starting_seed,
        dtype=torch.float32,
        device=device,
    ).clamp_min(1e-30)

    if starting_mask is None:
        seed_mask = torch.ones_like(seed_genome, dtype=torch.float32, device=device)
    else:
        seed_mask = torch.tensor(
            starting_mask,
            dtype=torch.float32,
            device=device,
        )
        seed_mask = (seed_mask > 0.5).float()

    if seed_genome.numel() != best_genome.numel():
        raise ValueError(
            "Starting seed and best genome have different sizes: "
            f"{seed_genome.numel()} vs {best_genome.numel()}"
        )

    saved_path = plot_weights_mask_scatter(
        gen1=seed_genome,
        m1=seed_mask,
        gen2=best_genome,
        m2=best_mask,
        title=(
            f"Starting seed vs best population individual "
            f"| best_idx={best_idx}, fit={best_fit:.6f}, fit_wmask={best_fit_wmask:.6f}"
        ),
        output_path=output_path,
    )

    return {
        "best_index": best_idx,
        "best_genome": best_genome,
        "best_mask": best_mask,
        "best_fit": best_fit,
        "best_fit_wmask": best_fit_wmask,
        "output_path": str(saved_path),
    }