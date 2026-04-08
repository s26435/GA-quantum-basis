from datetime import datetime
from typing import Union, Tuple, List
from .globals import log_handle, BLOCKS

import torch
import random

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
    seed_alphas: List[float], pop_size: int, device: Union[str, torch.device] = "cpu"
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
    off = 0
    for _ in range(pop_size):
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
