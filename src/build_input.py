#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

# Regex objects for searching headers and exponent place in INPUT template
RE_NPRIM_NCONTR = re.compile(r"^(\s*)(\d+)\s+(\d+)\s*$")
RE_NEW = re.compile(r"^\s*NEW(\d+)\s*$")


def load_params(path: Path) -> Tuple[List[float], Optional[List[bool]]]:
    """
    Loads json with exponents and mask

    :param path: path to json file
    :type path: pathlib.Path

    :return: to fields tuple: (list of exponents, mask as a list)
    :rtype: Tuple[List[float], Optional[List[bool]]]
    """
    s = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not s:
        raise ValueError(f"Pusty plik: {path}")

    if s[0] in "[{":
        obj = json.loads(s)
        if isinstance(obj, list):
            return [float(x) for x in obj], None
        if isinstance(obj, dict):
            alphas = None
            for k in ("alphas", "alpha"):
                if k in obj:
                    alphas = [float(x) for x in obj[k]]
                    break
            if alphas is None:
                raise ValueError(
                    "JSON dict musi zawierać klucz 'alphas' (albo 'alpha')."
                )

            mask = None
            if "mask" in obj and obj["mask"] is not None:
                mask = [bool(x) for x in obj["mask"]]
            return alphas, mask

        raise ValueError("JSON musi być listą albo dictem {alphas:..., mask:...}.")

    return [float(x) for x in s.split()], None


def next_nonempty_idx(lines: List[str], i: int) -> int:
    """
    Returns idx of first non empty line from the list form i-th position

    :param lines: target list of string
    :type lines: list of string
    :param i: starting idx
    :type i: int
    :return: idx of first non empty string in lines
    :rtype: int
    """
    j = i
    while j < len(lines) and lines[j].strip() == "":
        j += 1
    return j


def parse_matrix(
    lines: List[str], start: int, nrows: int
) -> Optional[List[List[float]]]:
    """
    Loads contraction matrix from lines. Returns None if parsing matrix failed.

    :param lines: target lines
    :type lines: List[str]
    :param start: idx of line that matrix starts
    :type start: int
    :param nrows: num of rows of matrix
    :type nrows: int
    :return: loaded matrix
    :rtype: Optional[List[List[float]]]
    """
    if start + nrows > len(lines):
        return None
    mat: List[List[float]] = []
    for r in range(nrows):
        row_s = lines[start + r].strip()
        if not row_s:
            return None
        toks = row_s.split()
        try:
            mat.append([float(t) for t in toks])
        except Exception:
            return None
    return mat


def format_row(vals: List[float]) -> str:
    """
    Formats given list of floats into one line with 16 digits of precision. Adds a new line at the end.

    :param vals: list of floats
    :type vals: List[float]
    :return: formatted line
    :rtype: str
    """
    return " ".join(f"{v:.16g}" for v in vals) + "\n"


def submatrix_or_identity(
    orig: Optional[List[List[float]]], idx: List[int]
) -> List[List[float]]:
    """
    Gives subpatrix of orginal matrix by given idxs, if failed returns identity matrix of given size.

    :param orig: template matrix
    :type orig: Optional[List[List[float]]]
    :param idx: list of included rows and columns
    :type idx: List[int]
    :return: submatrix of orginal matrix
    :rtype: List[List[float]]

    :example:
    >>> orig = [
    [11, 12, 13, 14, 15],
    [21, 22, 23, 24, 25],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45],
    [51, 52, 53, 54, 55],
    ]

    >>> idx = [0, 2, 4]


    >>> submatrix_or_identity(orig, idx)
    [[11, 13, 15],[31, 33, 35],[51, 53, 55]]
    """

    k = len(idx)
    if orig is None:
        return [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]

    if len(orig) == 0 or any(len(r) < max(idx) + 1 for r in orig):
        return [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]

    sub: List[List[float]] = []
    for r in idx:
        sub.append([orig[r][c] for c in idx])
    return sub


def build_block(
    chunk_a: List[float],
    chunk_m: List[bool],
    indent: str,
    orig_matrix: Optional[List[List[float]]],
    min_alpha: float,
) -> Tuple[List[str], int]:
    """
    Builds a new part of input file - based on exponents and mask builds. Build contraction matrix
    and writes exponents above matrix. Creates header of orbital.

    :param chunk_a: list of exponent values for given orbitals
    :type chunk_a: List[float]
    :param chunk_m: mask for `chunk_a` list of exponents
    :type chunk_m: List[bool]
    :param indent: Indentation prefix for the header line
    :type indent: str
    :param orig_matrix: template matrix
    :type orig_matrix: Optional[List[List[float]]]
    :param min_alpha: minimum alowed value of exponent
    :type min_alpha: float
    :return: A tuple containing the formatted output lines and the number
    of active exponents.
    :rtype: Tuple[List[str], int]
    :raises ValueError: if mask and exponent have different length
    """
    nprim = len(chunk_a)
    if len(chunk_m) != nprim:
        raise ValueError(f"len(mask_chunk)={len(chunk_m)} != len(alpha_chunk)={nprim}")

    active = [i for i, m in enumerate(chunk_m) if bool(m)]

    if len(active) == 0:
        best_i = max(
            range(nprim),
            key=lambda i: chunk_a[i] if math.isfinite(chunk_a[i]) else -1e300,
        )
        active = [best_i]

    active_alphas: List[float] = []
    for i in active:
        v = float(chunk_a[i])
        if not math.isfinite(v):
            v = min_alpha
        if v < min_alpha:
            v = min_alpha
        active_alphas.append(v)

    k = len(active_alphas)

    out: List[str] = [f"{indent}{k:5d} {k:5d}\n"]

    out.append(format_row(active_alphas))

    mat = submatrix_or_identity(orig_matrix, active)
    out.extend(format_row(row) for row in mat)

    return out, k


def inject(
    template_lines: List[str],
    alphas: List[float],
    mask: Optional[List[bool]],
    use_mask: bool,
    min_alpha: float,
) -> List[str]:
    """
    Injects exponents values int template and creates valid INPUT file for molcas calculations

    :param template_lines: INPUT template as list of lines
    :type template_lines: List[str]
    :param alphas: list of exponents values
    :type alphas: List[float]
    :param mask: optional mask for exponents
    :type mask: Optional[List[bool]]
    :param use_mask: whether the mask should be used
    :type use_mask: bool
    :param min_alpha: minimum alowed value of exponent (molcas don't like 0s)
    :type min_alpha: float
    :return: valid input for molcas file as a list of lines
    :rtype: List[str]
    :raises ValueError: if number of exponent is not valid for template file
    :raises ValueError: if `use_mask` is set true, but json have no mask
    :raises ValueError: if number used mask is not equal to the number of provided exponents
    """
    block_nprims: List[int] = []
    for i, line in enumerate(template_lines):
        m = RE_NPRIM_NCONTR.match(line)
        if not m:
            continue
        j = next_nonempty_idx(template_lines, i + 1)
        if j < len(template_lines) and RE_NEW.match(template_lines[j]):
            block_nprims.append(int(m.group(2)))

    need = sum(block_nprims)
    if len(alphas) != need:
        raise ValueError(
            f"Template needs {need} alpha (suma of nprim), but got {len(alphas)}."
        )

    if use_mask:
        if mask is None:
            raise ValueError("use_mask=True, but JSON file have no 'mask'.")
        if len(mask) != len(alphas):
            raise ValueError(
                f"Mask needs have lenght {len(alphas)}, but have {len(mask)}."
            )

    out: List[str] = []
    a_ptr = 0
    i = 0

    while i < len(template_lines):
        m = RE_NPRIM_NCONTR.match(template_lines[i])
        if not m:
            out.append(template_lines[i])
            i += 1
            continue

        indent = m.group(1)
        nprim = int(m.group(2))
        ncontr = int(m.group(3))

        j = next_nonempty_idx(template_lines, i + 1)
        mnew = RE_NEW.match(template_lines[j]) if j < len(template_lines) else None
        if not mnew:
            out.append(template_lines[i])
            i += 1
            continue

        orig_mat = parse_matrix(template_lines, j + 1, ncontr)

        chunk_a = alphas[a_ptr : a_ptr + nprim]
        chunk_m = (
            mask[a_ptr : a_ptr + nprim]
            if (use_mask and mask is not None)
            else [True] * nprim
        )

        block_out, _k = build_block(
            chunk_a, chunk_m, indent, orig_mat, min_alpha=min_alpha
        )

        out.extend(block_out[:1])
        out.extend(template_lines[i + 1 : j])
        out.extend(block_out[1:])

        i = j + 1 + ncontr
        a_ptr += nprim

    if a_ptr != len(alphas):
        raise ValueError(
            f"Used {a_ptr} alpha, but {len(alphas)} are given. Maybe something wrong with template?"
        )

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "params_file",
        type=Path,
        help="JSON: list of alphas or dict {alphas:[...], mask:[...]}",
    )
    ap.add_argument("out_inp", type=Path, help="path for saving OUTPUT INPUT")
    ap.add_argument("--template", type=Path, required=True, help="INPUT.template")
    ap.add_argument(
        "--use-mask",
        action="store_true",
        help="force using mask",
    )
    ap.add_argument(
        "--min-alpha",
        type=float,
        default=1e-12,
        help="min alpha for active exponents",
    )
    args = ap.parse_args()

    alphas, mask = load_params(args.params_file)
    tmpl_lines = args.template.read_text(encoding="utf-8", errors="ignore").splitlines(
        keepends=True
    )

    use_mask = args.use_mask or (mask is not None)

    new_lines = inject(
        tmpl_lines,
        alphas=alphas,
        mask=mask,
        use_mask=use_mask,
        min_alpha=float(args.min_alpha),
    )

    args.out_inp.parent.mkdir(parents=True, exist_ok=True)
    args.out_inp.write_text("".join(new_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
