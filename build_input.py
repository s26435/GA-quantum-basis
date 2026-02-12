#!/usr/bin/env python3
import re
import json
from pathlib import Path
from typing import List
import argparse

RE_NPRIM_NCONTR = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")
RE_NEW = re.compile(r"^\s*NEW\d+\s*$")

def load_alphas(path: Path) -> List[float]:
    s = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not s:
        raise ValueError(f"Pusty plik: {path}")

    if s[0] in "[{":
        obj = json.loads(s)
        if isinstance(obj, list):
            return [float(x) for x in obj]
        if isinstance(obj, dict):
            for key in ("alphas", "alpha"):
                if key in obj:
                    return [float(x) for x in obj[key]]
        raise ValueError("Error during reading JSON: JSON needs to be list or dict")

    return [float(x) for x in s.split()]

def inject_alphas_into_template(template_lines: List[str], alphas: List[float]) -> List[str]:
    block_sizes: List[int] = []

    def next_nonempty_idx(i: int) -> int:
        j = i
        while j < len(template_lines) and template_lines[j].strip() == "":
            j += 1
        return j

    for i, line in enumerate(template_lines):
        m = RE_NPRIM_NCONTR.match(line)
        if not m:
            continue
        nprim = int(m.group(1))
        j = next_nonempty_idx(i + 1)
        if j < len(template_lines) and RE_NEW.match(template_lines[j]):
            block_sizes.append(nprim)

    out: List[str] = []
    a_idx = 0
    blk_idx = 0

    for line in template_lines:
        mnew = RE_NEW.match(line)
        if mnew:
            if blk_idx >= len(block_sizes):
                raise ValueError(f"Error during injecting into template: NEW{mnew.group(1)} without corresponding label nprim")
            nprim = block_sizes[blk_idx]
            blk_idx += 1

            if a_idx + nprim > len(alphas):
                raise ValueError(f"Error during injecting into template: Amount alpha ({blk_idx}) needs {nprim}, {len(alphas)-a_idx} left")

            chunk = alphas[a_idx:a_idx+nprim]
            out.append(" ".join(f"{v:.16g}" for v in chunk) + "\n")
            a_idx += nprim
            continue

        out.append(line)

    if a_idx != len(alphas):
        raise ValueError(f"Error during injecting into template: {a_idx} alpha used, {len(alphas)} are given")

    return out
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("alphas_file", type=Path, help="json file with alphas")
    ap.add_argument("out_inp", type=Path, help="output file locations")
    ap.add_argument("--template", type=Path, default=Path("/mnt/data/INPUT"), help="template INPUT")
    args = ap.parse_args()

    alphas = load_alphas(args.alphas_file)
    tmpl_lines = args.template.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

    new_lines = inject_alphas_into_template(tmpl_lines, alphas)
    args.out_inp.parent.mkdir(parents=True, exist_ok=True)
    args.out_inp.write_text("".join(new_lines), encoding="utf-8")
