#!/usr/bin/env python3
import re
import sys
from pathlib import Path

FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"

PATTERNS = [
    re.compile(rf"\bCASPT2\b.*?\bTotal\s+energy\b.*?({FLOAT})", re.IGNORECASE),
    re.compile(rf"\bTotal\s+SCF\s+energy\b.*?({FLOAT})", re.IGNORECASE),
]

BAD = re.compile(r"\bGB of memory\b|\bpid:\b|Orthogonality violated", re.IGNORECASE)

def main() -> int:
    if len(sys.argv) != 2:
        print("nan")
        return 2

    p = Path(sys.argv[1])
    txt_lines = p.read_text(errors="ignore").splitlines()

    txt = "\n".join([ln for ln in txt_lines if not BAD.search(ln)])

    for pat in PATTERNS:
        m = list(pat.finditer(txt))
        if m:
            print(float(m[-1].group(1)))
            return 0

    print("nan")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
