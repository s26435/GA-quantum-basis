#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

RE_SMALL_OVERLAP = re.compile(r"small overlap", re.IGNORECASE)
RE_OVERLAP = re.compile(r"overlap[^0-9\-+]*([0-9]*\.[0-9]+)", re.IGNORECASE)

def parse_cmocorr_text(text: str, t1: float) -> dict:
    warning_count = len(RE_SMALL_OVERLAP.findall(text))
    overlaps = []

    for m in RE_OVERLAP.finditer(text):
        try:
            overlaps.append(float(m.group(1)))
        except Exception:
            pass

    min_overlap = min(overlaps) if overlaps else None

    penalty = 0.0
    penalty += 0.25 * warning_count

    if min_overlap is not None and math.isfinite(min_overlap):
        penalty += max(0.0, t1 - min_overlap) * 100.0

    return {
        "ok": True,
        "warning_count": warning_count,
        "min_overlap": min_overlap,
        "penalty": penalty,
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmocorr_log", type=Path)
    ap.add_argument("--t1", type=float, default=0.90)
    args = ap.parse_args()

    text = args.cmocorr_log.read_text(encoding="utf-8", errors="ignore")
    result = parse_cmocorr_text(text, t1=args.t1)
    print(json.dumps(result, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())