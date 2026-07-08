#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CASE_RE = re.compile(r"^(?P<case>pop|gen|reference)_case_(?P<idx>\d+)$")
GEN_RE = re.compile(r"^gen_(?P<gen>\d+)$")

FLOAT_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
)

MAIN_MOLCAS_DEBUG_MARKER = "[DEBUG] molcas_cmd="
BUILD_FAILED_MARKER = "[build_input FAILED]"
MOLCAS_TIMEOUT_MARKER = "[molcas TIMEOUT]"
MOLCAS_FAILED_MARKER = "[molcas FAILED]"
PARSE_ENERGY_FAILED_MARKER = "[parse_energy FAILED]"
WORKER_CRASH_MARKER = "[WORKER CRASH]"


def read_text_safe(path: Path, max_bytes: int | None = None) -> str:
    if not path.exists() or not path.is_file():
        return ""

    try:
        if max_bytes is None:
            return path.read_text(encoding="utf-8", errors="ignore")

        with path.open("rb") as f:
            data = f.read(max_bytes)

        return data.decode("utf-8", errors="ignore")

    except Exception:
        return ""


def read_tail_safe(path: Path, tail_bytes: int = 256_000) -> str:
    if not path.exists() or not path.is_file():
        return ""

    try:
        size = path.stat().st_size

        with path.open("rb") as f:
            f.seek(max(0, size - tail_bytes))
            data = f.read()

        return data.decode("utf-8", errors="ignore")

    except Exception:
        return ""


def parse_float_or_none(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None

    m = FLOAT_RE.search(text)
    if not m:
        return None

    try:
        value = float(m.group(0).replace("D", "E").replace("d", "e"))
    except ValueError:
        return None

    if not math.isfinite(value):
        return None

    return value


def parse_energy_txt(path: Path) -> float | None:
    if not path.exists() or not path.is_file():
        return None

    return parse_float_or_none(read_text_safe(path))


def find_generation(case_dir: Path, root: Path) -> int | None:
    try:
        parts = case_dir.relative_to(root).parts
    except ValueError:
        parts = case_dir.parts

    for part in parts:
        m = GEN_RE.match(part)
        if m:
            return int(m.group("gen"))

    return None


def format_generation(gen: int | None) -> str:
    if gen is None:
        return "unknown"

    return f"gen_{gen:04d}"


def find_case_type_and_idx(case_dir: Path) -> tuple[str, int | None]:
    m = CASE_RE.match(case_dir.name)
    if not m:
        return "unknown", None

    return m.group("case"), int(m.group("idx"))


def iter_case_dirs(root: Path):
    seen: set[Path] = set()

    for p in root.rglob("*"):
        if not p.is_dir():
            continue

        if CASE_RE.match(p.name):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                yield rp
            continue

        # fallback, gdyby nazwa katalogu była niestandardowa
        if (p / "run.out").exists() and list(p.glob("params_*.json")):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                yield rp


def find_params_file(case_dir: Path) -> Path | None:
    files = sorted(case_dir.glob("params_*.json"))

    if not files:
        return None

    return files[0]


def load_params_mask(case_dir: Path) -> list[bool] | None:
    params_file = find_params_file(case_dir)

    if params_file is None:
        return None

    try:
        data = json.loads(params_file.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    mask = data.get("mask")

    if mask is None:
        return None

    try:
        return [bool(x) for x in mask]
    except Exception:
        return None


def mask_signature(mask: list[bool] | None) -> str:
    if mask is None:
        return ""

    return "".join("1" if bool(x) else "0" for x in mask)


def classify_main_status(run_out: Path, energy_txt: Path) -> tuple[str, bool, float | None]:
    head = read_text_safe(run_out, max_bytes=128_000)
    tail = read_tail_safe(run_out, tail_bytes=256_000)
    text = head + "\n" + tail

    energy = parse_energy_txt(energy_txt)

    build_failed = BUILD_FAILED_MARKER in text
    molcas_timeout = MOLCAS_TIMEOUT_MARKER in text
    molcas_failed = MOLCAS_FAILED_MARKER in text
    parse_failed = PARSE_ENERGY_FAILED_MARKER in text
    worker_crash = WORKER_CRASH_MARKER in text

    has_main_debug_marker = MAIN_MOLCAS_DEBUG_MARKER in head

    # Ten marker jest zapisywany przed odpaleniem:
    # molcas INPUT
    main_molcas_attempted = has_main_debug_marker and not build_failed

    if build_failed:
        return "build_input_failed", main_molcas_attempted, energy

    if molcas_timeout:
        return "molcas_timeout", main_molcas_attempted, energy

    if molcas_failed:
        return "molcas_failed", main_molcas_attempted, energy

    if parse_failed:
        return "parse_energy_failed", main_molcas_attempted, energy

    if worker_crash:
        return "worker_crash", main_molcas_attempted, energy

    if energy is not None:
        return "ok", main_molcas_attempted, energy

    if main_molcas_attempted:
        return "unknown_after_molcas_attempt", main_molcas_attempted, energy

    if run_out.exists():
        return "run_out_exists_but_no_molcas_marker", main_molcas_attempted, energy

    return "no_run_out", main_molcas_attempted, energy


def read_initial_reference_signatures(workdir: Path) -> set[str]:
    """
    Czyta referencje zapisane przez GA:
        workspace/reference/reference_sig_<signature>.orb

    Jeśli istnieją, to te sygnatury miały referencję CMOCORR
    już na początku albo po wcześniejszym zapisie.
    """

    refs: set[str] = set()
    ref_dir = workdir / "reference"

    if not ref_dir.exists():
        return refs

    for p in ref_dir.glob("reference_sig_*.orb"):
        stem = p.stem
        sig = stem.removeprefix("reference_sig_")

        if sig:
            refs.add(sig)

    return refs


def build_case_row(case_dir: Path, root: Path) -> dict[str, Any]:
    case_dir = case_dir.resolve()

    run_out = case_dir / "run.out"
    energy_txt = case_dir / "energy.txt"
    input_rasorb = case_dir / "INPUT.RasOrb"
    input_status = case_dir / "INPUT.status"
    params_file = find_params_file(case_dir)

    case_type, case_idx = find_case_type_and_idx(case_dir)
    generation = find_generation(case_dir, root)

    main_status, main_molcas_attempted, energy = classify_main_status(
        run_out=run_out,
        energy_txt=energy_txt,
    )

    mask = load_params_mask(case_dir)
    sig = mask_signature(mask)

    has_orbital = input_rasorb.exists() and input_rasorb.is_file()
    has_status = input_status.exists() and input_status.is_file()

    # Do rejestracji referencji potrzebne było:
    # - valid == 1
    # - energy finite
    # - orbital_file is not None
    #
    # Po sprzątaniu traktujemy to jako:
    # - energy.txt zawiera skończoną energię
    # - INPUT.RasOrb istnieje
    eligible_as_reference = (
        main_status == "ok"
        and energy is not None
        and has_orbital
        and bool(sig)
    )

    return {
        "path": str(case_dir),
        "generation": generation,
        "case_type": case_type,
        "case_idx": case_idx,

        "params_file": str(params_file) if params_file is not None else "",
        "mask_signature": sig,
        "mask_len": sum(mask) if mask is not None else "",

        "has_run_out": run_out.exists(),
        "has_energy_txt": energy_txt.exists(),
        "has_input_rasorb": has_orbital,
        "has_input_status": has_status,

        "main_molcas_attempted": main_molcas_attempted,
        "main_status": main_status,
        "energy": energy,

        "eligible_as_reference": eligible_as_reference,

        # Uzupełniane później przez rekonstrukcję batchy.
        "cmocorr_reconstructed": False,
        "cmocorr_reason": "",
        "molcas_processes": int(main_molcas_attempted),
    }


def batch_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    """
    Kolejność zgodna z GA.run():
    1. reference
    2. dla każdej generacji:
       - pop fitness
       - gen fitness

    W samym batchu kolejność case_idx nie wpływa na CMOCORR,
    bo ref_path jest ustalany przed wysłaniem batcha do ProcessPool.
    """

    gen = row["generation"]

    if row["case_type"] == "reference":
        gen_key = -1
        batch_kind = 0
    else:
        gen_key = 10**12 if gen is None else int(gen)

        if row["case_type"] == "pop":
            batch_kind = 1
        elif row["case_type"] == "gen":
            batch_kind = 2
        else:
            batch_kind = 9

    idx = row["case_idx"]
    idx_key = 10**12 if idx is None else int(idx)

    return gen_key, batch_kind, idx_key


def group_batches(rows: list[dict[str, Any]]) -> list[tuple[tuple[int | None, str], list[dict[str, Any]]]]:
    grouped: dict[tuple[int | None, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[(row["generation"], row["case_type"])].append(row)

    def key_fn(item: tuple[tuple[int | None, str], list[dict[str, Any]]]) -> tuple[int, int]:
        (gen, case_type), _batch_rows = item

        if case_type == "reference":
            return -1, 0

        gen_key = 10**12 if gen is None else int(gen)

        if case_type == "pop":
            kind = 1
        elif case_type == "gen":
            kind = 2
        else:
            kind = 9

        return gen_key, kind

    out = []

    for batch_key, batch_rows in sorted(grouped.items(), key=key_fn):
        out.append(
            (
                batch_key,
                sorted(batch_rows, key=batch_sort_key),
            )
        )

    return out


def reconstruct_cmocorr(
    rows: list[dict[str, Any]],
    initial_refs: set[str],
    cmocorr_enabled: bool,
) -> None:
    """
    Rekonstrukcja według logiki GA:

    W fitness():
    - dla całego batcha najpierw ustalany jest ref_path_i,
    - cmocorr_enabled_i = cfg.cmocorr_enabled and ref_path_i is not None,
    - potem batch leci równolegle,
    - dopiero po zakończeniu case’y bez CMOCORR mogą zarejestrować nową referencję.

    Wniosek:
    - jeśli sygnatura była w known_refs PRZED batchem, case z energią i orbitalem miał CMOCORR,
    - jeśli sygnatury nie było PRZED batchem, case nie miał CMOCORR, ale może dodać referencję dla następnych batchy.
    """

    known_refs = set(initial_refs)

    for (_gen, case_type), batch_rows in group_batches(rows):
        refs_before_batch = set(known_refs)

        # Bootstrap/reference case w Twoim kodzie jest liczony z cmocorr_enabled=False,
        # więc nie doliczamy mu CMOCORR.
        is_reference_batch = case_type == "reference"

        for row in batch_rows:
            sig = row["mask_signature"]
            main_ok = row["main_status"] == "ok"
            has_orb = bool(row["has_input_rasorb"])

            if not cmocorr_enabled:
                row["cmocorr_reconstructed"] = False
                row["cmocorr_reason"] = "cmocorr_disabled"
                row["molcas_processes"] = int(row["main_molcas_attempted"])
                continue

            if is_reference_batch:
                row["cmocorr_reconstructed"] = False
                row["cmocorr_reason"] = "reference_batch_no_cmocorr"
                row["molcas_processes"] = int(row["main_molcas_attempted"])
                continue

            if not row["main_molcas_attempted"]:
                row["cmocorr_reconstructed"] = False
                row["cmocorr_reason"] = "main_molcas_not_attempted"
                row["molcas_processes"] = int(row["main_molcas_attempted"])
                continue

            if not main_ok:
                row["cmocorr_reconstructed"] = False
                row["cmocorr_reason"] = "main_not_ok"
                row["molcas_processes"] = int(row["main_molcas_attempted"])
                continue

            if not has_orb:
                row["cmocorr_reconstructed"] = False
                row["cmocorr_reason"] = "no_input_rasorb"
                row["molcas_processes"] = int(row["main_molcas_attempted"])
                continue

            if not sig:
                row["cmocorr_reconstructed"] = False
                row["cmocorr_reason"] = "no_mask_signature"
                row["molcas_processes"] = int(row["main_molcas_attempted"])
                continue

            if sig in refs_before_batch:
                row["cmocorr_reconstructed"] = True
                row["cmocorr_reason"] = "signature_had_reference_before_batch"
                row["molcas_processes"] = int(row["main_molcas_attempted"]) + 1
            else:
                row["cmocorr_reconstructed"] = False
                row["cmocorr_reason"] = "first_batch_for_signature_registers_reference_afterwards"
                row["molcas_processes"] = int(row["main_molcas_attempted"])

        # Po batchu rejestrują się nowe referencje z udanych case'ów.
        for row in batch_rows:
            sig = row["mask_signature"]

            if row["eligible_as_reference"] and sig:
                known_refs.add(sig)


def print_summary(rows: list[dict[str, Any]], root: Path, initial_refs: set[str]) -> None:
    main_status_counts = Counter(row["main_status"] for row in rows)
    cmocorr_reason_counts = Counter(row["cmocorr_reason"] for row in rows)
    case_type_counts = Counter(row["case_type"] for row in rows)

    total_case_dirs = len(rows)
    total_main = sum(int(row["main_molcas_attempted"]) for row in rows)
    total_cmocorr = sum(int(row["cmocorr_reconstructed"]) for row in rows)
    total_processes = sum(int(row["molcas_processes"]) for row in rows)

    total_ok = main_status_counts["ok"]

    by_generation: dict[int | None, dict[str, int]] = defaultdict(
        lambda: {
            "cases": 0,
            "main": 0,
            "cmocorr": 0,
            "total": 0,
            "ok": 0,
        }
    )

    for row in rows:
        gen = row["generation"]

        by_generation[gen]["cases"] += 1
        by_generation[gen]["main"] += int(row["main_molcas_attempted"])
        by_generation[gen]["cmocorr"] += int(row["cmocorr_reconstructed"])
        by_generation[gen]["total"] += int(row["molcas_processes"])
        by_generation[gen]["ok"] += int(row["main_status"] == "ok")

    print(f"WORKDIR: {root}")
    print()
    print(f"Initial reference signatures:       {len(initial_refs)}")
    print(f"Case directories found:             {total_case_dirs}")
    print()
    print(f"Main Molcas INPUT runs:             {total_main}")
    print(f"CMOCORR runs reconstructed:         {total_cmocorr}")
    print(f"Total reconstructed Molcas runs:    {total_processes}")
    print()
    print(f"Successful parsed energies:         {total_ok}")

    print()
    print("Main status:")
    for status, count in main_status_counts.most_common():
        print(f"  {status:56s} {count}")

    print()
    print("CMOCORR reconstruction reasons:")
    for reason, count in cmocorr_reason_counts.most_common():
        print(f"  {reason:56s} {count}")

    print()
    print("Case type:")
    for case_type, count in case_type_counts.most_common():
        print(f"  {case_type:12s} {count}")

    print()
    print("By generation:")
    for gen in sorted(by_generation, key=lambda x: -1 if x is None else x):
        label = format_generation(gen)
        v = by_generation[gen]

        print(
            f"  {label:12s} "
            f"cases={v['cases']:5d} "
            f"main={v['main']:5d} "
            f"cmocorr={v['cmocorr']:5d} "
            f"total={v['total']:5d} "
            f"ok={v['ok']:5d}"
        )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "generation",
        "case_type",
        "case_idx",

        "params_file",
        "mask_signature",
        "mask_len",

        "has_run_out",
        "has_energy_txt",
        "has_input_rasorb",
        "has_input_status",

        "main_molcas_attempted",
        "main_status",
        "energy",

        "eligible_as_reference",
        "cmocorr_reconstructed",
        "cmocorr_reason",

        "molcas_processes",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json_summary(rows: list[dict[str, Any]], root: Path, initial_refs: set[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    main_status_counts = Counter(row["main_status"] for row in rows)
    cmocorr_reason_counts = Counter(row["cmocorr_reason"] for row in rows)
    case_type_counts = Counter(row["case_type"] for row in rows)

    payload = {
        "workdir": str(root),
        "initial_reference_signatures": len(initial_refs),
        "total_case_dirs": len(rows),
        "total_main_molcas_input_runs": sum(int(row["main_molcas_attempted"]) for row in rows),
        "total_cmocorr_runs_reconstructed": sum(int(row["cmocorr_reconstructed"]) for row in rows),
        "total_reconstructed_molcas_runs": sum(int(row["molcas_processes"]) for row in rows),
        "successful_parsed_energies": main_status_counts["ok"],
        "by_main_status": dict(main_status_counts),
        "by_cmocorr_reason": dict(cmocorr_reason_counts),
        "by_case_type": dict(case_type_counts),
    }

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_cmocorr_cases(rows: list[dict[str, Any]]) -> None:
    print()
    print("Reconstructed CMOCORR cases:")

    any_found = False

    for row in rows:
        if row["cmocorr_reconstructed"]:
            any_found = True
            print(
                f"  {format_generation(row['generation']):10s} "
                f"{row['case_type']:10s} "
                f"case={str(row['case_idx']):>5s} "
                f"mask_len={str(row['mask_len']):>4s} "
                f"{row['path']}"
            )

    if not any_found:
        print("  none")


def print_no_cmocorr_cases(rows: list[dict[str, Any]]) -> None:
    print()
    print("Cases without reconstructed CMOCORR:")

    any_found = False

    for row in rows:
        if row["main_status"] == "ok" and not row["cmocorr_reconstructed"]:
            any_found = True
            print(
                f"  {row['cmocorr_reason']:56s} "
                f"{format_generation(row['generation']):10s} "
                f"{row['case_type']:10s} "
                f"case={str(row['case_idx']):>5s} "
                f"mask_len={str(row['mask_len']):>4s} "
                f"{row['path']}"
            )

    if not any_found:
        print("  none")


def infer_full_mask_signature(rows: list[dict[str, Any]]) -> str:
    max_len = 0

    for row in rows:
        sig = row["mask_signature"]

        if sig:
            max_len = max(max_len, len(sig))

    if max_len <= 0:
        return ""

    return "1" * max_len


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Count Molcas runs after molcas_work cleanup. "
            "Main runs are counted from run.out/energy.txt. "
            "CMOCORR runs are reconstructed from GA reference-signature logic "
            "using params_*.json and INPUT.RasOrb."
        )
    )

    parser.add_argument(
        "workdir",
        type=Path,
        help="Path to GA work_root, e.g. /home/jwolski/GA-quantum-basis/workspace",
    )

    parser.add_argument(
        "--cmocorr-disabled",
        action="store_true",
        help="Assume CMOCORR was disabled. Then reconstructed CMOCORR count is zero.",
    )

    parser.add_argument(
        "--no-assume-full-mask-reference",
        action="store_true",
        help=(
            "Do not assume full-mask reference when reference_sig_*.orb files are absent. "
            "By default, if no reference_sig files exist, the script assumes bootstrap created full-mask reference."
        ),
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional detailed CSV report path.",
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )

    parser.add_argument(
        "--show-cmocorr-cases",
        action="store_true",
        help="Print cases reconstructed as having CMOCORR.",
    )

    parser.add_argument(
        "--show-no-cmocorr-ok-cases",
        action="store_true",
        help="Print OK energy cases reconstructed as not having CMOCORR.",
    )

    args = parser.parse_args()

    root = args.workdir.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Workdir does not exist: {root}")

    if not root.is_dir():
        raise SystemExit(f"Workdir is not a directory: {root}")

    rows = [
        build_case_row(case_dir, root)
        for case_dir in sorted(iter_case_dirs(root), key=lambda p: str(p))
    ]

    initial_refs = read_initial_reference_signatures(root)

    # Jeśli reference_sig_*.orb zniknęły, ale używasz domyślnego bootstrapu,
    # to pełna maska jest pierwszą referencją.
    if not initial_refs and not args.no_assume_full_mask_reference:
        full_sig = infer_full_mask_signature(rows)

        if full_sig:
            initial_refs.add(full_sig)

    reconstruct_cmocorr(
        rows=rows,
        initial_refs=initial_refs,
        cmocorr_enabled=not args.cmocorr_disabled,
    )

    print_summary(rows, root, initial_refs)

    if args.show_cmocorr_cases:
        print_cmocorr_cases(rows)

    if args.show_no_cmocorr_ok_cases:
        print_no_cmocorr_cases(rows)

    if args.csv is not None:
        write_csv(rows, args.csv)
        print()
        print(f"CSV report saved: {args.csv}")

    if args.json is not None:
        write_json_summary(rows, root, initial_refs, args.json)
        print(f"JSON summary saved: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
