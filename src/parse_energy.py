import re
import sys
from pathlib import Path
from typing import Optional


FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"

HARD_FATAL = re.compile(
    r"Orthogonality violated|"
    r"_INTERNAL_ERROR_|"
    r"Premature abort|"
    r"Program aborted",
    re.IGNORECASE,
)

REFERENCE_WEIGHT = re.compile(
    rf"Reference\s+weight\s*:\s*({FLOAT})",
    re.IGNORECASE,
)

MIN_REFERENCE_WEIGHT = 0.6


def parse_float(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))


def reference_weight_too_small(text: str, threshold: float = MIN_REFERENCE_WEIGHT) -> bool:
    weights = [parse_float(m.group(1)) for m in REFERENCE_WEIGHT.finditer(text)]
    return bool(weights) and min(weights) < threshold


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def collect_root_energies(lines: list[str], method: str) -> list[tuple[int, float]]:
    patterns = [
        re.compile(
            rf"^\s*::.*?\b{method}\b.*?\bRoot\s*(?:number)?\s*(\d+)\b.*?\bTotal\s+energy\b\s*[:=]?\s*({FLOAT})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"^\s*::.*?\bRoot\s*(?:number)?\s*(\d+)\b.*?\b{method}\b.*?\bTotal\s+energy\b\s*[:=]?\s*({FLOAT})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"^\s*::.*?\b{method}\b.*?\broot\s+number\s*(\d+).*?\bTotal\s+energy\b\s*[:=]?\s*({FLOAT})",
            re.IGNORECASE,
        ),
    ]

    root_energies: list[tuple[int, float]] = []

    for line in lines:
        for pat in patterns:
            m = pat.search(line)
            if m:
                root = int(m.group(1))
                energy = parse_float(m.group(2))
                root_energies.append((root, energy))
                break

    return root_energies


def split_into_root_sets(root_energies: list[tuple[int, float]]) -> list[list[float]]:
    sets: list[list[float]] = []
    current: list[float] = []
    seen_roots: set[int] = set()

    for root, energy in root_energies:
        if root in seen_roots and current:
            sets.append(current)
            current = []
            seen_roots = set()

        current.append(energy)
        seen_roots.add(root)

    if current:
        sets.append(current)

    return sets


def average_roots(root_energies: list[tuple[int, float]]) -> float | None:
    if not root_energies:
        return None

    root_sets = split_into_root_sets(root_energies)

    root_means = [mean(root_set) for root_set in root_sets if root_set]

    if not root_means:
        return None

    return mean(root_means)


def parse_method_energy(lines: list[str], method: str) -> float | None:
    return average_roots(collect_root_energies(lines, method))


def main_with_return(path: Path | str) -> Optional[float]:
    path = Path(path)

    if not path.exists():
        return None

    txt = path.read_text(errors="ignore")

    if HARD_FATAL.search(txt) or reference_weight_too_small(txt):
        return None

    official_lines = [
        line for line in txt.splitlines() if line.lstrip().startswith("::")
    ]

    caspt2_energy = parse_method_energy(official_lines, "CASPT2")

    if caspt2_energy is not None:
        return caspt2_energy

    rasscf_energy = parse_method_energy(official_lines, "RASSCF")

    if rasscf_energy is not None:
        return rasscf_energy

    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("nan")
        return 2

    path = Path(sys.argv[1])

    if not path.exists():
        print("nan")
        return 2

    txt = path.read_text(errors="ignore")

    if HARD_FATAL.search(txt) or reference_weight_too_small(txt):
        print("nan")
        return 1

    official_lines = [
        line for line in txt.splitlines() if line.lstrip().startswith("::")
    ]

    caspt2_energy = parse_method_energy(official_lines, "CASPT2")

    if caspt2_energy is not None:
        print(caspt2_energy)
        return 0

    rasscf_energy = parse_method_energy(official_lines, "RASSCF")

    if rasscf_energy is not None:
        print(rasscf_energy)
        return 0

    print("nan")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())