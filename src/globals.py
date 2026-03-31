from pathlib import Path

log_handle = Path("out.log")
log_handle.touch(exist_ok=True)

population_handle = Path("populations.csv")
population_handle.touch(exist_ok=True)

BASE_DIR = Path("/home/tezriem/Documents/GA-quantum-basis/") # Path(__file__).resolve().parent
DEFAULT_WORK_ROOT = (BASE_DIR / "workspace").resolve()
DEFAULT_REFERENCE_DIR = (DEFAULT_WORK_ROOT / "reference").resolve()

BLOCKS = [11, 7, 3, 2]
# BLOCKS = [14, 9, 4, 2]  # Be
# BLOCKS = [20, 16, 6, 4]  # Ca

