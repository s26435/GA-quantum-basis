from pathlib import Path

log_handle = Path("out.log")
log_handle.touch(exist_ok=True)

population_handle = Path("populations.csv")
population_handle.touch(exist_ok=True)

BASE_DIR = Path("/home/jwolski/GA-quantum-basis/") # Path(__file__).resolve().parent
DEFAULT_WORK_ROOT = (BASE_DIR / "workspace").resolve()
DEFAULT_REFERENCE_DIR = (DEFAULT_WORK_ROOT / "reference").resolve()

# BLOCKS = [11, 7, 3, 2]
# BLOCKS = [14, 9, 4, 2]  # Be
# BLOCKS = [20, 16, 6, 4, 2]  # Ca
BLOCKS = [27, 24, 18, 14, 6, 3] # Thorium
# BLOCKS = [25, 18, 12, 8, 5, 3] # Cs
# BLOCKS = [23, 16, 12, 8, 6, 4] # Ru
# BLOCKS = [16, 11, 8, 6, 4, 3] # Si
# BLOCKS = [21, 15, 9, 7, 5, 3] # Se
