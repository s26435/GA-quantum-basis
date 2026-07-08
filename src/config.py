from dataclasses import dataclass, asdict
from typing import Optional, Tuple
from .globals import BLOCKS, DEFAULT_WORK_ROOT, DEFAULT_REFERENCE_DIR

from pathlib import Path
import json

@dataclass()
class GA_cfg:
    # true value of energy - used in early stopping, may be None
    ground_truth: Optional[float] = None
    error_ground_truth: Optional[float] = None
    # GA
    population_size: int = 30
    device: str = "cpu"
    generations: int = 1000
    genome_size: int = sum(BLOCKS)

    # when mask will be smaller then that algorithm will give it penalty
    min_mask_size: int = 19

    # mask weight multiplier
    start_lambda: float = 5e-2 
    end_lambda: float = 5e-3
    error_start: float = 5e-1 
    error_end:float = 3e-4

    # ga auto stops when error is lower
    error_threshold_early_stopping: float = 1e-6 
    early_stopping_patience: int = 500

    # GA FUNCTIONS
    elite_frac: float = 0.2
    tournament_k: int = 3
    crossover_p: float = 0.9
    # exponent mutation probability
    mutation_p: float = 0.6
    # strenght of mutation
    mutation_sigma: float = 10
    # mask mutation probability
    mask_flip_p: float = 0.01

    # GENERATOR PARAMETERS
    # latent space size
    zdim: int = 16
    # % generated genomes in population
    gen_percent: float = 0.3

    # generator training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    local_max_workers: int = 15

    # directories
    work_root: str = str(
        DEFAULT_WORK_ROOT
    )  # "/home/tezriem/Documents/GA-quantum-basis/workspace"
    python_bin: str = "/home/jwolski/GA-quantum-basis/.venv/bin/python"
    molcas_cmd: str = "molcas"
    molcas_root: str = "/home/jwolski/8.6.NEW.Valera.Valence.gradient.cmocorr/"
    molcas_dir: str = "/home/jwolski/8.6.NEW.Valera.Valence.gradient.cmocorr/"
    # CMOCORR
    cmocorr_enabled: bool = True
    cmocorr_ref_orb: Optional[str] = None
    cmocorr_bootstrap_from_seed: bool = True
    cmocorr_reference_dir: str = str(
        DEFAULT_REFERENCE_DIR
    )  # "/home/tezriem/Documents/GA-quantum-basis/workspace/reference"
    include_orginal_seed: bool = False
    
    cmocorr_t1: float = 0.90
    cmocorr_t2: float = 0.95
    cmocorr_lambda: float = 1e-6
    cmocorr_fail_penalty: float = 1e4

    # logging
    log_level: int = 1

    # cache storage
    db_path: str = "cache/energy.sqlite"  # ca = "cache/energy_ca.sqlite"

    # model storage
    model_from_file: bool = False
    model_load_path: str = "model_be.ckpt"  #  ca = "model_ca.ckpt"
    model_save_path: str = "model_be.ckpt"

    energy_run_timeout: int = 60*5

def save_config(config, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4)

def load_config(path: str | Path) -> GA_cfg:
    with open(path, "r") as f:
        data = json.load(f)
    return GA_cfg(**data)
