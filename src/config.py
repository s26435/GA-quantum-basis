from dataclasses import dataclass
from typing import Optional, Tuple
from .globals import BLOCKS, DEFAULT_WORK_ROOT, DEFAULT_REFERENCE_DIR


@dataclass(frozen=True)
class GA_cfg:
    # true value of energy - used in early stopping, may be None
    ground_truth: Optional[float] = -677.5

    # GA
    population_size: int = 30
    device: str = "cpu"
    generations: int = 1000
    genome_size: int = sum(BLOCKS)

    # when mask will be smaller then that algorithm will give it penalty
    min_mask_size: int = 14

    # mask weight multiplier
    start_lambda: float = 5e-4

    # ga auto stops whrn error is lower
    error_threshold_early_stopping: float = 0.0001
    early_stopping_patience: int = 50

    # GA FUNCTIONS
    elite_frac: float = 0.2
    tournament_k: int = 2
    crossover_p: float = 0.9
    # exponent mutation probability
    mutation_p: float = 0.4
    # strenght of mutation
    mutation_sigma: float = 0.6
    # mask mutation probability
    mask_flip_p: float = 0.01

    # GENERATOR PARAMETERS
    # latent space size
    zdim: int = 16
    # % generated genomes in population
    gen_percent: float = 0.5

    # generator training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    local_max_workers: int = 10

    # directories
    work_root: str = str(
        DEFAULT_WORK_ROOT
    )  # "/home/tezriem/Documents/GA-quantum-basis/workspace"
    python_bin: str = "/home/tezriem/miniconda3/envs/gabasis/bin/python"
    molcas_cmd: str = "molcas"
    molcas_root: str = "/home/tezriem/molcas"
    molcas_dir: str = "/home/tezriem/molcas"
    # CMOCORR
    cmocorr_enabled: bool = True
    cmocorr_ref_orb: Optional[str] = None
    cmocorr_bootstrap_from_seed: bool = True
    cmocorr_reference_dir: str = str(
        DEFAULT_REFERENCE_DIR
    )  # "/home/tezriem/Documents/GA-quantum-basis/workspace/reference"

    # TODO zmienić bo brzydkie af
    cmocorr_orbital_candidates: Tuple[str, ...] = ("RASORB",)
    cmocorr_t1: float = 0.90
    cmocorr_t2: float = 0.95
    cmocorr_lambda: float = 1.0
    cmocorr_fail_penalty: float = 1e4

    # logging
    log_level: int = 1

    # cache storage
    db_path: str = "cache/energy.sqlite"  # ca = "cache/energy_ca.sqlite"

    # model storage
    model_from_file: bool = False
    model_load_path: str = "model_be.ckpt"  #  ca = "model_ca.ckpt"
    model_save_path: str = "model_be.ckpt"