import shutil
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import traceback
import hashlib
import os

import numpy as np

from contextlib import contextmanager

from datetime import datetime

import torch
from torch import nn
import torch.distributions as D

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

from pathlib import Path

import csv

import sqlite3

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

BLOCKS = [11, 7, 3, 2]
# BLOCKS = [14, 9, 4, 2]  # Be
# BLOCKS = [20, 16, 6, 4]  # Ca

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WORK_ROOT = (BASE_DIR / "workspace").resolve()
DEFAULT_REFERENCE_DIR = (DEFAULT_WORK_ROOT / "reference").resolve()


@dataclass(frozen=True)
class GA_cfg:
    # true value of energy - used in early stopping, may be None
    ground_truth: Optional[float] = -14.668

    # GA
    population_size: int = 30
    device: str = "cpu"
    generations: int = 1000
    genome_size: int = sum(BLOCKS)

    # when mask will be smaller then that algorithm will give it penalty
    min_mask_size: int = 14

    # mask weight multiplier
    start_lambda: float = 5e-3

    # ga auto stops whrn error is lower
    error_threshold_early_stopping: float = 0.001
    early_stopping_patience: int = 30

    # GA FUNCTIONS
    elite_frac: float = 0.2
    tournament_k: int = 2
    crossover_p: float = 0.9
    # exponent mutation probability
    mutation_p: float = 0.4
    # strenght of mutation
    mutation_sigma: float = 0.2
    # mask mutation probability
    mask_flip_p: float = 0.05

    # GENERATOR PARAMETERS
    # latent space size
    zdim: int = 16
    # % generated genomes in population
    gen_percent: float = 0.5

    # generator training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    local_max_workers: int = 4

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


log_handle = Path("out.log")
log_handle.touch(exist_ok=True)

population_handle = Path("populations.csv")
population_handle.touch(exist_ok=True)


def materialize_rasorb_from_jobiph(
    wd: Path,
    molcas_cmd: str,
    molcas_root: str,
    fail_log: Path,
) -> Optional[Path]:
    wd = wd.resolve()

    rasorb = wd / "RASORB"
    if rasorb.exists() and rasorb.is_file():
        return rasorb

    jobiph_candidates = [
        wd / "JOBIPH",
        wd / "JobIph",
        wd / "INPUT.JobIph",
        wd / "molcas_work" / "JOBIPH",
        wd / "molcas_work" / "JobIph",
        wd / "molcas_work" / "INPUT.JobIph",
    ]
    jobiph = next((p for p in jobiph_candidates if p.exists() and p.is_file()), None)
    if jobiph is None:
        return None

    local_jobold = wd / "JOBOLD"
    if local_jobold.exists():
        local_jobold.unlink()
    shutil.copy2(jobiph, local_jobold)

    orbonly_inp = wd / "make_rasorb.input"
    orbonly_log = wd / "make_rasorb.log"

    orbonly_inp.write_text(
        "&RASSCF\nORBOnly\nOUTOrbital=Natural; 1\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    for k in [
        "MOLCAS",
        "WorkDir",
        "MOLCAS_WORKDIR",
        "MOLCAS_NEW_WORKDIR",
        "MOLCAS_OUTPUT",
        "MOLCAS_PROJECT",
        "MOLCAS_NPROCS",
    ]:
        env.pop(k, None)

    workdir = (wd / "orbonly_work").resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    env["MOLCAS"] = str(Path(molcas_root).resolve())
    env["WorkDir"] = str(workdir)
    env["MOLCAS_WORKDIR"] = str(workdir)
    env["MOLCAS_NEW_WORKDIR"] = "YES"
    env["MOLCAS_OUTPUT"] = "WORKDIR"
    env["MOLCAS_PROJECT"] = "NAME"
    env["MOLCAS_NPROCS"] = "1"

    with orbonly_log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [molcas_cmd, "make_rasorb.input"],
            cwd=str(wd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    for p in [wd / "RASORB", workdir / "RASORB"]:
        if p.exists() and p.is_file():
            if p.parent != wd:
                shutil.copy2(p, wd / "RASORB")
            return wd / "RASORB"

    with fail_log.open("a", encoding="utf-8") as f:
        f.write("\n[ORBONLY FAILED TO MATERIALIZE RASORB]\n")
        f.write(f"jobiph={jobiph}\n")
        f.write(f"orbonly_log={orbonly_log}\n")
        f.write(f"returncode={proc.returncode}\n")

    return None


def lg(text: str, log_level: int = 1):
    """
    Logs *text* into stdout and into out.log file.

    :param text: Text to log
    :type text: str
    :param log_level: Level of logging
    :type log_level: int
    """
    msg = "[" + str(datetime.now()) + "] " + text
    if log_level > 0:
        print(msg)
        with open(log_handle, "a") as file:
            file.write(msg + "\n")


@dataclass
class CaseResult:
    idx: int
    energy: float
    valid: int
    orbital_penalty: float
    total_loss: float
    mask_len: int
    orbital_file: Optional[str]
    run_out_path: str
    cmocorr_log_path: Optional[str]
    failure_reason: Optional[str] = None


class CacheDatabase:
    def __init__(self, db_path: Union[str, Path]):
        """
        Constructor for database for caching energy for coressponding alpha.

        :param db_path: path to .sqlite file
        :type db_path: Union[str, Path]
        """
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS energy_cache (
                    key TEXT PRIMARY KEY,
                    energy REAL NOT NULL,
                    orbital_penalty REAL NOT NULL DEFAULT 0.0,
                    total_loss REAL NOT NULL DEFAULT 0.0,
                    valid INTEGER NOT NULL,
                    status_json TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
            """)
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def check(self, key: str) -> Optional[dict]:
        """
        Checking if given key is in database

        :param key: hashed key of genome
        :type key: str
        :return: energy, valid or None if there is no key in database
        :rtype: Tuple[float, int] | None
        """
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT energy, orbital_penalty, total_loss, valid, status_json
                FROM energy_cache
                WHERE key=?;
                """,
                (key,),
            ).fetchone()
            if row is None:
                return None
            return {
                "energy": float(row[0]),
                "orbital_penalty": float(row[1]),
                "total_loss": float(row[2]),
                "valid": int(row[3]),
                "status_json": row[4],
            }

    def load(self, key: str, result: CaseResult):
        """
        Uploads genome to database

        :param key: hashed genome
        :type key: str
        :param energy: value of energy
        :type energy: float
        :param valid: 1 if is valid 0 if not
        :type valid: int
        """
        payload = {
            "idx": result.idx,
            "orbital_file": result.orbital_file,
            "run_out_path": result.run_out_path,
            "cmocorr_log_path": result.cmocorr_log_path,
            "failure_reason": result.failure_reason,
            "mask_len": result.mask_len,
        }
        with self.connect() as conn:
            conn.execute(
                """
                    INSERT OR REPLACE INTO energy_cache
                    (key, energy, orbital_penalty, total_loss, valid, status_json)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                (
                    key,
                    float(result.energy) if math.isfinite(result.energy) else 0.0,
                    float(result.orbital_penalty),
                    float(result.total_loss)
                    if math.isfinite(result.total_loss)
                    else 0.0,
                    int(result.valid),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )


def random_variation_ordered(seq):
    """
    Generates ordered ranomized variation of given seqence

    :param seq: template sequence
    """
    seq = list(seq)
    n = len(seq)
    sorted_seq = sorted(seq, reverse=True)
    new_sorted = [None] * n
    orig_max = sorted_seq[0]
    orig_second = sorted_seq[1] if n > 1 else orig_max
    lo = orig_second
    hi = orig_max * 1.1
    if hi < lo:
        hi = lo

    new_sorted[0] = random.uniform(lo, hi)
    for i in range(1, n):
        higher = new_sorted[i - 1]
        orig_val = sorted_seq[i]
        if i == n - 1:
            lo = orig_val * 0.9
            hi = min(orig_val * 1.1, higher)
        else:
            orig_lower_neighbor = sorted_seq[i + 1]
            lo = orig_lower_neighbor
            hi = higher
        if hi < lo:
            hi = lo

        new_sorted[i] = random.uniform(lo, hi)
    return new_sorted


def make_initial_population_from_seed(
    seed_alphas: List[float], pop_size: int, device: Union[str, torch.device] = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates initial population for GA

    :param seed_alphas: Seed
    :type seed_alphas: List[float]
    :param pop_size: size of given and returned population
    :type pop_size: int
    :param device: device of target tensors
    :type device: Union[str, torch.device]
    :return: tuple of exponents and masks
    :rtype: Tuple[Tensor, Tensor]
    """
    assert len(seed_alphas) == sum(BLOCKS), f"{len(seed_alphas)} is not {sum(BLOCKS)}"

    pop = []
    off = 0
    for _ in range(pop_size):
        indiv = []
        off = 0
        for b in BLOCKS:
            block = seed_alphas[off : off + b]
            new_block = random_variation_ordered(block)
            indiv.extend(new_block)
            off += b
        pop.append(indiv)

    alphas = torch.tensor(pop, dtype=torch.float32, device=device)

    genome = torch.log(alphas.clamp_min(1e-12))
    return genome, torch.ones_like(genome)


def sanitize_block_with_mask(
    a_block: torch.Tensor,
    m_block: torch.Tensor,
    lo: float,
    hi: float,
    min_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Makes sure that genome is valid and not extends given range

    :param a_block: genome tensor
    :type a_block: torch.Tensor
    :param m_block: mask tensor
    :type m_block: torch.Tensor
    :param lo: low value of clamp
    :type lo: float
    :param hi: high value of clamp
    :type hi: float
    :param min_ratio: maximum diffrence between exponents
    :type min_ratio: float
    :return: tuple of sanitized genome and mask
    :rtype: Tuple[Tensor, Tensor]
    """

    m = m_block > 0.5

    out_a = torch.zeros_like(a_block)
    out_m = torch.zeros_like(m_block, dtype=torch.float32)

    if m.sum().item() == 0:
        j = torch.argmax(a_block)
        m = torch.zeros_like(m, dtype=torch.bool)
        m[j] = True

    vals = a_block[m]
    vals = torch.nan_to_num(vals, nan=hi, posinf=hi, neginf=lo)
    vals = vals.clamp(lo, hi)
    vals, _ = torch.sort(vals, descending=True)

    for k in range(vals.numel() - 1):
        vals[k + 1] = torch.minimum(vals[k + 1], vals[k] / min_ratio)

    vals = vals.clamp(lo, hi)

    k = vals.numel()
    out_a[:k] = vals
    out_m[:k] = 1.0
    return out_a, out_m


def sanitize_blocks(
    a: torch.Tensor,
    mask: torch.Tensor,
    blocks=BLOCKS,
    lo: float = 1e-2,
    hi: float = 1e2,
    min_ratio: float = 1.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sanitizes blocks - aranges exponents in order

    :param a: Description
    :type a: torch.Tensor
    :param mask: Description
    :type mask: torch.Tensor
    :param blocks: Description
    :param lo: low value of clamp
    :type lo: float
    :param hi: high value of clamp
    :type hi: float
    :param min_ratio: maximum diffrence between exponents
    :type min_ratio: float
    :return: tuple of sanitized genome and mask
    :rtype: Tuple[Tensor, Tensor]
    """

    assert a.numel() == mask.numel(), (
        f"a and mask must have same length: {a.numel()} vs {mask.numel()}"
    )

    parts_a = []
    parts_m = []
    off = 0
    for n in blocks:
        a_block = a[off : off + n]
        m_block = mask[off : off + n]
        sa, sm = sanitize_block_with_mask(
            a_block, m_block, lo=lo, hi=hi, min_ratio=min_ratio
        )
        parts_a.append(sa)
        parts_m.append(sm)
        off += n

    return torch.cat(parts_a, dim=0), torch.cat(parts_m, dim=0)


def file_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_case_context(
    alphas: torch.Tensor,
    mask: torch.Tensor,
    template_path: Union[str, Path],
    ref_orb_path: Optional[Union[str, Path]],
    cmocorr_t1: float,
    cmocorr_t2: float,
    cmocorr_enabled: bool,
) -> str:
    base = hash_alphas(alphas, mask)
    h = hashlib.blake2b(digest_size=16)
    h.update(base.encode("utf-8"))
    h.update(str(Path(template_path)).encode("utf-8"))
    if Path(template_path).exists():
        h.update(file_sha256(template_path).encode("utf-8"))
    h.update(str(bool(cmocorr_enabled)).encode("utf-8"))
    h.update(f"{cmocorr_t1:.6f}|{cmocorr_t2:.6f}".encode("utf-8"))
    if ref_orb_path is not None:
        ref_orb_path = Path(ref_orb_path)
        h.update(str(ref_orb_path).encode("utf-8"))
        if ref_orb_path.exists():
            h.update(file_sha256(ref_orb_path).encode("utf-8"))
    return h.hexdigest()


def find_candidate_orbital_file(
    wd: Path,
    candidates: Tuple[str, ...] = ("INPUT.RasOrb",),
    project_name: Optional[str] = None,
    started_at: Optional[float] = None,
) -> Optional[Path]:
    p = (wd / "INPUT.RasOrb").resolve()
    if p.exists() and p.is_file():
        return p
    return None


def run_cmocorr(
    wd: Path,
    molcas_cmd: str,
    molcas_root: str,
    python_bin: str,
    ref_orb: Union[str, Path],
    chk_orb: Union[str, Path],
    t1: float,
    t2: float,
    fail_penalty: float,
) -> Tuple[float, Optional[Path], Optional[str]]:
    wd = Path(wd).resolve()
    cmowd = (wd / "molcas_work").resolve() #/ "cmocorr").resolve() # TODO
    cmowd.mkdir(parents=True, exist_ok=True)

    ref_orb = Path(ref_orb).resolve()
    chk_orb = Path(chk_orb).resolve()

    if not ref_orb.exists():
        return fail_penalty, None, f"CMOCORR ref orbital file not found: {ref_orb}"
    if not chk_orb.exists():
        return fail_penalty, None, f"CMOCORR chk orbital file not found: {chk_orb}"

    cmoref = cmowd / "CMOREF"
    cmochk = cmowd / "CMOCHK"
    inp = cmowd / "cmocorr.input"
    log = cmowd / "cmocorr.log"

    for p in (cmoref, cmochk, inp, log):
        try:
            if p.exists() or p.is_symlink():
                p.unlink()
        except IsADirectoryError:
            shutil.rmtree(p)

    shutil.copy2(ref_orb, cmoref)
    shutil.copy2(chk_orb, cmochk)

    # copy_runfile_as_cmocorr(
    #     cmowd=cmowd
    # )

    inp.write_text(
        f"&CMOCORR\nDoOrbitals\nThresholds\n {t1} {t2}\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    for k in [
        "MOLCAS",
        "WorkDir",
        "MOLCAS_WORKDIR",
        "MOLCAS_NEW_WORKDIR",
        "MOLCAS_OUTPUT",
        "MOLCAS_PROJECT",
        "MOLCAS_NPROCS",
    ]:
        env.pop(k, None)

    env["MOLCAS"] = str(Path(molcas_root).resolve())
    env["WorkDir"] = str(cmowd)
    env["MOLCAS_WORKDIR"] = str(cmowd)
    env["MOLCAS_NEW_WORKDIR"] = "YES"
    env["MOLCAS_OUTPUT"] = "WORKDIR"
    env["MOLCAS_PROJECT"] = "NAME"
    env["MOLCAS_NPROCS"] = "1"

    assert (cmowd / "CMOREF").exists(), f"Missing {cmowd / 'CMOREF'}"
    assert (cmowd / "CMOCHK").exists(), f"Missing {cmowd / 'CMOCHK'}"

    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [molcas_cmd, "cmocorr.input"],
            cwd=str(cmowd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    
    with log.open("w", encoding="utf-8") as f:
        f.write(f"[DEBUG] cwd={cmowd}\n")
        f.write(f"[DEBUG] CMOREF exists={(cmowd / 'CMOREF').exists()}\n")
        f.write(f"[DEBUG] CMOCHK exists={(cmowd / 'CMOCHK').exists()}\n")
        f.write(f"[DEBUG] files={sorted(p.name for p in cmowd.iterdir())}\n\n")
        proc = subprocess.run(
            [molcas_cmd, "cmocorr.input"],
            cwd=str(cmowd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        debug = sorted([p.name for p in cmowd.iterdir()])
        return (
            fail_penalty,
            log,
            f"CMOCORR failed, rc={proc.returncode}, files_in_cmowd={debug}",
        )

    parse_cmd = [
        python_bin,
        str(wd.parent / "parse_cmocorr.py"),
        str(log),
        "--t1",
        str(t1),
    ]
    p = subprocess.run(parse_cmd, capture_output=True, text=True)

    if p.returncode != 0:
        return (
            fail_penalty,
            log,
            f"parse_cmocorr failed, rc={p.returncode}, stderr={p.stderr}",
        )

    try:
        data = json.loads(p.stdout)
        penalty = float(data.get("penalty", fail_penalty))
        return penalty, log, None
    except Exception as e:
        return fail_penalty, log, f"parse_cmocorr json decode failed: {e!r}"


def copy_runfile_as_cmocorr(
    cmowd: Union[str, Path]
) -> Path:
    """
    Copies an existing RunFile into cmowd as 'cmocorr.RunFile'.
    """
    cmowd = Path(cmowd).resolve()
    cmowd.mkdir(parents=True, exist_ok=True)

    dst = cmowd / "cmocorr.RunFile"
    target = Path("../molcas_work/INPUT.RunFile")

    shutil.copy2(target, dst)
    return dst

def _run_energy_case(
    i: int,
    alphas: List[float],
    mask: List[bool],
    gen_dir_str: str,
    python_bin: str,
    molcas_cmd: str,
    molcas_root: str,
    case: str,
    mask_lambda: float,
    cmocorr_enabled: bool,
    cmocorr_ref_orb: Optional[str],
    cmocorr_orbital_candidates: Tuple[str, ...],
    cmocorr_t1: float,
    cmocorr_t2: float,
    cmocorr_lambda: float,
    cmocorr_fail_penalty: float,
) -> CaseResult:
    gen_dir = Path(gen_dir_str).resolve()
    wd = (gen_dir / f"{case}_case_{i:04d}").resolve()
    wd.mkdir(parents=True, exist_ok=True)

    molcas_workdir = (wd / "molcas_work").resolve()
    molcas_workdir.mkdir(parents=True, exist_ok=True)

    run_out = wd / "run.out"
    run_out.write_text("", encoding="utf-8")
    energy_txt = wd / "energy.txt"

    mask_len = int(sum(bool(x) for x in mask))

    def _fail(reason: str) -> CaseResult:
        return CaseResult(
            idx=i,
            energy=float("nan"),
            valid=0,
            orbital_penalty=cmocorr_fail_penalty if cmocorr_enabled else 0.0,
            total_loss=float("inf"),
            mask_len=mask_len,
            orbital_file=None,
            run_out_path=str(run_out),
            cmocorr_log_path=None,
            failure_reason=reason,
        )

    try:
        if len(alphas) != len(mask):
            run_out.write_text(
                f"[BAD INPUT] len(alphas)={len(alphas)} != len(mask)={len(mask)}\n",
                encoding="utf-8",
            )
            energy_txt.write_text("nan\n", encoding="utf-8")
            return _fail("bad_input_length")

        params_path = wd / f"params_{case}_{i:04d}.json"
        params_path.write_text(
            json.dumps({"alphas": alphas, "mask": mask}),
            encoding="utf-8",
        )

        build_cmd = [
            python_bin,
            str(gen_dir / "build_input.py"),
            str(params_path),
            str(wd / "INPUT"),
            "--template",
            str(gen_dir / "INPUT.template"),
            "--use-mask",
        ]
        p = subprocess.run(build_cmd, capture_output=True, text=True)
        if p.returncode != 0:
            energy_txt.write_text("nan\n", encoding="utf-8")
            run_out.write_text(
                "[build_input FAILED]\n"
                f"returncode: {p.returncode}\n"
                f"cmd: {build_cmd}\n\n"
                f"STDOUT:\n{p.stdout}\n\n"
                f"STDERR:\n{p.stderr}\n",
                encoding="utf-8",
            )
            return _fail("build_input_failed")

        env = os.environ.copy()
        for k in [
            "MOLCAS",
            "WorkDir",
            "MOLCAS_WORKDIR",
            "MOLCAS_NEW_WORKDIR",
            "MOLCAS_OUTPUT",
            "MOLCAS_PROJECT",
            "MOLCAS_NPROCS",
        ]:
            env.pop(k, None)

        env["MOLCAS"] = str(Path(molcas_root).resolve())
        env["WorkDir"] = str(molcas_workdir)
        env["MOLCAS_WORKDIR"] = str(molcas_workdir)
        env["MOLCAS_NEW_WORKDIR"] = "YES"
        env["MOLCAS_OUTPUT"] = "WORKDIR"
        env["MOLCAS_PROJECT"] = "NAME"
        env["MOLCAS_NPROCS"] = "1"

        # jeden open("w"), bez kasowania własnego debugu
        with run_out.open("w", encoding="utf-8") as f:
            f.write(f"[DEBUG] molcas_cmd={molcas_cmd}\n")
            f.write(f"[DEBUG] molcas_root={molcas_root}\n")
            f.write(f"[DEBUG] cwd={wd}\n")
            f.write(f"[DEBUG] workdir={molcas_workdir}\n\n")

            p3 = subprocess.run(
                [molcas_cmd, "INPUT"],
                cwd=str(wd),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        if p3.returncode != 0:
            energy_txt.write_text("nan\n", encoding="utf-8")

            try:
                tail_lines = run_out.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()[-120:]
                tail_text = "\n".join(tail_lines)
            except Exception:
                tail_text = "<could not read run.out>"

            with run_out.open("a", encoding="utf-8") as f:
                f.write(f"\n[molcas FAILED] returncode={p3.returncode}\n")
                f.write(f"[molcas_workdir] {molcas_workdir}\n")
                f.write("[tail run.out]\n")
                f.write(tail_text + "\n")

            return _fail(
                f"molcas_failed_rc_{p3.returncode}; "
                f"run_out={run_out}; workdir={molcas_workdir}; "
                f"tail={tail_text[-1000:]}"
            )

        parse_cmd = [python_bin, str(gen_dir / "parse_energy.py"), str(run_out)]
        p2 = subprocess.run(parse_cmd, capture_output=True, text=True)
        if p2.returncode != 0:
            energy_txt.write_text("nan\n", encoding="utf-8")
            with run_out.open("a", encoding="utf-8") as f:
                f.write("\n[parse_energy FAILED]\n")
                f.write(f"returncode: {p2.returncode}\n")
                f.write(f"cmd: {parse_cmd}\n")
                f.write("\nSTDOUT:\n" + p2.stdout + "\n")
                f.write("\nSTDERR:\n" + p2.stderr + "\n")
            return _fail("parse_energy_failed")

        energy = float(p2.stdout.strip())
        energy_txt.write_text(f"{energy}\n", encoding="utf-8")

        orbital_penalty = 0.0
        cmocorr_log_path = None
        failure_reason = None

        project = "INPUT"

        orb_candidates = [
            molcas_workdir / f"{project}.RasOrb.1",
            molcas_workdir / f"{project}.RasOrb",
            wd / f"{project}.RasOrb.1",
            wd / f"{project}.RasOrb",
            molcas_workdir / "RASORB",
            wd / "RASORB",
        ]

        orb = None
        for p in orb_candidates:
            if p.exists() and p.is_file():
                orb = p.resolve()
                break

        if orb is not None and orb.parent != wd:
            target = wd / orb.name
            shutil.copy2(orb, target)
            orb = target.resolve()

        orbital_file = str(orb) if orb else None

        if cmocorr_enabled:
            if not cmocorr_ref_orb:
                orbital_penalty = cmocorr_fail_penalty
                failure_reason = "cmocorr_ref_orb_missing"
            elif orb is None:
                orbital_penalty = cmocorr_fail_penalty
                failure_reason = "candidate_orbital_not_found"
            else:
                raw_penalty, cmolog, cmofail = run_cmocorr(
                    wd=wd,
                    molcas_cmd=molcas_cmd,
                    molcas_root=molcas_root,
                    python_bin=python_bin,
                    ref_orb=cmocorr_ref_orb,
                    chk_orb=orb,
                    t1=cmocorr_t1,
                    t2=cmocorr_t2,
                    fail_penalty=cmocorr_fail_penalty,
                )
                orbital_penalty = raw_penalty
                cmocorr_log_path = str(cmolog) if cmolog else None
                if cmofail is not None:
                    failure_reason = (
                        f"{cmofail}; ref={cmocorr_ref_orb}; chk={orb}; "
                        f"wd_files={[p.name for p in sorted(wd.iterdir())]}"
                    )

        total_loss = (
            float(energy) + mask_lambda * mask_len + cmocorr_lambda * orbital_penalty
        )

        return CaseResult(
            idx=i,
            energy=float(energy),
            valid=1,
            orbital_penalty=float(orbital_penalty),
            total_loss=float(total_loss),
            mask_len=mask_len,
            orbital_file=orbital_file,
            run_out_path=str(run_out),
            cmocorr_log_path=cmocorr_log_path,
            failure_reason=failure_reason,
        )

    except Exception as e:
        tb = traceback.format_exc()
        energy_txt.write_text("nan\n", encoding="utf-8")
        with run_out.open("a", encoding="utf-8") as f:
            f.write("\n[WORKER CRASH]\n")
            f.write(f"type: {type(e).__name__}\n")
            f.write(f"repr: {repr(e)}\n")
            f.write("\nTRACEBACK:\n")
            f.write(tb)
        return _fail(f"worker_crash_{type(e).__name__}: {e!r}")


def hash_alphas(alphas: torch.Tensor, mask: torch.Tensor) -> str:
    """
    Hashing algorithm, that turns tensor of exponents into key for caching database

    :param alphas: tensor of exponents
    :type alphas: torch.Tensor
    :param mask: mask
    :type mask: torch.Tensor
    :return: hashed alphas
    :rtype: str
    """

    if isinstance(alphas, torch.Tensor):
        a = alphas.detach().cpu().numpy().astype(np.float64, copy=False)
    elif isinstance(alphas, np.ndarray):
        a = alphas.astype(np.float64, copy=False)
    else:
        a = np.asarray(list(alphas), dtype=np.float64)

    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    elif isinstance(mask, np.ndarray):
        m = mask.astype(np.uint8, copy=False)
    else:
        m = np.asarray(list(mask), dtype=np.uint8)

    a = np.concatenate(
        [
            np.round(
                np.ravel(a),
                decimals=7,
            ),
            m,
        ]
    )

    return hashlib.blake2b(a.tobytes(order="C"), digest_size=16).hexdigest()


class PopGenerator(nn.Module):
    def __init__(
        self, population_size: int, genome_size: int, zdim: int, *args, **kwargs
    ):
        """
        Constructor for popultion generator. population_size, genome_size and zdim have to match with rest of the code

        :param population_size: size of populaion
        :type population_size: int
        :param genome_size: size of one genome
        :typr genome_size: int
        :param zdim: size of latent space
        :type zdim: int
        """
        super(PopGenerator, self).__init__(*args, **kwargs)

        self.zdim = zdim
        self.gen_size = genome_size
        self.pop_size = population_size

        self.net = nn.Sequential(
            nn.Linear(zdim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 2 * genome_size),
        )

        self.mask_encoder = nn.Sequential(
            nn.Linear(zdim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, genome_size),
        )

    def sample(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples given latent space z

        :param z: latent tensor
        :type z: torch.Tensor
        :return: tuple of generated exponents, log of probability of generated exponents, generated mask logits, and log of probability of generated mask logits
        :rtype: Tuple[Tensor, Tensor, Tensor, Tensor]
        """
        out = self.net(z)
        mu, log_std = out.chunk(2, dim=1)
        log_std = log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std)

        eps = torch.randn_like(std)
        x = mu + eps * std
        x_det = x.detach()
        logp_per_dim = -0.5 * (
            ((x_det - mu) / std) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        logp = logp_per_dim.sum(dim=1)

        logits = self.mask_encoder(z)
        probs = torch.sigmoid(logits)
        b = D.Bernoulli(probs=probs)
        m_sample = b.sample()
        logp_m = b.log_prob(m_sample).sum(dim=1)

        m_st = m_sample + probs - probs.detach()

        logp = logp + logp_m
        return x, logp, log_std, m_st

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrapper for sample - used only for use when you don't want to train the net

        :param z: latent tensor
        :type z: torch.Tensor
        :return: tuple of exponends and mask
        :rtype: Tuple[Tensor, Tensor]
        """
        x, _, _, m = self.sample(z)
        return x, m


class GaussianPolicy(nn.Module):
    def __init__(self, pop_size: int, zdim: int):
        """
        Constructor for latent space. population_size and zdim have to match with rest of the code

        :param pop_size: size of populations
        :type pop_size: int
        :param zdim: size of latent space
        :type zdim: int
        """
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(pop_size, zdim))
        self.log_std = nn.Parameter(torch.zeros(pop_size, zdim))

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples latent space

        :return: tuple of latent and log probability of latent
        :rtype: Tuple[Tensor, Tensor]
        """
        log_std = self.log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        x = self.mu + eps * std
        x_det = x.detach()
        logp_per_dim = -0.5 * (
            ((x_det - self.mu) / std) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        logp = logp_per_dim.sum(dim=1)
        return x, logp


class GA:
    def __init__(self, config: GA_cfg):
        """
        Constructor for genetic algorithm object

        :param config: configuration object
        :type config: GA_cfg
        """

        self.cfg = config
        self.device = config.device

        self.baseline = None
        self.baseline_beta = 0.9

        self.energy_cache = CacheDatabase(config.db_path)
        self.n_elite = max(1, int(self.cfg.elite_frac * self.cfg.population_size))
        self.n_gen = max(
            0,
            min(
                int(self.cfg.population_size * self.cfg.gen_percent),
                self.cfg.population_size - self.n_elite,
            ),
        )

        if self.n_gen <= 0:
            raise ValueError(
                f"Number of generated genomes must be greater then 0 but is {self.n_gen}"
            )

        self.n_offspring = max(0, self.cfg.population_size - self.n_elite - self.n_gen)

        self.generator = PopGenerator(
            population_size=self.n_gen,
            genome_size=config.genome_size,
            zdim=config.zdim,
        ).to(self.device)

        self.policy = GaussianPolicy(self.n_gen, self.cfg.zdim).to(self.device)

        self.opt = torch.optim.AdamW(
            self.policy.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        self.opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        if not Path(self.cfg.model_load_path).exists() and self.cfg.model_from_file:
            raise FileNotFoundError(
                f"Model checkpoint file not found: {self.cfg.model_load_path}"
            )

        if self.cfg.model_from_file:
            ckpt = torch.load(self.cfg.model_load_path, map_location=self.cfg.device)

            self.generator.load_state_dict(ckpt["generator"])
            self.policy.load_state_dict(ckpt["policy"])

            self.opt.load_state_dict(ckpt["opt"])
            self.opt_g.load_state_dict(ckpt["opt_g"])

        self.err_ema = None
        self.err_ema_beta = 0.9

        self._cmocorr_refs_by_sig: dict[str, str] = {}
        self._resolved_cmocorr_ref_orb: Optional[str] = None
        self._last_raw_energies: Optional[torch.Tensor] = None

    def _reference_file_for_len(self, mask_len: int) -> Path:
        ref_dir = Path(self.cfg.cmocorr_reference_dir).resolve()
        ref_dir.mkdir(parents=True, exist_ok=True)
        return ref_dir / f"reference_len_{mask_len}.orb"

    def _load_persisted_references(self):
        ref_dir = Path(self.cfg.cmocorr_reference_dir).resolve()
        if not ref_dir.exists():
            return

        for p in ref_dir.glob("reference_len_*.orb"):
            try:
                mask_len = int(p.stem.split("_")[-1])
            except ValueError:
                continue
            self._cmocorr_refs_by_len[mask_len] = str(p.resolve())

    def _mask_signature(self, mask: Union[List[bool], torch.Tensor]) -> str:
        """
        Signature of sanitized mask.
        Because sanitize_blocks packs active orbitals to the front of each block,
        this signature uniquely determines active basis structure.
        """
        if isinstance(mask, torch.Tensor):
            mask = (mask.detach().cpu() > 0.5).to(torch.uint8).tolist()

        return "".join("1" if bool(x) else "0" for x in mask)

    def _reference_file_for_signature(self, sig: str) -> Path:
        ref_dir = Path(self.cfg.cmocorr_reference_dir).resolve()
        ref_dir.mkdir(parents=True, exist_ok=True)
        return ref_dir / f"reference_sig_{sig}.orb"

    def _load_persisted_references(self):
        ref_dir = Path(self.cfg.cmocorr_reference_dir).resolve()
        if not ref_dir.exists():
            return

        for p in ref_dir.glob("reference_sig_*.orb"):
            name = p.stem  # reference_sig_101110...
            sig = name[len("reference_sig_") :]
            if sig:
                self._cmocorr_refs_by_sig[sig] = str(p.resolve())

    def _register_reference_for_signature(
        self,
        sig: str,
        orbital_file: Union[str, Path],
        overwrite: bool = False,
    ):
        src = Path(orbital_file).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Reference source orbital does not exist: {src}")

        dst = self._reference_file_for_signature(sig)

        if dst.exists() and not overwrite:
            self._cmocorr_refs_by_sig[sig] = str(dst.resolve())
            return

        shutil.copy2(src, dst)
        self._cmocorr_refs_by_sig[sig] = str(dst.resolve())

    def _pick_reference_for_signature(self, sig: str) -> Optional[str]:
        if not self.cfg.cmocorr_enabled:
            return None
        return self._cmocorr_refs_by_sig.get(sig)

    def _register_reference_for_length(
        self,
        mask_len: int,
        orbital_file: Union[str, Path],
        overwrite: bool = False,
    ):
        src = Path(orbital_file).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Reference source orbital does not exist: {src}")

        dst = self._reference_file_for_len(mask_len)

        if dst.exists() and not overwrite:
            self._cmocorr_refs_by_len[mask_len] = str(dst.resolve())
            return

        shutil.copy2(src, dst)
        self._cmocorr_refs_by_len[mask_len] = str(dst.resolve())

        lg(
            f"Registered CMOCORR reference for mask_len={mask_len}: {dst}",
            self.cfg.log_level,
        )

    def _pick_reference_for_length(self, mask_len: int) -> Optional[str]:
        """
        Exact match if available.
        Otherwise pick the smallest known reference with length >= mask_len.
        If none exists, fall back to the largest known reference.
        """
        if not self.cfg.cmocorr_enabled:
            return None

        if mask_len in self._cmocorr_refs_by_len:
            return self._cmocorr_refs_by_len[mask_len]

        if not self._cmocorr_refs_by_len:
            return None

        known = sorted(self._cmocorr_refs_by_len.keys())

        longer_or_equal = [k for k in known if k >= mask_len]
        chosen = longer_or_equal[0] if longer_or_equal else known[-1]
        return self._cmocorr_refs_by_len[chosen]

    def _prepare_case_files(self, target_dir: Path):
        target_dir.mkdir(parents=True, exist_ok=True)
        base = Path(__file__).resolve().parent

        shutil.copy2(base / "INPUT", target_dir / "INPUT.template")
        shutil.copy2(base / "build_input.py", target_dir / "build_input.py")
        shutil.copy2(base / "parse_energy.py", target_dir / "parse_energy.py")

        if self.cfg.cmocorr_enabled:
            shutil.copy2(base / "parse_cmocorr.py", target_dir / "parse_cmocorr.py")

    def _resolve_cmocorr_reference(self, seed_alphas: List[float]):
        if not self.cfg.cmocorr_enabled:
            self._cmocorr_refs_by_sig = {}
            self._resolved_cmocorr_ref_orb = None
            return

        self._load_persisted_references()

        full_mask = [True] * len(seed_alphas)
        seed_sig = self._mask_signature(full_mask)

        if self.cfg.cmocorr_ref_orb is not None:
            p = Path(self.cfg.cmocorr_ref_orb).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"CMOCORR reference orbital file not found: {p}")

            self._cmocorr_refs_by_sig[seed_sig] = str(p)
            self._resolved_cmocorr_ref_orb = str(p)

            lg(
                f"Using external CMOCORR reference for signature={seed_sig}: {p}",
                self.cfg.log_level,
            )
            return

        cached_ref = self._reference_file_for_signature(seed_sig)
        if cached_ref.exists():
            self._cmocorr_refs_by_sig[seed_sig] = str(cached_ref.resolve())
            self._resolved_cmocorr_ref_orb = str(cached_ref.resolve())

            lg(
                f"Using cached bootstrap CMOCORR reference for signature={seed_sig}: {cached_ref}",
                self.cfg.log_level,
            )
            return

        if not self.cfg.cmocorr_bootstrap_from_seed:
            raise RuntimeError(
                "CMOCORR is enabled, but no cmocorr_ref_orb was given and bootstrap is disabled."
            )

        ref_dir = Path(self.cfg.cmocorr_reference_dir).resolve()
        self._prepare_case_files(ref_dir)

        molcas_cmd = getattr(self.cfg, "molcas_cmd", self.cfg.molcas_dir)

        lg(
            f"Building bootstrap CMOCORR reference from original seed for signature={seed_sig}...",
            self.cfg.log_level,
        )

        result = _run_energy_case(
            i=0,
            alphas=list(seed_alphas),
            mask=full_mask,
            gen_dir_str=str(ref_dir),
            python_bin=self.cfg.python_bin,
            molcas_cmd=molcas_cmd,
            molcas_root=self.cfg.molcas_root,
            case="reference",
            mask_lambda=0.0,
            cmocorr_enabled=False,
            cmocorr_ref_orb=None,
            cmocorr_orbital_candidates=self.cfg.cmocorr_orbital_candidates,
            cmocorr_t1=self.cfg.cmocorr_t1,
            cmocorr_t2=self.cfg.cmocorr_t2,
            cmocorr_lambda=0.0,
            cmocorr_fail_penalty=self.cfg.cmocorr_fail_penalty,
        )

        if result.valid != 1 or not math.isfinite(result.energy):
            raise RuntimeError(
                "Bootstrap reference calculation failed:\n"
                f"reason: {result.failure_reason}\n"
                f"run_out: {result.run_out_path}\n"
                f"orbital_file: {result.orbital_file}\n"
            )

        if result.orbital_file is None:
            raise RuntimeError(
                "Bootstrap reference calculation finished, but no orbital file was found.\n"
                f"run_out: {result.run_out_path}\n"
                f"failure_reason: {result.failure_reason}\n"
            )

        self._register_reference_for_signature(
            sig=seed_sig,
            orbital_file=result.orbital_file,
            overwrite=True,
        )

        self._resolved_cmocorr_ref_orb = self._cmocorr_refs_by_sig[seed_sig]

        lg(
            f"Bootstrap CMOCORR reference created for signature={seed_sig}: "
            f"{self._resolved_cmocorr_ref_orb}",
            self.cfg.log_level,
        )

    def fitness(
        self, population: torch.Tensor, pop_mask: torch.Tensor, case: str
    ) -> torch.Tensor:
        """
        Fitness function for population.

        total_loss = energy + lambda_mask * mask_len + lambda_cmocorr * orbital_penalty

        CMOCORR reference is chosen by sanitized mask signature, not by mask length.
        If a new valid individual with unseen signature appears, its orbital file
        becomes the reference for that signature for future evaluations.
        """

        B = population.size(0)
        gen_dir = Path(self.cfg.work_root) / f"gen_{self._gen_idx:04d}"
        self._prepare_case_files(gen_dir)

        pop_cpu = population.detach().cpu()
        mask_cpu = pop_mask.detach().cpu()

        alphas_all: List[List[float]] = []
        mask_all: List[List[bool]] = []
        mask_lens: List[int] = []
        signatures: List[str] = []
        ref_paths: List[Optional[str]] = []
        keys: List[str] = []
        cmocorr_enabled_flags: List[bool] = []

        mask_lambda = float(getattr(self, "_current_lambda", self.cfg.start_lambda))

        for i in range(B):
            a = torch.exp(pop_cpu[i])
            m_i = (mask_cpu[i] > 0.5).float()

            a_s, m_s = sanitize_blocks(a, m_i, lo=1e-6, hi=1e5, min_ratio=1.2)

            alphas_i = a_s.tolist()
            mask_i = [bool(x) for x in m_s.tolist()]
            mask_len_i = int(sum(mask_i))
            sig_i = self._mask_signature(mask_i)

            ref_path_i = self._pick_reference_for_signature(sig_i)

            # CMOCORR only if we actually have a compatible reference for this signature
            cmocorr_enabled_i = bool(
                self.cfg.cmocorr_enabled and ref_path_i is not None
            )

            key = hash_case_context(
                alphas=a_s,
                mask=m_s,
                template_path=gen_dir / "INPUT.template",
                ref_orb_path=ref_path_i,
                cmocorr_t1=self.cfg.cmocorr_t1,
                cmocorr_t2=self.cfg.cmocorr_t2,
                cmocorr_enabled=cmocorr_enabled_i,
            )

            alphas_all.append(alphas_i)
            mask_all.append(mask_i)
            mask_lens.append(mask_len_i)
            signatures.append(sig_i)
            ref_paths.append(ref_path_i)
            keys.append(key)
            cmocorr_enabled_flags.append(cmocorr_enabled_i)

        losses: List[Optional[float]] = [None] * B
        raw_energies: List[Optional[float]] = [None] * B

        miss: List[
            Tuple[
                int,  # idx
                List[float],  # alphas
                List[bool],  # mask
                int,  # mask_len
                str,  # signature
                Optional[str],  # ref_path
                bool,  # cmocorr_enabled_i
                str,  # cache_key
            ]
        ] = []

        for i, key in enumerate(keys):
            hit = self.energy_cache.check(key)

            if hit is None:
                miss.append(
                    (
                        i,
                        alphas_all[i],
                        mask_all[i],
                        mask_lens[i],
                        signatures[i],
                        ref_paths[i],
                        cmocorr_enabled_flags[i],
                        key,
                    )
                )
                continue

            total_loss = float(hit["total_loss"])
            energy = float(hit["energy"])
            valid = int(hit["valid"])

            if valid == 1 and math.isfinite(total_loss):
                losses[i] = total_loss
                raw_energies[i] = energy
            else:
                miss.append(
                    (
                        i,
                        alphas_all[i],
                        mask_all[i],
                        mask_lens[i],
                        signatures[i],
                        ref_paths[i],
                        cmocorr_enabled_flags[i],
                        key,
                    )
                )

        penalty = 1e4

        if miss:
            python_bin = self.cfg.python_bin
            molcas_cmd = getattr(self.cfg, "molcas_cmd", self.cfg.molcas_dir)
            max_workers = max(1, int(self.cfg.local_max_workers))

            miss_ok: List[
                Tuple[
                    int,
                    List[float],
                    List[bool],
                    int,
                    str,
                    Optional[str],
                    bool,
                    str,
                ]
            ] = []

            for (
                idx,
                alphas,
                msk,
                mask_len,
                sig,
                ref_path,
                cmocorr_enabled_i,
                key,
            ) in miss:
                if sum(msk) < self.cfg.min_mask_size:
                    result = CaseResult(
                        idx=idx,
                        energy=float("nan"),
                        valid=0,
                        orbital_penalty=0.0,
                        total_loss=penalty * 10.0,
                        mask_len=sum(msk),
                        orbital_file=None,
                        run_out_path="",
                        cmocorr_log_path=None,
                        failure_reason="mask_too_small",
                    )
                    losses[idx] = result.total_loss
                    raw_energies[idx] = float("nan")
                    self.energy_cache.load(key, result)
                else:
                    miss_ok.append(
                        (
                            idx,
                            alphas,
                            msk,
                            mask_len,
                            sig,
                            ref_path,
                            cmocorr_enabled_i,
                            key,
                        )
                    )

            if miss_ok:
                futures = {}

                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    for (
                        idx,
                        alphas_i,
                        mask_i,
                        mask_len_i,
                        sig_i,
                        ref_path_i,
                        cmocorr_enabled_i,
                        cache_key,
                    ) in miss_ok:
                        fut = ex.submit(
                            _run_energy_case,
                            idx,
                            alphas_i,
                            mask_i,
                            str(gen_dir),
                            python_bin,
                            molcas_cmd,
                            self.cfg.molcas_root,
                            case,
                            mask_lambda,
                            cmocorr_enabled_i,  # <- per candidate
                            ref_path_i,  # <- compatible ref or None
                            self.cfg.cmocorr_orbital_candidates,
                            self.cfg.cmocorr_t1,
                            self.cfg.cmocorr_t2,
                            self.cfg.cmocorr_lambda,
                            self.cfg.cmocorr_fail_penalty,
                        )
                        futures[fut] = (
                            cache_key,
                            sig_i,
                            mask_len_i,
                            cmocorr_enabled_i,
                        )

                    for fut in as_completed(futures):
                        cache_key, sig_i, mask_len_i, cmocorr_enabled_i = futures[fut]

                        try:
                            result: CaseResult = fut.result()
                        except Exception as e:
                            lg(
                                f"Worker future crashed for signature={sig_i}: {e!r}",
                                self.cfg.log_level,
                            )
                            continue

                        idx = result.idx

                        losses[idx] = (
                            float(result.total_loss)
                            if math.isfinite(float(result.total_loss))
                            else penalty
                        )

                        raw_energies[idx] = (
                            float(result.energy)
                            if math.isfinite(float(result.energy))
                            else float("nan")
                        )

                        self.energy_cache.load(cache_key, result)

                        # If this signature had no reference yet, first valid result becomes reference
                        if (
                            self.cfg.cmocorr_enabled
                            and (
                                not cmocorr_enabled_i
                            )  # no ref existed before this run
                            and result.valid == 1
                            and math.isfinite(float(result.energy))
                            and result.orbital_file is not None
                            and sig_i not in self._cmocorr_refs_by_sig
                        ):
                            try:
                                self._register_reference_for_signature(
                                    sig=sig_i,
                                    orbital_file=result.orbital_file,
                                    overwrite=False,
                                )
                                lg(
                                    f"Registered new CMOCORR reference for signature={sig_i} "
                                    f"(mask_len={mask_len_i}) from {result.orbital_file}",
                                    self.cfg.log_level,
                                )
                            except Exception as e:
                                lg(
                                    f"Failed to register CMOCORR reference for signature={sig_i}: {e!r}",
                                    self.cfg.log_level,
                                )

        out = [
            penalty if (v is None or not math.isfinite(float(v))) else float(v)
            for v in losses
        ]

        out_energy = [
            float("nan") if (v is None or not math.isfinite(float(v))) else float(v)
            for v in raw_energies
        ]

        self._last_raw_energies = torch.tensor(
            out_energy, dtype=torch.float32, device=population.device
        )

        return torch.tensor(out, dtype=torch.float32, device=population.device)

    def lambda_from_error(
        self, err_abs, lam_min=1e-7, lam_max=5e-3, e_low=0.003, e_high=0.05
    ):
        x = (err_abs - e_low) / (e_high - e_low)
        x = max(0.0, min(1.0, x))
        return lam_min + (lam_max - lam_min) * x

    def _tournament_select(self, fit: torch.Tensor, n_select: int) -> torch.Tensor:
        """
        Docstring for _tournament_select

        :param fit: tensor of energy values
        :type fit: torch.Tensor
        :param n_select: num of choosen genomes
        :type n_select: int
        :return: tensor of choosen genomes
        :rtype: Tensor
        """
        B = fit.size(0)
        k = self.cfg.tournament_k
        idx = torch.randint(0, B, (n_select, k), device=fit.device)
        cand_fit = fit[idx]
        winners = idx[
            torch.arange(n_select, device=fit.device), torch.argmin(cand_fit, dim=1)
        ]
        return winners

    def _uniform_crossover(
        self, p1: torch.Tensor, m1: torch.Tensor, p2: torch.Tensor, m2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Docstring for _uniform_crossover

        :param p1: first parent
        :type p1: torch.Tensor
        :param m1: first parent's mask
        :type m1: torch.Tensor
        :param p2: second parent
        :type p2: torch.Tensor
        :param m2: second parent's mask
        :type m2: torch.Tensor
        :return: created child with mask
        :rtype: Tuple[Tensor, Tensor]
        """
        if torch.rand(()) > self.cfg.crossover_p:
            return p1.clone(), m1.clone()

        sw = torch.rand_like(p1) < 0.5
        child_p = torch.where(sw, p1, p2)
        child_m = torch.where(sw, m1, m2)
        return child_p, child_m

    def _ensure_block_nonempty(
        self, a_row: torch.Tensor, m_row: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensures that every block in exponent tensor have at least one exponent

        :param a_row: genome
        :type a_row: torch.Tensor
        :param m_row: mask
        :type m_row: torch.Tensor
        :return: corrected mask tensor
        :rtype: Tensor
        """
        off = 0
        m = m_row.clone()
        for n in BLOCKS:
            mb = m[off : off + n]
            if mb.sum() < 0.5:
                ab = a_row[off : off + n]
                j = torch.argmax(ab)
                m[off + j] = 1.0
            off += n
        return m

    def _mutate(
        self, x: torch.Tensor, m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mutation function. Mutates given genome (with cfg.mutation_p prob) and mask (with mask_flip_p prob for each element in mask)

        :param x: tensor of exponents
        :type x: torch.Tensor
        :param m: mask tensor
        :type m: torch.Tensor
        :return: tuple fo mutated genome and mask
        :rtype: Tuple[Tensor, Tensor]
        """
        B, _ = x.shape
        if self.cfg.mutation_p > 0:
            do_mut = (torch.rand(B, 1, device=x.device) < self.cfg.mutation_p).float()
            noise = torch.randn_like(x) * self.cfg.mutation_sigma
            x = x + do_mut * noise

        if self.cfg.mask_flip_p > 0:
            flip = (torch.rand_like(m) < self.cfg.mask_flip_p).float()
            m = (m > 0.5).float()
            m = torch.abs(m - flip)  # xor

        for i in range(B):
            m[i] = self._ensure_block_nonempty(torch.exp(x[i]), m[i])

        return x, m

    def run(self, seed_alphas: List[float]):
        """
        Runs the genetic algorithm

        :param seed_alphas: Template for initial population
        :type seed_alphas: List[float]
        """

        lg("Initializing population...", self.cfg.log_level)
        pop, pop_mask = make_initial_population_from_seed(
            seed_alphas, self.cfg.population_size, device=self.device
        )

        lg("Initializing policy model...", self.cfg.log_level)

        best_genome = pop[0].clone()
        best_mask = pop_mask[0].clone()
        best_energy = float("inf")
        best_fit = float("inf")

        best_err_seen = float("inf")
        patience_counter = 0

        curr_lambda = self.cfg.start_lambda
        self._current_lambda = curr_lambda
        self._resolve_cmocorr_reference(seed_alphas)

        lg("Starting GA...", self.cfg.log_level)
        handle = Path("metrics.csv")
        handle.touch(exist_ok=True)
        with open(handle, "a") as f:
            f.write(
                "generation,generator_loss,gen_non_penalty_rate,ga_non_penalty_rate,best_fit,mean_fit,average_length,best_len,best_energy,error,current_lambda\n"
            )

        for gen in range(self.cfg.generations):
            lg(f"Starting Gen: {gen}", self.cfg.log_level)

            self._gen_idx = gen
            self._current_lambda = curr_lambda
            fit = self.fitness(pop, pop_mask, "pop")
            raw_energy_pop = self._last_raw_energies.clone()
            fit_wmask = fit

            lg(
                f"=======================\nFitness\n=======================\n{fit}\n=======================\n",
                self.cfg.log_level,
            )
            with open(population_handle, "a") as file:
                w = csv.writer(file)
                for p, msk, f, fwm in zip(
                    pop.detach().cpu().tolist(),
                    pop_mask.detach().cpu().tolist(),
                    fit.detach().cpu().tolist(),
                    fit_wmask.detach().cpu().tolist(),
                ):
                    w.writerow(
                        [
                            torch.exp(torch.tensor(p, dtype=torch.float64)).tolist(),
                            msk,
                            f,
                            fwm,
                        ]
                    )

            minfit, argmin = torch.min(fit_wmask, dim=0)

            lg(f"Minfit: {minfit}, Argmin: {argmin}", self.cfg.log_level)

            min_energy = float(raw_energy_pop[argmin])

            if minfit < best_fit:
                best_fit = float(minfit)
                best_energy = min_energy
                best_genome = pop[argmin].detach().clone()
                best_mask = pop_mask[argmin].detach().clone()

            elite_idx = torch.argsort(fit_wmask)[: self.n_elite]

            lg(f"Choosen elites: {elite_idx}", self.cfg.log_level)

            elite = pop[elite_idx].detach().clone()
            elite_mask = pop_mask[elite_idx].detach().clone()

            lg("Cross and mutating...", self.cfg.log_level)

            if self.n_offspring > 0:
                w = self._tournament_select(fit_wmask, self.n_offspring * 2)
                parents = pop[w]
                parents_mask = pop_mask[w]
                p1 = parents[0::2]
                p2 = parents[1::2]
                m1 = parents_mask[0::2]
                m2 = parents_mask[1::2]
                children, children_mask = [], []
                for a, b, c, d in zip(p1, m1, p2, m2):
                    child, child_mask = self._uniform_crossover(a, b, c, d)
                    child, child_mask = self._mutate(
                        child.unsqueeze(0), child_mask.unsqueeze(0)
                    )
                    children.append(child.squeeze(0))
                    children_mask.append(child_mask.squeeze(0))
                children = torch.stack(children, 0)
                children_mask = torch.stack(children_mask, 0)

            else:
                children = pop.new_empty((0, pop.size(1)))
                children_mask = pop_mask.new_empty((0, pop_mask.size(1)))

            next_base = torch.cat([elite, children], dim=0)
            next_base_mask = torch.cat([elite_mask, children_mask], dim=0)

            lg("Sampling...", self.cfg.log_level)
            z, logp_z = self.policy.sample()
            z = z.to(self.device)
            logp_z = logp_z.to(self.device)

            lg("Generating population using generation model...", self.cfg.log_level)
            gen_part, logp_g, g_log_std, mask = self.generator.sample(z.detach())
            mask = (mask > 0.5).float()

            lg(f"Generated {len(gen_part)}", self.cfg.log_level)
            lg(f"Generated {len(mask)}", self.cfg.log_level)
            lg("Caltulationg loss for generated population...", self.cfg.log_level)
            self._current_lambda = curr_lambda
            gen_energy = self.fitness(gen_part, mask, "gen")
            gen_raw_energy = self._last_raw_energies.clone()

            lg(
                f"=======================\nGen E\n=======================\n{gen_energy}\n=======================\n",
                self.cfg.log_level,
            )

            reward = -gen_energy

            r_mean = reward.mean()

            if self.baseline is None:
                self.baseline = r_mean
            else:
                self.baseline = (
                    self.baseline_beta * self.baseline
                    + (1 - self.baseline_beta) * r_mean
                )

            adv = reward - self.baseline
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            logp_total = logp_z + logp_g
            loss = -(adv.detach() * logp_total).mean()
            entropy_g = (
                (0.5 * (1.0 + math.log(2.0 * math.pi)) + g_log_std).sum(dim=1).mean()
            )
            loss = loss - 1e-3 * entropy_g

            err = abs(best_energy - self.cfg.ground_truth)
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (
                    self.err_ema_beta * self.err_ema + (1 - self.err_ema_beta) * err
                )

            curr_lambda = self.lambda_from_error(self.err_ema)

            lg(
                f"gen {gen:04d} | loss(gen) {float(loss.detach()):.6f} | % of correct Generator Population {(gen_energy < (1e4 * 0.999)).float().mean().item():.3} | % of correct in whole population {(fit < (1e4 * 0.999)).float().mean().item():.3} | best(pop) fitness {best_fit:.6f} | average num. of exp {pop_mask.sum(dim=1).mean().item():.3} | num. of exp in best genome {best_mask.sum().item():.3} | energy of best genome {best_energy} | abs error of best genome {abs(err)} | current lambda {curr_lambda:.2e}\n",
                self.cfg.log_level,
            )

            with open(handle, "a") as f:
                f.write(
                    f"{gen},{loss},{(gen_energy < (1e4 * 0.999)).float().mean().item():.3},{(fit < (1e4 * 0.999)).float().mean().item():.3},{best_fit},{fit.mean().item():.3},{pop_mask.sum(dim=1).mean().item():.3},{best_mask.sum().item():.3},{best_energy},{abs(err)},{curr_lambda:.2e}\n"
                )

            self.opt.zero_grad()
            self.opt_g.zero_grad()
            loss.backward()
            self.opt.step()
            self.opt_g.step()

            pop = torch.cat([next_base, gen_part], dim=0)
            pop_mask = torch.cat([next_base_mask, mask], dim=0)

            if err < best_err_seen - 1e-8:
                best_err_seen = err
                patience_counter = 0
            else:
                patience_counter += 1

            if err <= self.cfg.error_threshold_early_stopping:
                break

            if patience_counter >= self.cfg.early_stopping_patience:
                break

            if (self.cfg.ground_truth is not None) and (
                abs(err) <= self.cfg.error_threshold_early_stopping
            ):
                break

            if patience_counter >= self.cfg.early_stopping_patience:
                break

        torch.save(
            {
                "generator": self.generator.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt": self.opt.state_dict(),
                "policy": self.policy.state_dict(),
            },
            self.cfg.model_save_path,
        )
        return best_genome, best_mask, best_fit, best_energy


if __name__ == "__main__":
    cfg = GA_cfg()

    seed = [
        3691.66806090,
        574.05640852,
        124.00283202,
        34.69632525,
        11.53183174,
        4.16574637,
        1.51489133,
        0.52752984,
        0.17434522,
        0.05729263,
        0.01882728,
        21.12969131,
        8.89027185,
        3.48780524,
        1.28729033,
        0.50437998,
        0.22167796,
        0.09053350,
        0.73233854,
        0.33641238,
        0.14089237,
        0.45925197,
        0.19865529,
    ]

    # seed = [
    #     22628.599,
    #     3372.3181,
    #     760.35040,
    #     211.74048,
    #     67.223468,
    #     23.372177,
    #     8.7213730,
    #     3.4680910,
    #     1.4521440,
    #     0.60861500,
    #     0.25768600,
    #     0.10417600,
    #     0.04242700,
    #     0.01484900,
    #     0.001,
    #     0.0001,
    #     33.710184,
    #     8.0576495,
    #     2.8364714,
    #     1.0999657,
    #     0.44339640,
    #     0.18222640,
    #     0.07572410,
    #     0.03168540,
    #     0.01108990,
    #     0.001,
    #     1.4000000,
    #     0.49000000,
    #     0.17150000,
    # ]

    # Calcium
    # seed = [
    #     6809719.36,
    #     829347.437,
    #     162290.662,
    #     41026.8198,
    #     12263.7623,
    #     4131.35161,
    #     1523.09560,
    #     602.564047,
    #     252.124225,
    #     110.346869,
    #     49.9958150,
    #     22.9719819,
    #     9.95680985,
    #     4.57299757,
    #     2.05419310,
    #     0.88535650,
    #     0.37372268,
    #     0.06601407,
    #     0.02635672,
    #     0.01054268,
    #     21599.8143,
    #     3884.20155,
    #     1096.96697,
    #     385.187516,
    #     153.934262,
    #     66.9982541,
    #     30.9708648,
    #     14.8465963,
    #     7.28779623,
    #     3.62364204,
    #     1.73391125,
    #     0.81737794,
    #     0.37811791,
    #     0.15762725,
    #     0.06305090,
    #     0.02522036,
    #     2.798365,
    #     1.097115,
    #     0.430130,
    #     0.168635,
    #     0.0661143,
    #     0.02644572,
    #     3.731153,
    #     1.462819,
    #     0.224846,
    #     0.0881523,
    # ]

    ga = GA(cfg)
    best_genome, best_mask, best_fit, best_energy = ga.run(seed)
    lg("BEST_FIT:" + str(best_fit), 1)
    lg("BEST_energy:" + str(best_energy), 1)
    lg("BEST_ALPHA:" + str(torch.exp(best_genome).tolist()), 1)
    lg("MASK: " + str(best_mask.tolist()), 1)
