import shutil
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import traceback
import hashlib

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


@dataclass(frozen=True)
class GA_cfg:
    # GA
    population_size: int = 20
    device: str = "cpu"
    generations: int = 1000
    genome_size: int = 29

    # GA functions
    elite_frac: float = 0.2
    tournament_k: int = 2
    crossover_p: float = 0.9
    mutation_p: float = 0.4
    mask_flip_p: float = 0.05
    mutation_sigma: float = 0.2

    # generator parameters
    mask_lambda: float = 0.5
    zdim: int = 16
    gen_percent: float = 0.5

    # generator training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    local_max_workers: int = 4

    # directories
    work_root: str = "/home/janek/GA_proj"
    python_bin: str = "/home/janek/GA_proj/venv/bin/python3"

    # logging
    log_level: int = 1

    # cache
    db_path: str = "cache/energy.sqlite"

    # model storage
    model_from_file: bool = False
    model_save_path: str = "model.ckpt"


BLOCKS = [14, 9, 4, 2]

log_handle = Path("out.log")
log_handle.touch(exist_ok=True)

population_handle = Path("populations.csv")
population_handle.touch(exist_ok=True)


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
                valid INTEGER NOT NULL,
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

    def check(self, key: str) -> Optional[Tuple[float, int]]:
        """
        Checking if given key is in database

        :param key: hashed key of genome
        :type key: str
        :return: energy, valid or None if there is no key in database
        :rtype: Tuple[float, int] | None
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT energy, valid FROM energy_cache WHERE key=?;",
                (key,),
            ).fetchone()
            return None if row is None else (float(row[0]), int(row[1]))

    def load(self, key: str, energy: float, valid: int):
        """
        Uploads genome to database

        :param key: hashed genome
        :type key: str
        :param energy: value of energy
        :type energy: float
        :param valid: 1 if is valid 0 if not
        :type valid: int
        """
        try:
            e = float(energy)
        except Exception:
            e = 0.0
            valid = 0

        if not math.isfinite(e):
            e = 0.0
            valid = 0

        try:
            with self.connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO energy_cache(key, energy, valid) VALUES(?,?,?);",
                    (key, e, int(valid)),
                )
        except Exception as err:
            print(f"{err} | e={e} valid={valid} key={key}")


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
    mask_thresh: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:

    m = m_block > mask_thresh

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
    mask_thresh: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:

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
            a_block,
            m_block,
            lo=lo,
            hi=hi,
            min_ratio=min_ratio,
            mask_thresh=mask_thresh,
        )
        parts_a.append(sa)
        parts_m.append(sm)
        off += n

    return torch.cat(parts_a, dim=0), torch.cat(parts_m, dim=0)


def _run_energy_case(
    i: int,
    alphas: List[float],
    mask: List[bool],
    gen_dir_str: str,
    python_bin: str,
    engine_cmd: str,
    case: str,
) -> Tuple[int, float]:
    gen_dir = Path(gen_dir_str)
    wd = gen_dir / f"{case}_case_{i:04d}"
    wd.mkdir(parents=True, exist_ok=True)

    run_out = wd / "run.out"
    run_out.write_text("", encoding="utf-8")
    energy_txt = wd / "energy.txt"

    if len(alphas) != len(mask):
        run_out.write_text(
            f"[BAD INPUT] len(alphas)={len(alphas)} != len(mask)={len(mask)}\n",
            encoding="utf-8",
        )
        energy_txt.write_text("nan\n", encoding="utf-8")
        return i, float("nan")

    try:
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
            return i, float("nan")

        with run_out.open("w", encoding="utf-8") as f:
            p3 = subprocess.run(
                [engine_cmd, "INPUT"],
                cwd=str(wd),
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        if p3.returncode != 0:
            energy_txt.write_text("nan\n", encoding="utf-8")
            with run_out.open("a", encoding="utf-8") as f:
                f.write(f"\n[molcas FAILED] returncode={p3.returncode}\n")
            return i, float("nan")

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
            return i, float("nan")

        s = p2.stdout.strip()
        e = float(s)
        energy_txt.write_text(f"{e}\n", encoding="utf-8")
        return i, e

    except Exception as e:
        energy_txt.write_text("nan\n", encoding="utf-8")
        with run_out.open("a", encoding="utf-8") as f:
            f.write("\n[WORKER CRASH]\n")
            f.write(f"type: {type(e).__name__}\n")
            f.write(f"repr: {repr(e)}\n")
            f.write("\nTRACEBACK:\n")
            f.write(traceback.format_exc())
        return i, float("nan")


def hash_alphas(alphas: torch.Tensor, mask: torch.Tensor) -> str:
    """
    Hashing algorithm, that turns tensor of althas into key for caching database

    :param alphas: Description
    :type alphas: torch.Tensor
    :return:
    :rtype:
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
    def __init__(self, population_size, genome_size, zdim, *args, **kwargs):
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x, _, _, _ = self.sample(z)
        return x


class GaussianPolicy(nn.Module):
    def __init__(self, pop_size, zdim):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(pop_size, zdim))
        self.log_std = nn.Parameter(torch.zeros(pop_size, zdim))

    def sample(self):
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

        if not Path(self.cfg.model_save_path).exists() and self.cfg.model_from_file:
            raise FileNotFoundError(
                f"Model checkpoint file not found: {self.cfg.model_save_path}"
            )

        if self.cfg.model_from_file:
            ckpt = torch.load(self.cfg.model_save_path, map_location=self.cfg.device)

            self.generator.load_state_dict(ckpt["generator"])
            self.policy.load_state_dict(ckpt["policy"])

            self.opt.load_state_dict(ckpt["opt"])
            self.opt_g.load_state_dict(ckpt["opt_g"])

    def fitness(
        self, population: torch.Tensor, pop_mask: torch.Tensor, case: str
    ) -> torch.Tensor:
        B = population.size(0)

        gen_dir = Path(self.cfg.work_root) / f"gen_{self._gen_idx:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        base = Path(__file__).resolve().parent
        shutil.copy2(base / "INPUT", gen_dir / "INPUT.template")
        shutil.copy2(base / "build_input.py", gen_dir / "build_input.py")
        shutil.copy2(base / "parse_energy.py", gen_dir / "parse_energy.py")

        for p in ["INPUT.template", "build_input.py", "parse_energy.py"]:
            if not (gen_dir / p).exists():
                raise RuntimeError(f"File do not exist: {(gen_dir / p)}")

        pop_cpu = population.detach().cpu()
        mask_cpu = pop_mask.detach().cpu()

        alphas_all: List[List[float]] = []
        mask_all: List[List[bool]] = []
        keys: List[str] = []

        for i in range(B):
            a = torch.exp(pop_cpu[i])
            m_i = mask_cpu[i]
            m_i = (m_i > 0.5).float()

            a_s, m_s = sanitize_blocks(a, m_i, lo=0.0, hi=1e5, min_ratio=1.2)

            alphas_all.append(a_s.tolist())
            mask_all.append([bool(x) for x in m_s.tolist()])

            keys.append(hash_alphas(a_s, m_s))

        energies: List[float | None] = [None] * B
        miss: List[tuple[int, List[float], List[bool], str]] = []

        for i, key in enumerate(keys):
            hit = self.energy_cache.check(key)
            if hit is None:
                miss.append((i, alphas_all[i], mask_all[i], key))
            else:
                e, valid = hit
                if valid == 1 and math.isfinite(e):
                    energies[i] = float(e)
                else:
                    miss.append((i, alphas_all[i], mask_all[i], key))

        if miss:
            python_bin = self.cfg.python_bin
            engine_cmd = "molcas"
            max_workers = max(1, int(self.cfg.local_max_workers))

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        _run_energy_case,
                        i,
                        alphas,
                        msk,
                        str(gen_dir),
                        python_bin,
                        engine_cmd,
                        case,
                    )
                    for (i, alphas, msk, _key) in miss
                ]

                i2key = {i: key for (i, _a, _m, key) in miss}

                for fut in as_completed(futures):
                    i, e = fut.result()
                    energies[i] = float(e)

                    valid = 1 if (isinstance(e, float) and math.isfinite(e)) else 0
                    self.energy_cache.load(i2key[i], float(e), valid)

        penalty = 1e4
        out = [
            penalty if (e is None or not math.isfinite(float(e))) else float(e)
            for e in energies
        ]
        return torch.tensor(out, dtype=torch.float32, device=population.device)

    def _tournament_select(self, fit: torch.Tensor, n_select: int) -> torch.Tensor:
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
        if torch.rand(()) > self.cfg.crossover_p:
            return p1.clone(), m1.clone()

        sw = torch.rand_like(p1) < 0.5
        child_p = torch.where(sw, p1, p2)
        child_m = torch.where(sw, m1, m2)
        return child_p, child_m

    def _ensure_block_nonempty(
        self, a_row: torch.Tensor, m_row: torch.Tensor
    ) -> torch.Tensor:
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

        lg("Initializing population...", self.cfg.log_level)
        pop, pop_mask = make_initial_population_from_seed(
            seed_alphas, self.cfg.population_size, device=self.device
        )

        lg("Initializing policy model...", self.cfg.log_level)

        best_genome = pop[0].clone()
        best_mask = pop_mask[0].clone()

        best_fit = float("inf")

        lg("Starting GA...", self.cfg.log_level)
        handle = Path("metrics.csv")
        handle.touch(exist_ok=True)
        with open(handle, "a") as f:
            f.write(
                "generation,generator_loss,gen_non_penalty_rate,ga_non_penalty_rate,best_fit,mean_fit,average_length,lr\n"
            )

        for gen in range(self.cfg.generations):
            lg(f"Starting Gen: {gen}", self.cfg.log_level)

            self._gen_idx = gen
            fit = self.fitness(pop, pop_mask, "pop")
            fit_wmask = fit + self.cfg.mask_lambda * (
                pop_mask.sum(dim=1) / self.cfg.genome_size
            )

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
                    w.writerow([torch.exp(torch.tensor(p, dtype=torch.float64)).tolist(), msk, f, fwm])

            minfit, argmin = torch.min(fit_wmask, dim=0)

            lg(f"Minfit: {minfit}, Argmin: {argmin}", self.cfg.log_level)

            if float(minfit) < best_fit:
                best_fit = float(minfit)
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
            for i in range(mask.size(0)):
                mask[i] = self._ensure_block_nonempty(
                    torch.exp(gen_part[i].detach()), mask[i]
                )
            lg(f"Generated {len(gen_part)}", self.cfg.log_level)
            lg(f"Generated {len(mask)}", self.cfg.log_level)
            lg("Caltulationg loss for generated population...", self.cfg.log_level)
            gen_energy = self.fitness(gen_part, mask, "gen")

            lg(
                f"=======================\nGen E\n=======================\n{gen_energy}\n=======================\n",
                self.cfg.log_level,
            )

            reward = -gen_energy - self.cfg.mask_lambda * mask.sum(dim=1)
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

            lg(
                f"gen {gen:04d} | loss(gen) {float(loss):.6f} |  {(gen_energy < (1e4 * 0.999)).float().mean().item():.3} | {(fit < (1e4 * 0.999)).float().mean().item():.3} | best(pop) {best_fit:.6f} | {pop_mask.sum(dim=1).mean().item():.3} | lr {self.cfg.lr:.2e}\n",
                self.cfg.log_level,
            )

            with open(handle, "a") as f:
                f.write(
                    f"{gen},{loss},{(gen_energy < (1e4 * 0.999)).float().mean().item():.3},{(fit < (1e4 * 0.999)).float().mean().item():.3},{best_fit},{fit.mean().item():.3},{pop_mask.sum(dim=1).mean().item():.3},{self.cfg.lr:.2e}\n"
                )

            self.opt.zero_grad()
            self.opt_g.zero_grad()
            loss.backward()
            self.opt.step()
            self.opt_g.step()

            pop = torch.cat([next_base, gen_part], dim=0)
            pop_mask = torch.cat([next_base_mask, mask], dim=0)

        torch.save(
            {
                "generator": self.generator.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt": self.opt.state_dict(),
                "policy": self.policy.state_dict(),
            },
            self.cfg.model_save_path,
        )
        return best_genome, best_mask, best_fit


if __name__ == "__main__":
    cfg = GA_cfg(
        population_size=30,
        genome_size=29,
        generations=10,
        device="cpu",
    )

    seed = [
        22628.599,
        3372.3181,
        760.35040,
        211.74048,
        67.223468,
        23.372177,
        8.7213730,
        3.4680910,
        1.4521440,
        0.60861500,
        0.25768600,
        0.10417600,
        0.04242700,
        0.01484900,
        0.001,
        0.0001,
        33.710184,
        8.0576495,
        2.8364714,
        1.0999657,
        0.44339640,
        0.18222640,
        0.07572410,
        0.03168540,
        0.01108990,
        0.001,
        1.4000000,
        0.49000000,
        0.17150000,
    ]

    ga = GA(cfg)
    (
        best_genome,
        best_mask,
        best_fit,
    ) = ga.run(seed)
    lg("BEST_FIT:" + str(best_fit), 1)
    lg("BEST_ALPHA:" + str(torch.exp(best_genome).tolist()), 1)
    lg("MASK: " + str(best_mask.tolist()), 1)
