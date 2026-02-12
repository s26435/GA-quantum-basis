import shutil
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import traceback
import os
import hashlib

import numpy as np

from contextlib import contextmanager

from datetime import datetime

import torch
from torch import nn

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

from pathlib import Path

import csv

import sqlite3


from tqdm import tqdm  # noqa: F401

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


@dataclass(frozen=True)
class GA_cfg:
    # GA
    population_size: int = 20
    device: str = "cpu"
    generations: int = 1000
    genome_size: int = 29
    zdim: int = 16
    gen_percent: float = 0.5

    # GA functions
    elite_frac: float = 0.2
    tournament_k: int = 2
    crossover_p: float = 0.9
    mutation_p: float = 0.4
    mutation_sigma: float = 0.2

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


class CacheDatabase:
    def __init__(self, db_path: Union[str, Path]):
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
        with self.connect() as conn:
            row = conn.execute(
                "SELECT energy, valid FROM energy_cache WHERE key=?;",
                (key,),
            ).fetchone()
            return None if row is None else (float(row[0]), int(row[1]))

    def load(self, key: str, energy: float, valid: int):
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
):

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
    return genome


def sanitize_alphas(
    a: torch.Tensor, lo: float = 0, hi: float = 1e5, min_ratio: float = 1.2
) -> torch.Tensor:
    a = torch.clamp(a, lo, hi)
    a, _ = torch.sort(a, descending=True)
    for k in range(len(a) - 1):
        a[k + 1] = torch.minimum(a[k + 1], a[k] / min_ratio)
    a = torch.clamp(a, lo, hi)

    return a


def sanitize_blocks(
    a: torch.Tensor, blocks=BLOCKS, lo=1e-2, hi=1e2, min_ratio=1.2
) -> torch.Tensor:
    parts = []
    off = 0
    for n in blocks:
        part = sanitize_alphas(a[off : off + n], lo=lo, hi=hi, min_ratio=min_ratio)
        parts.append(part)
        off += n
    return torch.cat(parts, dim=0)


def _run_energy_case(
    i: int,
    alphas: List[float],
    gen_dir_str: str,
    python_bin: str,
    engine_cmd: str,
) -> Tuple[int, float]:
    gen_dir = Path(gen_dir_str)
    wd = gen_dir / f"case_{i:04d}"
    wd.mkdir(parents=True, exist_ok=True)

    run_out = wd / "run.out"
    run_out.touch(exist_ok=True)
    energy_txt = wd / "energy.txt"

    try:
        alphas_path = gen_dir / f"alphas_{i:04d}.json"
        alphas_path.write_text(json.dumps(alphas), encoding="utf-8")

        build_cmd = [
            python_bin,
            str(gen_dir / "build_input.py"),
            str(alphas_path),
            str(wd / "INPUT"),
            "--template",
            str(gen_dir / "INPUT.template"),
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

        txt = run_out.read_text(errors="ignore")
        if "_INTERNAL_ERROR_" in txt or "Program aborted" in txt:
            energy_txt.write_text("nan\n", encoding="utf-8")
            return i, float("nan")

        with run_out.open("w", encoding="utf-8") as f:
            subprocess.run(
                [engine_cmd, "INPUT"],
                cwd=str(wd),
                stdout=f,
                stderr=subprocess.STDOUT,
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
            return i, float("nan")

        s = p2.stdout.strip()
        try:
            e = float(s)
        except Exception:
            energy_txt.write_text("nan\n", encoding="utf-8")
            with run_out.open("a", encoding="utf-8") as f:
                f.write(f"\n[parse_energy OUTPUT NOT FLOAT]\n{s}\n")
            return i, float("nan")

        energy_txt.write_text(f"{e}\n", encoding="utf-8")
        return i, e

    except Exception as e:
        energy_txt.write_text("nan\n", encoding="utf-8")
        with run_out.open("a", encoding="utf-8") as f:
            f.write("\n[WORKER CRASH]\n")
            f.write(f"type: {type(e).__name__}\n")
            f.write(f"repr: {repr(e)}\n")
            if isinstance(e, FileNotFoundError):
                f.write(f"missing filename: {getattr(e, 'filename', None)}\n")
            f.write(f"cwd: {os.getcwd()}\n")
            f.write(f"wd: {wd}\n")
            f.write(f"exists wd: {wd.exists()}\n")
            f.write(f"exists INPUT.template: {(gen_dir / 'INPUT.template').exists()}\n")
            f.write(f"exists build_input.py: {(gen_dir / 'build_input.py').exists()}\n")
            f.write(
                f"exists parse_energy.py: {(gen_dir / 'parse_energy.py').exists()}\n"
            )
            f.write(f"python_bin: {python_bin} exists={Path(python_bin).exists()}\n")
            f.write(f"engine_cmd: {engine_cmd} which={shutil.which(engine_cmd)}\n")
            f.write("\nTRACEBACK:\n")
            f.write(traceback.format_exc())
        return i, float("nan")


def hash_alphas(alphas: torch.Tensor):
    if isinstance(alphas, torch.Tensor):
        a = alphas.detach().cpu().numpy().astype(np.float64, copy=False)
    elif isinstance(alphas, np.ndarray):
        a = alphas.astype(np.float64, copy=False)
    else:
        a = np.asarray(list(alphas), dtype=np.float64)

    a = np.round(
        np.ravel(a),
        decimals=7,
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

    def sample(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(z)  # (B, 2*G)
        mu, log_std = out.chunk(2, dim=1)
        log_std = log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std)

        eps = torch.randn_like(std)
        x = mu + eps * std

        log2pi = math.log(2.0 * math.pi)
        logp_per_dim = -0.5 * (((x - mu) / std) ** 2 + 2.0 * log_std + log2pi)
        logp = logp_per_dim.sum(dim=1)

        return x, logp, log_std

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x, _, _ = self.sample(z)
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
        log2pi = math.log(2.0 * math.pi)
        logp_per_dim = -0.5 * (((x - self.mu) / std) ** 2 + 2.0 * log_std + log2pi)
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
        self.n_gen = max(0, min(int(self.cfg.population_size * self.cfg.gen_percent), self.cfg.population_size - self.n_elite))

        if self.n_gen <= 0:
            raise ValueError(f"Number of generated genomes must be greater then 0 but is {self.n_gen}")

        self.n_offspring = max(0, self.cfg.population_size - self.n_elite - self.n_gen)

        self.generator = PopGenerator(
            population_size=self.n_gen,
            genome_size=config.genome_size,
            zdim=config.zdim,
        ).to(self.device)

    def fitness(self, population: torch.Tensor) -> torch.Tensor:
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

        alphas_all: List[List[float]] = []
        keys: List[str] = []
        for i in range(B):
            a = torch.exp(pop_cpu[i])
            a = sanitize_blocks(a, lo=1e-6, hi=1e5, min_ratio=1.2)
            alphas_all.append(a.tolist())
            keys.append(hash_alphas(a,))

        energies: List[float | None] = [None] * B
        miss: List[tuple[int, List[float], str]] = []

        for i, key in enumerate(keys):
            hit = self.energy_cache.check(key)
            if hit is None:
                miss.append((i, alphas_all[i], key))
            else:
                e, valid = hit
                if valid == 1 and math.isfinite(e):
                    energies[i] = float(e)
                else:
                    miss.append((i, alphas_all[i], key))

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
                        str(gen_dir),
                        python_bin,
                        engine_cmd,
                    )
                    for (i, alphas, _key) in miss
                ]

                i2key = {i: key for (i, _alphas, key) in miss}

                for fut in as_completed(futures):
                    i, e = fut.result()
                    energies[i] = float(e)

                    valid = 1 if (isinstance(e, float) and math.isfinite(e)) else 0
                    self.energy_cache.load(i2key[i], float(e), valid)

        penalty = 1e4
        out = []
        for e in energies:
            if e is None or (isinstance(e, float) and (math.isnan(e) or math.isinf(e))):
                out.append(penalty)
            else:
                out.append(float(e))

        return torch.tensor(out, dtype=torch.float32, device=population.device)

    def _tournament_select(
        self, pop: torch.Tensor, fit: torch.Tensor, n_select: int
    ) -> torch.Tensor:
        B = pop.size(0)
        k = self.cfg.tournament_k
        idx = torch.randint(0, B, (n_select, k), device=pop.device)
        cand_fit = fit[idx]
        winners = idx[
            torch.arange(n_select, device=pop.device), torch.argmin(cand_fit, dim=1)
        ]
        return pop[winners]

    def _uniform_crossover(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.cfg.crossover_p:
            return p1.clone()

        mask = torch.rand_like(p1) < 0.5
        child = torch.where(mask, p1, p2)
        return child

    def _mutate(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.mutation_p <= 0:
            return x
        do_mut = (
            torch.rand(x.size(0), 1, device=x.device) < self.cfg.mutation_p
        ).float()
        noise = torch.randn_like(x) * self.cfg.mutation_sigma
        return x + do_mut * noise

    def run(self, seed_alphas: List[float]):

        lg("Initializing population...", self.cfg.log_level)
        pop = make_initial_population_from_seed(
            seed_alphas, self.cfg.population_size, device=self.device
        )

        lg("Initializing policy model...", self.cfg.log_level)
        policy = GaussianPolicy(self.n_gen, self.cfg.zdim).to(self.device)
        opt = torch.optim.AdamW(
            policy.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        best_genome = pop[0].clone()
        best_fit = float("inf")

        lg("Starting GA...", self.cfg.log_level)
        handle = Path("metrics.csv")
        handle.touch(exist_ok=True)
        with open(handle, "a") as f:
            f.write("generation,generator_loss,gen_non_penalty,ga_non_penalty_rate,best_fit,mean_fit,lr\n")


        for gen in range(self.cfg.generations):
            lg(f"Starting Gen: {gen}", self.cfg.log_level)

            self._gen_idx = gen
            fit = self.fitness(pop)
            lg(
                f"=======================\nFitness\n=======================\n{fit}\n=======================\n",
                self.cfg.log_level,
            )
            with open(population_handle, "a") as file:
                w = csv.writer(file)
                for p, f in zip(
                    pop.detach().cpu().tolist(), fit.detach().cpu().tolist()
                ):
                    w.writerow([p, f])

            minfit, argmin = torch.min(fit, dim=0)

            lg(f"Minfit: {minfit}, Argmin: {argmin}", self.cfg.log_level)

            if float(minfit) < best_fit:
                best_fit = float(minfit)
                best_genome = pop[argmin].detach().clone()


            elite_idx = torch.argsort(fit)[:self.n_elite]

            lg(f"Choosen elites: {elite_idx}", self.cfg.log_level)

            elite = pop[elite_idx].detach().clone()

            lg("Cross and mutating...", self.cfg.log_level)

            if self.n_offspring > 0:
                parents = self._tournament_select(pop, fit, self.n_offspring * 2)
                p1 = parents[0::2]
                p2 = parents[1::2]
                children = []
                for a, b in zip(p1, p2):
                    child = self._uniform_crossover(a, b)
                    child = self._mutate(child.unsqueeze(0)).squeeze(0)
                    children.append(child)
                children = torch.stack(children, dim=0)
            else:
                children = pop.new_empty((0, pop.size(1)))

            next_base = torch.cat([elite, children], dim=0)

            lg("Sampling...", self.cfg.log_level)
            z, logp_z = policy.sample()
            z = z.to(self.device)
            logp_z = logp_z.to(self.device)

            lg("Generating population using generation model...", self.cfg.log_level)
            gen_part, logp_g, g_log_std = self.generator.sample(z.detach())
            lg(f"Generated {len(gen_part)}", self.cfg.log_level)

            lg("Caltulationg loss for generated population...", self.cfg.log_level)
            gen_energy = self.fitness(gen_part)

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
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            logp_total = logp_z + logp_g
            loss = -(adv.detach() * logp_total).mean()
            entropy_g = (
                (0.5 * (1.0 + math.log(2.0 * math.pi)) + g_log_std).sum(dim=1).mean()
            )
            loss = loss - 1e-3 * entropy_g

            lg(
                f"gen {gen:04d} | loss(gen) {float(loss):.6f} | {(gen_energy < (1e-4 * .999)).float().mean().item():.3} | {(fit < (1e4 * 0.999)).float().mean().item():.3} | best(pop) {best_fit:.6f} | lr {self.cfg.lr:.2e}\n",
                self.cfg.log_level,
            )

            with open(handle, "a") as f:
                f.write(
                    f"{gen},{loss},{(fit < (1e4 * 0.999)).float().mean().item():.3},{best_fit},{fit.mean()},{self.cfg.lr:.2e}\n"
                )

            opt.zero_grad()
            opt_g.zero_grad()
            loss.backward()
            opt.step()
            opt_g.step()

            pop = torch.cat([next_base, gen_part], dim=0)

        return best_genome, best_fit


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
    best_genome, best_fit = ga.run(seed)
    print("BEST_FIT:", best_fit)
    print("BEST_ALPHA:", torch.exp(best_genome).tolist())
