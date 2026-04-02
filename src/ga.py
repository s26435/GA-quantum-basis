import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

from typing import List, Tuple, Union, Optional

from pathlib import Path

import csv

from src.util import lg, make_initial_population_from_seed, sanitize_blocks
from src.globals import population_handle, BLOCKS
from src.cache import CaseResult, CacheDatabase, hash_case_context
from src.config import GA_cfg
from src.model import PopGenerator, GaussianPolicy
from src.energy import run_energy_case

import torch


class GA:
    """
    Genetic Algorithm object 

    :param config: configuration object
    :type config: GA_cfg
    """

    def __init__(self, config: GA_cfg):
        """
        Constructor for GA

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
                raise FileNotFoundError(
                    f"CMOCORR reference orbital file not found: {p}"
                )

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

        result = run_energy_case(
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
            f"Bootstrap CMOCORR reference created for signature={seed_sig}: \n{self._resolved_cmocorr_ref_orb}",
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
                            run_energy_case,
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

            lg(
                "Generating population using generation model...",
                self.cfg.log_level,
            )
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
