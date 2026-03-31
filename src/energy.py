from typing import List, Optional, Tuple
from .cache import CaseResult
from .cmocorr_util import run_cmocorr
from pathlib import Path
import json
import subprocess
import os
import shutil
import traceback

def run_energy_case(
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

