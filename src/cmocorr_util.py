from pathlib import Path

from typing import Optional, Tuple, Union
import subprocess
import shutil
import os
import json

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
    """
    Runs CMOCORR calculations

    :param wd: absolute path for working directory
    :type wd: Path
    :param molcas_cmd: command for running molcas
    :type molcas_cmd: str
    :param molcas_root: molcas root directory path
    :type molcas_root: str
    :param python_bin: pyhton binary path 
    :type python_bin: str
    :param ref_orb: reference orbital file path
    :type ref_orb: str|Path
    :param chk_orb: check orbital file path
    :type chk_orb: str|Path
    :param t1: lower threshold
    :type t1: float
    :param t2: higher threshold
    :type t2: float
    :param fail_penalty: value of penalty for fail
    :type fail_penalty: float
    :return: tuple of loss value, log file path, error info
    :rtype: Tuple[float, Path|None, str|None]:
    """

    wd = Path(wd).resolve()
    cmowd = (wd / "molcas_work").resolve()  # / "cmocorr").resolve() # TODO
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

def copy_runfile_as_cmocorr(cmowd: Union[str, Path]) -> Path:
    """
    Copies an existing RunFile into cmowd as 'cmocorr.RunFile'

    :param cmowd: CMOCORR working directory
    :type cmowd: str|Path
    :return: copied file path
    :rtype: Path
    """
    cmowd = Path(cmowd).resolve()
    cmowd.mkdir(parents=True, exist_ok=True)

    dst = cmowd / "cmocorr.RunFile"
    target = Path("../molcas_work/INPUT.RunFile")

    shutil.copy2(target, dst)
    return dst
