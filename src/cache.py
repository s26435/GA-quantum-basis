from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import math
import json
import hashlib
import torch
import numpy as np

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

