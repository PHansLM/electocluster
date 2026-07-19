"""Trazabilidad minima para las ejecuciones integradas de la iteracion 5."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROVENANCE_SCHEMA = "iteration5-provenance-v1"


def sha256_file(path: str | Path) -> str:
    """Calcula SHA-256 sobre los bytes exactos de un archivo."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Calcula un hash estable para una estructura serializable como JSON."""
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_run_provenance(
    dataset_path: str | Path,
    df: pd.DataFrame,
    algorithm: str,
    params: dict,
    weights: dict,
) -> dict:
    """Construye la identidad reproducible minima de una ejecucion."""
    columns = [str(column) for column in df.columns]
    dataset = {
        "path": _dataset_label(dataset_path),
        "sha256": sha256_file(dataset_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_order": columns,
        "feature_order_sha256": sha256_json(columns),
    }
    execution_identity = {
        "algorithm": algorithm,
        "params": params,
        "weights": weights,
        "feature_order": columns,
    }
    git_commit, git_dirty = _git_state()
    return {
        "schema": PROVENANCE_SCHEMA,
        "dataset": dataset,
        "code": {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
        "environment": {
            "python": platform.python_version(),
        },
        "configuration": {
            "params_sha256": sha256_json(params),
            "weights_sha256": sha256_json(weights),
            "execution_sha256": sha256_json(execution_identity),
        },
    }


def stored_dataset_sha256(run: dict) -> str | None:
    """Obtiene el hash persistido, tolerando runs historicos sin procedencia."""
    metadata = run.get("metadata") or {}
    provenance = metadata.get("provenance") or {}
    dataset = provenance.get("dataset") or {}
    value = dataset.get("sha256")
    return str(value) if value else None


def _dataset_label(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
        return commit or None, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None
