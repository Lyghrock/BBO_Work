"""
Mopta08 benchmark wrapper.

Mopta08 is evaluated through the original binary executable.
Expected protocol (same as common TuRBO/HPO benchmark wrappers):
- write decision vector to input.txt
- run executable in working directory
- read output.txt
  - line 1: objective value
  - remaining lines: constraint values (<= 0 means feasible)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


MOPTA08_DIMS = 124


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_mopta08_executable(executable_path: Optional[str] = None) -> str:
    """Resolve Mopta08 executable path."""
    candidates = []

    if executable_path:
        candidates.append(Path(executable_path).expanduser())

    env_path = os.getenv("MOPTA08_EXECUTABLE")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    root = _project_root()
    candidates.extend([
        root / "functions" / "mopta08" / "mopta08_elf64.bin",
        root / "functions" / "mopta08" / "mopta08_linux.bin",
        root / "functions" / "mopta08" / "mopta08.bin",
    ])

    for c in candidates:
        if c.exists() and c.is_file():
            return str(c)

    checked = "\n".join([str(c) for c in candidates])
    raise FileNotFoundError(
        "Mopta08 executable not found. Checked paths:\n"
        f"{checked}\n"
        "Provide --mopta-executable or set MOPTA08_EXECUTABLE."
    )


class Mopta08FuncWrapper:
    """Unified wrapper exposing Mopta08 as a minimization black-box objective."""

    def __init__(
        self,
        executable_path: Optional[str] = None,
        constraint_penalty: float = 10.0,
        is_minimizing: bool = True,
    ):
        self.executable = resolve_mopta08_executable(executable_path)
        self.constraint_penalty = float(constraint_penalty)
        self.is_minimizing = is_minimizing
        self.sign = 1.0 if is_minimizing else -1.0

        self.lb = np.zeros(MOPTA08_DIMS, dtype=np.float64)
        self.ub = np.ones(MOPTA08_DIMS, dtype=np.float64)
        self.dims = MOPTA08_DIMS
        self.call_count = 0

        # Expose raw callable for optimizers that inspect func_wrapper.func directly.
        self.func = lambda x: self._evaluate_raw(np.asarray(x).flatten())[0]

    def _evaluate_raw(self, x: np.ndarray) -> Tuple[float, float, np.ndarray]:
        x = np.asarray(x, dtype=np.float64).flatten()
        if x.shape[0] != self.dims:
            raise ValueError(f"Mopta08 expects {self.dims} dims, got {x.shape[0]}")

        x = np.clip(x, self.lb, self.ub)

        with tempfile.TemporaryDirectory(prefix="mopta08_") as tmp_dir:
            workdir = Path(tmp_dir)
            input_file = workdir / "input.txt"
            output_file = workdir / "output.txt"

            with open(input_file, 'w') as f:
                for i, v in enumerate(x):
                    if i > 0:
                        f.write('=')
                    f.write(f"{v:.18e}")

            try:
                proc = subprocess.run(
                    [self.executable],
                    cwd=str(workdir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=300,
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError(
                    f"Mopta08 executable timed out after 300s for input of "
                    f"dimension {x.shape[0]}. The binary may be hanging."
                )

            if proc.returncode != 0:
                raise RuntimeError(
                    "Mopta08 executable failed with return code "
                    f"{proc.returncode}: {proc.stderr.strip()}"
                )

            if output_file.exists():
                with open(output_file) as f:
                    lines = [l.strip() for l in f if l.strip()]
                values = np.array([float(l.split('=', 1)[1].strip()) for l in lines if '=' in l])
            else:
                values = np.fromstring(proc.stdout, sep="\n")

            if values.size < 1:
                raise RuntimeError(
                    "Mopta08 output is empty. Expected objective and optional constraints."
                )

            objective = float(values[0])
            constraints = np.asarray(values[1:], dtype=np.float64)
            positive_violations = np.maximum(constraints, 0.0)
            penalty = self.constraint_penalty * float(np.sum(positive_violations))
            penalized = objective + penalty

            return penalized, objective, constraints

    def __call__(self, x):
        self.call_count += 1
        penalized, _, _ = self._evaluate_raw(x)
        return self.sign * penalized

    def evaluate_with_metrics(self, x) -> Dict[str, float]:
        penalized, objective, constraints = self._evaluate_raw(np.asarray(x).flatten())
        positive = np.maximum(constraints, 0.0)
        max_violation = float(np.max(positive)) if positive.size > 0 else 0.0

        return {
            "penalized_fx": float(penalized),
            "objective": float(objective),
            "num_constraints": int(constraints.size),
            "num_violations": int(np.sum(constraints > 0.0)),
            "max_violation": max_violation,
        }

    def gen_random_inputs(self, n: int) -> np.ndarray:
        return np.random.uniform(self.lb, self.ub, size=(n, self.dims))


def create_mopta08_benchmark(
    executable_path: Optional[str] = None,
    constraint_penalty: float = 10.0,
) -> Mopta08FuncWrapper:
    return Mopta08FuncWrapper(
        executable_path=executable_path,
        constraint_penalty=constraint_penalty,
        is_minimizing=True,
    )
