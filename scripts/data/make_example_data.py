"""Generate the bundled synthetic example dataset.

Writes a small, reproducible regression table -- entirely synthetic, no
real-world data -- so the README quickstart runs end to end out of the box. The
columns match the default ``Settings.required_columns`` (``x1``, ``x2``, ``y``):
two float features and a target ``y = 2*x1 - 0.5*x2 + 1`` plus light Gaussian
noise. Re-run it (optionally pointing ``--output`` elsewhere) to regenerate.

Usage::

    python scripts/data/make_example_data.py [-o OUT.csv] [-n ROWS] [--seed SEED]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_OUTPUT = Path("data/example/orig/example.csv")


def build_parser() -> argparse.ArgumentParser:
    """Wire the command-line contract: output path, row count, RNG seed."""
    parser = argparse.ArgumentParser(
        prog="make_example_data",
        description="Generate a small synthetic regression CSV with columns x1, x2, y.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Path to write the CSV (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "-n", "--rows", type=int, default=256, help="Number of rows (default: 256)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for reproducibility (default: 0)."
    )
    return parser


def make_frame(rows: int, seed: int) -> pd.DataFrame:
    """Build a deterministic synthetic regression frame: features x1, x2 and target y."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.0, 1.0, size=rows)
    x2 = rng.uniform(0.0, 10.0, size=rows)
    noise = rng.normal(0.0, 0.05, size=rows)
    y = 2.0 * x1 - 0.5 * x2 + 1.0 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    frame = make_frame(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"DONE: wrote {len(frame)} rows -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
