"""Eksekusi notebook secara headless untuk memastikan semua sel berjalan tanpa error.

Contoh:
    python scripts/run_notebooks.py                       # semua notebook di root repo
    python scripts/run_notebooks.py 05_level_expert.ipynb # notebook tertentu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]


def run(path: Path, timeout: int) -> bool:
    nb = nbformat.read(path, as_version=4)
    start = time.perf_counter()
    try:
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    except CellExecutionError as exc:
        print(f"[GAGAL] {path.name}\n{exc}")
        return False
    print(f"[OK]    {path.name} ({time.perf_counter() - start:.1f} detik)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Path notebook (default: semua *.ipynb di root).")
    parser.add_argument("--timeout", type=int, default=1200, help="Batas waktu per sel (detik).")
    args = parser.parse_args()

    paths = [Path(p) for p in args.notebooks] or sorted(ROOT.glob("*.ipynb"))
    results = [run(p.resolve(), args.timeout) for p in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
