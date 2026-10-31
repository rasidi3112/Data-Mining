import sys
from pathlib import Path

# Pastikan paket dmtoolkit dapat diimpor tanpa instalasi (pip install -e .).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
