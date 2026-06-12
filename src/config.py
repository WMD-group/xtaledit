"""Project-wide path configuration.

Machine-specific paths can be set in .env (gitignored):
    INPUT_DIR=/path/to/input
    RESULTS_DIR=/path/to/results
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = Path(os.environ.get("INPUT_DIR", str(ROOT / "input")))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(ROOT / "results")))
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis"
