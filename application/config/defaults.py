from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

CONFIG_DIR = SRC_DIR / "config"
DATA_DIR = SRC_DIR / "data"
DATA_STAT_DIR = DATA_DIR / "stat"
MODEL_DIR = SRC_DIR / "model"
OUTPUT_DIR = SRC_DIR / "output"

RISK_PROFILE_NAMES = ["aggressive", "moderate", "conservative"]
