from __future__ import annotations

from config.defaults import CONFIG_DIR, DATA_DIR, DATA_STAT_DIR, MODEL_DIR, OUTPUT_DIR
from utils.io import save_json

DEFAULT_RISK = {
    "aggressive": {"lambda_risk": 0.05, "turnover_penalty": 0.0005, "drawdown_penalty": 0.01},
    "moderate": {"lambda_risk": 0.5, "turnover_penalty": 0.001, "drawdown_penalty": 0.05},
    "conservative": {"lambda_risk": 2.0, "turnover_penalty": 0.004, "drawdown_penalty": 0.1},
}

DEFAULT_SENTIMENT = [
    {"market_event": "COVID", "crash_factor": -0.9, "recovery_factor": 0.6},
    {"market_event": "RussiaUkraine", "crash_factor": -0.5, "recovery_factor": 0.3},
    {"market_event": "AmericanSlowdown", "crash_factor": -0.3, "recovery_factor": 0.4},
    {"market_event": "TrumpTariff", "crash_factor": -0.4, "recovery_factor": 0.6},
    {"market_event": "IranWar", "crash_factor": -0.8, "recovery_factor": 0.5},
]

DEFAULT_RUN = {
    "start_date": "2019-01-01",
    "end_date": "2026-03-31",
    "train_split": 0.8,
    "use_sentiment": False,
}

DEFAULT_TICKERS = {"tickers": ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]}


def bootstrap_project_files():
    for d in [CONFIG_DIR, DATA_DIR, DATA_STAT_DIR, MODEL_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for p in ["aggressive", "moderate", "conservative"]:
        (OUTPUT_DIR / p).mkdir(parents=True, exist_ok=True)

    save_json(CONFIG_DIR / "risk_profiles.json", DEFAULT_RISK, if_missing=True)
    save_json(CONFIG_DIR / "sentiment.json", DEFAULT_SENTIMENT, if_missing=True)
    save_json(CONFIG_DIR / "run_config.json", DEFAULT_RUN, if_missing=True)
    save_json(CONFIG_DIR / "ticker.json", DEFAULT_TICKERS, if_missing=True)
