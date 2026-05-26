
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from config.defaults import CONFIG_DIR, DATA_DIR, DATA_STAT_DIR
from utils.io import load_json


# TICKER_TO_SENTIMENT_MAP = {
#     # Growth Assets
#     "TCS.NS": "Growth Sentiment",
#     "RELIANCE.NS": "Growth Sentiment",
#     "INFY.NS": "Growth Sentiment",
    
#     # Anchor Assets
#     "ITC.NS": "Anchor Sentiment",
#     "HDFCBANK.NS": "Anchor Sentiment",
#     "HUL.NS": "Anchor Sentiment",
    
#     # Safe Havens
#     "GOLDBEES.NS": "Gold Sentiment",
#     "LIQUIDBEES.NS": "Gov Bonds Sentiment",
#     "GOVT_BOND_TICKER": "Gov Bonds Sentiment" 
# }

# def _build_excel_sentiment(price_index: pd.DatetimeIndex, csv_path: Path) -> pd.DataFrame:
#     # 1. Load your Excel/CSV sheet
#     raw_sentiment = pd.read_csv(csv_path)
#     raw_sentiment["Start Date"] = pd.to_datetime(raw_sentiment["Start Date"])
#     raw_sentiment["End Date"] = pd.to_datetime(raw_sentiment["End Date"])
    
#     # 2. Create an empty DataFrame with 0.0s for the 9 assets
#     tickers = list(TICKER_TO_SENTIMENT_MAP.keys())
#     daily_sentiment = pd.DataFrame(0.0, index=price_index, columns=tickers)
    
#     # 3. Paint the sentiment scores onto the daily timeline
#     for _, row in raw_sentiment.iterrows():
#         start = row["Start Date"]
#         end = row["End Date"]
        
#         # Find which days in our price data fall into this macro event
#         mask = (daily_sentiment.index >= start) & (daily_sentiment.index <= end)
        
#         # Map the Excel columns to the specific tickers
#         for ticker, sentiment_column in TICKER_TO_SENTIMENT_MAP.items():
#             if sentiment_column in raw_sentiment.columns:
#                 daily_sentiment.loc[mask, ticker] = float(row[sentiment_column])
                
#     return daily_sentiment

@dataclass
class PipelineArtifacts:
    prices: pd.DataFrame
    returns: pd.DataFrame
    volatility: pd.DataFrame
    #sentiment: pd.DataFrame
    z_score: pd.DataFrame
    combined: pd.DataFrame


def _resolve_tickers() -> List[str]:
    ticker_cfg = load_json(CONFIG_DIR / "ticker.json", default={"tickers": []})
    tickers = [str(t).strip().upper() for t in ticker_cfg.get("tickers", []) if str(t).strip()]
    if not tickers:
        raise ValueError("No tickers found in src/config/ticker.json")
    return tickers


def _resolve_run_cfg() -> dict:
    run_cfg = load_json(CONFIG_DIR / "run_config.json", default={})
    run_cfg.setdefault("start_date", "2018-01-01")
    run_cfg.setdefault("end_date", "2026-12-31")
    run_cfg.setdefault("train_split", 0.8)
    run_cfg.setdefault("use_sentiment", False)
    return run_cfg


def _resolve_sentiment_cfg() -> Dict[str, Dict[str, float]]:
    rows = load_json(CONFIG_DIR / "sentiment.json", default=[])
    # Expected rows:
    # [{"market_event":"COVID","crash_factor":-0.9,"recovery_factor":0.6}, ...]
    out = {}
    for r in rows:
        ev = r.get("market_event")
        if not ev:
            continue
        out[ev] = {
            "crash_factor": float(r.get("crash_factor", 0.0)),
            "recovery_factor": float(r.get("recovery_factor", 0.0)),
        }
    return out


def _macro_overlay(ts: pd.Timestamp, event_factors: Dict[str, Dict[str, float]]) -> float:
    ym = ts.strftime("%Y-%m")
    for event_name, windows in EVENT_WINDOWS.items():
        factors = event_factors.get(event_name)
        if not factors:
            continue
        for start_ym, end_ym, phase in windows:
            if start_ym <= ym <= end_ym:
                if phase == "crash":
                    return factors["crash_factor"]
                return factors["recovery_factor"]
    return 0.0


def _download_close_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    # Typical shape: MultiIndex columns -> ('Close', ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            data = raw["Close"].copy()
        else:
            # fallback: first top-level if Close not present
            lvl0 = raw.columns.get_level_values(0).unique()[0]
            data = raw[lvl0].copy()
    else:
        # single ticker case
        data = raw.to_frame(name=tickers[0])

    # Flatten any remaining MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)

    # align requested tickers ordering (if some missing, they become NaN then filled)
    data = data.reindex(columns=tickers)

    # NOTE: as requested, fill missing using ffill + bfill
    data = data.ffill().bfill()

    # Remove rows still fully empty (very rare)
    # data = data.dropna(how="all")
    return data


def _build_features(
    prices: pd.DataFrame,
    use_sentiment: bool,
    #sentiment_cfg: Dict[str, Dict[str, float]],
    vol_window: int = 20,
) -> PipelineArtifacts:
    # Log returns (as per notebook)
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.ffill().bfill()

    # Volatility
    volatility = log_returns.rolling(window=vol_window).std()
    volatility = volatility.ffill().bfill()

    # 20-Day Normalized Z-Score Price
    rolling_mean = prices.rolling(window=20).mean()
    rolling_std = prices.rolling(window=20).std()
    z_score = (prices - rolling_mean) / rolling_std
    z_score = z_score.ffill().bfill() 
    
    if use_sentiment:
        macro_series = pd.Series(
            [_macro_overlay(ts, sentiment_cfg) for ts in prices.index],
            index=prices.index,
            name="sentiment",
        )
    else:
        macro_series = pd.Series(0.0, index=prices.index, name="sentiment")
    
    csv_path = "sentiment_scores.xlsx" # Or wherever you save it
    #sentiment = _build_excel_sentiment(prices.index, csv_path) 
    sentiment_df = pd.DataFrame(0.0, index=prices.index, columns=["sentiment"])

    # Combine into the final requested frame
    combined = pd.concat(
        [prices, log_returns, volatility, z_score],
        axis=1,
        keys=["prices", "returns", "vol", "z_score"],
    )
    

    # Optional strict alignment intentionally not forced.

    # Split back
    prices_out = combined["prices"]
    returns_out = combined["returns"]
    vol_out = combined["vol"]
    z_score_out = combined["z_score"]
    # sentiment_out = combined["sentiment"]  # shape: (T,1)

    # Safety fill again after concat/split
    prices_out = prices_out.ffill().bfill()
    returns_out = returns_out.ffill().bfill()
    vol_out = vol_out.ffill().bfill()
    z_score_out = z_score_out.ffill().bfill()
    # sentiment_out = sentiment_out.ffill().bfill()
    print(combined.head())
    return PipelineArtifacts(
        prices=prices_out,
        returns=returns_out,
        volatility=vol_out,
        #sentiment=sentiment_out,
        z_score=z_score_out,
        combined=combined,
    )

def _save_visualizations(prices: pd.DataFrame):
    DATA_STAT_DIR.mkdir(parents=True, exist_ok=True)

    # Asset price plot
    plt.figure(figsize=(12, 6))
    for col in prices.columns:
        plt.plot(prices.index, prices[col], label=col)
    plt.title("Asset Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DATA_STAT_DIR / "asset_prices.png", dpi=170)
    plt.close()



def _save_splits(art: PipelineArtifacts, split_ratio: float):
    split_idx = int(len(art.prices) * split_ratio)

    # Wide splits (for notebook-style checks/backtests)
    train_prices = art.prices.iloc[:split_idx]
    test_prices = art.prices.iloc[split_idx:]

    train_returns = art.returns.iloc[:split_idx]
    test_returns = art.returns.iloc[split_idx:]

    train_vol = art.volatility.iloc[:split_idx]
    test_vol = art.volatility.iloc[split_idx:]

    train_zscore = art.z_score.iloc[:split_idx]
    test_zscore = art.z_score.iloc[split_idx:]

    # train_sent = art.sentiment.iloc[:split_idx]
    # test_sent = art.sentiment.iloc[split_idx:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_prices.to_parquet(DATA_DIR / "train_prices.parquet")
    test_prices.to_parquet(DATA_DIR / "test_prices.parquet")

    train_returns.to_parquet(DATA_DIR / "train_returns.parquet")
    test_returns.to_parquet(DATA_DIR / "test_returns.parquet")

    train_vol.to_parquet(DATA_DIR / "train_volatility.parquet")
    test_vol.to_parquet(DATA_DIR / "test_volatility.parquet")

    train_zscore.to_parquet(DATA_DIR / "train_zscore.parquet")
    test_zscore.to_parquet(DATA_DIR / "test_zscore.parquet")

    #train_sent.to_parquet(DATA_DIR / "train_sentiment.parquet")
    #test_sent.to_parquet(DATA_DIR / "test_sentiment.parquet")

    # Long panel for training env (date/ticker rows)
    # Includes sentiment scalar broadcast to each ticker per date
    panel = []
    for tkr in art.prices.columns:
        tmp = pd.DataFrame(
            {
                "date": art.prices.index,
                "ticker": tkr,
                "price": art.prices[tkr].values,
                "log_return": art.returns[tkr].values,
                "volatility": art.volatility[tkr].values,
                "z_score": art.z_score[tkr].values,
                #"sentiment": art.sentiment["sentiment"].values,
            }
        )
        panel.append(tmp)
    panel_df = pd.concat(panel, axis=0, ignore_index=True).sort_values(["date", "ticker"])

    split_date = art.prices.index[split_idx - 1] if split_idx > 0 else art.prices.index.min()
    train_panel = panel_df[panel_df["date"] <= split_date].copy()
    test_panel = panel_df[panel_df["date"] > split_date].copy()

    train_panel.to_parquet(DATA_DIR / "train.parquet")
    test_panel.to_parquet(DATA_DIR / "test.parquet")

    # Save full artifacts too
    art.prices.to_parquet(DATA_DIR / "prices.parquet")
    art.returns.to_parquet(DATA_DIR / "returns.parquet")
    art.volatility.to_parquet(DATA_DIR / "volatility.parquet")
    art.z_score.to_parquet(DATA_DIR / "z_score.parquet")
    #art.sentiment.to_parquet(DATA_DIR / "sentiment.parquet")

    # MultiIndex columns can be awkward in parquet for some readers; save as CSV too
    art.combined.to_csv(DATA_DIR / "combined.csv")

    return {
        "split_idx": split_idx,
        "train_prices_shape": train_prices.shape,
        "test_prices_shape": test_prices.shape,
        "train_returns_shape": train_returns.shape,
        "test_returns_shape": test_returns.shape,
        "train_vol_shape": train_vol.shape,
        "test_vol_shape": test_vol.shape,
        "train_zscore_shape": train_zscore.shape,
        "test_zscore_shape": test_zscore.shape,
        #"train_sent_shape": train_sent.shape,
        #"test_sent_shape": test_sent.shape,
    }


def run_data_pipeline() -> dict:
    """
    Reads Streamlit user configs from src/config and writes outputs to src/data.
    Returns validation metadata (shapes, NaN checks) for logging/UI.
    """
    tickers = _resolve_tickers()
    run_cfg = _resolve_run_cfg()
    #sentiment_cfg = _resolve_sentiment_cfg()

    prices_raw = _download_close_prices(
        tickers=tickers,
        start_date=run_cfg["start_date"],
        end_date=run_cfg["end_date"],
    )

    artifacts = _build_features(
        prices=prices_raw,
        use_sentiment=bool(run_cfg.get("use_sentiment", False)),
        #sentiment_cfg=sentiment_cfg,
        vol_window=20,
    )

    _save_visualizations(artifacts.prices)
    split_stats = _save_splits(artifacts, float(run_cfg["train_split"]))

    validation = {
        "final_shapes": {
            "prices": artifacts.prices.shape,
            "returns": artifacts.returns.shape,
            "volatility": artifacts.volatility.shape,
            #"sentiment": artifacts.sentiment.shape,  # should be (T, 1)
        },
        "nan_check": {
            "prices": int(artifacts.prices.isna().sum().sum()),
            "returns": int(artifacts.returns.isna().sum().sum()),
            "volatility": int(artifacts.volatility.isna().sum().sum()),
            #"sentiment": int(artifacts.sentiment.isna().sum().sum()),
        },
        "split": split_stats,
        "saved_files": {
            "prices": str(DATA_DIR / "prices.parquet"),
            "returns": str(DATA_DIR / "returns.parquet"),
            "volatility": str(DATA_DIR / "volatility.parquet"),
            #"sentiment": str(DATA_DIR / "sentiment.parquet"),
            "combined_csv": str(DATA_DIR / "combined.csv"),
            "asset_plot": str(DATA_STAT_DIR / "asset_prices.png"),
            #"sentiment_plot": str(DATA_STAT_DIR / "market_sentiment.png"),
        },
    }
    return validation
