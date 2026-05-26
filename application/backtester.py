from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import newton
from stable_baselines3 import PPO

from config.defaults import CONFIG_DIR, DATA_DIR, OUTPUT_DIR, RISK_PROFILE_NAMES
from env import EnvConfig, MultiAssetTradingEnv
from utils.io import load_json


@dataclass
class BacktestConfig:
    manifest_path: Path
    run_mode: str = "test_data"  # "test_data" | "custom_data"
    initial_investment: float = 100000.0
    start_date: str | None = None
    end_date: str | None = None


EVENT_WINDOWS: Dict[str, List[Tuple[str, str, str]]] = {
    "COVID": [("2020-03", "2020-06", "crash"), ("2021-01", "2021-12", "recovery")],
    "RussiaUkraine": [("2022-02", "2022-09", "crash"), ("2022-10", "2022-12", "recovery")],
    "AmericanSlowdown": [("2023-09", "2023-10", "crash"), ("2023-11", "2024-01", "recovery")],
    "TrumpTariff": [("2025-02", "2025-04", "crash"), ("2025-05", "2025-07", "recovery")],
    "IranWar": [("2026-03", "2026-03", "crash")],
}


def _xnpv(rate, cashflows):
    return sum(cf / ((1 + rate) ** t) for t, cf in cashflows)


def _xirr(cashflows):
    try:
        return newton(lambda r: _xnpv(r, cashflows), x0=0.1, maxiter=100)
    except Exception:
        return np.nan


def _macro_overlay(ts: pd.Timestamp, event_factors: Dict[str, Dict[str, float]]) -> float:
    ym = ts.strftime("%Y-%m")
    for event_name, windows in EVENT_WINDOWS.items():
        factors = event_factors.get(event_name)
        if not factors:
            continue
        for start_ym, end_ym, phase in windows:
            if start_ym <= ym <= end_ym:
                return float(factors["crash_factor"] if phase == "crash" else factors["recovery_factor"])
    return 0.0


def _build_custom_data(start_date: str, end_date: str):
    ticker_cfg = load_json(CONFIG_DIR / "ticker.json", default={"tickers": []})
    tickers = [t.strip().upper() for t in ticker_cfg.get("tickers", []) if str(t).strip()]
    if not tickers:
        raise ValueError("No tickers found in src/config/ticker.json")

    run_cfg = load_json(CONFIG_DIR / "run_config.json", default={})
    use_sentiment = bool(run_cfg.get("use_sentiment", False))
    sentiment_rows = load_json(CONFIG_DIR / "sentiment.json", default=[])

    event_factors = {
        r["market_event"]: {
            "crash_factor": float(r.get("crash_factor", 0.0)),
            "recovery_factor": float(r.get("recovery_factor", 0.0)),
        }
        for r in sentiment_rows
        if "market_event" in r
    }

    raw = yf.download(tickers=tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError("No data downloaded for selected date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            lvl0 = raw.columns.get_level_values(0).unique()[0]
            prices = raw[lvl0].copy()
    else:
        prices = raw.to_frame(name=tickers[0])

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(-1)

    prices = prices.reindex(columns=tickers).ffill().bfill().dropna(how="all")
    if len(prices) < 35:
        raise ValueError("Not enough rows for backtest. Increase date range.")

    returns = np.log(prices / prices.shift(1)).ffill().bfill()
    volatility = returns.rolling(window=20).std().ffill().bfill()
    zscore = (returns - returns.rolling(window=20).mean()) / returns.rolling(window=20).std()
    # zscore.ffill().bfill()

    if use_sentiment:
        macro = pd.Series([_macro_overlay(ts, event_factors) for ts in prices.index], index=prices.index, name="macro")
    else:
        macro = pd.Series(0.0, index=prices.index, name="macro")

    base_sent = np.tanh(returns * 5)
    sentiment_full = base_sent.add(macro, axis=0)
    sentiment = sentiment_full.mean(axis=1).to_frame(name="sentiment").ffill().bfill()

    return prices, returns, volatility, zscore, sentiment


def _load_test_data():
    test_prices = pd.read_parquet(DATA_DIR / "test_prices.parquet")
    test_returns = pd.read_parquet(DATA_DIR / "test_returns.parquet")
    test_vol = pd.read_parquet(DATA_DIR / "test_volatility.parquet")
    test_zscore = pd.read_parquet(DATA_DIR / "test_zscore.parquet")
    test_sent = pd.read_parquet(DATA_DIR / "test_sentiment.parquet")
    return test_prices, test_returns, test_vol, test_zscore, test_sent


def _constant_weight_portfolio_values(
    returns: pd.DataFrame,
    start_t: int,
    periods: int,
    initial_investment: float,
    weights: np.ndarray,
) -> np.ndarray:
    """Value path for a no-action portfolio that never rebalances."""
    if periods <= 0:
        return np.array([], dtype=float)

    weights = np.asarray(weights, dtype=float)
    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        weights = np.ones(returns.shape[1], dtype=float) / returns.shape[1]
    else:
        weights = weights / weight_sum

    held_returns = returns.iloc[start_t : start_t + periods].to_numpy(dtype=float)
    held_returns = np.nan_to_num(held_returns, nan=0.0, posinf=0.0, neginf=0.0)
    daily_simple_returns = np.exp(held_returns @ weights) - 1.0
    return float(initial_investment) * np.cumprod(1.0 + daily_simple_returns)


def run_backtest_for_manifest(cfg: BacktestConfig):
    manifest = load_json(cfg.manifest_path, {})
    risk_profiles = load_json(CONFIG_DIR / "risk_profiles.json", {})

    if cfg.run_mode == "custom_data":
        if not cfg.start_date or not cfg.end_date:
            raise ValueError("start_date and end_date are required for custom_data mode.")
        test_prices, test_returns, test_vol, test_zscore, test_sent = _build_custom_data(cfg.start_date, cfg.end_date)
    else:
        test_prices, test_returns, test_vol, test_zscore, test_sent = _load_test_data()

    tickers = list(test_returns.columns)

    profile_charts: Dict[str, str] = {}
    profile_action_charts: Dict[str, str] = {}
    profile_csvs: Dict[str, str] = {}
    xirr_out: Dict[str, float] = {}
    constant_weight_xirr_out: Dict[str, float] = {}
    common_state_charts = []
    all_daily_returns = {}

    for profile in RISK_PROFILE_NAMES:
        if profile not in manifest.get("profiles", {}):
            continue

        model_path = Path(manifest["profiles"][profile]["model_path"])
        model = PPO.load(str(model_path))

        env_cfg = EnvConfig(
            initial_investment=float(cfg.initial_investment),
            transaction_cost=0.0002,
            settlement_day=2,
            window_size=30,
        )

        env = MultiAssetTradingEnv(
            prices=test_prices,
            returns=test_returns,
            volatility=test_vol,
            z_score=test_zscore,
            sentiment=test_sent,
            risk_profile=profile,
            risk_profiles=risk_profiles,
            cfg=env_cfg,
        )

        obs, _ = env.reset()
        initial_weights = env.weights.copy()
        rows = []
        done = False
        step = 0
        running_max = env_cfg.initial_investment

        while not done:
            current_t = env.t
            current_date = pd.to_datetime(test_prices.index[current_t])

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if "return_simple" in info:
                daily_ret = float(info["return_simple"])
            elif "return_log" in info:
                daily_ret = float(np.exp(info["return_log"]) - 1.0)
            else:
                daily_ret = float(info.get("return", 0.0))

            pv = float(info["portfolio_value"])
            running_max = max(running_max, pv)
            drawdown = float(info.get("drawdown", (running_max - pv) / max(running_max, 1e-12)))

            rows.append(
                {
                    "step": step,
                    "date": current_date,
                    "portfolio_value": pv,
                    "daily_return": daily_ret,
                    "drawdown": drawdown,
                    "transaction_cost": float(info["transaction_cost"]),
                    "turnover": float(info["turnover"]),
                    "sentiment": float(info["sentiment"]),
                    **{f"w_{t}": w for t, w in zip(tickers, info["weights"])},
                }
            )
            step += 1

        out_df = pd.DataFrame(rows)
        if out_df.empty:
            continue

        out_df["date"] = pd.to_datetime(out_df["date"])
        out_df["constant_weight_portfolio_value"] = _constant_weight_portfolio_values(
            returns=test_returns,
            start_t=env_cfg.window_size,
            periods=len(out_df),
            initial_investment=env_cfg.initial_investment,
            weights=initial_weights,
        )
        all_daily_returns[profile] = out_df[["date", "daily_return"]].copy()

        prof_dir = OUTPUT_DIR / profile
        prof_dir.mkdir(parents=True, exist_ok=True)

        csv_path = prof_dir / "backtest_detail.csv"
        out_df.to_csv(csv_path, index=False)
        profile_csvs[profile] = str(csv_path)

        plt.figure(figsize=(10, 4))
        plt.plot(out_df["date"], out_df["portfolio_value"], label=f"{profile.title()} RL Strategy")
        plt.plot(
            out_df["date"],
            out_df["constant_weight_portfolio_value"],
            label="No Action (Initial Weights Held)",
            linestyle="--",
        )
        plt.title(f"{profile.title()} Portfolio Value vs No Action")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        pv_path = prof_dir / "portfolio_value.png"
        plt.savefig(pv_path, dpi=160)
        plt.close()
        profile_charts[profile] = str(pv_path)

        w_cols = [c for c in out_df.columns if c.startswith("w_")]
        if w_cols:
            weights_df = out_df[["date"] + w_cols].copy().set_index("date")
            plt.figure(figsize=(14, 6))
            plt.stackplot(
                weights_df.index,
                weights_df[w_cols].T.values,
                labels=[c.replace("w_", "") for c in w_cols],
            )
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.title(f"Action Space (Portfolio Allocation) - {profile}")
            plt.xlabel("Date")
            plt.ylabel("Weight Allocation")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            action_path = prof_dir / "action_space.png"
            plt.savefig(action_path, dpi=160)
            plt.close()
            profile_action_charts[profile] = str(action_path)

        cashflows = [
            (0.0, -env_cfg.initial_investment),
            (len(out_df) / 252.0, float(out_df["portfolio_value"].iloc[-1])),
        ]
        xirr_out[profile] = _xirr(cashflows)
        constant_weight_cashflows = [
            (0.0, -env_cfg.initial_investment),
            (len(out_df) / 252.0, float(out_df["constant_weight_portfolio_value"].iloc[-1])),
        ]
        constant_weight_xirr_out[profile] = _xirr(constant_weight_cashflows)

    if all_daily_returns:
        plt.figure(figsize=(10, 4))
        for p, df_ret in all_daily_returns.items():
            rolling_vol = df_ret["daily_return"].rolling(20).std()
            plt.plot(df_ret["date"], rolling_vol, label=p)
        plt.legend()
        plt.title("Rolling Volatility (20)")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        vol_path = OUTPUT_DIR / "state_volatility.png"
        plt.savefig(vol_path, dpi=160)
        plt.close()
        common_state_charts.append(str(vol_path))

        plt.figure(figsize=(10, 4))
        for p, df_ret in all_daily_returns.items():
            cum = (1.0 + df_ret["daily_return"]).cumprod() - 1.0
            plt.plot(df_ret["date"], cum, label=p)
        plt.legend()
        plt.title("Cumulative Return")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        ret_path = OUTPUT_DIR / "state_return.png"
        plt.savefig(ret_path, dpi=160)
        plt.close()
        common_state_charts.append(str(ret_path))

    return {
        "profile_value_charts": profile_charts,
        "profile_action_charts": profile_action_charts,
        "profile_csvs": profile_csvs,
        "common_state_charts": common_state_charts,
        "xirr": xirr_out,
        "constant_weight_xirr": constant_weight_xirr_out,
    }
