from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from config.bootstrap import bootstrap_project_files
from config.defaults import CONFIG_DIR, DATA_DIR, MODEL_DIR, RISK_PROFILE_NAMES
from data_pipeline import run_data_pipeline
from train import TrainingConfig, train_all_profiles
from backtester import BacktestConfig, run_backtest_for_manifest
from utils.io import load_json, save_json

st.set_page_config(page_title="Algorithmic RL Trader", layout="wide")
bootstrap_project_files()

TICKER_FILE = CONFIG_DIR / "ticker.json"
RISK_FILE = CONFIG_DIR / "risk_profiles.json"
SENTIMENT_FILE = CONFIG_DIR / "sentiment.json"
RUN_FILE = CONFIG_DIR / "run_config.json"


def _alnum10(v: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9]{1,10}", v or ""))


def _is_valid_amount_2dp(v: str) -> bool:
    return bool(re.fullmatch(r"\d+(\.\d{1,2})?", v.strip()))


def _load_setup_defaults() -> dict:
    d = load_json(RUN_FILE, default={})
    d.setdefault("start_date", "2019-01-01")
    d.setdefault("end_date", "2026-03-31")
    d.setdefault("train_split", 0.8)
    return d


def _show_pipeline_plots():
    stat_dir = DATA_DIR / "stat"
    if not stat_dir.exists():
        st.warning("No plots found. Please run data pipeline first.")
        return

    png_files = sorted(stat_dir.glob("*.png"))
    if not png_files:
        st.warning("No PNG plots found in src/data/stat.")
        return

    st.subheader("Pipeline Plots")
    for p in png_files:
        st.markdown(f"**{p.stem.replace('_', ' ').title()}**")
        st.image(str(p), use_container_width=True)


def page_welcome():
    st.title("Algorithmic RL Trader")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("How to use")
        st.write("Instructions Go here")
    with c2:
        st.markdown(
            "<div style='font-size:120px;text-align:center;'>📈🤖</div>",
            unsafe_allow_html=True,
        )


def page_setup_data():
    st.header("Setup Your Data")

    if "pipeline_completed" not in st.session_state:
        st.session_state["pipeline_completed"] = False

    setup = _load_setup_defaults()
    risk_profiles = load_json(RISK_FILE, default={})
    if not risk_profiles:
        risk_profiles = {
            "aggressive": {"lambda_risk": 0.05, "turnover_penalty": 0.0005, "drawdown_penalty": 0.01},
            "moderate": {"lambda_risk": 0.5, "turnover_penalty": 0.001, "drawdown_penalty": 0.05},
            "conservative": {"lambda_risk": 2.0, "turnover_penalty": 0.004, "drawdown_penalty": 0.1},
        }

    ticker_json = load_json(TICKER_FILE, default={"tickers": ["AAPL", "MSFT", "GOOG"]})
    sentiments = load_json(SENTIMENT_FILE, default=[])

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start Date", value=date.fromisoformat(setup["start_date"]))
    with c2:
        end = st.date_input("End Date", value=date.fromisoformat(setup["end_date"]))
    st.caption(f"Selected Range: {start} to {end}")

    tickers_text = st.text_area(
        "Tickers (comma separated)",
        value=",".join(ticker_json.get("tickers", [])),
        help="Please provide correct ticker name available in Yahoo Finance, else this program will not run.",
    )

    st.subheader("Risk Profiles")
    st.info(
        "Risk: higher value penalizes volatility more.\n\n"
        "Turnover Penalty: higher value discourages frequent rebalancing.\n\n"
        "Drawdown Penalty: higher value penalizes deep portfolio drops."
    )

    rp_out = {}
    for p in RISK_PROFILE_NAMES:
        st.markdown(f"**{p.title()}**")
        default = risk_profiles.get(p, {})
        a, b, c = st.columns(3)
        with a:
            lam = st.slider(
                f"{p}-Risk",
                min_value=0.01,
                max_value=3.0,
                value=float(default.get("lambda_risk", 0.5)),
                step=0.01,
            )
        with b:
            to = st.slider(
                f"{p}-Turnover Penalty",
                min_value=0.0001,
                max_value=0.01,
                value=float(default.get("turnover_penalty", 0.001)),
                step=0.0001,
                format="%.4f",
            )
        with c:
            dd = st.slider(
                f"{p}-Drawdown Penalty",
                min_value=0.001,
                max_value=0.5,
                value=float(default.get("drawdown_penalty", 0.05)),
                step=0.001,
                format="%.3f",
            )
        rp_out[p] = {"lambda_risk": lam, "turnover_penalty": to, "drawdown_penalty": dd}

    split = st.slider("Train-Test Split", min_value=0.7, max_value=0.9, value=float(setup["train_split"]), step=0.01)

    use_sentiment = st.checkbox("Market Sentiment", value=False)
    sentiment_df = pd.DataFrame(
        sentiments
        if sentiments
        else [
            {"market_event": "COVID", "crash_factor": -0.9, "recovery_factor": 0.6},
            {"market_event": "RussiaUkraine", "crash_factor": -0.5, "recovery_factor": 0.3},
            {"market_event": "AmericanSlowdown", "crash_factor": -0.3, "recovery_factor": 0.4},
            {"market_event": "TrumpTariff", "crash_factor": -0.4, "recovery_factor": 0.6},
            {"market_event": "IranWar", "crash_factor": -0.8, "recovery_factor": 0.5},
        ]
    )

    if use_sentiment:
        st.caption("Crash Factor must be in [-1, 0). Recovery Factor must be in (0, 1].")
        sentiment_df = st.data_editor(
            sentiment_df,
            num_rows="fixed",
            column_config={
                "market_event": st.column_config.TextColumn("Market Event", disabled=True),
                "crash_factor": st.column_config.NumberColumn("Crash Factor", min_value=-1.0, max_value=-0.0001, step=0.01),
                "recovery_factor": st.column_config.NumberColumn("Recovery Factor", min_value=0.0001, max_value=1.0, step=0.01),
            },
            hide_index=True,
        )

    csave, crun = st.columns([1, 1])
    with csave:
        if st.button("Save Setup", use_container_width=True):
            tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
            save_json(TICKER_FILE, {"tickers": tickers})
            save_json(RISK_FILE, rp_out)
            save_json(
                RUN_FILE,
                {"start_date": str(start), "end_date": str(end), "train_split": split, "use_sentiment": use_sentiment},
            )
            if use_sentiment:
                rows = sentiment_df.to_dict(orient="records")
                rows = [
                    {
                        "market_event": r["market_event"],
                        "crash_factor": float(max(-1.0, min(-0.0001, r["crash_factor"]))),
                        "recovery_factor": float(max(0.0001, min(1.0, r["recovery_factor"]))),
                    }
                    for r in rows
                ]
                save_json(SENTIMENT_FILE, rows)
            st.success("Saved configuration to src/config.")

    with crun:
        if st.button("Run Data Pipeline", use_container_width=True):
            with st.spinner("Running data pipeline..."):
                run_data_pipeline()
            st.session_state["pipeline_completed"] = True
            st.success("Pipeline completed. Data saved in src/data and charts in src/data/stat.")

    if st.session_state.get("pipeline_completed", False):
        if st.button("Show Plots", use_container_width=True):
            _show_pipeline_plots()


def page_train():
    st.header("Train Your Model")

    init_invest = st.number_input("Initial Investment", min_value=1000.0, value=100000.0, step=1000.0)
    txn_pct = st.number_input("Transaction Cost (%)", min_value=0.0, value=0.02, step=0.01)
    settlement_day = st.number_input("Settlement Day", min_value=0, max_value=10, value=2, step=1)
    use_graph = st.checkbox("Asset Relationship Graph (with GCN-like extractor)", value=False)

    if use_graph:
        corr_file = DATA_DIR / "stat" / "correlation_matrix.csv"
        if corr_file.exists():
            corr_df = pd.read_csv(corr_file, index_col=0)
            st.dataframe(corr_df)
        else:
            st.info("Run Setup/Data Pipeline first to generate correlation matrix.")

    stepsize = st.number_input("Stepsize", min_value=1000, max_value=5000000, value=100000, step=1000)

    st.subheader("Hyperparameters")
    lr = st.slider("Learning Rate", min_value=1e-5, max_value=5e-3, value=3e-4, step=1e-5, format="%.5f")
    batch = st.slider("Batch Size", min_value=16, max_value=1024, value=64, step=16)
    gamma = st.slider("Gamma", min_value=0.80, max_value=0.999, value=0.99, step=0.001)
    ent = st.slider("Entropy Coef", min_value=0.0, max_value=0.05, value=0.005, step=0.001)

    use_seed = st.checkbox("Use Seed=42", value=True)
    model_tag = st.text_input("Model Tag (alphanumeric, max 10 chars)", value="RUN1", max_chars=10)

    if st.button("Start Training", use_container_width=True):
        if not _alnum10(model_tag):
            st.error("Model tag must be alphanumeric and length <= 10.")
            return

        log_box = st.empty()
        logs: List[str] = []

        def logger(msg: str):
            logs.append(msg)
            log_box.code("\n".join(logs[-25:]))

        cfg = TrainingConfig(
            initial_investment=float(init_invest),
            transaction_cost=float(txn_pct) / 100.0,
            settlement_day=int(settlement_day),
            use_graph_extractor=use_graph,
            total_timesteps=int(stepsize),
            learning_rate=float(lr),
            batch_size=int(batch),
            gamma=float(gamma),
            ent_coef=float(ent),
            use_seed=bool(use_seed),
            model_tag=model_tag,
        )
        train_all_profiles(cfg, logger=logger)
        st.success("Training completed for all profiles.")


def page_backtester():
    st.header("Backtester")

    manifests = sorted(MODEL_DIR.glob("model_manifest_*.json"), reverse=True)
    if not manifests:
        st.info("No model manifests found. Train first.")
        return

    selected_manifest = st.selectbox("Select Model Manifest", options=[m.name for m in manifests])
    manifest_path = MODEL_DIR / selected_manifest

    run_mode_label = st.radio(
        "Backtest Mode",
        options=["Run on test data", "Run on another data"],
        index=0,
        horizontal=True,
    )

    custom_amount = None
    custom_start = None
    custom_end = None

    if run_mode_label == "Run on another data":
        st.subheader("Custom Backtest Inputs")
        custom_amount_text = st.text_input(
            "Initial Amount (INR)",
            value="100000.00",
            help="Enter numeric value with up to 2 decimal places, e.g. 250000 or 250000.50",
        )
        c1, c2 = st.columns(2)
        with c1:
            custom_start = st.date_input("Start Date", value=date(2024, 10, 1), key="bt_custom_start")
        with c2:
            custom_end = st.date_input("End Date", value=date(2026, 1, 31), key="bt_custom_end")

        if custom_end <= custom_start:
            st.error("Validation failed: End Date must be greater than Start Date.")

        if _is_valid_amount_2dp(custom_amount_text):
            custom_amount = float(custom_amount_text)
        else:
            st.error("Initial Amount must be numeric with up to 2 decimal places.")

    current_run_signature = (
        selected_manifest,
        run_mode_label,
        str(custom_amount) if custom_amount is not None else "",
        str(custom_start) if custom_start is not None else "",
        str(custom_end) if custom_end is not None else "",
    )
    if st.session_state.get("last_backtest_signature") != current_run_signature:
        st.session_state.pop("last_backtest", None)

    if st.button("Run Backtester", use_container_width=True):
        if run_mode_label == "Run on another data":
            if custom_amount is None:
                st.error("Please provide a valid Initial Amount.")
                return
            if custom_start is None or custom_end is None or custom_end <= custom_start:
                st.error("Please provide valid Start Date and End Date (End Date > Start Date).")
                return

            bt_cfg = BacktestConfig(
                manifest_path=manifest_path,
                run_mode="custom_data",
                initial_investment=custom_amount,
                start_date=str(custom_start),
                end_date=str(custom_end),
            )
        else:
            bt_cfg = BacktestConfig(
                manifest_path=manifest_path,
                run_mode="test_data",
                initial_investment=100000.0,
            )

        with st.spinner("Running backtest..."):
            result = run_backtest_for_manifest(bt_cfg)

        st.session_state["last_backtest"] = result
        st.session_state["last_backtest_signature"] = current_run_signature
        st.success("Backtest completed.")

    if "last_backtest" in st.session_state:
        st.subheader("Show Results")
        result = st.session_state["last_backtest"]

        for profile, chart_path in result.get("profile_value_charts", {}).items():
            st.markdown(f"**{profile.title()} Portfolio Value**")
            st.image(str(chart_path))

        for profile, chart_path in result.get("profile_action_charts", {}).items():
            st.markdown(f"**{profile.title()} Action Space (Allocation)**")
            st.image(str(chart_path))

        st.markdown("**Your Portfolio Output**")
        for profile, csv_path in result.get("profile_csvs", {}).items():
            with open(csv_path, "rb") as f:
                st.download_button(
                    label=f"Download {profile} CSV",
                    data=f,
                    file_name=Path(csv_path).name,
                    mime="text/csv",
                    key=f"dl_{profile}_{selected_manifest}",
                )

        st.markdown("**Common State Charts (src/output)**")
        for p in result.get("common_state_charts", []):
            st.image(str(p))

        with st.expander("XIRR"):
            for profile, x in result.get("xirr", {}).items():
                st.write(f"{profile.title()}: {x:.2%}")


def main():
    st.sidebar.title("Algorithmic RL Trader")
    page = st.sidebar.radio(
        "Pages",
        ["Welcome", "Setup Your Data", "Train Your Model", "Backtester"],
    )
    if page == "Welcome":
        page_welcome()
    elif page == "Setup Your Data":
        page_setup_data()
    elif page == "Train Your Model":
        page_train()
    else:
        page_backtester()


if __name__ == "__main__":
    main()
