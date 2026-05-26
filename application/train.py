# train.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from feature_extractors import HybridGCNExtractor
from config.defaults import CONFIG_DIR, DATA_DIR, MODEL_DIR, RISK_PROFILE_NAMES
from env import EnvConfig, MultiAssetTradingEnv
from utils.io import load_json, save_json


@dataclass
class TrainingConfig:
    initial_investment: float
    transaction_cost: float
    settlement_day: int
    use_graph_extractor: bool
    total_timesteps: int
    learning_rate: float
    batch_size: int
    gamma: float
    ent_coef: float
    use_seed: bool
    model_tag: str


class GCNFeatureExtractor(BaseFeaturesExtractor):
    """
    Lightweight GCN-like extractor for dict observations:
    returns:(B,W,N), volatility:(B,W,N), sentiment:(B,W,1), weights:(B,N)
    """
    def __init__(self, observation_space, adjacency: np.ndarray, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_assets = observation_space.spaces["weights"].shape[0]
        self.n_assets = n_assets
        self.register_buffer("adj", torch.tensor(adjacency, dtype=torch.float32))

        # Per-asset summary features: mean/std of returns, mean vol, mean sentiment => 4 scalars per asset
        self.asset_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Linear(n_assets * 32 + n_assets, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r = obs["returns"]            # (B,W,N)
        v = obs["volatility"]         # (B,W,N)
        s = obs["sentiment"]          # (B,W,1)
        w = obs["weights"]            # (B,N)

        r_mean = r.mean(dim=1)        # (B,N)
        r_std = r.std(dim=1)          # (B,N)
        v_mean = v.mean(dim=1)        # (B,N)
        s_mean = s.mean(dim=1)        # (B,1) -> broadcast
        s_mean = s_mean.repeat(1, self.n_assets)

        x = torch.stack([r_mean, r_std, v_mean, s_mean], dim=-1)  # (B,N,4)

        # graph mixing
        ax = torch.einsum("ij,bjk->bik", self.adj.to(x.device), x)
        h = self.asset_mlp(ax)        # (B,N,32)
        h = h.reshape(h.shape[0], -1)

        return self.final(torch.cat([h, w], dim=1))


def _load_train_data():
    train_prices = pd.read_parquet(DATA_DIR / "train_prices.parquet")
    train_returns = pd.read_parquet(DATA_DIR / "train_returns.parquet")
    train_vol = pd.read_parquet(DATA_DIR / "train_volatility.parquet")
    train_zscore = pd.read_parquet(DATA_DIR / "train_zscore.parquet")
    train_sent = pd.read_parquet(DATA_DIR / "train_sentiment.parquet")
    return train_prices, train_returns, train_vol, train_zscore, train_sent


def _build_adjacency(train_returns: pd.DataFrame, threshold: float = 0.2) -> np.ndarray:
    corr = train_returns.corr().fillna(0.0).values.astype(np.float32)
    # Keep stronger links, preserve self loops
    adj = np.where(np.abs(corr) >= threshold, corr, 0.0).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    return adj


def train_all_profiles(cfg: TrainingConfig, logger: Optional[Callable[[str], None]] = None):
    logger = logger or (lambda _: None)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    risk_profiles = load_json(CONFIG_DIR / "risk_profiles.json", default={})
    train_prices, train_returns, train_vol, train_zscore, train_sent = _load_train_data()

    logger(f"Train prices shape: {train_prices.shape}")
    logger(f"Train returns shape: {train_returns.shape}")
    logger(f"Train volatility shape: {train_vol.shape}")
    logger(f"Train z-score shape: {train_zscore.shape}")
    logger(f"Train sentiment shape: {train_sent.shape}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {
        "timestamp": timestamp,
        "tag": cfg.model_tag,
        "hyperparameters": {
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "gamma": cfg.gamma,
            "ent_coef": cfg.ent_coef,
            "total_timesteps": cfg.total_timesteps,
            "use_seed": cfg.use_seed,
            "seed": 42 if cfg.use_seed else None,
            "use_graph_extractor": cfg.use_graph_extractor,
        },
        "profiles": {},
    }

    adjacency = _build_adjacency(train_returns) if cfg.use_graph_extractor else None

    for profile in RISK_PROFILE_NAMES:
        if profile not in risk_profiles:
            logger(f"Skipping {profile}: missing in risk_profiles.json")
            continue

        logger(f"Training started for {profile.title()}...")

        env_cfg = EnvConfig(
            initial_investment=cfg.initial_investment,
            transaction_cost=cfg.transaction_cost,
            settlement_day=cfg.settlement_day,
            window_size=30,
        )

        def make_env(profile_name=profile):
            return MultiAssetTradingEnv(
                prices=train_prices,
                returns=train_returns,
                volatility=train_vol,
                z_score=train_zscore,
                sentiment=train_sent,
                risk_profile=profile_name,
                risk_profiles=risk_profiles,
                cfg=env_cfg,
            )

        env = DummyVecEnv([make_env])

        policy_kwargs = {}
        if cfg.use_graph_extractor and adjacency is not None:
            policy_kwargs = {
                "features_extractor_class": HybridGCNExtractor,
                #"features_extractor_kwargs": {"adjacency": adjacency, "features_dim": 256},
                "features_extractor_kwargs": {
                    "n_assets": train_returns.shape[1],
                    "adjacency": adjacency,
                    "features_dim": 256,
},
                }
            
        print(policy_kwargs)

        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            ent_coef=cfg.ent_coef,
            policy_kwargs=policy_kwargs,
            seed=42 if cfg.use_seed else None,
            verbose=0,
        )

        model.learn(total_timesteps=int(cfg.total_timesteps))

        model_name = f"PPO_{timestamp}_{cfg.model_tag}_{profile}.zip"
        model_path = MODEL_DIR / model_name
        model.save(str(model_path))

        manifest["profiles"][profile] = {"model_path": str(model_path)}
        logger(f"Training completed for {profile.title()}.")

    manifest_path = MODEL_DIR / f"model_manifest_{timestamp}_{cfg.model_tag}.json"
    save_json(manifest_path, manifest)
    save_json(MODEL_DIR / "model_manifest.json", manifest)
    logger(f"Saved manifest: {manifest_path.name}")
