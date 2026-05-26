from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvConfig:
    initial_investment: float
    transaction_cost: float
    settlement_day: int
    window_size: int = 30


class MultiAssetTradingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices,
        returns,
        volatility,
        z_score,
        sentiment,
        risk_profile: str,
        risk_profiles: Dict[str, Dict[str, float]],
        cfg: EnvConfig,
    ):
        super().__init__()

        self.prices = np.asarray(prices.values, dtype=np.float32)
        self.returns = np.asarray(returns.values, dtype=np.float32)
        self.volatility = np.asarray(volatility.values, dtype=np.float32)
        self.z_score = np.asarray(z_score.values, dtype=np.float32)
        self.sentiment = np.asarray(sentiment.values, dtype=np.float32)

        self.risk_profile = risk_profile
        self.risk_profiles = risk_profiles
        self.cfg = cfg

        self.window_size = int(cfg.window_size)
        self.num_assets = self.returns.shape[1]
        self.num_features = 4

        rp = self.risk_profiles[risk_profile]

        self.lambda_risk = float(rp["lambda_risk"])
        self.turnover_penalty = float(rp["turnover_penalty"])
        self.drawdown_penalty = float(rp["drawdown_penalty"])

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_assets,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict({
            "features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self.window_size,
                    self.num_assets,
                    self.num_features
                ),
                dtype=np.float32,
            ),
            "weights": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_assets,),
                dtype=np.float32,
            ),
        })

        self.pending_actions = deque(
            maxlen=max(1, self.cfg.settlement_day)
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = self.window_size

        self.weights = (
            np.ones(self.num_assets, dtype=np.float32)
            / self.num_assets
        )

        self.portfolio_value = float(
            self.cfg.initial_investment
        )

        self.max_value = float(
            self.cfg.initial_investment
        )

        self.transaction_cost_value = 0.0

        self.pending_actions.clear()

        return self._get_obs(), {}

    def _get_obs(self):

        start = self.t - self.window_size

        r = self.returns[start:self.t]
        v = self.volatility[start:self.t]
        z = self.z_score[start:self.t]

        s = self.sentiment[start:self.t]

        if s.ndim == 2:
            s = np.repeat(
                s[:, :, np.newaxis],
                self.num_assets,
                axis=2
            )
            s = np.transpose(s, (0, 2, 1))

        features = np.stack(
            [r, v, z, s.squeeze(-1)],
            axis=-1
        )

        features = np.nan_to_num(
            features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

        return {
            "features": features,
            "weights": self.weights.astype(np.float32),
        }

    def step(self, action):

        action = np.clip(action, 1e-6, 1.0)

        new_weights = action / np.sum(action)

        self.pending_actions.append(new_weights)

        if len(self.pending_actions) == self.cfg.settlement_day:
            weights = self.pending_actions.popleft()
        else:
            weights = self.weights

        ret_log = float(
            np.dot(weights, self.returns[self.t])
        )

        ret_simple = float(np.exp(ret_log) - 1.0)

        prev_value = self.portfolio_value

        self.portfolio_value *= (1.0 + ret_simple)

        turnover = float(
            np.sum(np.abs(weights - self.weights))
        )

        self.transaction_cost_value = (
            self.cfg.transaction_cost
            * turnover
            * prev_value
        )

        self.portfolio_value -= self.transaction_cost_value

        vol = float(
            np.mean(self.volatility[self.t])
        )

        self.max_value = max(
            self.max_value,
            self.portfolio_value
        )

        drawdown = (
            self.max_value - self.portfolio_value
        ) / max(self.max_value, 1e-12)

        sentiment_t = float(
            np.mean(self.sentiment[self.t])
        )

        reward = (
            ret_simple
            - self.lambda_risk * vol
            - self.drawdown_penalty * drawdown
            - self.turnover_penalty * turnover
            - self.cfg.transaction_cost * turnover
        )
        sentiment_t = float(
            np.mean(self.sentiment[self.t])
        )
        reward += 0.05 * sentiment_t

        self.weights = weights

        self.t += 1

        done = self.t >= len(self.returns) - 1

        return (
            self._get_obs(),
            float(reward),
            done,
            False,
            {
                "portfolio_value": float(self.portfolio_value),
                "return_simple": float(ret_simple),
                "return_log": float(ret_log),

                "transaction_cost": float(
                    self.transaction_cost_value
                ),

                "turnover": float(turnover),

                "drawdown": float(drawdown),

                "sentiment": float(sentiment_t),

                "weights": weights.copy(),
            },
        )