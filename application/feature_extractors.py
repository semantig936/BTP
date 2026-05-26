from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HybridGCNExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space,
        n_assets: int,
        adjacency: np.ndarray,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        self.n_assets = n_assets

        self.adj = torch.tensor(
            adjacency,
            dtype=torch.float32
        )

        self.temporal_encoder = nn.LSTM(
            input_size=4,
            hidden_size=32,
            batch_first=True,
        )

        self.asset_mlp = nn.Sequential(
            nn.Linear(32, 32),
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

    def forward(self, obs):

        x = obs["features"]

        w = obs["weights"]

        B, W, N, F = x.shape

        x = x.permute(0, 2, 1, 3)

        x = x.reshape(B * N, W, F)

        _, (h_n, _) = self.temporal_encoder(x)

        h = h_n[-1]

        h = h.reshape(B, N, 32)

        ax = torch.einsum(
            "ij,bjk->bik",
            self.adj.to(h.device),
            h
        )

        h = self.asset_mlp(ax)

        h = h.reshape(B, -1)

        out = self.final(
            torch.cat([h, w], dim=1)
        )

        return out