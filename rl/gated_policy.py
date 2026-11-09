# -*- coding: utf-8 -*-
"""门控双专家特征提取器 (v19)

架构:
  obs(57) ─┬─ flat_expert  (平地疾行 MLP)
           ├─ slope_expert (陡坡攀爬 MLP)
           └─ 解析门控: g = sigmoid((‖重力水平分量‖ − sin15°)/0.08)
              (重力水平分量 = obs[3:5] 的范数 = sin(机身坡度), 免训练、可解释)
  features = (1−g)·flat + g·slope   →  策略/价值头
"""
import math

import numpy as np
import torch
import torch.nn as nn
import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GatedDualExpert(BaseFeaturesExtractor):
    def __init__(self, observation_space, feat_dim=128):
        super().__init__(observation_space, features_dim=feat_dim)
        d = int(observation_space.shape[0])

        def expert():
            return nn.Sequential(
                nn.Linear(d, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, feat_dim), nn.ReLU())

        self.flat_expert = expert()
        self.slope_expert = expert()
        self._thresh = math.sin(math.radians(15.0))
        self._slope_scale = 0.08

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        f = self.flat_expert(obs)
        s = self.slope_expert(obs)
        g = torch.linalg.norm(obs[:, 3:5], dim=1, keepdim=True)
        g = torch.sigmoid((g - self._thresh) / self._slope_scale)
        return (1.0 - g) * f + g * s


POLICY_KWARGS = dict(
    features_extractor_class=GatedDualExpert,
    features_extractor_kwargs=dict(feat_dim=128),
    net_arch=dict(pi=[128], vf=[128]),
)
