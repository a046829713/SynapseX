import numpy as np


class DownsidePenaltyReward:
    """
    Ex-ante volatility-normalized asymmetric downside penalty reward (Sortino utility):
      - Profitable or flat step (z >= 0): reward = z
      - Losing step (z < 0): reward = z - 0.5 * risk_lambda * z^2

    Since z = (p * r_market - cost) / sigma_asset, z^2 is proportional to p^2,
    which quadratically penalizes large losing positions without penalizing winning holds.
    """

    def __init__(
        self,
        risk_lambda: float = 0.5,
        clip: float = 5.0,
        min_sigma: float = 1e-3,
    ):
        assert risk_lambda >= 0.0, "risk_lambda must be non-negative"
        assert clip > 0.0, "clip must be positive"
        assert min_sigma > 0.0, "min_sigma must be positive"
        self.risk_lambda = float(risk_lambda)
        self.clip = float(clip)
        self.min_sigma = float(min_sigma)

    def step(self, step_return: float, asset_sigma: float) -> float:
        safe_sigma = max(float(asset_sigma), self.min_sigma)
        z_return = float(step_return) / safe_sigma
        downside = min(0.0, z_return)
        reward = z_return - 0.5 * self.risk_lambda * (downside * downside)
        return float(np.clip(reward, -self.clip, self.clip))
