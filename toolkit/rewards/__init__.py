"""Differentiable reward functions for DRaFT-K reward training (FedorAiToolkit).

- face: ArcFace (antelopev2) identity similarity, vendored from
  https://github.com/KONAKONA666/krea-2 (Apache-2.0).
- body: SAM 3D Body -> MHR shape params -> SOMA-X canonical body compare.
- combined: weighted face + body reward with graceful degradation.
"""

from .combined import CombinedReward, build_reward_from_config

__all__ = ["CombinedReward", "build_reward_from_config"]
