"""Weighted face + body reward with graceful degradation."""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn


class CombinedReward(nn.Module):
    """``w_face * face + w_body * body``.

    Either sub-reward may be None (weight 0 or failed setup); at least one
    must be present. ``pairwise_reward`` (expression-diversity regularizer)
    is forwarded to the face reward when available.
    """

    def __init__(
        self,
        face_reward: Optional[nn.Module] = None,
        body_reward: Optional[nn.Module] = None,
        face_weight: float = 1.0,
        body_weight: float = 0.5,
    ):
        super().__init__()
        if face_reward is None and body_reward is None:
            raise ValueError("at least one of face_reward / body_reward is required")
        self.face_reward = face_reward
        self.body_reward = body_reward
        self.face_weight = float(face_weight)
        self.body_weight = float(body_weight)
        self.last_components: dict[str, float] = {}

    def forward(self, image: torch.Tensor, prompt: str | None = None, **kwargs):
        total = None
        self.last_components = {}
        if self.face_reward is not None and self.face_weight != 0.0:
            face = self.face_reward(image, prompt, **kwargs)
            self.last_components["face"] = float(face.detach().mean())
            total = self.face_weight * face
        if self.body_reward is not None and self.body_weight != 0.0:
            body = self.body_reward(image, prompt, **kwargs)
            self.last_components["body"] = float(body.detach().mean())
            total = body * self.body_weight if total is None else total + self.body_weight * body
        return total

    def pairwise_reward(self, first, second, prompt=None, **kwargs):
        if self.face_reward is not None and hasattr(self.face_reward, "pairwise_reward"):
            return self.face_reward.pairwise_reward(first, second, prompt, **kwargs)
        return (first.sum() + second.sum()) * 0.0


def build_reward_from_config(cfg: dict, device=None) -> CombinedReward:
    """Build the combined reward from the trainer's ``draft.reward`` config dict.

    Expected keys (all optional except reference_images):
      reference_images: path (or list of paths/dirs) used by BOTH rewards
      face_weight / body_weight: component weights (face 1.0, body 0.5)
      face: kwargs forwarded to FaceSimilarityReward
      body: kwargs forwarded to BodyGeometryReward (checkpoint paths etc.)
    Body reward setup failures are non-fatal (warned + skipped) because the
    SAM 3D Body checkpoint is gated on Hugging Face.
    """
    cfg = dict(cfg or {})
    reference_images = cfg.get("reference_images", None)
    if reference_images is None:
        raise ValueError("draft.reward.reference_images is required")
    face_weight = float(cfg.get("face_weight", 1.0))
    body_weight = float(cfg.get("body_weight", 0.5))

    face = None
    if face_weight != 0.0:
        from .face import FaceSimilarityReward

        face_kwargs = dict(cfg.get("face", {}) or {})
        face_kwargs.setdefault("reference_images", reference_images)
        if device is not None:
            face_kwargs.setdefault("device", device)
        face = FaceSimilarityReward(**face_kwargs)
        if face.skipped_reference_images:
            print(
                f"face reward: skipped {len(face.skipped_reference_images)} "
                f"reference image(s) without a usable face"
            )
        print(
            f"face reward ready: {len(face.valid_reference_images)} reference "
            f"image(s), {face.reference_embeddings.shape[0]} embedding(s)"
        )

    body = None
    if body_weight != 0.0:
        try:
            from .body import BodyGeometryReward

            body_kwargs = dict(cfg.get("body", {}) or {})
            body_kwargs.setdefault("reference_images", reference_images)
            if device is not None:
                body_kwargs.setdefault("device", device)
            body = BodyGeometryReward(**body_kwargs)
            print(
                f"body reward ready: {len(body.valid_reference_images)} reference "
                f"image(s), tier={body.loss_tier}"
            )
        except Exception as exc:  # noqa: BLE001 - gated checkpoint / optional dep
            warnings.warn(
                f"body reward unavailable ({exc}); continuing with face reward "
                "only. To enable it: request access to facebook/sam-3d-body-vith "
                "on Hugging Face, run `hf auth login` in the FedorAiToolkit venv, "
                "and re-run."
            )
            body = None
            body_weight = 0.0

    return CombinedReward(
        face_reward=face,
        body_reward=body,
        face_weight=face_weight,
        body_weight=body_weight,
    )
