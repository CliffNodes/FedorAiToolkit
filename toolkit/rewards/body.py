"""Differentiable body-geometry reward for DRaFT-K training.

Pipeline:
  reference images --(no grad)--> SAM 3D Body -> MHR shape/scale params
                                  averaged into a reference body prototype
  generated images --(with grad)--> differentiable full-image crop
                                  -> SAM 3D Body forward -> shape/scale params
  loss tier 1 ("somax"): both parameter sets are mapped through SOMA-X
      (SOMALayer, identity_model_type="mhr") to canonical neutral-pose
      vertices, compared pose-independently (perceptual shape weighting).
  loss tier 2 ("shape_params"): direct saturated distance on the MHR shape
      parameters, pure torch (fallback when the Warp/vertex path misbehaves).

Pose parameters are discarded by design: only body identity (build /
proportions / height) is rewarded, never the pose in the image.

SAM 3D Body code is vendored via a git clone at <toolkit>/repositories/
sam-3d-body (no pip package exists). Its checkpoints are GATED on Hugging
Face -- request access to facebook/sam-3d-body-vith (or -dinov3) and run
`hf auth login`, after which the checkpoint auto-downloads on first use.
When the checkpoint (or a gradient path) is unavailable the reward degrades
gracefully: it returns a zero-gradient penalty term so face-only training
continues, and warns once.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOOLKIT_ROOT = Path(__file__).resolve().parents[2]
SAM3D_REPO_DIR = TOOLKIT_ROOT / "repositories" / "sam-3d-body"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def _ensure_sam3d_importable():
    if str(SAM3D_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(SAM3D_REPO_DIR))
    try:
        import sam_3d_body  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"sam_3d_body not importable. Expected a clone of "
            f"https://github.com/facebookresearch/sam-3d-body at {SAM3D_REPO_DIR}"
        ) from exc


def _resolve_checkpoint(
    checkpoint_path: Optional[str], hf_repo: str
) -> tuple[str, str]:
    """Resolve (model.ckpt, mhr_model.pt) locally or via the HF hub (gated)."""
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        mhr = ckpt.parent / "assets" / "mhr_model.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        if not mhr.is_file():
            raise FileNotFoundError(mhr)
        return str(ckpt), str(mhr)

    local = TOOLKIT_ROOT / "models" / hf_repo.split("/")[-1]
    if (local / "model.ckpt").is_file() and (local / "assets" / "mhr_model.pt").is_file():
        return str(local / "model.ckpt"), str(local / "assets" / "mhr_model.pt")

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=hf_repo, token=os.getenv("HF_TOKEN", None))
    return (
        os.path.join(local_dir, "model.ckpt"),
        os.path.join(local_dir, "assets", "mhr_model.pt"),
    )


def _reference_paths(reference_images) -> list[Path]:
    if reference_images is None:
        raise ValueError("reference_images is required")
    if isinstance(reference_images, (str, Path)):
        paths = [Path(reference_images)]
    else:
        paths = [Path(p) for p in reference_images]
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(
                sorted(
                    p
                    for p in path.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                )
            )
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(path)
    if not out:
        raise ValueError("reference_images did not resolve to any image files")
    return out


def _center_crop_grid(
    image: torch.Tensor, out_size: int, padding: float = 1.25
) -> torch.Tensor:
    """Differentiable equivalent of GetBBoxCenterScale + TopdownAffine on the
    full-image bbox: center-crop with the estimator's 1.25 padding factor,
    aspect preserved, resampled to (out_size, out_size) via grid_sample."""
    b, _, h, w = image.shape
    device = image.device
    side = max(h, w) * padding
    ys = torch.linspace(-side / 2, side / 2, out_size, device=device) + h / 2
    xs = torch.linspace(-side / 2, side / 2, out_size, device=device) + w / 2
    gy = (ys + 0.5) * (2.0 / h) - 1.0
    gx = (xs + 0.5) * (2.0 / w) - 1.0
    grid = torch.stack(
        (gx.view(1, -1).expand(out_size, -1), gy.view(-1, 1).expand(-1, out_size)),
        dim=-1,
    ).unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(
        image.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


class BodyGeometryReward(nn.Module):
    def __init__(
        self,
        reference_images=None,
        sam3d_checkpoint_path: Optional[str] = None,
        sam3d_hf_repo: str = "facebook/sam-3d-body-vith",
        loss_tier: str = "somax",  # "somax" (tier 1) or "shape_params" (tier 2)
        target_distance: float = 0.010,  # meters of mean canonical-vertex error
        saturation_temperature: float = 0.005,
        shape_target_distance: float = 0.35,  # tier-2 normalized param distance
        shape_saturation_temperature: float = 0.10,
        no_person_penalty: float = 0.25,
        soma_lod: str = "low",
        device: str | torch.device | None = None,
    ):
        super().__init__()
        if loss_tier not in {"somax", "shape_params"}:
            raise ValueError("loss_tier must be 'somax' or 'shape_params'")
        self.loss_tier = loss_tier
        self.target_distance = float(target_distance)
        self.saturation_temperature = float(saturation_temperature)
        self.shape_target_distance = float(shape_target_distance)
        self.shape_saturation_temperature = float(shape_saturation_temperature)
        self.no_person_penalty = float(no_person_penalty)
        self.soma_lod = soma_lod
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._warned_grad_fallback = False
        self._soma_layer = None
        self._soma_failed = False

        _ensure_sam3d_importable()
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        ckpt_path, mhr_path = _resolve_checkpoint(sam3d_checkpoint_path, sam3d_hf_repo)
        self.model, self.cfg = load_sam_3d_body(
            checkpoint_path=ckpt_path, device=str(self.device), mhr_path=mhr_path
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # detector-free estimator: with no detector, process_one_image uses the
        # full image as the person box, which fits single-subject datasets.
        self.estimator = SAM3DBodyEstimator(self.model, self.cfg)
        image_size = self.cfg.MODEL.IMAGE_SIZE
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]
        self.input_size = int(image_size)

        proto_shape, proto_scale = self._build_reference_prototype(
            _reference_paths(reference_images)
        )
        self.register_buffer("reference_shape", proto_shape)
        self.register_buffer("reference_scale", proto_scale)
        self.register_buffer(
            "reference_vertices", self._canonical_vertices_nograd(proto_shape, proto_scale)
        )

    # ------------------------------------------------------------------
    # Reference prototype (no grad, via the stock estimator)
    # ------------------------------------------------------------------
    def _build_reference_prototype(self, paths: list[Path]):
        import cv2

        shapes, scales = [], []
        self.valid_reference_images: list[str] = []
        self.skipped_reference_images: list[str] = []
        with torch.no_grad():
            for path in paths:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    self.skipped_reference_images.append(str(path))
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                try:
                    outs = self.estimator.process_one_image(rgb, inference_type="body")
                except Exception as exc:  # noqa: BLE001 - skip unreadable refs
                    warnings.warn(f"body reference {path} failed: {exc}")
                    self.skipped_reference_images.append(str(path))
                    continue
                if not outs:
                    self.skipped_reference_images.append(str(path))
                    continue
                out = outs[0]
                shapes.append(torch.as_tensor(np.asarray(out["shape_params"]), dtype=torch.float32))
                scales.append(torch.as_tensor(np.asarray(out["scale_params"]), dtype=torch.float32))
                self.valid_reference_images.append(str(path))
        if not shapes:
            raise RuntimeError(
                "no bodies regressed from reference images; check the images "
                "contain a visible person"
            )
        proto_shape = torch.stack(shapes).mean(dim=0, keepdim=True).to(self.device)
        proto_scale = torch.stack(scales).mean(dim=0, keepdim=True).to(self.device)
        return proto_shape.reshape(1, -1), proto_scale.reshape(1, -1)

    # ------------------------------------------------------------------
    # SOMA-X canonical neutral-pose vertices
    # ------------------------------------------------------------------
    def _get_soma_layer(self):
        if self._soma_layer is None and not self._soma_failed:
            try:
                from soma import SOMALayer

                self._soma_layer = SOMALayer(
                    device=self.device,
                    identity_model_type="mhr",
                    lod=self.soma_lod,
                    # the nvidia/soma-x HF snapshot ships without the
                    # procedural transform definition json; the legacy
                    # 78-joint rig is sufficient for neutral-pose identity
                    # comparison
                    enable_procedural_transforms=False,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"SOMALayer unavailable ({exc}); body reward falls back to "
                    "tier-2 shape-parameter distance"
                )
                self._soma_failed = True
        return self._soma_layer

    def _native_scales(self, scale_params: torch.Tensor) -> torch.Tensor:
        """Expand SAM 3D Body's 28 scale PCA coefficients to the native
        68-dim MHR scale vector SOMA-X expects (same op as mhr_forward:
        ``scale_mean + coeffs @ scale_comps``, differentiable)."""
        head = self.model.head_pose
        return head.scale_mean[None, :].to(scale_params) + scale_params @ head.scale_comps.to(scale_params)

    def _canonical_vertices(
        self, shape_params: torch.Tensor, scale_params: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Map MHR identity to canonical neutral-pose vertices (B, V, 3)."""
        layer = self._get_soma_layer()
        if layer is None:
            return None
        scale_params = self._native_scales(scale_params)
        b = shape_params.shape[0]
        # 77 user-facing joints (Root excluded), zero axis-angle = neutral pose
        poses = torch.zeros(b, 77, 3, device=self.device, dtype=torch.float32)
        out = layer(
            poses=poses,
            identity_coeffs=shape_params.to(self.device, torch.float32),
            scale_params=scale_params.to(self.device, torch.float32),
            apply_correctives=False,
        )
        verts = getattr(out, "vertices", None)
        if verts is None and isinstance(out, dict):
            verts = out.get("vertices")
        return verts

    def _canonical_vertices_nograd(self, shape_params, scale_params):
        with torch.no_grad():
            verts = self._canonical_vertices(shape_params, scale_params)
        if verts is None:
            return torch.zeros(1, 1, 3)
        return verts.detach()

    # ------------------------------------------------------------------
    # Differentiable regression on generated images
    # ------------------------------------------------------------------
    def _regress_differentiable(self, image: torch.Tensor):
        """Regress MHR shape/scale params with gradient wrt ``image``.

        ``image`` is (1, 3, H, W) in [-1, 1]. Bypasses the no-grad estimator
        wrapper: builds the same batch dict prepare_batch would, but with a
        differentiable full-image crop, and runs the model's body-only
        inference under enable_grad.
        """
        from sam_3d_body.data.utils.prepare_batch import NoCollate

        _, _, h, w = image.shape
        img01 = (image.float() + 1.0) / 2.0
        crop = _center_crop_grid(img01, self.input_size)  # (1, 3, S, S)

        side = max(h, w) * 1.25
        # full-image -> crop affine (what TopdownAffine's warp_mat encodes):
        # x' = s * (x - (cx - side/2)), s = input_size / side
        s = self.input_size / side
        affine = torch.tensor(
            [[[[s, 0.0, -s * (w / 2.0 - side / 2.0)],
               [0.0, s, -s * (h / 2.0 - side / 2.0)]]]],
            dtype=torch.float32,
        )
        batch = {
            "affine_trans": affine,  # (1, N=1, 2, 3)
            "img": crop.unsqueeze(0),  # (1, N=1, 3, S, S)
            "img_size": torch.tensor([[[self.input_size, self.input_size]]], dtype=torch.float32),
            "ori_img_size": torch.tensor([[[w, h]]], dtype=torch.float32),
            "bbox_center": torch.tensor([[[w / 2.0, h / 2.0]]], dtype=torch.float32),
            "bbox_scale": torch.tensor([[[side, side]]], dtype=torch.float32),
            "bbox": torch.tensor([[[0.0, 0.0, float(w), float(h)]]], dtype=torch.float32),
            "mask": torch.zeros(1, 1, 1, self.input_size, self.input_size, dtype=torch.float32),
            "mask_score": torch.zeros(1, 1, dtype=torch.float32),
            "person_valid": torch.ones(1, 1),
            "cam_int": torch.tensor(
                [[[(h**2 + w**2) ** 0.5, 0, w / 2.0],
                  [0, (h**2 + w**2) ** 0.5, h / 2.0],
                  [0, 0, 1]]],
                dtype=torch.float32,
            ),
        }
        batch = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        img_np = (
            (img01[0].detach().clamp(0, 1) * 255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        batch["img_ori"] = [NoCollate(img_np)]

        # note: bf16 autocast is NOT used here -- the torchscripted MHR module
        # inside the model fails under autocast, silently degrading the reward
        # to face-only. fp32 fits on 32 GB when the GPU is otherwise idle.
        with torch.enable_grad():
            self.model._initialize_batch(batch)
            pose_output = self.model.run_inference(
                img_np, batch, inference_type="body"
            )
        out = pose_output["mhr"]
        shape = out["shape"]
        scale = out["scale"]
        if isinstance(shape, np.ndarray):
            raise RuntimeError("SAM 3D Body returned numpy; gradient path broken")
        return shape.reshape(1, -1), scale.reshape(1, -1)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _saturated_reward(
        self, distance: torch.Tensor, target: float, temperature: float
    ) -> torch.Tensor:
        return -temperature * F.softplus((distance - target) / temperature)

    def _identity_distance(
        self, shape: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_tier == "somax":
            verts = self._canonical_vertices(shape, scale)
            if verts is not None and self.reference_vertices.shape[1] > 1:
                ref = self.reference_vertices.to(verts)
                # mean per-vertex L2 in meters between canonical neutral meshes
                d = torch.linalg.norm(verts - ref, dim=-1).mean()
                return self._saturated_reward(
                    d, self.target_distance, self.saturation_temperature
                )
            # SOMA path unavailable -> fall through to tier 2
        ref_shape = self.reference_shape.to(shape)
        d = torch.linalg.norm(shape - ref_shape, dim=-1).mean() / max(
            float(torch.linalg.norm(ref_shape)), 1e-6
        )
        return self._saturated_reward(
            d, self.shape_target_distance, self.shape_saturation_temperature
        )

    def forward(self, image: torch.Tensor, prompt: str | None = None, **kwargs):
        del prompt, kwargs
        if image.dim() == 3:
            image = image.unsqueeze(0)
        rewards = []
        for img in image:
            img = img.unsqueeze(0)
            try:
                shape, scale = self._regress_differentiable(img)
                value = self._identity_distance(shape, scale)
                if not value.requires_grad and img.requires_grad:
                    raise RuntimeError("body reward lost the gradient connection")
                rewards.append(value.reshape(()))
            except Exception as exc:  # noqa: BLE001 - degrade to face-only
                if not self._warned_grad_fallback:
                    warnings.warn(
                        f"body reward gradient path failed ({exc}); returning a "
                        "graph-preserving penalty so training continues face-only"
                    )
                    self._warned_grad_fallback = True
                rewards.append(img.mean() * 0.0 - self.no_person_penalty)
        return torch.stack(rewards)
