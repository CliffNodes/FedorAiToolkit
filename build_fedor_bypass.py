"""Build fedor_bypass.safetensors for Krea 2.

Surgical filter bypass: only cols 9 and 10 of txtfusion.projector (FB2 values).
All other columns stay at zero.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

TARGET_KEY = "diffusion_model.txtfusion.projector.diff"
OUT_PATH = Path(__file__).resolve().parent / "fedor_bypass.safetensors"

# 1-indexed cols 9 and 10 from community analysis (FilterBypass2 values)
COL_9_DELTA = -0.5117
COL_10_DELTA = -0.8906

METADATA = {
    "name": "fedor_bypass",
    "version": "v1 (cols 9+10 only, FB2 values)",
    "target_model": "Krea 2 (any variant)",
    "target_weight": "diffusion_model.txtfusion.projector.weight",
    "shape": "[1, 12]",
    "modifies_columns": "9 and 10 only",
    "leaves_untouched": "1-8, 11, 12",
    "recommended_strength": "3.0 to 5.0",
    "author": "Fedor / CliffNodes",
}


def build_tensor() -> torch.Tensor:
    t = torch.zeros((1, 12), dtype=torch.float32)
    t[0, 8] = COL_9_DELTA
    t[0, 9] = COL_10_DELTA
    return t


def main() -> int:
    tensor = build_tensor()
    meta_flat = {k: str(v) for k, v in METADATA.items()}
    save_file({TARGET_KEY: tensor}, str(OUT_PATH), metadata=meta_flat)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    print(f"values: {tensor.tolist()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
