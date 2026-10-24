"""Pure visualization helpers for the PatchCore demo."""

from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_anomaly_map(image: Image.Image, anomaly_map: np.ndarray) -> Image.Image:
    """Overlay a heatmap while respecting the model's actual map dimensions."""

    from matplotlib import colormaps

    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    if anomaly_map.ndim != 2 or 0 in anomaly_map.shape:
        raise ValueError("anomaly_map must be a non-empty 2-D array.")

    # IM224 models produce 224x224 maps, while IM320 models produce 320x320
    # maps. The output size must come from the model map, not a hard-coded 224.
    map_height, map_width = anomaly_map.shape
    base = np.asarray(
        image.convert("RGB").resize(
            (map_width, map_height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    heat = colormaps["magma"](np.clip(anomaly_map, 0.0, 1.0))[..., :3]
    blended = np.clip(0.58 * base + 0.42 * heat, 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8))

