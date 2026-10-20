"""Fixed-weight late fusion using the original benchmark normalization."""

import numpy as np


def normalize_weights(weights, n_branches, option="--fusion_weight"):
    if n_branches < 1:
        raise ValueError("At least one backbone is required.")
    values = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        values = np.ones(n_branches, dtype=np.float64)
    if values.shape != (n_branches,):
        raise ValueError(f"Provide one {option} per backbone, in backbone order.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Fusion weights must be finite and non-negative.")
    if values.max() == 0:
        raise ValueError("At least one fusion weight must be positive.")
    values = values / values.max()
    return values / values.sum()


def resolve_task_weights(n_branches, shared=(), image=(), pixel=()):
    """Task-specific options override the shared fallback independently."""
    common = normalize_weights(shared, n_branches)
    image_weights = normalize_weights(
        image if len(image) else common, n_branches, "--image_fusion_weight"
    )
    pixel_weights = normalize_weights(
        pixel if len(pixel) else common, n_branches, "--pixel_fusion_weight"
    )
    return image_weights, pixel_weights


def restore_task_weights(config, n_branches):
    """Read v2 task weights, v1 shared weights, or an old unweighted ensemble."""
    if config is None:
        return resolve_task_weights(n_branches)
    if config.get("normalization") != "evaluation_set_minmax":
        raise ValueError("Unsupported saved fusion normalization.")
    version = config.get("version", 1)
    if version == 1:
        return resolve_task_weights(n_branches, shared=config["weights"])
    if version == 2:
        image, pixel = config.get("image_weights"), config.get("pixel_weights")
        if image is None or pixel is None or len(image) == 0 or len(pixel) == 0:
            raise ValueError("Version 2 fusion config requires both task weight lists.")
        return resolve_task_weights(n_branches, image=image, pixel=pixel)
    raise ValueError(f"Unsupported fusion config version: {version}")


def fuse_predictions(predictions, weights):
    """Normalize each branch over the WHOLE evaluation set, then weight it.

    Input: [branches, images] or [branches, images, height, width].
    Image scores and maps must be passed separately. Constant branches map
    to zero instead of producing NaNs. Returns output and normalization bounds.
    No test labels are used, but the bounds depend on the evaluation population.
    """
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim not in (2, 4) or any(size == 0 for size in values.shape):
        raise ValueError("Expected non-empty [K,N] scores or [K,N,H,W] maps.")
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinity.")
    weights = normalize_weights(weights, len(values))
    flat = values.reshape(len(values), -1)
    lower, upper = flat.min(axis=1), flat.max(axis=1)
    broadcast = (len(values),) + (1,) * (values.ndim - 1)
    span = (upper - lower).reshape(broadcast)
    normalized = np.divide(
        values - lower.reshape(broadcast),
        span,
        out=np.zeros_like(values),
        where=span != 0,
    )
    fused = np.sum(normalized * weights.reshape(broadcast), axis=0)
    return fused, {"min": lower.tolist(), "max": upper.tolist()}
