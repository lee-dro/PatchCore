"""Fixed and calibration-based task-specific fusion utilities."""

import numpy as np


def shrink_weights(base_weights, target_weights, shrinkage, epsilon=1e-12):
    """Move normalized target weights toward normalized base weights."""
    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1.")
    base_weights = np.asarray(base_weights, dtype=np.float64)
    target_weights = np.asarray(target_weights, dtype=np.float64)
    if base_weights.shape != target_weights.shape or base_weights.ndim != 1:
        raise ValueError(
            "base_weights and target_weights must be 1-D with equal shape."
        )
    if base_weights.size == 0:
        raise ValueError("At least one fusion weight is required.")
    if not np.isfinite(base_weights).all() or not np.isfinite(target_weights).all():
        raise ValueError("Fusion weights must be finite.")
    if (base_weights < 0).any() or (target_weights < 0).any():
        raise ValueError("Fusion weights must be non-negative.")
    if (
        abs(base_weights.sum() - 1.0) > epsilon
        or abs(target_weights.sum() - 1.0) > epsilon
    ):
        raise ValueError("base_weights and target_weights must each sum to 1.")
    return (1.0 - float(shrinkage)) * base_weights + float(shrinkage) * target_weights


def _bootstrap_median_lower_bound(
    values, confidence_level=0.9, bootstrap_samples=2000, random_state=0
):
    """Estimate a deterministic lower confidence bound from group summaries."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("At least one group margin is required.")
    if values.size == 1:
        return float(values[0])
    rng = np.random.default_rng(random_state)
    samples = rng.choice(
        values, size=(int(bootstrap_samples), values.size), replace=True
    )
    medians = np.median(samples, axis=1)
    alpha = 1.0 - float(confidence_level)
    return float(np.quantile(medians, alpha / 2.0))


def sensitivity_gate(
    diagnostics,
    min_group_consistency=0.8,
    max_layer2_weight=0.8,
    min_layer3_stability=0.9,
    mode="legacy",
    min_group_margin=0.5,
    confidence_level=0.9,
    bootstrap_samples=2000,
):
    """Decide whether calibration supports enabling conservative shrinkage.

    The gate only reads calibration diagnostics. It is deliberately a safety
    rule rather than a test-metric optimizer: require agreement across
    synthetic defect groups, cap layer2 dominance, and reject a markedly
    unstable layer3 branch.

    ``legacy`` preserves the original majority-only rule. ``conservative``
    additionally requires a minimum oriented sensitivity margin for most
    defect groups and a non-negative bootstrap lower bound for the median
    group margin. The confidence interval is based on group summaries saved
    by the calibration stage; it is not a substitute for sample-level
    uncertainty when those raw outputs are unavailable.
    """
    if not 0.0 <= float(min_group_consistency) <= 1.0:
        raise ValueError("min_group_consistency must be between 0 and 1.")
    if not 0.0 <= float(max_layer2_weight) <= 1.0:
        raise ValueError("max_layer2_weight must be between 0 and 1.")
    if not 0.0 <= float(min_layer3_stability) <= 1.0:
        raise ValueError("min_layer3_stability must be between 0 and 1.")
    if mode not in ("legacy", "conservative"):
        raise ValueError("Gate mode must be either legacy or conservative.")
    if float(min_group_margin) < 0.0:
        raise ValueError("min_group_margin must be non-negative.")
    if not 0.5 <= float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be in [0.5, 1.0).")
    if int(bootstrap_samples) < 100:
        raise ValueError("bootstrap_samples must be at least 100.")
    pixel = diagnostics["pixel"]
    group_values = np.asarray(list(pixel["sensitivity_by_group"].values()), dtype=float)
    if group_values.ndim != 2 or group_values.shape[1] != 2 or not len(group_values):
        raise ValueError("Gate diagnostics must contain two branch values per group.")
    if not np.isfinite(group_values).all():
        raise ValueError("Gate group responses must be finite.")
    preferences = np.argmax(group_values, axis=1)
    group_consistency = float(
        max(np.sum(preferences == 0), np.sum(preferences == 1)) / len(preferences)
    )
    group_margins = group_values[:, 0] - group_values[:, 1]
    majority_branch = int(
        0 if np.sum(preferences == 0) >= np.sum(preferences == 1) else 1
    )
    oriented_margins = group_margins if majority_branch == 0 else -group_margins
    margin_consistency = float(np.mean(oriented_margins >= float(min_group_margin)))
    margin_lower_confidence = _bootstrap_median_lower_bound(
        oriented_margins,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
    )
    weights = np.asarray(pixel["weights"], dtype=float)
    relative_stability = np.asarray(pixel["relative_stability"], dtype=float)
    if weights.shape != (2,) or relative_stability.shape != (2,):
        raise ValueError("Gate requires exactly two branch weights and stabilities.")
    normalize_weights(weights, 2)
    if not np.isfinite(relative_stability).all() or (relative_stability < 0).any():
        raise ValueError("Gate stabilities must be finite and non-negative.")
    checks = {
        "group_consistency": group_consistency,
        "group_consistency_ok": group_consistency >= min_group_consistency,
        "layer2_weight": float(weights[0]),
        "layer2_weight_ok": float(weights[0]) <= max_layer2_weight,
        "layer3_relative_stability": float(relative_stability[1]),
        "layer3_stability_ok": float(relative_stability[1]) >= min_layer3_stability,
    }
    checks.update(
        {
            "mode": mode,
            "majority_branch": majority_branch,
            "group_margins": group_margins.tolist(),
            "oriented_group_margins": oriented_margins.tolist(),
            "margin_consistency": margin_consistency,
            "margin_consistency_ok": margin_consistency >= float(min_group_consistency),
            "margin_lower_confidence": margin_lower_confidence,
            "margin_confidence_ok": margin_lower_confidence >= 0.0,
        }
    )
    basic_checks = all(
        checks[key]
        for key in ("group_consistency_ok", "layer2_weight_ok", "layer3_stability_ok")
    )
    checks["use_shrinkage"] = basic_checks and (
        mode == "legacy"
        or (checks["margin_consistency_ok"] and checks["margin_confidence_ok"])
    )
    checks["thresholds"] = {
        "min_group_consistency": float(min_group_consistency),
        "max_layer2_weight": float(max_layer2_weight),
        "min_layer3_stability": float(min_layer3_stability),
        "min_group_margin": float(min_group_margin),
        "confidence_level": float(confidence_level),
        "bootstrap_samples": int(bootstrap_samples),
    }
    return checks


def _robust_scale(values, epsilon=1e-12):
    """Return a robust scale estimate, with a standard-deviation fallback."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826
    if mad > epsilon:
        return float(mad)
    std = np.std(values)
    return float(max(std, epsilon))


def _normalize_positive(values, fallback_size, epsilon=1e-12):
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values) & (values > epsilon), values, 0.0)
    if values.sum() <= epsilon:
        return np.full(fallback_size, 1.0 / fallback_size, dtype=np.float64)
    return values / values.sum()


def _adaptive_utility(stability, sensitivity, epsilon=1e-12):
    """Combine relative stability and synthetic-defect sensitivity.

    Stability is normalized so that it only expresses a branch's relative
    behavior among branches. Sensitivity is a robust paired increase between
    a synthetic image and its clean source. The geometric mean keeps either
    a noisy or an insensitive branch from receiving a large weight.
    """
    stability = np.asarray(stability, dtype=np.float64)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    stability = stability / max(np.max(stability), epsilon)
    sensitivity = np.maximum(sensitivity, 0.0)
    utility = np.sqrt(np.maximum(stability, 0.0) * sensitivity)
    return utility


def _strategy_utility(
    stability, sensitivity, strategy, stability_penalty=0.0, epsilon=1e-12
):
    if strategy == "stability":
        return np.asarray(stability, dtype=np.float64)
    if strategy == "sensitivity":
        return np.maximum(np.asarray(sensitivity, dtype=np.float64), 0.0)
    if strategy == "combined":
        return _adaptive_utility(stability, sensitivity, epsilon)
    if strategy == "sensitivity_aux":
        volatility = 1.0 / np.maximum(np.asarray(stability, dtype=np.float64), epsilon)
        relative_volatility = volatility / max(np.median(volatility), epsilon)
        instability = np.maximum(relative_volatility - 1.0, 0.0)
        sensitivity = np.maximum(np.asarray(sensitivity, dtype=np.float64), 0.0)
        return sensitivity / (1.0 + stability_penalty * instability)
    raise ValueError(
        "Adaptive strategy must be one of: stability, sensitivity, combined, "
        "sensitivity_aux."
    )


def adaptive_task_weights(
    normal_scores,
    synthetic_scores,
    normal_maps,
    synthetic_maps,
    synthetic_masks,
    synthetic_groups=None,
    strategy="combined",
    stability_penalty=0.0,
    epsilon=1e-12,
):
    """Estimate image/pixel fusion weights from a clean calibration split.

    The inputs contain outputs from each independent PatchCore branch. The
    clean and synthetic arrays must be paired by image. Only calibration data
    are used; test labels are not involved.

    The estimate is intentionally simple and inspectable:

    * stability: inverse robust dispersion on clean calibration outputs;
    * sensitivity: robust paired increase caused by a synthetic defect;
    * utility: geometric mean of relative stability and sensitivity.

    Returns normalized image weights, pixel weights, and a JSON-serializable
    diagnostics dictionary.
    """
    scores_clean = np.asarray(normal_scores, dtype=np.float64)
    scores_synthetic = np.asarray(synthetic_scores, dtype=np.float64)
    maps_clean = np.asarray(normal_maps, dtype=np.float64)
    maps_synthetic = np.asarray(synthetic_maps, dtype=np.float64)
    masks = np.asarray(synthetic_masks, dtype=np.float64)
    if synthetic_groups is None:
        synthetic_groups = ["all"] * len(masks)
    synthetic_groups = np.asarray(synthetic_groups, dtype=object)

    if scores_clean.ndim != 2 or scores_synthetic.shape != scores_clean.shape:
        raise ValueError(
            "Calibration image scores must have shape [branches, samples]."
        )
    if maps_clean.ndim != 4 or maps_synthetic.shape != maps_clean.shape:
        raise ValueError(
            "Calibration anomaly maps must have shape [branches, samples, height, width]."
        )
    if maps_clean.shape[:2] != scores_clean.shape:
        raise ValueError(
            "Calibration scores and maps must have matching branches/samples."
        )
    if masks.shape != maps_clean.shape[1:]:
        raise ValueError("Synthetic masks must have shape [samples, height, width].")
    if synthetic_groups.shape != (maps_clean.shape[1],):
        raise ValueError("Synthetic groups must have one label per calibration sample.")
    if not np.isfinite(scores_clean).all() or not np.isfinite(scores_synthetic).all():
        raise ValueError("Calibration image scores contain NaN or infinity.")
    if not np.isfinite(maps_clean).all() or not np.isfinite(maps_synthetic).all():
        raise ValueError("Calibration anomaly maps contain NaN or infinity.")
    if not np.isfinite(masks).all() or not np.all((masks == 0) | (masks == 1)):
        raise ValueError("Synthetic masks must be finite binary masks.")
    if not (masks.reshape(len(masks), -1).sum(axis=1) > 0).all():
        raise ValueError("Each synthetic mask must contain defect pixels.")
    if not np.isfinite(stability_penalty) or stability_penalty < 0:
        raise ValueError("stability_penalty must be finite and non-negative.")

    branch_count = scores_clean.shape[0]
    if branch_count < 1 or scores_clean.shape[1] < 2:
        raise ValueError(
            "At least one branch and two calibration samples are required."
        )

    score_scales = np.asarray(
        [_robust_scale(row, epsilon) for row in scores_clean], dtype=np.float64
    )
    score_stability = 1.0 / (score_scales + epsilon)
    score_deltas = (scores_synthetic - scores_clean) / (score_scales[:, None] + epsilon)
    group_names = list(dict.fromkeys(str(group) for group in synthetic_groups))
    group_image_sensitivity = {}
    grouped_image_values = []
    for group_name in group_names:
        selected = np.asarray(
            [str(group) == group_name for group in synthetic_groups], dtype=bool
        )
        group_values = np.median(score_deltas[:, selected], axis=1)
        group_image_sensitivity[group_name] = group_values.tolist()
        grouped_image_values.append(group_values)
    score_sensitivity = np.median(np.asarray(grouped_image_values), axis=0)
    image_utility = _strategy_utility(
        score_stability,
        score_sensitivity,
        strategy,
        stability_penalty,
        epsilon,
    )
    image_weights = _normalize_positive(image_utility, branch_count, epsilon)

    clean_high = np.quantile(
        maps_clean.reshape(branch_count, maps_clean.shape[1], -1), 0.99, axis=2
    )
    map_scales = np.asarray(
        [_robust_scale(row, epsilon) for row in clean_high], dtype=np.float64
    )
    map_stability = 1.0 / (map_scales + epsilon)
    mask = masks > 0.5
    pixel_sample_deltas = []
    for branch in range(branch_count):
        clean_inside = []
        synthetic_inside = []
        for sample in range(maps_clean.shape[1]):
            selected = mask[sample]
            if not np.any(selected):
                continue
            clean_inside.append(np.mean(maps_clean[branch, sample][selected]))
            synthetic_inside.append(np.mean(maps_synthetic[branch, sample][selected]))
        if not clean_inside:
            pixel_sample_deltas.append(np.zeros(maps_clean.shape[1]))
        else:
            pixel_sample_deltas.append(
                (np.asarray(synthetic_inside) - np.asarray(clean_inside))
                / (map_scales[branch] + epsilon)
            )
    pixel_sample_deltas = np.asarray(pixel_sample_deltas, dtype=np.float64)
    group_pixel_sensitivity = {}
    grouped_pixel_values = []
    for group_name in group_names:
        selected = np.asarray(
            [str(group) == group_name for group in synthetic_groups], dtype=bool
        )
        group_values = np.median(pixel_sample_deltas[:, selected], axis=1)
        group_pixel_sensitivity[group_name] = group_values.tolist()
        grouped_pixel_values.append(group_values)
    pixel_sensitivity = np.median(np.asarray(grouped_pixel_values), axis=0)
    pixel_utility = _strategy_utility(
        map_stability,
        pixel_sensitivity,
        strategy,
        stability_penalty,
        epsilon,
    )
    pixel_weights = _normalize_positive(pixel_utility, branch_count, epsilon)

    diagnostics = {
        "method": f"clean_dispersion_plus_synthetic_suite_{strategy}",
        "stability_penalty": float(stability_penalty),
        "calibration_samples": int(scores_clean.shape[1]),
        "image": {
            "clean_robust_scale": score_scales.tolist(),
            "relative_stability": (score_stability / score_stability.max()).tolist(),
            "synthetic_sensitivity": score_sensitivity.tolist(),
            "sensitivity_by_group": group_image_sensitivity,
            "utility": image_utility.tolist(),
            "weights": image_weights.tolist(),
        },
        "pixel": {
            "clean_high_quantile_robust_scale": map_scales.tolist(),
            "relative_stability": (map_stability / map_stability.max()).tolist(),
            "synthetic_sensitivity": pixel_sensitivity.tolist(),
            "sensitivity_by_group": group_pixel_sensitivity,
            "utility": pixel_utility.tolist(),
            "weights": pixel_weights.tolist(),
        },
    }
    return image_weights, pixel_weights, diagnostics


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


def select_output_variant(variants):
    """Choose the requested method, never choose using test metrics."""
    for name in (
        "gate_selected",
        "image_fixed_pixel_sensitivity_shrinkage",
        "image_fixed_pixel_sensitivity",
        "adaptive",
        "combined",
        "combined_old",
        "sensitivity_both",
        "fixed",
    ):
        if name in variants:
            return name
    raise ValueError("No supported output fusion variant was found.")


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
        # Earlier adaptive checkpoints saved base weights at the top level.
        # Recover the actual requested variant when its record is available.
        weights_config = config
        if config.get("variants"):
            selected = config.get("selected_variant") or select_output_variant(
                config["variants"]
            )
            if selected not in config["variants"]:
                raise ValueError("Saved selected_variant is missing from variants.")
            weights_config = config["variants"][selected]
        image, pixel = weights_config.get("image_weights"), weights_config.get(
            "pixel_weights"
        )
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
