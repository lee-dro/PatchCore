import contextlib
import json
import logging
import os
import sys
import time

import click
import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.fusion
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


class _SyntheticAnomalyDataset(torch.utils.data.Dataset):
    """Small deterministic calibration set built from clean validation images."""

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "mask": self.masks[index].unsqueeze(0),
            "is_anomaly": 1,
        }


def _make_synthetic_calibration_loader(calibration_loader, seed, max_samples):
    """Create paired, deterministic small-point, scratch, and texture defects."""
    clean_images = []
    for batch in calibration_loader:
        clean_images.append(batch["image"].detach().cpu())
    if not clean_images:
        raise ValueError("The calibration split contains no clean images.")
    clean_images = torch.cat(clean_images, dim=0)
    if len(clean_images) == 0:
        raise ValueError("The calibration split contains no clean images.")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    synthetic_images = []
    masks = torch.zeros(
        (max_samples, clean_images.shape[-2], clean_images.shape[-1]),
        dtype=torch.float32,
    )
    height, width = clean_images.shape[-2:]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    reference_indices = []
    group_names = []
    defect_types = ("low_contrast_spot", "fine_scratch", "texture_patch")
    area_scales = (0.02, 0.04, 0.08)
    for index in range(max_samples):
        reference_index = index % len(clean_images)
        variant_index = index % (len(defect_types) * len(area_scales))
        defect_type = defect_types[variant_index % len(defect_types)]
        area_scale = area_scales[variant_index // len(defect_types)]
        image = clean_images[reference_index].clone()
        mask = torch.zeros((height, width), dtype=torch.float32)
        extent = max(3, int(min(height, width) * area_scale))
        top = int(torch.randint(0, height - extent + 1, (1,), generator=generator))
        left = int(torch.randint(0, width - extent + 1, (1,), generator=generator))

        if defect_type == "low_contrast_spot":
            radius = max(2, extent // 2)
            center_y, center_x = top + radius, left + radius
            mask = (
                (y_grid - center_y) ** 2 + (x_grid - center_x) ** 2 <= radius**2
            ).to(torch.float32)
            image = torch.clamp(image + mask.unsqueeze(0) * 0.45, -3.0, 3.0)
        elif defect_type == "fine_scratch":
            thickness = max(1, extent // 5)
            length = max(extent, int(extent * 2.5))
            length = min(length, width - left)
            mask[top : top + thickness, left : left + length] = 1.0
            image = torch.clamp(image + mask.unsqueeze(0) * 0.75, -3.0, 3.0)
        else:
            patch_mask = torch.zeros((height, width), dtype=torch.float32)
            patch_mask[top : top + extent, left : left + extent] = 1.0
            noise = torch.randn(
                (3, extent, extent), generator=generator, dtype=image.dtype
            )
            image[:, top : top + extent, left : left + extent] = torch.clamp(
                image[:, top : top + extent, left : left + extent] + noise * 0.35,
                -3.0,
                3.0,
            )
            mask = patch_mask

        synthetic_images.append(image)
        masks[index] = mask
        reference_indices.append(reference_index)
        group_names.append(f"{defect_type}_area_{area_scale:.2f}")

    synthetic_images = torch.stack(synthetic_images, dim=0)

    dataset = _SyntheticAnomalyDataset(synthetic_images, masks)
    return (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=calibration_loader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        ),
        masks.numpy(),
        np.asarray(reference_indices, dtype=np.int64),
        group_names,
    )


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
@click.option(
    "--adaptive_fusion",
    is_flag=True,
    help="Calibrate task-specific weights from clean validation images and synthetic defects.",
)
@click.option(
    "--ablation_fusion",
    is_flag=True,
    help="Evaluate fixed, stability-only, sensitivity-only, and combined fusion together.",
)
@click.option(
    "--regularized_ablation",
    is_flag=True,
    help="Evaluate fixed, sensitivity-led, stability-regularized, and legacy combined fusion.",
)
@click.option(
    "--task_decoupled_ablation",
    is_flag=True,
    help="Compare fixed fusion, fixed image plus sensitivity pixel fusion, and sensitivity-only fusion.",
)
@click.option(
    "--pixel_sensitivity_shrinkage",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Shrink sensitivity-derived pixel weights toward the base pixel weights.",
)
@click.option(
    "--sensitivity_gate",
    is_flag=True,
    help="Select fixed or shrinkage fusion using calibration-only safety checks.",
)
@click.option(
    "--gate_mode",
    type=click.Choice(["legacy", "conservative"]),
    default="legacy",
    show_default=True,
    help="Gate rule: legacy majority checks or conservative margin-aware checks.",
)
@click.option(
    "--gate_min_group_consistency",
    type=click.FloatRange(0.0, 1.0),
    default=0.8,
    show_default=True,
)
@click.option(
    "--gate_max_layer2_weight",
    type=click.FloatRange(0.0, 1.0),
    default=0.8,
    show_default=True,
)
@click.option(
    "--gate_min_layer3_stability",
    type=click.FloatRange(0.0, 1.0),
    default=0.9,
    show_default=True,
)
@click.option(
    "--gate_min_group_margin",
    type=click.FloatRange(0.0, None),
    default=0.5,
    show_default=True,
)
@click.option(
    "--gate_confidence_level",
    type=click.FloatRange(0.5, 0.999),
    default=0.9,
    show_default=True,
)
@click.option(
    "--adaptive_max_samples", type=click.IntRange(2), default=64, show_default=True
)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
    adaptive_fusion=False,
    ablation_fusion=False,
    regularized_ablation=False,
    task_decoupled_ablation=False,
    pixel_sensitivity_shrinkage=None,
    sensitivity_gate=False,
    gate_mode="legacy",
    gate_min_group_consistency=0.8,
    gate_max_layer2_weight=0.8,
    gate_min_layer3_stability=0.9,
    gate_min_group_margin=0.5,
    gate_confidence_level=0.9,
    adaptive_max_samples=64,
):
    if (
        sum(
            bool(value)
            for value in (
                adaptive_fusion,
                ablation_fusion,
                regularized_ablation,
                task_decoupled_ablation,
            )
        )
        > 1
    ):
        raise click.UsageError(
            "Use only one of --adaptive_fusion, --ablation_fusion, "
            "--regularized_ablation, or --task_decoupled_ablation."
        )
    if pixel_sensitivity_shrinkage is not None and not task_decoupled_ablation:
        raise click.UsageError(
            "--pixel_sensitivity_shrinkage requires --task_decoupled_ablation."
        )
    if sensitivity_gate and (
        not task_decoupled_ablation or pixel_sensitivity_shrinkage is None
    ):
        raise click.UsageError(
            "--sensitivity_gate requires --task_decoupled_ablation and "
            "--pixel_sensitivity_shrinkage."
        )
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        dataset_started = time.perf_counter()
        needs_calibration = any(
            (
                adaptive_fusion,
                ablation_fusion,
                regularized_ablation,
                task_decoupled_ablation,
            )
        )
        if needs_calibration and dataloaders.get("validation") is None:
            raise click.ClickException(
                "Adaptive fusion requires --train_val_split below 1.0; "
                "calibration images must be excluded from the Memory Bank."
            )
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](
                device,
            )
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            if sensitivity_gate and [
                list(m.layers_to_extract_from) for m in PatchCore_list
            ] != [["layer2"], ["layer3"]]:
                raise click.UsageError(
                    "The gate requires two ordered branches: 0.layer2, 1.layer3."
                )
            image_weights = [model.image_fusion_weight for model in PatchCore_list]
            pixel_weights = [model.pixel_fusion_weight for model in PatchCore_list]
            branch_records = []
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                LOGGER.debug(
                    "[Fusion] branch=%d layers=%s image_weight=%.6f pixel_weight=%.6f",
                    i,
                    PatchCore.layers_to_extract_from,
                    image_weights[i],
                    pixel_weights[i],
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                PatchCore.fit(dataloaders["training"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                bank_shape = list(PatchCore.anomaly_scorer.detection_features.shape)
                branch_records.append(
                    {
                        "backbone": PatchCore.backbone.name,
                        "layers": list(PatchCore.layers_to_extract_from),
                        "layer_fusion_weights": getattr(
                            PatchCore, "layer_fusion_weights", None
                        ),
                        "image_weight": image_weights[i],
                        "pixel_weight": pixel_weights[i],
                        "fit_time_s": elapsed,
                        "memory_bank_shape": bank_shape,
                    }
                )
                LOGGER.info("Fit time: %.2f seconds", elapsed)

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                branch_records[i]["predict_time_s"] = elapsed
                LOGGER.info("Predict time: %.2f seconds", elapsed)
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            base_image_weights = np.asarray(image_weights, dtype=np.float64)
            base_pixel_weights = np.asarray(pixel_weights, dtype=np.float64)
            fusion_variants = {
                "fixed": {
                    "image_weights": base_image_weights,
                    "pixel_weights": base_pixel_weights,
                    "diagnostics": None,
                }
            }
            calibration_time = 0.0
            if (
                adaptive_fusion
                or ablation_fusion
                or regularized_ablation
                or task_decoupled_ablation
            ):
                calibration_started = time.perf_counter()
                calibration_loader = dataloaders.get("validation")
                if calibration_loader is None:
                    raise click.ClickException(
                        "Adaptive fusion requires --train_val_split below 1.0 "
                        "so clean calibration images are kept outside the Memory Bank."
                    )
                (
                    synthetic_loader,
                    synthetic_masks,
                    synthetic_reference_indices,
                    synthetic_groups,
                ) = _make_synthetic_calibration_loader(
                    calibration_loader, seed, adaptive_max_samples
                )
                calibration_scores = []
                calibration_maps = []
                synthetic_scores = []
                synthetic_maps = []
                for i, PatchCore in enumerate(PatchCore_list):
                    LOGGER.info(
                        "Adaptive calibration branch %d/%d: clean validation outputs",
                        i + 1,
                        len(PatchCore_list),
                    )
                    clean_scores, clean_maps, _, _ = PatchCore.predict(
                        calibration_loader
                    )
                    LOGGER.info(
                        "Adaptive calibration branch %d/%d: synthetic defect outputs",
                        i + 1,
                        len(PatchCore_list),
                    )
                    branch_scores, branch_maps, _, _ = PatchCore.predict(
                        synthetic_loader
                    )
                    clean_scores = np.asarray(clean_scores)
                    clean_maps = np.asarray(clean_maps)
                    calibration_scores.append(clean_scores[synthetic_reference_indices])
                    calibration_maps.append(clean_maps[synthetic_reference_indices])
                    synthetic_scores.append(np.asarray(branch_scores))
                    synthetic_maps.append(np.asarray(branch_maps))

                if task_decoupled_ablation:
                    variant_specs = []
                    if pixel_sensitivity_shrinkage is None:
                        variant_specs.append(("sensitivity", "sensitivity_both", 0.0))
                    variant_specs.append(
                        ("sensitivity", "image_fixed_pixel_sensitivity", 0.0)
                    )
                    if pixel_sensitivity_shrinkage is not None:
                        variant_specs.append(
                            (
                                "sensitivity",
                                "image_fixed_pixel_sensitivity_shrinkage",
                                0.0,
                            )
                        )
                elif regularized_ablation:
                    variant_specs = [
                        ("sensitivity_aux", "sensitivity_l0", 0.0),
                        ("sensitivity_aux", "sensitivity_aux_l01", 0.1),
                        ("sensitivity_aux", "sensitivity_aux_l025", 0.25),
                        ("combined", "combined_old", 0.0),
                    ]
                elif ablation_fusion:
                    variant_specs = [
                        ("stability", "stability_only", 0.0),
                        ("sensitivity", "sensitivity_only", 0.0),
                        ("combined", "combined", 0.0),
                    ]
                else:
                    variant_specs = [("combined", "adaptive", 0.0)]

                weight_cache = {}
                for strategy, variant_name, stability_penalty in variant_specs:
                    cache_key = (strategy, stability_penalty)
                    if cache_key not in weight_cache:
                        weight_cache[cache_key] = (
                            patchcore.fusion.adaptive_task_weights(
                                calibration_scores,
                                synthetic_scores,
                                calibration_maps,
                                synthetic_maps,
                                synthetic_masks,
                                synthetic_groups,
                                strategy=strategy,
                                stability_penalty=stability_penalty,
                            )
                        )
                    adaptive_image_weights, adaptive_pixel_weights, diagnostics = (
                        weight_cache[cache_key]
                    )
                    if (
                        task_decoupled_ablation
                        and variant_name == "image_fixed_pixel_sensitivity"
                    ):
                        adaptive_image_weights = base_image_weights.copy()
                    if (
                        task_decoupled_ablation
                        and variant_name == "image_fixed_pixel_sensitivity_shrinkage"
                    ):
                        adaptive_image_weights = base_image_weights.copy()
                        adaptive_pixel_weights = patchcore.fusion.shrink_weights(
                            base_pixel_weights,
                            adaptive_pixel_weights,
                            pixel_sensitivity_shrinkage,
                        )
                    fusion_variants[variant_name] = {
                        "image_weights": adaptive_image_weights,
                        "pixel_weights": adaptive_pixel_weights,
                        "diagnostics": diagnostics,
                    }
                    LOGGER.info(
                        "[Fusion:%s] image_weights=%s pixel_weights=%s",
                        variant_name,
                        np.round(adaptive_image_weights, 6).tolist(),
                        np.round(adaptive_pixel_weights, 6).tolist(),
                    )

                if sensitivity_gate:
                    gate_diagnostics = patchcore.fusion.sensitivity_gate(
                        fusion_variants["image_fixed_pixel_sensitivity_shrinkage"][
                            "diagnostics"
                        ],
                        min_group_consistency=gate_min_group_consistency,
                        max_layer2_weight=gate_max_layer2_weight,
                        min_layer3_stability=gate_min_layer3_stability,
                        mode=gate_mode,
                        min_group_margin=gate_min_group_margin,
                        confidence_level=gate_confidence_level,
                    )
                    selected_name = (
                        "image_fixed_pixel_sensitivity_shrinkage"
                        if gate_diagnostics["use_shrinkage"]
                        else "fixed"
                    )
                    selected_variant = fusion_variants[selected_name]
                    fusion_variants["gate_selected"] = {
                        "image_weights": selected_variant["image_weights"].copy(),
                        "pixel_weights": selected_variant["pixel_weights"].copy(),
                        "diagnostics": {
                            "gate": gate_diagnostics,
                            "selected_variant": selected_name,
                        },
                    }
                    LOGGER.info(
                        "[Fusion:gate_selected] use_shrinkage=%s selected=%s checks=%s",
                        gate_diagnostics["use_shrinkage"],
                        selected_name,
                        gate_diagnostics,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                calibration_time = time.perf_counter() - calibration_started
                LOGGER.info("Calibration time: %.2f seconds", calibration_time)
                del (
                    calibration_scores,
                    calibration_maps,
                    synthetic_scores,
                    synthetic_maps,
                    weight_cache,
                )

            started = time.perf_counter()
            variant_outputs = {}
            output_cache = {}
            for variant_name, variant in fusion_variants.items():
                weights_key = (
                    tuple(variant["image_weights"]),
                    tuple(variant["pixel_weights"]),
                )
                if weights_key in output_cache:
                    variant_outputs[variant_name] = output_cache[weights_key]
                    continue
                variant_started = time.perf_counter()
                variant_scores, score_bounds = patchcore.fusion.fuse_predictions(
                    aggregator["scores"], variant["image_weights"]
                )
                variant_segmentations, map_bounds = patchcore.fusion.fuse_predictions(
                    aggregator["segmentations"], variant["pixel_weights"]
                )
                variant_outputs[variant_name] = {
                    "scores": variant_scores,
                    "segmentations": variant_segmentations,
                    "score_bounds": score_bounds,
                    "map_bounds": map_bounds,
                    "fusion_time_s": time.perf_counter() - variant_started,
                }
                output_cache[weights_key] = variant_outputs[variant_name]
            fusion_time = time.perf_counter() - started
            fit_time = sum(row["fit_time_s"] for row in branch_records)
            selected_variant = patchcore.fusion.select_output_variant(fusion_variants)
            predict_time = (
                sum(row["predict_time_s"] for row in branch_records)
                + variant_outputs[selected_variant]["fusion_time_s"]
            )
            fusion_config = {
                "version": 2,
                "dataset": dataset_name,
                "seed": seed,
                "selected_variant": selected_variant,
                "image_weights": fusion_variants[selected_variant][
                    "image_weights"
                ].tolist(),
                "pixel_weights": fusion_variants[selected_variant][
                    "pixel_weights"
                ].tolist(),
                "base_image_weights": base_image_weights.tolist(),
                "base_pixel_weights": base_pixel_weights.tolist(),
                "data_split": dataloaders.get("split_manifest", {}),
                "adaptive_max_samples": (
                    adaptive_max_samples if needs_calibration else 0
                ),
                "pixel_sensitivity_shrinkage": pixel_sensitivity_shrinkage,
                "sensitivity_gate": sensitivity_gate,
                "gate_thresholds": {
                    "mode": gate_mode,
                    "min_group_consistency": gate_min_group_consistency,
                    "max_layer2_weight": gate_max_layer2_weight,
                    "min_layer3_stability": gate_min_layer3_stability,
                    "min_group_margin": gate_min_group_margin,
                    "confidence_level": gate_confidence_level,
                },
                "fusion_method": (
                    "task_decoupled_sensitivity_pixel_shrinkage_ablation"
                    if task_decoupled_ablation
                    and pixel_sensitivity_shrinkage is not None
                    else (
                        "task_decoupled_sensitivity_pixel_ablation"
                        if task_decoupled_ablation
                        else (
                            "sensitivity_led_stability_regularized_ablation"
                            if regularized_ablation
                            else (
                                "four_way_ablation"
                                if ablation_fusion
                                else (
                                    "clean_dispersion_plus_synthetic_suite_combined"
                                    if adaptive_fusion
                                    else "fixed_task_weights"
                                )
                            )
                        )
                    )
                ),
                "variants": {
                    name: {
                        "image_weights": variant["image_weights"].tolist(),
                        "pixel_weights": variant["pixel_weights"].tolist(),
                        "adaptive_diagnostics": variant["diagnostics"],
                        "score_bounds": variant_outputs[name]["score_bounds"],
                        "map_bounds": variant_outputs[name]["map_bounds"],
                        "fusion_time_s": variant_outputs[name]["fusion_time_s"],
                    }
                    for name, variant in fusion_variants.items()
                },
                "branches": branch_records,
                "normalization": "evaluation_set_minmax",
                "fit_time_s": fit_time,
                "predict_time_s": predict_time,
                "calibration_time_s": calibration_time,
                "fit_calibration_predict_time_s": fit_time
                + calibration_time
                + predict_time,
                "fusion_time_s": fusion_time,
                "command": sys.argv,
            }
            with open(
                os.path.join(run_save_path, dataset_name + "_fusion.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(fusion_config, handle, indent=2)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # Metrics, visualization and saved/reloaded weights share this selection.
            scores = variant_outputs[selected_variant]["scores"]
            segmentations = variant_outputs[selected_variant]["segmentations"]

            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            LOGGER.info("Computing evaluation metrics.")
            evaluation_started = time.perf_counter()
            metrics_cache = {}
            for variant_name, outputs in variant_outputs.items():
                variant_scores = outputs["scores"]
                variant_segmentations = outputs["segmentations"]
                output_key = id(outputs)
                if output_key not in metrics_cache:
                    metrics_cache[output_key] = (
                        patchcore.metrics.compute_imagewise_retrieval_metrics(
                            variant_scores, anomaly_labels
                        )["auroc"],
                        patchcore.metrics.compute_pixelwise_retrieval_metrics(
                            variant_segmentations, masks_gt
                        )["auroc"],
                        patchcore.metrics.compute_pixelwise_retrieval_metrics(
                            [variant_segmentations[i] for i in sel_idxs],
                            [masks_gt[i] for i in sel_idxs],
                        )["auroc"],
                    )
                auroc, full_pixel_auroc, anomaly_pixel_auroc = metrics_cache[output_key]
                result = {
                    "dataset_name": (
                        dataset_name
                        if len(variant_outputs) == 1
                        else f"{dataset_name}__{variant_name}"
                    ),
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
                result_collect.append(result)
                LOGGER.info(
                    "[%s] instance_auroc: %.3f full_pixel_auroc: %.3f "
                    "anomaly_pixel_auroc: %.3f",
                    variant_name,
                    auroc,
                    full_pixel_auroc,
                    anomaly_pixel_auroc,
                )
            fusion_config["evaluation_time_s"] = (
                time.perf_counter() - evaluation_started
            )
            fusion_config["dataset_wall_time_s_before_save"] = (
                time.perf_counter() - dataset_started
            )
            with open(
                os.path.join(run_save_path, dataset_name + "_fusion.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(fusion_config, handle, indent=2)

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                with open(
                    os.path.join(patchcore_save_path, "fusion.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(fusion_config, handle, indent=2)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
        row_groups=[
            name.split("__", 1)[-1] if "__" in name else "fixed"
            for name in result_dataset_names
        ],
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option(
    "--fusion_weight",
    type=float,
    multiple=True,
    help="One non-negative weight per backbone; default: equal weights.",
)
@click.option(
    "--image_fusion_weight",
    type=float,
    multiple=True,
    help="Image-score weights in backbone order; overrides --fusion_weight for images.",
)
@click.option(
    "--pixel_fusion_weight",
    type=float,
    multiple=True,
    help="Anomaly-map weights in backbone order; overrides --fusion_weight for pixels.",
)
@click.option(
    "--layer_fusion_weight",
    type=float,
    multiple=True,
    help="Feature-level weights in extracted-layer order for one PatchCore model.",
)
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    fusion_weight,
    image_fusion_weight,
    pixel_fusion_weight,
    layer_fusion_weight,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    try:
        image_weights, pixel_weights = patchcore.fusion.resolve_task_weights(
            len(backbone_names),
            shared=fusion_weight,
            image=image_fusion_weight,
            pixel=pixel_fusion_weight,
        )
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            try:
                idx = int(layer.split(".")[0])
            except ValueError as error:
                raise click.BadParameter(
                    "Use indexed layers, e.g. 0.layer2 and 1.layer3.", param_hint="-le"
                ) from error
            if idx < 0 or idx >= len(backbone_names) or "." not in layer:
                raise click.BadParameter(
                    "Layer index must match a backbone.", param_hint="-le"
                )
            layer = ".".join(layer.split(".")[1:])
            if not layer:
                raise click.BadParameter(
                    "Layer name cannot be empty.", param_hint="-le"
                )
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    if any(not layers for layers in layers_to_extract_from_coll):
        raise click.BadParameter(
            "Each backbone needs at least one layer.", param_hint="-le"
        )

    layer_fusion_weights = list(layer_fusion_weight)
    total_layer_count = sum(len(layers) for layers in layers_to_extract_from_coll)
    if layer_fusion_weights and len(layer_fusion_weights) != total_layer_count:
        raise click.UsageError(
            "--layer_fusion_weight requires one value per extracted layer "
            "in command-line order."
        )

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        layer_weight_offset = 0
        for backbone_name, layers_to_extract_from, image_weight, pixel_weight in zip(
            backbone_names, layers_to_extract_from_coll, image_weights, pixel_weights
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            model_layer_weights = None
            if layer_fusion_weights:
                layer_weight_end = layer_weight_offset + len(layers_to_extract_from)
                model_layer_weights = layer_fusion_weights[
                    layer_weight_offset:layer_weight_end
                ]
                layer_weight_offset = layer_weight_end

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_score_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
                layer_fusion_weights=model_layer_weights,
            )
            patchcore_instance.image_fusion_weight = float(image_weight)
            patchcore_instance.pixel_fusion_weight = float(pixel_weight)
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument(
    "name", type=click.Choice(["identity", "greedy_coreset", "approx_greedy_coreset"])
)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option(
    "--train_val_split",
    type=click.FloatRange(0.0, 1.0, min_open=True),
    default=1,
    show_default=True,
)
@click.option("--batch_size", default=2, type=click.IntRange(1), show_default=True)
@click.option("--num_workers", default=8, type=click.IntRange(0), show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            full_train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=1.0,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            if train_val_split < 1.0:
                if not 0.0 < train_val_split < 1.0:
                    raise click.BadParameter(
                        "--train_val_split must be between 0 and 1.",
                        param_hint="--train_val_split",
                    )
                indices = np.random.RandomState(seed).permutation(
                    len(full_train_dataset)
                )
                split_index = int(len(indices) * train_val_split)
                if split_index < 1 or len(indices) - split_index < 2:
                    raise click.BadParameter(
                        "Split needs at least one training image and two calibration images."
                    )
                if augment:
                    raise click.BadParameter(
                        "Paired calibration requires deterministic transforms; disable --augment."
                    )
                train_indices = indices[:split_index].tolist()
                validation_indices = indices[split_index:].tolist()
                train_dataset = torch.utils.data.Subset(
                    full_train_dataset, train_indices
                )
                val_dataset = torch.utils.data.Subset(
                    full_train_dataset, validation_indices
                )
                # Preserve the attributes consumed by the training loop.
                train_dataset.imagesize = full_train_dataset.imagesize
                val_dataset.imagesize = full_train_dataset.imagesize
            else:
                train_dataset = full_train_dataset
                val_dataset = None
                train_indices = list(range(len(full_train_dataset)))
                validation_indices = []

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if val_dataset is not None:
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
                "split_manifest": {
                    "train_val_split": train_val_split,
                    "split_seed": seed,
                    "train_indices": train_indices,
                    "calibration_indices": validation_indices,
                    "train_files": [
                        os.path.relpath(
                            full_train_dataset.data_to_iterate[i][2], data_path
                        )
                        for i in train_indices
                    ],
                    "calibration_files": [
                        os.path.relpath(
                            full_train_dataset.data_to_iterate[i][2], data_path
                        )
                        for i in validation_indices
                    ],
                    "test_files": [
                        os.path.relpath(row[2], data_path)
                        for row in test_dataset.data_to_iterate
                    ],
                    "resize": resize,
                    "imagesize": imagesize,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "augment": augment,
                },
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
