"""Regression tests for calibration, selected output, checkpoints and reporting."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_fusion import load_script
from patchcore import fusion, utils


@pytest.mark.parametrize(
    "options,selected",
    [
        ({"adaptive_fusion": True}, "adaptive"),
        ({"ablation_fusion": True}, "combined"),
        ({"regularized_ablation": True}, "combined_old"),
        ({"task_decoupled_ablation": True}, "image_fixed_pixel_sensitivity"),
        (
            {"task_decoupled_ablation": True, "pixel_sensitivity_shrinkage": 0.25},
            "image_fixed_pixel_sensitivity_shrinkage",
        ),
        (
            {
                "task_decoupled_ablation": True,
                "pixel_sensitivity_shrinkage": 0.25,
                "sensitivity_gate": True,
            },
            "gate_selected",
        ),
    ],
)
def test_all_adaptive_modes_save_and_plot_the_selected_weights(
    tmp_path, monkeypatch, options, selected
):
    run = load_script("run_patchcore")
    raw_scores = np.array([[0, 1, 2], [2, 1, 0]], dtype=float)
    raw_maps = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
    raw_maps[1] = raw_maps[1, ::-1]
    models = [
        SimpleNamespace(
            image_fusion_weight=wi,
            pixel_fusion_weight=0.5,
            backbone=SimpleNamespace(name="tiny", seed=None),
            layers_to_extract_from=[layer],
            anomaly_scorer=SimpleNamespace(detection_features=np.zeros((2, 4))),
            fit=lambda data: None,
            predict=lambda data, i=i: (
                raw_scores[i],
                raw_maps[i],
                [0, 1, 1],
                np.ones((3, 2, 2)),
            ),
            save_to_path=lambda *args: None,
        )
        for i, (wi, layer) in enumerate(zip([0.9, 0.1], ["layer2", "layer3"]))
    ]
    dataset = SimpleNamespace(
        imagesize=(3, 2, 2),
        data_to_iterate=[
            (0, "good", "good.png", None),
            (1, "bad", "bad.png", "mask.png"),
            (2, "bad", "bad2.png", "mask2.png"),
        ],
    )
    loader = SimpleNamespace(name="mvtec_mock", dataset=dataset)
    monkeypatch.setattr(utils, "set_torch_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(
        run,
        "_make_synthetic_calibration_loader",
        lambda *args: (loader, np.ones((3, 2, 2)), np.arange(3), ["spot"] * 3),
    )
    monkeypatch.setattr(
        fusion,
        "adaptive_task_weights",
        lambda *args, **kw: (
            np.array([0.2, 0.8]),
            np.array([0.2, 0.8]),
            {
                "pixel": {
                    "weights": [0.2, 0.8],
                    "relative_stability": [1, 1],
                    "sensitivity_by_group": {"spot": [1, 2], "scratch": [1, 2]},
                }
            },
        ),
    )
    monkeypatch.setattr(
        run.patchcore.metrics,
        "compute_imagewise_retrieval_metrics",
        lambda *args: {"auroc": 0.5},
    )
    monkeypatch.setattr(
        run.patchcore.metrics,
        "compute_pixelwise_retrieval_metrics",
        lambda *args: {"auroc": 0.5},
    )
    seen = {}

    def plot(folder, images, maps, scores, *args, **kwargs):
        seen.update(maps=maps, scores=scores)

    monkeypatch.setattr(utils, "plot_segmentation_images", plot)
    run.run(
        [
            (
                "get_dataloaders",
                lambda _: [
                    {"training": loader, "testing": loader, "validation": loader}
                ],
            ),
            ("get_sampler", lambda _: None),
            ("get_patchcore", lambda *args: models),
        ],
        str(tmp_path),
        [],
        0,
        "run",
        "project",
        True,
        True,
        **options,
    )
    path = tmp_path / "project/run"
    config = json.loads((path / "mvtec_mock_fusion.json").read_text())
    saved = json.loads((path / "models/mvtec_mock/fusion.json").read_text())
    assert config == saved
    assert config["selected_variant"] == selected
    wi, wp = fusion.restore_task_weights(saved, 2)
    np.testing.assert_allclose(wi, config["image_weights"])
    np.testing.assert_allclose(wp, config["pixel_weights"])
    np.testing.assert_allclose(seen["maps"], fusion.fuse_predictions(raw_maps, wp)[0])
    np.testing.assert_allclose(
        seen["scores"], fusion.fuse_predictions(raw_scores, wi)[0]
    )
    if options.get("task_decoupled_ablation"):
        np.testing.assert_allclose(wi, [0.9, 0.1])
    if options.get("pixel_sensitivity_shrinkage") is not None:
        np.testing.assert_allclose(wp, [0.425, 0.575])
    assert config["calibration_time_s"] >= 0
    rows = list(csv.DictReader((path / "results.csv").open()))
    assert "Mean" not in [r["Row Names"] for r in rows]
    assert any(r["Row Names"] == "Mean__" + selected for r in rows)


def test_old_adaptive_checkpoint_recovers_variant_instead_of_base():
    cfg = {
        "version": 2,
        "normalization": "evaluation_set_minmax",
        "image_weights": [0.9, 0.1],
        "pixel_weights": [0.5, 0.5],
        "variants": {
            "gate_selected": {"image_weights": [0.9, 0.1], "pixel_weights": [0.4, 0.6]}
        },
    }
    np.testing.assert_allclose(fusion.restore_task_weights(cfg, 2)[1], [0.4, 0.6])


def test_gate_legacy_rejects_unstable_branch_and_conservative_rejects_ties():
    diag = {
        "pixel": {
            "weights": [0.5, 0.5],
            "relative_stability": [1, 0.5],
            "sensitivity_by_group": {"spot": [1, 2], "scratch": [1, 2]},
        }
    }
    assert not fusion.sensitivity_gate(diag)["use_shrinkage"]
    diag["pixel"]["relative_stability"] = [1, 1]
    diag["pixel"]["sensitivity_by_group"] = {"spot": [0, 0]}
    assert not fusion.sensitivity_gate(diag, mode="conservative")["use_shrinkage"]


@pytest.mark.parametrize("kind", ["empty_mask", "nonbinary_mask", "shape_mismatch"])
def test_invalid_calibration_inputs_are_rejected(kind):
    scores = np.zeros((2, 3))
    maps = np.zeros((2, 3, 2, 2))
    masks = np.ones((3, 2, 2))
    if kind == "empty_mask":
        masks[0] = 0
    elif kind == "nonbinary_mask":
        masks[0] = np.nan
    else:
        scores = np.zeros((1, 3))
    with pytest.raises(ValueError):
        fusion.adaptive_task_weights(scores, scores, maps, maps, masks)


def test_shrinkage_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        fusion.shrink_weights([-0.1, 1.1], [0.5, 0.5], 0.25)


def test_synthetic_suite_is_deterministic_paired_and_confined_to_masks():
    run = load_script("run_patchcore")
    loader = torch.utils.data.DataLoader(
        [
            {"image": torch.zeros(3, 32, 32)},
            {"image": torch.ones(3, 32, 32)},
        ],
        batch_size=2,
    )
    a, masks, refs, groups = run._make_synthetic_calibration_loader(loader, 3, 18)
    b, masks_b, refs_b, groups_b = run._make_synthetic_calibration_loader(loader, 3, 18)
    torch.testing.assert_close(a.dataset.images, b.dataset.images)
    np.testing.assert_array_equal(refs, refs_b)
    assert groups == groups_b and len(set(groups)) == 9
    clean = torch.stack([loader.dataset[i]["image"] for i in refs])
    changes = (a.dataset.images - clean).abs().sum(dim=1).numpy()
    assert np.all(changes[masks == 0] == 0)
    assert np.all(masks.sum(axis=(1, 2)) > 0)


def test_heatmap_filenames_keep_defect_type_and_allow_no_masks(tmp_path):
    from PIL import Image

    paths = []
    for defect in ("good", "scratch"):
        folder = tmp_path / defect
        folder.mkdir()
        file = folder / "000.png"
        Image.new("RGB", (8, 8)).save(file)
        paths.append(str(file))
    utils.plot_segmentation_images(str(tmp_path / "plots"), paths, np.zeros((2, 8, 8)))
    assert len(list((tmp_path / "plots").glob("*.png"))) == 2


def test_evaluator_can_reuse_one_model_for_two_loaders(tmp_path, monkeypatch):
    evaluator = load_script("load_and_evaluate_patchcore")
    dataset = SimpleNamespace(data_to_iterate=[(0, "good"), (1, "bad")])
    loaders = [
        {"testing": SimpleNamespace(name=name, dataset=dataset)} for name in ("a", "b")
    ]
    calls = []

    def predict(loader):
        calls.append(loader.name)
        return [0, 1], np.arange(8).reshape(2, 2, 2), [0, 1], np.ones((2, 2, 2))

    model = SimpleNamespace(
        image_fusion_weight=1.0, pixel_fusion_weight=1.0, predict=predict
    )
    monkeypatch.setattr(utils, "set_torch_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(
        evaluator.patchcore.metrics,
        "compute_imagewise_retrieval_metrics",
        lambda *args: {"auroc": 0.5},
    )
    monkeypatch.setattr(
        evaluator.patchcore.metrics,
        "compute_pixelwise_retrieval_metrics",
        lambda *args: {"auroc": 0.5},
    )
    evaluator.run(
        [
            ("get_dataloaders_iter", (lambda _: iter(loaders), 2)),
            ("get_patchcore_iter", (lambda _: iter([[model]]), 1)),
        ],
        str(tmp_path),
        [],
        0,
        False,
    )
    assert calls == ["a", "b"]


def test_dataset_split_manifest_is_disjoint_and_seeded(tmp_path):
    from PIL import Image

    run = load_script("run_patchcore")
    folder = tmp_path / "bottle/train/good"
    folder.mkdir(parents=True)
    for i in range(10):
        Image.new("RGB", (32, 32)).save(folder / f"{i:03d}.png")
    for defect in ("good", "broken"):
        test_folder = tmp_path / "bottle/test" / defect
        test_folder.mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(test_folder / "000.png")
    ground = tmp_path / "bottle/ground_truth/broken"
    ground.mkdir(parents=True)
    Image.new("L", (32, 32), 255).save(ground / "000_mask.png")
    _, factory = run.dataset.callback(
        "mvtec", str(tmp_path), ("bottle",), 0.8, 2, 32, 32, 0, False
    )
    a = factory(0)[0]["split_manifest"]
    b = factory(1)[0]["split_manifest"]
    assert len(a["train_files"]) == 8 and len(a["calibration_files"]) == 2
    assert not set(a["train_files"]) & set(a["calibration_files"])
    assert a["train_indices"] != b["train_indices"]
    assert a == factory(0)[0]["split_manifest"]


def test_statistics_use_category_level_resampling():
    benchmark = load_script("benchmark_fusion")
    rows = []
    for category, gain in (("a", 0.1), ("b", -0.1)):
        for seed in range(5):
            for method, value in (("fixed", 0.5), ("adaptive", 0.5 + gain)):
                rows.append(
                    dict(
                        category=category,
                        seed=seed,
                        method=method,
                        **{m: value for m in benchmark.METRICS},
                    )
                )
    _, pairs, stats = benchmark.paired_statistics(rows, "fixed")
    assert len(pairs) == 10
    assert all(s["categories"] == 2 and s["pairs"] == 10 for s in stats)
    assert all(
        abs(s["mean_delta"]) < 1e-12 and s["category_signflip_p"] == 1 for s in stats
    )
