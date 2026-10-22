import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from patchcore.fusion import (
    adaptive_task_weights,
    fuse_predictions,
    normalize_weights,
    resolve_task_weights,
    restore_task_weights,
    sensitivity_gate,
    shrink_weights,
)


def test_adaptive_task_weights_follow_clean_stability_and_synthetic_sensitivity():
    rng = np.random.RandomState(0)
    clean_scores = np.asarray([0.01 * rng.normal(size=8), 0.5 * rng.normal(size=8)])
    synthetic_scores = clean_scores + np.asarray([[1.0], [0.1]])
    clean_maps = np.asarray(
        [0.01 * rng.normal(size=(8, 4, 4)), 0.5 * rng.normal(size=(8, 4, 4))]
    )
    synthetic_maps = clean_maps.copy()
    masks = np.zeros((8, 4, 4), dtype=float)
    masks[:, 1:3, 1:3] = 1.0
    synthetic_maps += masks[None] * np.asarray([[[[1.0]]], [[[0.1]]]])

    image_weights, pixel_weights, diagnostics = adaptive_task_weights(
        clean_scores,
        synthetic_scores,
        clean_maps,
        synthetic_maps,
        masks,
    )

    assert image_weights[0] > image_weights[1]
    assert pixel_weights[0] > pixel_weights[1]
    assert diagnostics["method"] == "clean_dispersion_plus_synthetic_suite_combined"
    np.testing.assert_allclose(image_weights.sum(), 1.0)
    np.testing.assert_allclose(pixel_weights.sum(), 1.0)


def test_sensitivity_aux_zero_penalty_matches_sensitivity_only():
    rng = np.random.RandomState(1)
    clean_scores = np.asarray([0.05 * rng.normal(size=10), 0.25 * rng.normal(size=10)])
    synthetic_scores = clean_scores + np.asarray([[0.8], [0.3]])
    clean_maps = np.asarray(
        [0.05 * rng.normal(size=(10, 4, 4)), 0.25 * rng.normal(size=(10, 4, 4))]
    )
    synthetic_maps = clean_maps.copy()
    masks = np.zeros((10, 4, 4), dtype=float)
    masks[:, 1:3, 1:3] = 1.0
    synthetic_maps += masks[None] * np.asarray([[[[0.8]]], [[[0.3]]]])
    groups = ["spot"] * 10

    sensitivity = adaptive_task_weights(
        clean_scores,
        synthetic_scores,
        clean_maps,
        synthetic_maps,
        masks,
        groups,
        strategy="sensitivity",
    )
    sensitivity_aux = adaptive_task_weights(
        clean_scores,
        synthetic_scores,
        clean_maps,
        synthetic_maps,
        masks,
        groups,
        strategy="sensitivity_aux",
        stability_penalty=0.0,
    )

    np.testing.assert_allclose(sensitivity_aux[0], sensitivity[0])
    np.testing.assert_allclose(sensitivity_aux[1], sensitivity[1])
    assert sensitivity_aux[2]["stability_penalty"] == 0.0


def test_sensitivity_aux_penalty_reduces_weight_of_more_volatile_branch():
    rng = np.random.RandomState(2)
    clean_scores = np.asarray([0.05 * rng.normal(size=12), 0.5 * rng.normal(size=12)])
    synthetic_scores = clean_scores + np.asarray([[0.5], [0.5]])
    clean_maps = np.asarray(
        [0.05 * rng.normal(size=(12, 4, 4)), 0.5 * rng.normal(size=(12, 4, 4))]
    )
    synthetic_maps = clean_maps.copy()
    masks = np.zeros((12, 4, 4), dtype=float)
    masks[:, 1:3, 1:3] = 1.0
    synthetic_maps += masks[None] * np.asarray([[[[0.5]]], [[[0.5]]]])
    groups = ["spot"] * 12

    no_penalty = adaptive_task_weights(
        clean_scores,
        synthetic_scores,
        clean_maps,
        synthetic_maps,
        masks,
        groups,
        strategy="sensitivity_aux",
        stability_penalty=0.0,
    )
    with_penalty = adaptive_task_weights(
        clean_scores,
        synthetic_scores,
        clean_maps,
        synthetic_maps,
        masks,
        groups,
        strategy="sensitivity_aux",
        stability_penalty=0.25,
    )

    assert with_penalty[0][0] > no_penalty[0][0]
    assert with_penalty[1][0] > no_penalty[1][0]
    assert with_penalty[2]["stability_penalty"] == 0.25


def test_shrink_weights_interpolates_between_base_and_sensitivity_weights():
    base = np.asarray([0.5, 0.5])
    sensitivity = np.asarray([0.8, 0.2])

    np.testing.assert_allclose(shrink_weights(base, sensitivity, 0.0), base)
    np.testing.assert_allclose(shrink_weights(base, sensitivity, 1.0), sensitivity)
    np.testing.assert_allclose(
        shrink_weights(base, sensitivity, 0.25), np.asarray([0.575, 0.425])
    )


def test_shrink_weights_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="between 0 and 1"):
        shrink_weights(np.asarray([0.5, 0.5]), np.asarray([0.8, 0.2]), 1.1)
    with pytest.raises(ValueError, match="sum to 1"):
        shrink_weights(np.asarray([0.4, 0.4]), np.asarray([0.8, 0.2]), 0.25)


def test_sensitivity_gate_requires_calibration_safety_checks():
    diagnostics = {
        "pixel": {
            "sensitivity_by_group": {
                "spot": [2.0, 1.0],
                "scratch": [1.8, 1.0],
            },
            "weights": [0.7, 0.3],
            "relative_stability": [0.7, 1.0],
        }
    }
    result = sensitivity_gate(diagnostics)
    assert result["use_shrinkage"] is True
    assert result["group_consistency_ok"] is True

    diagnostics["pixel"]["weights"] = [0.9, 0.1]
    result = sensitivity_gate(diagnostics)
    assert result["use_shrinkage"] is False


def test_conservative_sensitivity_gate_requires_margin_confidence():
    diagnostics = {
        "pixel": {
            "sensitivity_by_group": {
                "spot": [5.0, 1.0],
                "scratch": [4.0, 1.0],
                "texture": [3.0, 1.0],
            },
            "weights": [0.7, 0.3],
            "relative_stability": [0.4, 1.0],
        }
    }
    result = sensitivity_gate(diagnostics, mode="conservative")
    assert result["use_shrinkage"] is True
    assert result["margin_consistency"] == 1.0
    assert result["margin_confidence_ok"] is True

    diagnostics["pixel"]["sensitivity_by_group"]["texture"] = [1.0, 5.0]
    result = sensitivity_gate(diagnostics, mode="conservative")
    assert result["use_shrinkage"] is False
    assert result["layer2_weight_ok"] is True
    assert result["margin_consistency_ok"] is False


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "bin" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", [(2, 5), (2, 5, 4, 3)])
def test_equal_weights_match_original(shape):
    values = np.random.RandomState(0).rand(*shape)
    axes = tuple(range(1, len(shape)))
    lower = values.min(axis=axes, keepdims=True)
    expected = ((values - lower) / (values.max(axis=axes, keepdims=True) - lower)).mean(
        axis=0
    )
    actual, _ = fuse_predictions(values, [])
    np.testing.assert_allclose(actual, expected)


def test_weight_direction_endpoints_and_scaling():
    scores = [[10, 15, 20], [4, 2, 0]]
    np.testing.assert_allclose(fuse_predictions(scores, [0.7, 0.3])[0], [0.3, 0.5, 0.7])
    np.testing.assert_allclose(fuse_predictions(scores, [7, 3])[0], [0.3, 0.5, 0.7])
    np.testing.assert_allclose(fuse_predictions(scores, [1, 0])[0], [0, 0.5, 1])
    np.testing.assert_allclose(fuse_predictions(scores, [0, 1])[0], [1, 0.5, 0])


def test_constant_branch_and_single_branch():
    np.testing.assert_allclose(
        fuse_predictions([[1, 1, 1], [0, 1, 2]], [0.5, 0.5])[0], [0, 0.25, 0.5]
    )
    np.testing.assert_allclose(fuse_predictions([[1, 2, 3]], [])[0], [0, 0.5, 1])


@pytest.mark.parametrize(
    "weights", [[-1, 2], [0, 0], [float("nan"), 1], [float("inf"), 1], [1]]
)
def test_bad_weights(weights):
    with pytest.raises(ValueError):
        normalize_weights(weights, 2)


def test_launch_cli_constructs_independent_layer_branches(monkeypatch, tmp_path):
    run = load_script("run_patchcore")
    config = json.loads(
        (Path(__file__).resolve().parents[1] / ".vscode/launch.json").read_text()
    )
    assert len(config["configurations"]) == 1
    args = list(config["configurations"][0]["args"])
    assert args.count("-b") == 2
    assert args[args.index("-b") + 1] == "wideresnet50"
    assert args.count("-le") == 2
    assert args[args.index("-le") + 1] == "0.layer2"
    assert args[args.index("-le", args.index("-le") + 1) + 1] == "1.layer3"
    assert args.count("-d") == 1
    assert "--layer_fusion_weight" not in args
    assert "--task_decoupled_ablation" in args
    args[-1] = str(tmp_path)
    args[args.index("--anomaly_scorer_num_nn") + 1] = "3"
    captured = {}

    class Backbone:
        pass

    class Model:
        def __init__(self, device):
            pass

        def load(self, **kwargs):
            self.configuration = kwargs

    monkeypatch.setattr(run.patchcore.backbones, "load", lambda name: Backbone())
    monkeypatch.setattr(run.patchcore.patchcore, "PatchCore", Model)
    monkeypatch.setattr(run.patchcore.common, "FaissNN", lambda *args: object())

    def callback(methods, **kwargs):
        captured["models"] = dict(methods)["get_patchcore"](
            (3, 224, 224), object(), "cpu"
        )

    run.main.result_callback(replace=True)(callback)
    # Exercise the actual launch argument chain, without reading the dataset.
    result = CliRunner().invoke(run.main, args)
    assert result.exit_code == 0, result.output
    models = captured["models"]
    assert len(models) == 2
    assert [list(m.configuration["layers_to_extract_from"]) for m in models] == [
        ["layer2"],
        ["layer3"],
    ]
    assert all(m.configuration["layer_fusion_weights"] is None for m in models)
    np.testing.assert_allclose([m.image_fusion_weight for m in models], [0.9, 0.1])
    np.testing.assert_allclose([m.pixel_fusion_weight for m in models], [0.5, 0.5])
    assert all(model.configuration["anomaly_score_num_nn"] == 3 for model in models)


@pytest.mark.parametrize(
    "config,expected_image,expected_pixel",
    [
        (None, [0.5, 0.5], [0.5, 0.5]),
        (
            {"weights": [0.7, 0.3], "normalization": "evaluation_set_minmax"},
            [0.7, 0.3],
            [0.7, 0.3],
        ),
        (
            {
                "version": 2,
                "image_weights": [0.9, 0.1],
                "pixel_weights": [0.5, 0.5],
                "normalization": "evaluation_set_minmax",
            },
            [0.9, 0.1],
            [0.5, 0.5],
        ),
    ],
)
def test_loader_restores_saved_weights(
    tmp_path, monkeypatch, config, expected_image, expected_pixel
):
    loader = load_script("load_and_evaluate_patchcore")
    for i in (1, 2):
        (tmp_path / ("Ensemble-%d-2_nnscorer_search_index.faiss" % i)).touch()
    if config is not None:
        (tmp_path / "fusion.json").write_text(json.dumps(config))

    class Model:
        def __init__(self, device):
            pass

        def load_from_path(self, **kwargs):
            self.prefix = kwargs.get("prepend")

    monkeypatch.setattr(loader.patchcore.patchcore, "PatchCore", Model)
    monkeypatch.setattr(loader.patchcore.common, "FaissNN", lambda *args: None)
    _, (factory, _) = loader.patch_core_loader.callback((str(tmp_path),), False, 1)
    models = next(factory("cpu"))
    np.testing.assert_allclose(
        [model.image_fusion_weight for model in models], expected_image
    )
    np.testing.assert_allclose(
        [model.pixel_fusion_weight for model in models], expected_pixel
    )
    assert [model.prefix for model in models] == ["Ensemble-1-2_", "Ensemble-2-2_"]


def test_shared_weight_fallback_and_independent_override():
    image, pixel = resolve_task_weights(2, shared=[0.7, 0.3], image=[9, 1])
    np.testing.assert_allclose(image, [0.9, 0.1])
    np.testing.assert_allclose(pixel, [0.7, 0.3])
    image, pixel = resolve_task_weights(2, shared=[0.7, 0.3], pixel=[1, 1])
    np.testing.assert_allclose(image, [0.7, 0.3])
    np.testing.assert_allclose(pixel, [0.5, 0.5])
    image, pixel = resolve_task_weights(1)
    np.testing.assert_allclose(image, [1])
    np.testing.assert_allclose(pixel, [1])


@pytest.mark.parametrize("task", ["image", "pixel"])
@pytest.mark.parametrize(
    "weights", [[1], [-1, 2], [0, 0], [float("nan"), 1], [float("inf"), 1]]
)
def test_reject_invalid_task_weights(task, weights):
    with pytest.raises(ValueError):
        resolve_task_weights(2, **{task: weights})


def test_reject_incomplete_saved_task_weights():
    with pytest.raises(ValueError, match="requires both"):
        restore_task_weights(
            {
                "version": 2,
                "normalization": "evaluation_set_minmax",
                "image_weights": [0.9, 0.1],
            },
            2,
        )


def test_training_routes_weights_to_correct_outputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch

    run = load_script("run_patchcore")
    raw_scores = [[10, 15, 20], [4, 2, 0]]
    raw_maps = np.array(
        [[[[0, 1]], [[2, 3]], [[4, 5]]], [[[5, 4]], [[3, 2]], [[1, 0]]]], dtype=float
    )
    models = []
    for i, (wi, wp) in enumerate(zip([0.9, 0.1], [0.5, 0.5])):
        models.append(
            SimpleNamespace(
                image_fusion_weight=wi,
                pixel_fusion_weight=wp,
                backbone=SimpleNamespace(name="tiny", seed=None),
                layers_to_extract_from=["layer2" if i == 0 else "layer3"],
                anomaly_scorer=SimpleNamespace(detection_features=np.zeros((3, 4))),
                fit=lambda data: None,
                predict=lambda data, i=i: (
                    raw_scores[i],
                    raw_maps[i],
                    [0, 1, 1],
                    np.ones((3, 1, 2)),
                ),
            )
        )
    dataset = SimpleNamespace(
        imagesize=(3, 1, 2), data_to_iterate=[(0, "good"), (1, "bad"), (2, "bad")]
    )
    loader = SimpleNamespace(name="mvtec_pill", dataset=dataset)
    monkeypatch.setattr(
        run.patchcore.utils, "set_torch_device", lambda _: torch.device("cpu")
    )
    seen = {}

    def image_metrics(scores, labels):
        seen["scores"] = scores
        return {"auroc": 0.5}

    def pixel_metrics(maps, masks):
        seen["maps"] = maps
        return {"auroc": 0.5}

    monkeypatch.setattr(
        run.patchcore.metrics, "compute_imagewise_retrieval_metrics", image_metrics
    )
    monkeypatch.setattr(
        run.patchcore.metrics, "compute_pixelwise_retrieval_metrics", pixel_metrics
    )
    run.run(
        [
            ("get_dataloaders", lambda seed: [{"training": loader, "testing": loader}]),
            ("get_sampler", lambda device: None),
            ("get_patchcore", lambda *args: models),
        ],
        str(tmp_path),
        [],
        0,
        "test",
        "project",
        False,
        False,
    )
    np.testing.assert_allclose(seen["scores"], [0.1, 0.5, 0.9])
    np.testing.assert_allclose(seen["maps"], np.full((3, 1, 2), 0.5))
    config = json.loads((tmp_path / "project/test/mvtec_pill_fusion.json").read_text())
    assert config["version"] == 2
    assert config["image_weights"] == [0.9, 0.1]
    assert config["pixel_weights"] == [0.5, 0.5]


def test_neighbour_count_and_predictions_survive_real_reload(tmp_path, monkeypatch):
    import copy
    from collections import OrderedDict

    import torch
    import patchcore.backbones
    from patchcore import common, patchcore as model_module

    torch.manual_seed(0)
    backbone = torch.nn.Sequential(
        OrderedDict([("layer2", torch.nn.Conv2d(3, 4, kernel_size=1))])
    )
    backbone.name = "tiny_test_backbone"
    saved_backbone = copy.deepcopy(backbone)
    monkeypatch.setattr(
        patchcore.backbones, "load", lambda _: copy.deepcopy(saved_backbone)
    )
    model = model_module.PatchCore(torch.device("cpu"))
    model.load(
        backbone,
        ["layer2"],
        torch.device("cpu"),
        (3, 8, 8),
        8,
        8,
        anomaly_score_num_nn=3,
        nn_method=common.FaissNN(False, 1),
    )
    model.fit(torch.utils.data.DataLoader(torch.rand(2, 3, 8, 8), batch_size=2))
    query = torch.rand(2, 3, 8, 8)
    expected_scores, expected_maps = model.predict(query)
    model.save_to_path(str(tmp_path))
    restored = model_module.PatchCore(torch.device("cpu"))
    restored.load_from_path(
        str(tmp_path), torch.device("cpu"), common.FaissNN(False, 1)
    )
    assert restored.anomaly_scorer.n_nearest_neighbours == 3
    scores, maps = restored.predict(query)
    np.testing.assert_allclose(scores, expected_scores)
    np.testing.assert_allclose(maps, expected_maps)
