import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from patchcore.fusion import (
    fuse_predictions,
    normalize_weights,
    resolve_task_weights,
    restore_task_weights,
)


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


@pytest.mark.parametrize(
    "image_weights,pixel_weights",
    [
        ((0.9, 0.1), (0.5, 0.5)),
        ((0.7, 0.3), (0.3, 0.7)),
        ((1.0, 0.0), (0.0, 1.0)),
    ],
)
def test_launch_cli_constructs_independent_branches(
    monkeypatch, tmp_path, image_weights, pixel_weights
):
    run = load_script("run_patchcore")
    config = json.loads(
        (Path(__file__).resolve().parents[1] / ".vscode/launch.json").read_text()
    )
    assert len(config["configurations"]) == 1
    args = list(config["configurations"][0]["args"])
    # Exercise editable weights without relying on the user's current values
    # or requiring their dataset path to exist on another developer's computer.
    for option, weights in [
        ("--image_fusion_weight", image_weights),
        ("--pixel_fusion_weight", pixel_weights),
    ]:
        positions = [i + 1 for i, value in enumerate(args) if value == option]
        assert len(positions) == 2
        for position, weight in zip(positions, weights):
            args[position] = str(weight)
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
    assert models[0].configuration["layers_to_extract_from"] == ["layer2"]
    assert models[1].configuration["layers_to_extract_from"] == ["layer3"]
    assert (
        models[0].configuration["backbone"] is not models[1].configuration["backbone"]
    )
    np.testing.assert_allclose(
        [model.image_fusion_weight for model in models], image_weights
    )
    np.testing.assert_allclose(
        [model.pixel_fusion_weight for model in models], pixel_weights
    )
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
