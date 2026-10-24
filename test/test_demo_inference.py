import json
import pickle

import numpy as np
import pytest
from PIL import Image

from demo.inference import (
    DemoConfigurationError,
    FUSION_MODE_ORIGINAL,
    FUSION_MODE_FIXED,
    FUSION_MODE_GATE,
    FUSION_MODE_SENSITIVITY,
    PatchCoreDemoEngine,
    PredictionResult,
    classify_image_score,
    describe_fusion_modes,
    describe_model_directory,
    display_category_name,
    infer_model_category,
    load_saved_threshold,
    resolve_device,
    validate_model_directory,
)
import demo.inference as inference_module
from demo.visualization import overlay_anomaly_map


def test_resolve_cpu_device():
    assert resolve_device("cpu").type == "cpu"


def test_image_decision_uses_raw_image_score_not_heatmap_peak():
    assert classify_image_score(0.21859, 0.5) == "正常"
    assert classify_image_score(0.95000, 0.5) == "异常"


def test_image_decision_rejects_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        classify_image_score(0.2, -0.1)


def test_resolve_device_rejects_unknown_value():
    with pytest.raises(DemoConfigurationError):
        resolve_device("tpu")


def test_validate_model_directory_rejects_missing_directory(tmp_path):
    with pytest.raises(DemoConfigurationError, match="does not exist"):
        validate_model_directory(tmp_path / "missing")


def test_validate_model_directory_checks_parameter_file(tmp_path):
    (tmp_path / "branch_nnscorer_search_index.faiss").touch()
    with pytest.raises(DemoConfigurationError, match="Missing parameter file"):
        validate_model_directory(tmp_path)


def test_model_category_is_inferred_from_saved_directory_name(tmp_path):
    model_directory = tmp_path / "model_set" / "models" / "mvtec_metal_nut"
    model_directory.mkdir(parents=True)
    (model_directory / "branch_nnscorer_search_index.faiss").touch()
    assert infer_model_category(model_directory) == "metal_nut"
    assert display_category_name("metal_nut") == "Metal Nut"
    metadata = describe_model_directory(model_directory)
    assert metadata["category_label"] == "Metal Nut"
    assert metadata["branch_count"] == 1
    assert not metadata["has_fusion_config"]
    assert metadata["valid_structure"]


def test_original_patchcore_mode_requires_one_equal_single_memory_bank(tmp_path):
    model_directory = tmp_path / "models" / "mvtec_bottle"
    model_directory.mkdir(parents=True)
    (model_directory / "nnscorer_search_index.faiss").touch()
    with (model_directory / "patchcore_params.pkl").open("wb") as handle:
        pickle.dump(
            {
                "backbone.name": "wideresnet50",
                "layers_to_extract_from": ["layer2", "layer3"],
                "layer_fusion_weights": None,
            },
            handle,
        )

    metadata = describe_model_directory(model_directory)
    assert metadata["original_patchcore"]
    modes = describe_fusion_modes(model_directory)
    assert modes[FUSION_MODE_ORIGINAL]["available"]
    assert modes[FUSION_MODE_FIXED]["available"]
    assert not modes[FUSION_MODE_SENSITIVITY]["available"]
    assert not modes[FUSION_MODE_GATE]["available"]


def test_original_patchcore_mode_rejects_dual_branch_directory(tmp_path):
    model_directory = tmp_path / "models" / "mvtec_bottle"
    model_directory.mkdir(parents=True)
    for prefix in ("Ensemble-1-2_", "Ensemble-2-2_"):
        (model_directory / f"{prefix}nnscorer_search_index.faiss").touch()
        with (model_directory / f"{prefix}patchcore_params.pkl").open("wb") as handle:
            pickle.dump(
                {"layers_to_extract_from": ["layer2"]},
                handle,
            )

    modes = describe_fusion_modes(model_directory)
    assert not modes[FUSION_MODE_ORIGINAL]["available"]
    assert "一个 FAISS 记忆库" in modes[FUSION_MODE_ORIGINAL]["reason"]


def test_saved_threshold_is_loaded_and_validated(tmp_path):
    model_directory = tmp_path / "models" / "mvtec_tile"
    model_directory.mkdir(parents=True)
    (model_directory / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": 0.42,
                "source": "正常校准集 train/good 的 99% 分位数",
                "quantile": 0.99,
                "calibration_images": 20,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = load_saved_threshold(model_directory)
    assert config is not None
    assert config["threshold"] == pytest.approx(0.42)
    assert config["quantile"] == pytest.approx(0.99)
    assert config["calibration_images"] == 20


def _independent_fusion_config():
    return {
        "version": 2,
        "fusion_method": "task_decoupled_sensitivity_pixel_shrinkage",
        "base_image_weights": [0.9, 0.1],
        "base_pixel_weights": [0.5, 0.5],
        "branches": [
            {"layers": ["layer2"]},
            {"layers": ["layer3"]},
        ],
        "variants": {
            "fixed": {
                "image_weights": [0.9, 0.1],
                "pixel_weights": [0.5, 0.5],
            },
            "image_fixed_pixel_sensitivity_shrinkage": {
                "image_weights": [0.9, 0.1],
                "pixel_weights": [0.6, 0.4],
                "adaptive_diagnostics": {
                    "pixel": {"weights": [0.6, 0.4]}
                },
            },
            "gate_selected": {
                "image_weights": [0.9, 0.1],
                "pixel_weights": [0.6, 0.4],
                "adaptive_diagnostics": {
                    "gate": {
                        "use_shrinkage": True,
                        "reason": "synthetic groups agree",
                    },
                    "selected_variant": "image_fixed_pixel_sensitivity_shrinkage",
                },
            },
        },
    }


def test_fusion_mode_selection_requires_compatible_saved_config(tmp_path, monkeypatch):
    # Keep this test independent from archived fusion configs in the repository.
    monkeypatch.setattr(inference_module, "_PROJECT_ROOT", tmp_path)
    model_directory = tmp_path / "models" / "mvtec_tile"
    model_directory.mkdir(parents=True)
    (model_directory / "layer2_nnscorer_search_index.faiss").touch()
    (model_directory / "layer3_nnscorer_search_index.faiss").touch()

    modes = describe_fusion_modes(model_directory)
    assert modes[FUSION_MODE_FIXED]["available"]
    assert not modes[FUSION_MODE_SENSITIVITY]["available"]
    assert not modes[FUSION_MODE_GATE]["available"]

    (model_directory / "fusion.json").write_text(
        json.dumps(_independent_fusion_config()), encoding="utf-8"
    )
    modes = describe_fusion_modes(model_directory)
    assert modes[FUSION_MODE_SENSITIVITY]["available"]
    assert modes[FUSION_MODE_GATE]["available"]


def test_fusion_config_is_auto_discovered_from_saved_demo_output(tmp_path, monkeypatch):
    model_directory = tmp_path / "models" / "family" / "models" / "mvtec_bottle"
    model_directory.mkdir(parents=True)
    for prefix in ("layer2_", "layer3_"):
        (model_directory / f"{prefix}nnscorer_search_index.faiss").touch()
    demo_directory = tmp_path / "results" / "demo" / "run" / "models" / "mvtec_bottle"
    demo_directory.mkdir(parents=True)
    (demo_directory / "fusion.json").write_text(
        json.dumps(_independent_fusion_config()), encoding="utf-8"
    )
    monkeypatch.setattr(inference_module, "_PROJECT_ROOT", tmp_path)

    metadata = describe_model_directory(model_directory)
    assert metadata["has_fusion_config"]
    assert metadata["fusion_config_path"].endswith("mvtec_bottle\\fusion.json")
    modes = describe_fusion_modes(model_directory)
    assert modes[FUSION_MODE_SENSITIVITY]["available"]
    assert modes[FUSION_MODE_GATE]["available"]


def test_fusion_weights_use_requested_mode_without_silent_fallback():
    config = _independent_fusion_config()
    image, pixel, label, metadata, _ = inference_module._weights(
        config, 2, FUSION_MODE_SENSITIVITY
    )
    np.testing.assert_allclose(image, [0.9, 0.1])
    np.testing.assert_allclose(pixel, [0.6, 0.4])
    assert "敏感性收缩融合" in label
    assert metadata["selected_variant"] == "image_fixed_pixel_sensitivity_shrinkage"

    with pytest.raises(DemoConfigurationError, match="没有 fusion.json"):
        inference_module._weights(None, 2, FUSION_MODE_GATE)


def test_threshold_calibration_uses_normal_scores_and_persists_config(
    tmp_path, monkeypatch
):
    model_directory = tmp_path / "models" / "mvtec_tile"
    model_directory.mkdir(parents=True)
    normal_directory = tmp_path / "mvtec" / "tile" / "test" / "good"
    normal_directory.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (8, 8), color=(index, index, index)).save(
            normal_directory / f"{index}.png"
        )

    monkeypatch.setattr(
        inference_module, "_candidate_mvtec_roots", lambda: [tmp_path / "mvtec"]
    )
    engine = object.__new__(PatchCoreDemoEngine)
    engine.model_directory = model_directory
    engine.fusion_mode = "fixed"
    engine.warnings = []
    scores = iter([0.1, 0.2, 0.3])
    engine._predict_image_score = lambda image: next(scores)

    config = engine._load_or_calibrate_threshold()

    assert config["threshold"] == pytest.approx(np.quantile([0.1, 0.2, 0.3], 0.99))
    assert config["calibration_images"] == 3
    assert config["calibration_root"] == "datasets/mvtec"
    assert load_saved_threshold(model_directory)["threshold"] == pytest.approx(
        config["threshold"]
    )
    assert load_saved_threshold(model_directory)["calibration_root"] == "datasets/mvtec"


def test_prediction_result_json_is_serializable():
    result = PredictionResult(
        image=Image.new("RGB", (4, 4)),
        fused_map=np.zeros((4, 4), dtype=np.float32),
        branch_maps=[],
        branch_scores=[],
        branch_map_peaks=[],
        fused_image_score=0.2,
        fused_map_peak=0.25,
        decision="正常",
        threshold=0.5,
        elapsed_seconds=0.01,
        branches=[],
        model_directory="models/example",
        fusion_mode="equal-weight fallback",
        warnings=[],
    )
    json.dumps(result.as_json(), ensure_ascii=False)


@pytest.mark.parametrize("map_shape", [(224, 224), (320, 320), (240, 320)])
def test_overlay_matches_model_heatmap_shape(map_shape):
    overlay = overlay_anomaly_map(Image.new("RGB", (640, 480)), np.zeros(map_shape))
    assert overlay.size == (map_shape[1], map_shape[0])

