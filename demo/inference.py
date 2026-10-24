"""Model loading and single-image inference for the PatchCore demo.

The module deliberately keeps Streamlit out of the inference path.  This makes
the same code usable from a notebook, a CLI wrapper, or a future HTTP service.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Keep ``python demo/app.py`` and direct unit-test imports working from a
# checkout that has not been installed as a package yet.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import patchcore.common
import patchcore.fusion
import patchcore.patchcore


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_DEMO_THRESHOLD = 0.5
DEFAULT_THRESHOLD_QUANTILE = 0.99
THRESHOLD_FILENAME = "threshold.json"
THRESHOLD_FILENAMES = {
    "original": "threshold_original.json",
    "fixed": THRESHOLD_FILENAME,
    "sensitivity": "threshold_sensitivity.json",
    "gate": "threshold_gate.json",
}
PORTABLE_MVTEC_ROOT = "datasets/mvtec"

FUSION_MODE_ORIGINAL = "original"
FUSION_MODE_FIXED = "fixed"
FUSION_MODE_SENSITIVITY = "sensitivity"
FUSION_MODE_GATE = "gate"
FUSION_MODE_LABELS = {
    FUSION_MODE_ORIGINAL: "原论文 PatchCore（layer2+layer3 单记忆库）",
    FUSION_MODE_FIXED: "固定融合基线",
    FUSION_MODE_SENSITIVITY: "敏感性收缩融合（实验）",
    FUSION_MODE_GATE: "风险受控门控融合（实验）",
}

_MVTEC_CATEGORIES = {
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
}


class DemoConfigurationError(ValueError):
    """Raised when a model directory is not a valid PatchCore model set."""


def classify_image_score(image_score: float, threshold: float) -> str:
    """Classify an image using the fused image-level anomaly score.

    The heatmap is independently normalized for visualization, so its peak is
    intentionally not used as an image-level decision score.
    """

    image_score = float(image_score)
    threshold = float(threshold)
    if not np.isfinite(image_score) or not np.isfinite(threshold):
        raise ValueError("image_score and threshold must be finite.")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    return "异常" if image_score >= threshold else "正常"


def infer_model_category(model_directory: str | os.PathLike[str]) -> Optional[str]:
    """Infer the product category from a saved MVTec model directory name."""

    directory_name = Path(model_directory).expanduser().name.lower()
    for prefix in ("mvtec_", "mvtec-"):
        if directory_name.startswith(prefix):
            category = directory_name[len(prefix) :]
            return category if category in _MVTEC_CATEGORIES else category or None
    if directory_name in _MVTEC_CATEGORIES:
        return directory_name
    return None


def display_category_name(category: Optional[str]) -> str:
    """Return a readable category label for the UI."""

    if not category:
        return "未识别类别"
    return category.replace("_", " ").title()


def describe_model_directory(model_directory: str | os.PathLike[str]) -> Dict[str, Any]:
    """Return lightweight model metadata without loading backbone weights."""

    path = Path(model_directory).expanduser()
    index_count = len(list(path.glob("*nnscorer_search_index.faiss"))) if path.is_dir() else 0
    category = infer_model_category(path)
    model_family = path.parent.parent.name if path.parent.name == "models" else path.parent.name
    fusion_config_path = (
        _find_fusion_config_path(path, branch_count=index_count)
        if path.is_dir() and index_count
        else None
    )
    has_fusion_config = fusion_config_path is not None
    has_threshold_config = (path / THRESHOLD_FILENAME).is_file() if path.is_dir() else False
    return {
        "path": str(path),
        "category": category,
        "category_label": display_category_name(category),
        "model_family": model_family,
        "branch_count": index_count,
        "has_fusion_config": has_fusion_config,
        "fusion_config_path": (
            str(fusion_config_path) if fusion_config_path is not None else None
        ),
        "has_threshold_config": has_threshold_config,
        "fusion_modes": describe_fusion_modes(path),
        "exists": path.is_dir(),
        "valid_structure": path.is_dir() and index_count > 0,
        "original_patchcore": _original_patchcore_compatibility(path)[0]
        if path.is_dir()
        else False,
    }


def _threshold_path(
    model_directory: str | os.PathLike[str], fusion_mode: str = FUSION_MODE_FIXED
) -> Path:
    filename = THRESHOLD_FILENAMES.get(fusion_mode, THRESHOLD_FILENAME)
    return Path(model_directory).expanduser() / filename


def _portable_calibration_root(calibration_root: Optional[Path]) -> str:
    """Return calibration metadata without leaking a machine-specific path.

    Calibration may use an external local MVTec directory such as
    ``E:/Datasets/MVTecAD/mvtec``.  That path is useful at runtime but is not
    portable metadata for a saved model.  The repository convention is
    ``datasets/mvtec``; the path is descriptive only and is not used to load
    an already saved threshold.
    """

    if calibration_root is None:
        return ""
    try:
        resolved_root = calibration_root.expanduser().resolve()
        project_root = _PROJECT_ROOT.resolve()
        relative_root = resolved_root.relative_to(project_root)
    except (OSError, ValueError):
        return PORTABLE_MVTEC_ROOT
    return relative_root.as_posix() or PORTABLE_MVTEC_ROOT


def load_saved_threshold(
    model_directory: str | os.PathLike[str],
    fusion_mode: str = FUSION_MODE_FIXED,
) -> Optional[Dict[str, Any]]:
    """Load a validated model-specific threshold configuration if present."""

    path = _threshold_path(model_directory, fusion_mode)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoConfigurationError(f"无法读取阈值配置 {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise DemoConfigurationError(f"阈值配置必须是 JSON 对象: {path}")
    try:
        threshold = float(config["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DemoConfigurationError(
            f"阈值配置缺少有效的 threshold 字段: {path}"
        ) from exc
    if not np.isfinite(threshold) or threshold < 0.0:
        raise DemoConfigurationError(f"阈值必须是非负有限数: {path}")
    quantile = float(config.get("quantile", DEFAULT_THRESHOLD_QUANTILE))
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise DemoConfigurationError(f"quantile 必须位于 [0, 1] 内: {path}")
    calibration_images = int(config.get("calibration_images", 0))
    if calibration_images < 0:
        raise DemoConfigurationError(f"calibration_images 不能为负数: {path}")
    return {
        "threshold": threshold,
        "source": str(config.get("source", "模型目录 threshold.json")),
        "quantile": quantile,
        "calibration_images": calibration_images,
        "calibration_root": str(config.get("calibration_root", "")),
        "fusion_mode": str(config.get("fusion_mode", fusion_mode)),
    }


def _candidate_mvtec_roots() -> List[Path]:
    roots: List[Path] = []
    configured_root = os.environ.get("PATCHCORE_MVTEC_ROOT")
    if configured_root:
        roots.append(Path(configured_root).expanduser())
    roots.extend(
        [
            _PROJECT_ROOT / "datasets" / "mvtec",
            Path("E:/Datasets/MVTecAD/mvtec"),
        ]
    )
    unique_roots: List[Path] = []
    seen = set()
    for root in roots:
        normalized = str(root.resolve()).lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_roots.append(root)
    return unique_roots


def find_normal_calibration_images(
    model_directory: str | os.PathLike[str], max_images: int = 100
) -> Tuple[List[Path], str, Optional[Path]]:
    """Find normal images without using anomaly labels for threshold fitting."""

    category = infer_model_category(model_directory)
    if not category:
        return [], "未识别类别，无法自动校准", None
    for root in _candidate_mvtec_roots():
        for split in ("val", "test", "train"):
            image_directory = root / category / split / "good"
            if not image_directory.is_dir():
                continue
            paths = sorted(
                path
                for path in image_directory.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            if paths:
                return (
                    paths[: int(max_images)],
                    f"正常校准集 {split}/good 的 99% 分位数",
                    root,
                )
    return [], "未找到正常校准集，使用演示回退值", None


@dataclass(frozen=True)
class BranchMetadata:
    """Metadata shown in the UI for one loaded PatchCore branch."""

    name: str
    backbone: str
    layers: Tuple[str, ...]
    image_weight: float
    pixel_weight: float


@dataclass
class PredictionResult:
    """All values needed for rendering one prediction."""

    image: Image.Image
    fused_map: np.ndarray
    branch_maps: List[np.ndarray]
    branch_scores: List[float]
    branch_map_peaks: List[float]
    fused_image_score: float
    fused_map_peak: float
    decision: str
    threshold: float
    elapsed_seconds: float
    branches: List[BranchMetadata]
    model_directory: str
    fusion_mode: str
    warnings: List[str]
    threshold_source: str = "手动指定"
    threshold_quantile: Optional[float] = None
    threshold_calibration_images: int = 0
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary without embedding image pixels."""

        return {
            "decision": self.decision,
            "fused_image_score": float(self.fused_image_score),
            "fused_map_peak": float(self.fused_map_peak),
            "threshold": float(self.threshold),
            "threshold_source": self.threshold_source,
            "threshold_quantile": (
                None
                if self.threshold_quantile is None
                else float(self.threshold_quantile)
            ),
            "threshold_calibration_images": int(self.threshold_calibration_images),
            "elapsed_seconds": float(self.elapsed_seconds),
            "model_directory": self.model_directory,
            "fusion_mode": self.fusion_mode,
            "fusion_metadata": self.fusion_metadata,
            "branches": [
                {
                    "name": branch.name,
                    "backbone": branch.backbone,
                    "layers": list(branch.layers),
                    "image_weight": branch.image_weight,
                    "pixel_weight": branch.pixel_weight,
                    "raw_score": float(score),
                    "map_peak": float(peak),
                }
                for branch, score, peak in zip(
                    self.branches, self.branch_scores, self.branch_map_peaks
                )
            ],
            "warnings": list(self.warnings),
        }


def resolve_device(device_name: str) -> torch.device:
    """Resolve a user-facing device choice without silently falling back."""

    normalized = str(device_name).strip().lower()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise DemoConfigurationError(
                "CUDA was requested, but torch.cuda.is_available() is false."
            )
        return torch.device("cuda")
    if normalized != "cpu":
        raise DemoConfigurationError("device_name must be either 'cpu' or 'cuda'.")
    return torch.device("cpu")


def _list_index_files(model_directory: Path) -> List[Path]:
    return sorted(model_directory.glob("*nnscorer_search_index.faiss"))


def _original_patchcore_compatibility(
    model_directory: str | os.PathLike[str],
) -> Tuple[bool, str]:
    """Check whether a directory contains the original single-bank model.

    The original PatchCore setup extracts layer2 and layer3 inside one
    PatchCore instance and stores one FAISS index.  A dual-branch directory
    with two indices is therefore deliberately rejected here, even if its
    fusion weights happen to be 1.0/1.0.
    """

    path = Path(model_directory).expanduser()
    index_files = _list_index_files(path) if path.is_dir() else []
    if len(index_files) != 1:
        return False, "原论文模式要求目录中恰好有一个 FAISS 记忆库。"
    params_path = path / "patchcore_params.pkl"
    if not params_path.is_file():
        return False, "原论文模式缺少 patchcore_params.pkl。"
    try:
        with params_path.open("rb") as handle:
            params = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError) as exc:
        return False, f"无法读取 PatchCore 参数文件：{exc}"
    layers = params.get("layers_to_extract_from")
    if list(layers or []) != ["layer2", "layer3"]:
        return (
            False,
            "原论文模式要求 layers_to_extract_from 严格为 ['layer2', 'layer3']。",
        )
    layer_weights = params.get("layer_fusion_weights")
    if layer_weights is not None and list(layer_weights) != []:
        return False, "该单库模型包含额外的层内加权参数，不作为原论文等权模式。"
    return True, "单个 FAISS 记忆库，层为 layer2+layer3。"


def validate_model_directory(model_directory: str | os.PathLike[str]) -> Path:
    """Validate a model directory before loading heavyweight backbones."""

    path = Path(model_directory).expanduser().resolve()
    if not path.is_dir():
        raise DemoConfigurationError(f"Model directory does not exist: {path}")
    index_files = _list_index_files(path)
    if not index_files:
        raise DemoConfigurationError(
            "No '*nnscorer_search_index.faiss' file was found in "
            f"{path}. Select a saved PatchCore model directory."
        )
    for index_file in index_files:
        prefix = _prefix_for_index(index_file, len(index_files))
        params = path / f"{prefix}patchcore_params.pkl"
        if not params.exists():
            raise DemoConfigurationError(
                f"Missing parameter file for {index_file.name}: {params.name}"
            )
    return path


def _prefix_for_index(index_file: Path, branch_count: int) -> str:
    suffix = "nnscorer_search_index.faiss"
    if branch_count == 1:
        return ""
    if not index_file.name.endswith(suffix):
        raise DemoConfigurationError(f"Unexpected index filename: {index_file.name}")
    return index_file.name[: -len(suffix)]


def _read_fusion_config(config_path: Path) -> Dict[str, Any]:
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoConfigurationError(f"Could not read {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise DemoConfigurationError(f"Fusion config must be a JSON object: {config_path}")
    return config


def _fusion_config_branch_count(config: Dict[str, Any]) -> Optional[int]:
    """Infer how many saved branches a fusion config belongs to."""

    branches = config.get("branches")
    if isinstance(branches, list) and branches:
        return len(branches)
    variants = config.get("variants")
    if isinstance(variants, dict):
        for variant in variants.values():
            if not isinstance(variant, dict):
                continue
            for key in ("image_weights", "pixel_weights"):
                weights = variant.get(key)
                if isinstance(weights, list) and weights:
                    return len(weights)
    for key in ("base_image_weights", "base_pixel_weights"):
        weights = config.get(key)
        if isinstance(weights, list) and weights:
            return len(weights)
    return None


def _fusion_config_candidates(model_directory: Path) -> List[Path]:
    """Return direct and archived fusion configs for the same MVTec category."""

    candidates: List[Path] = []
    direct = model_directory / "fusion.json"
    if direct.is_file():
        candidates.append(direct)

    category = infer_model_category(model_directory)
    # Only associate archived Demo outputs, not arbitrary experiment tables
    # under results/, whose configs may use a different protocol or split.
    results_root = _PROJECT_ROOT / "results" / "demo"
    # Restrict this convenience lookup to the repository's conventional
    # ``models/<family>/models/<category>`` layout.  Without this boundary, a
    # temporary test directory with the same category name can accidentally
    # consume a real repository config.
    conventional_layout = (
        model_directory.parent.name.lower() == "models"
        and model_directory.parent.parent.parent.name.lower() == "models"
    )
    in_demo_results = results_root in model_directory.parents
    if (
        category
        and results_root.is_dir()
        and not candidates
        and (conventional_layout or in_demo_results)
    ):
        expected_directory = f"mvtec_{category}"
        for candidate in results_root.rglob("fusion.json"):
            if candidate.parent.name.lower() == expected_directory:
                candidates.append(candidate)
        for candidate in results_root.rglob(f"mvtec_{category}_fusion.json"):
            if candidate.is_file():
                candidates.append(candidate)

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _find_fusion_config_path(
    model_directory: Path, branch_count: Optional[int] = None
) -> Optional[Path]:
    """Find a compatible config without applying it to a wrong model."""

    candidates = _fusion_config_candidates(model_directory)
    if not candidates:
        return None
    direct = (model_directory / "fusion.json").resolve()
    if candidates[0] == direct:
        config = _read_fusion_config(candidates[0])
        declared_count = _fusion_config_branch_count(config)
        if branch_count is None or declared_count is None or declared_count == branch_count:
            return candidates[0]
        return None

    compatible: List[Path] = []
    for candidate in candidates:
        try:
            config = _read_fusion_config(candidate)
        except DemoConfigurationError:
            continue
        declared_count = _fusion_config_branch_count(config)
        if branch_count is None or declared_count == branch_count:
            compatible.append(candidate)
    if not compatible:
        return None
    return max(compatible, key=lambda item: item.stat().st_mtime)


def _load_fusion_config(
    model_directory: Path, branch_count: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    config_path = _find_fusion_config_path(model_directory, branch_count=branch_count)
    if config_path is None:
        return None
    return _read_fusion_config(config_path)


def _fusion_variants(config: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not config:
        return {}
    variants = config.get("variants")
    if isinstance(variants, dict):
        return {
            str(name): value
            for name, value in variants.items()
            if isinstance(value, dict)
        }
    return {"saved": config}


def _independent_layer_branch_config(
    config: Optional[Dict[str, Any]], branch_count: int
) -> bool:
    """Check that a config belongs to the independent layer2/layer3 setup."""

    if not config:
        return False
    branches = config.get("branches")
    if not isinstance(branches, list) or len(branches) != branch_count:
        return False
    layer_lists = [
        tuple(branch.get("layers", ()))
        for branch in branches
        if isinstance(branch, dict)
    ]
    return layer_lists == [("layer2",), ("layer3",)]


def _variant_weights(
    variant: Dict[str, Any], branch_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        image = patchcore.fusion.normalize_weights(
            variant["image_weights"], branch_count, "saved image weights"
        )
        pixel = patchcore.fusion.normalize_weights(
            variant["pixel_weights"], branch_count, "saved pixel weights"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DemoConfigurationError(f"fusion.json 中的权重配置无效：{exc}") from exc
    return np.asarray(image), np.asarray(pixel)


def _experimental_variant_name(
    config: Optional[Dict[str, Any]], requested_mode: str, branch_count: int
) -> str:
    if not config:
        raise DemoConfigurationError(
            f"当前模型没有 fusion.json，无法启用{FUSION_MODE_LABELS[requested_mode]}。"
            "请先用独立 layer2/layer3 双分支实验生成配置。"
        )
    if not _independent_layer_branch_config(config, branch_count):
        raise DemoConfigurationError(
            f"当前 fusion.json 不属于独立 layer2/layer3 双分支模型，"
            f"不能启用{FUSION_MODE_LABELS[requested_mode]}。"
        )
    variants = _fusion_variants(config)
    if requested_mode == FUSION_MODE_GATE:
        candidates = ("gate_selected",)
    else:
        candidates = (
            "image_fixed_pixel_sensitivity_shrinkage",
            "image_fixed_pixel_sensitivity",
            "sensitivity_only",
            "sensitivity",
            "adaptive",
        )
    for candidate in candidates:
        if candidate in variants:
            return candidate
    expected = "、".join(candidates)
    raise DemoConfigurationError(
        f"当前 fusion.json 缺少{FUSION_MODE_LABELS[requested_mode]}所需的配置项。"
        f"至少需要以下变体之一：{expected}。"
    )


def describe_fusion_modes(
    model_directory: str | os.PathLike[str],
) -> Dict[str, Dict[str, Any]]:
    """Describe which fusion modes can be selected for a saved model."""

    path = Path(model_directory).expanduser()
    branch_count = (
        len(list(path.glob("*nnscorer_search_index.faiss")))
        if path.is_dir()
        else 0
    )
    modes: Dict[str, Dict[str, Any]] = {
        FUSION_MODE_ORIGINAL: {
            "available": False,
            "label": FUSION_MODE_LABELS[FUSION_MODE_ORIGINAL],
            "reason": "需要一个同时提取 layer2 和 layer3 的单记忆库模型。",
            "variant": "single_memory_bank",
        },
        FUSION_MODE_FIXED: {
            "available": bool(path.is_dir() and branch_count > 0),
            "label": FUSION_MODE_LABELS[FUSION_MODE_FIXED],
            "reason": "使用模型保存的 fixed 配置；没有配置时使用等权回退。",
            "variant": "fixed",
        },
        FUSION_MODE_SENSITIVITY: {
            "available": False,
            "label": FUSION_MODE_LABELS[FUSION_MODE_SENSITIVITY],
            "reason": "需要自动找到与当前模型分支数兼容的独立 layer2/layer3 fusion.json。",
            "variant": None,
        },
        FUSION_MODE_GATE: {
            "available": False,
            "label": FUSION_MODE_LABELS[FUSION_MODE_GATE],
            "reason": "需要自动找到与当前模型分支数兼容的独立 layer2/layer3 门控 fusion.json。",
            "variant": None,
        },
    }
    if not path.is_dir() or branch_count == 0:
        modes[FUSION_MODE_FIXED]["reason"] = "当前目录不是可用的 PatchCore 模型目录。"
        return modes
    original_available, original_reason = _original_patchcore_compatibility(path)
    modes[FUSION_MODE_ORIGINAL]["available"] = original_available
    modes[FUSION_MODE_ORIGINAL]["reason"] = original_reason
    try:
        config = _load_fusion_config(path, branch_count=branch_count)
    except DemoConfigurationError as exc:
        for mode in (FUSION_MODE_SENSITIVITY, FUSION_MODE_GATE):
            modes[mode]["reason"] = str(exc)
        return modes
    for mode in (FUSION_MODE_SENSITIVITY, FUSION_MODE_GATE):
        try:
            variant = _experimental_variant_name(config, mode, branch_count)
        except DemoConfigurationError as exc:
            modes[mode]["reason"] = str(exc)
        else:
            modes[mode]["available"] = True
            modes[mode]["variant"] = variant
            modes[mode]["reason"] = f"可加载 fusion.json 的 {variant} 配置。"
    return modes


def _weights(
    config: Optional[Dict[str, Any]], branch_count: int, requested_mode: str
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any], List[str]]:
    if requested_mode not in FUSION_MODE_LABELS:
        raise DemoConfigurationError(f"未知推理模式：{requested_mode}")
    warnings: List[str] = []
    metadata: Dict[str, Any] = {
        "requested_mode": requested_mode,
        "requested_mode_label": FUSION_MODE_LABELS[requested_mode],
        "config_present": config is not None,
        "selected_variant": None,
        "gate": None,
    }
    if requested_mode == FUSION_MODE_ORIGINAL:
        # Structural validation is performed by PatchCoreDemoEngine and the
        # mode availability panel.  This branch intentionally does not read
        # fusion.json: the single PatchCore model already owns both layers and
        # emits one image score plus one anomaly map.
        weights = np.ones(branch_count, dtype=np.float64)
        metadata.update(
            {
                "selected_variant": "single_memory_bank",
                "fusion_method": "original_patchcore_single_memory_bank",
                "config_present": False,
                "layer_fusion": "layer2+layer3 inside one PatchCore memory bank",
            }
        )
        return (
            weights,
            weights.copy(),
            FUSION_MODE_LABELS[FUSION_MODE_ORIGINAL],
            metadata,
            warnings,
        )
    if not config:
        if requested_mode != FUSION_MODE_FIXED:
            _experimental_variant_name(config, requested_mode, branch_count)
        weights = np.full(branch_count, 1.0 / branch_count, dtype=np.float64)
        warnings.append(
            f"当前模型未提供 fusion.json，使用 {branch_count} 个模型分支等权融合 "
            f"（每个分支权重 {1.0 / branch_count:.3f}）。"
            "这是当前保存模型的默认回退，不是加载错误。"
        )
        metadata["fallback_reason"] = "fusion.json 不存在"
        return weights, weights.copy(), "固定融合基线（等权回退）", metadata, warnings

    variants = _fusion_variants(config)
    if requested_mode == FUSION_MODE_FIXED:
        if "fixed" in variants:
            variant_name = "fixed"
            image, pixel = _variant_weights(variants[variant_name], branch_count)
        elif config.get("base_image_weights") is not None:
            variant_name = "base"
            image = patchcore.fusion.normalize_weights(
                config["base_image_weights"], branch_count, "base image weights"
            )
            pixel = patchcore.fusion.normalize_weights(
                config.get("base_pixel_weights", config["base_image_weights"]),
                branch_count,
                "base pixel weights",
            )
        else:
            weights = np.full(branch_count, 1.0 / branch_count, dtype=np.float64)
            image, pixel = weights, weights.copy()
            variant_name = "equal-weight fallback"
            warnings.append(
                "fusion.json 未提供 fixed 或 base 权重，固定模式使用等权回退。"
            )
        display_name = (
            "固定融合基线"
            if variant_name != "equal-weight fallback"
            else "固定融合基线（等权回退）"
        )
    else:
        variant_name = _experimental_variant_name(config, requested_mode, branch_count)
        image, pixel = _variant_weights(variants[variant_name], branch_count)
        display_name = FUSION_MODE_LABELS[requested_mode]
        if requested_mode == FUSION_MODE_GATE:
            gate_record = variants[variant_name].get("adaptive_diagnostics")
            gate = gate_record.get("gate") if isinstance(gate_record, dict) else None
            if not isinstance(gate, dict):
                raise DemoConfigurationError(
                    "gate_selected 配置缺少门控诊断，无法解释门控结果。"
                )
            metadata["gate"] = gate
            selected_variant = gate_record.get("selected_variant")
            metadata["gate_selected_variant"] = selected_variant
            display_name = (
                f"{display_name} → "
                f"{'敏感性收缩' if gate.get('use_shrinkage') else '固定回退'}"
            )
    metadata["selected_variant"] = variant_name
    metadata["fusion_method"] = config.get("fusion_method")
    metadata["base_image_weights"] = config.get("base_image_weights")
    metadata["base_pixel_weights"] = config.get("base_pixel_weights")
    if config.get("normalization") == "evaluation_set_minmax":
        warnings.append(
            "The saved experiment used evaluation-set min-max normalization. "
            "The single-image demo uses display normalization instead."
        )
    return np.asarray(image), np.asarray(pixel), display_name, metadata, warnings


class PatchCoreDemoEngine:
    """Load saved PatchCore branches and run single-image inference."""

    def __init__(
        self,
        model_directory: str | os.PathLike[str],
        device: str = "cpu",
        fusion_mode: str = FUSION_MODE_FIXED,
    ):
        self.model_directory = validate_model_directory(model_directory)
        self.device = resolve_device(device)
        self.requested_fusion_mode = fusion_mode
        index_files = _list_index_files(self.model_directory)
        if fusion_mode == FUSION_MODE_ORIGINAL:
            compatible, reason = _original_patchcore_compatibility(self.model_directory)
            if not compatible:
                raise DemoConfigurationError(
                    f"当前模型不能使用{FUSION_MODE_LABELS[FUSION_MODE_ORIGINAL]}：{reason}"
                )
        if fusion_mode == FUSION_MODE_ORIGINAL:
            self.config_path = None
            self.config = None
        else:
            self.config_path = _find_fusion_config_path(
                self.model_directory, branch_count=len(index_files)
            )
            self.config = _load_fusion_config(
                self.model_directory, branch_count=len(index_files)
            )
        (
            self.image_weights,
            self.pixel_weights,
            self.fusion_mode,
            self.fusion_metadata,
            self.warnings,
        ) = _weights(
            self.config, len(index_files), fusion_mode
        )
        self.fusion_metadata["config_path"] = (
            str(self.config_path) if self.config_path is not None else None
        )
        self.models: List[patchcore.patchcore.PatchCore] = []
        self.branches: List[BranchMetadata] = []
        self._load_models(index_files)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.threshold_config = self._load_or_calibrate_threshold(fusion_mode)
        self.auto_threshold = float(self.threshold_config["threshold"])

    def _load_models(self, index_files: Sequence[Path]) -> None:
        branch_count = len(index_files)
        for index, index_file in enumerate(index_files):
            prepend = _prefix_for_index(index_file, branch_count)
            model = patchcore.patchcore.PatchCore(self.device)
            model.load_from_path(
                load_path=str(self.model_directory),
                device=self.device,
                nn_method=patchcore.common.FaissNN(False, 4),
                prepend=prepend,
            )
            self.models.append(model)
            self.branches.append(
                BranchMetadata(
                    name=(
                        "原论文单记忆库"
                        if self.requested_fusion_mode == FUSION_MODE_ORIGINAL
                        else prepend.rstrip("_") or f"branch-{index + 1}"
                    ),
                    backbone=str(model.backbone.name),
                    layers=tuple(model.layers_to_extract_from),
                    image_weight=float(self.image_weights[index]),
                    pixel_weight=float(self.pixel_weights[index]),
                )
            )

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image instance.")
        return self.transform(image.convert("RGB")).unsqueeze(0)

    def _predict_image_score(self, image: Image.Image) -> float:
        """Return only the fused image-level score for threshold calibration."""

        tensor = self.preprocess(image)
        branch_scores: List[float] = []
        with torch.no_grad():
            for model in self.models:
                scores, _ = model.predict(tensor)
                branch_scores.append(float(np.asarray(scores[0]).reshape(-1)[0]))
        return float(np.dot(self.image_weights, branch_scores))

    def _load_or_calibrate_threshold(
        self, fusion_mode: str = FUSION_MODE_FIXED
    ) -> Dict[str, Any]:
        """Load a saved threshold or calibrate one from normal images."""

        saved = load_saved_threshold(self.model_directory, fusion_mode)
        if saved is not None:
            return saved

        image_paths, source, calibration_root = find_normal_calibration_images(
            self.model_directory
        )
        if not image_paths:
            self.warnings.append(
                f"{source}；当前使用演示回退阈值 {DEFAULT_DEMO_THRESHOLD:.2f}。"
            )
            return {
                "threshold": DEFAULT_DEMO_THRESHOLD,
                "source": "未找到正常校准集，使用演示回退值",
                "quantile": DEFAULT_THRESHOLD_QUANTILE,
                "calibration_images": 0,
                "calibration_root": "",
                "fusion_mode": fusion_mode,
            }

        scores: List[float] = []
        failed_images = 0
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    scores.append(self._predict_image_score(image.convert("RGB")))
            except (OSError, RuntimeError, ValueError):
                failed_images += 1

        if not scores:
            self.warnings.append(
                "正常校准图片均无法完成推理；当前使用演示回退阈值 "
                f"{DEFAULT_DEMO_THRESHOLD:.2f}。"
            )
            return {
                "threshold": DEFAULT_DEMO_THRESHOLD,
                "source": "正常校准集不可用，使用演示回退值",
                "quantile": DEFAULT_THRESHOLD_QUANTILE,
                "calibration_images": 0,
                "calibration_root": _portable_calibration_root(calibration_root),
                "fusion_mode": fusion_mode,
            }

        threshold = float(np.quantile(scores, DEFAULT_THRESHOLD_QUANTILE))
        config: Dict[str, Any] = {
            "threshold": threshold,
            "source": source,
            "quantile": DEFAULT_THRESHOLD_QUANTILE,
            "calibration_images": len(scores),
            "calibration_root": _portable_calibration_root(calibration_root),
            "category": infer_model_category(self.model_directory),
            "model_family": self.model_directory.parent.parent.name,
            "fusion_mode": self.fusion_mode,
        }
        try:
            with _threshold_path(self.model_directory, fusion_mode).open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(config, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            self.warnings.append(f"阈值已在内存中校准，但无法保存 threshold.json：{exc}")
        if failed_images:
            self.warnings.append(
                f"阈值校准跳过了 {failed_images} 张无法读取或推理的正常图片。"
            )
        return config

    @staticmethod
    def _display_normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        lower = float(np.percentile(values, 1.0))
        upper = float(np.percentile(values, 99.0))
        if upper <= lower:
            lower = float(values.min())
            upper = float(values.max())
        if upper <= lower:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values - lower) / (upper - lower), 0.0, 1.0).astype(
            np.float32
        )

    def predict(
        self, image: Image.Image, threshold: Optional[float] = None
    ) -> PredictionResult:
        threshold_is_override = threshold is not None
        used_threshold = self.auto_threshold if threshold is None else float(threshold)
        if used_threshold < 0.0:
            raise ValueError("threshold must be non-negative.")
        image = image.convert("RGB")
        tensor = self.preprocess(image)
        branch_maps: List[np.ndarray] = []
        branch_scores: List[float] = []
        started = time.perf_counter()
        with torch.no_grad():
            for model in self.models:
                # PatchCore returns two values for a Tensor input. The
                # four-value form is only used by the DataLoader path.
                scores, maps = model.predict(tensor)
                branch_scores.append(float(np.asarray(scores[0]).reshape(-1)[0]))
                branch_maps.append(self._display_normalize(np.asarray(maps[0])))
        fused_map = np.sum(
            np.asarray(branch_maps) * self.pixel_weights[:, None, None], axis=0
        ).astype(np.float32)
        fused_map = np.clip(fused_map, 0.0, 1.0)
        branch_map_peaks = [float(np.max(item)) for item in branch_maps]
        fused_image_score = float(np.dot(self.image_weights, branch_scores))
        fused_map_peak = float(np.max(fused_map))
        decision = classify_image_score(fused_image_score, used_threshold)
        elapsed = time.perf_counter() - started
        result_warnings = list(self.warnings)
        if threshold_is_override:
            result_warnings.append("当前使用手动阈值覆盖自动校准阈值，仅用于实验。")
        return PredictionResult(
            image=image,
            fused_map=fused_map,
            branch_maps=branch_maps,
            branch_scores=branch_scores,
            branch_map_peaks=branch_map_peaks,
            fused_image_score=fused_image_score,
            fused_map_peak=fused_map_peak,
            decision=decision,
            threshold=used_threshold,
            elapsed_seconds=elapsed,
            branches=list(self.branches),
            model_directory=str(self.model_directory),
            fusion_mode=self.fusion_mode,
            warnings=result_warnings,
            threshold_source=(
                "手动覆盖（实验）"
                if threshold_is_override
                else str(self.threshold_config["source"])
            ),
            threshold_quantile=(
                None
                if threshold_is_override
                else float(self.threshold_config["quantile"])
            ),
            threshold_calibration_images=(
                0
                if threshold_is_override
                else int(self.threshold_config["calibration_images"])
            ),
            fusion_metadata=dict(self.fusion_metadata),
        )


def iter_saved_model_directories(models_root: str | os.PathLike[str]) -> Iterable[Path]:
    """Yield valid-looking saved model directories for a UI selector."""

    root = Path(models_root).expanduser()
    if not root.exists():
        return []
    candidates = set()
    for index_file in root.rglob("*nnscorer_search_index.faiss"):
        candidates.add(index_file.parent)
    return sorted(candidates)

