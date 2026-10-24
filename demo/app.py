"""Streamlit front-end for the PatchCore industrial anomaly demo."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
# Streamlit executes ``demo/app.py`` as a script. In that mode Python may put
# only the ``demo`` directory on sys.path, so the sibling package import below
# needs the repository root explicitly available.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - exercised by the launch command.
    raise SystemExit(
        "Streamlit is not installed. Install requirements-demo.txt in the "
        "PatchCore environment, then run: streamlit run demo/app.py"
    ) from exc

from demo.inference import (  # noqa: E402
    DemoConfigurationError,
    FUSION_MODE_ORIGINAL,
    FUSION_MODE_FIXED,
    FUSION_MODE_GATE,
    FUSION_MODE_LABELS,
    FUSION_MODE_SENSITIVITY,
    PatchCoreDemoEngine,
    describe_fusion_modes,
    describe_model_directory,
    display_category_name,
    iter_saved_model_directories,
)
from demo.visualization import overlay_anomaly_map  # noqa: E402


st.set_page_config(
    page_title="PatchCore 工业检测台",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _default_model_directory() -> str:
    candidates = _available_model_directories()
    if candidates:
        return str(candidates[0])
    return str(PROJECT_ROOT / "models")


def _available_model_directories() -> list[Path]:
    """Discover repository models and saved Demo models with metadata."""

    roots = [PROJECT_ROOT / "models", PROJECT_ROOT / "results" / "demo"]
    candidates = {
        path
        for root in roots
        for path in iter_saved_model_directories(root)
    }
    return sorted(candidates, key=lambda path: str(path).lower())


def _model_option_label(path: str) -> str:
    metadata = describe_model_directory(path)
    config_label = "已绑定 fusion.json" if metadata["has_fusion_config"] else "无兼容 fusion.json"
    original_label = "原论文单库可用" if metadata["original_patchcore"] else "双分支/其他模型"
    return (
        f"{display_category_name(metadata['category'])} | "
        f"{metadata['model_family']} | {metadata['branch_count']} 分支 | "
        f"{original_label} | {config_label}"
    )


@st.cache_resource(show_spinner="正在加载 PatchCore 模型，请稍候…")
def load_engine(
    model_directory: str, device: str, fusion_mode: str
) -> PatchCoreDemoEngine:
    return PatchCoreDemoEngine(
        model_directory, device=device, fusion_mode=fusion_mode
    )


def _json_download(result) -> bytes:
    return json.dumps(result.as_json(), ensure_ascii=False, indent=2).encode("utf-8")


def _show_model_health(model_directory: str) -> None:
    metadata = describe_model_directory(model_directory)
    if not metadata["exists"]:
        st.warning(
            "模型目录不存在。请在上方选择或填写有效目录。",
            icon=":material/folder_off:",
        )
        return
    if not metadata["valid_structure"]:
        st.warning(
            "该目录中没有可用的 PatchCore FAISS 索引文件。",
            icon=":material/error:",
        )
        return
    st.badge("模型已就绪", icon=":material/check_circle:", color="green")
    st.markdown(f"**{metadata['category_label']}**")
    st.caption(f"建议上传：{metadata['category_label']} 产品图片")
    st.caption(f"{metadata['branch_count']} 个 PatchCore 分支")
    original_status = (
        "可用于原论文单记忆库模式"
        if metadata["original_patchcore"]
        else "不符合原论文单记忆库模式"
    )
    st.caption(f"原论文模式：{original_status}")
    with st.expander("查看模型详情", expanded=False, icon=":material/info:"):
        st.caption(f"模型版本：{metadata['model_family']}")
        st.caption(f"模型路径：{metadata['path']}")
    if metadata["has_fusion_config"]:
        st.caption("融合配置：已自动加载")
        st.caption(f"配置来源：{metadata['fusion_config_path']}")
    else:
        st.caption(
            "未找到兼容 fusion.json；固定模式可等权回退，"
            "敏感性/门控模式不可用"
        )


st.title("PatchCore 工业检测台", icon=":material/analytics:")
st.caption("单图异常检测 · 多层特征融合 · 可解释诊断")
with st.container(border=True):
    intro_left, intro_right = st.columns([3, 1], vertical_alignment="center")
    with intro_left:
        st.markdown(
            "上传一张待检产品图像，系统将完成预处理、PatchCore 推理、"
            "异常定位和分支诊断。"
        )
        st.caption(
            "整图判定使用图像级异常分数；热力图用于定位，不作为整图阈值。"
        )
    with intro_right:
        st.badge("本地推理", icon=":material/bolt:", color="blue")
        st.caption("模型与图片均在本机处理")

with st.sidebar:
    st.header("检测工作台", icon=":material/tune:")
    st.caption("先选择同类别模型，再上传对应产品图片。")
    discovered = _available_model_directories()
    discovered_strings = [str(path) for path in discovered]
    default_path = _default_model_directory()
    with st.container(border=True):
        st.subheader("模型与设备", icon=":material/lan:")
        if discovered_strings:
            selected = st.selectbox(
                "模型目录",
                options=discovered_strings,
                index=discovered_strings.index(default_path)
                if default_path in discovered_strings
                else 0,
                format_func=_model_option_label,
                help="模型名称会显示类别、版本、分支数量和可用配置。",
            )
            with st.expander(
                "加载外部模型目录",
                expanded=False,
                icon=":material/folder_open:",
            ):
                custom_path = st.text_input(
                    "自定义模型目录",
                    value="",
                    placeholder="留空则使用上方选择的模型",
                    help=(
                        "只有模型不在仓库 models/ 目录中时才需要填写。"
                        "目录中应包含 FAISS 索引和对应的 patchcore_params.pkl。"
                    ),
                )
            model_directory = custom_path.strip() or selected
        else:
            model_directory = st.text_input("模型目录", value=default_path).strip()
        device = st.selectbox(
            "推理设备",
            ["cpu", "cuda"],
            index=0,
            format_func=lambda value: "CPU（兼容性优先）"
            if value == "cpu"
            else "CUDA（速度优先）",
        )

    mode_options = [
        FUSION_MODE_ORIGINAL,
        FUSION_MODE_FIXED,
        FUSION_MODE_SENSITIVITY,
        FUSION_MODE_GATE,
    ]
    with st.container(border=True):
        st.subheader("推理策略", icon=":material/alt_route:")
        selected_fusion_mode = st.segmented_control(
            "选择模式",
            options=mode_options,
            default=FUSION_MODE_FIXED,
            format_func=lambda mode: FUSION_MODE_LABELS[mode],
            key="selected_fusion_mode",
            help=(
                "原论文模式使用单记忆库；固定模式是稳定基线；敏感性和门控模式"
                "自动读取兼容的 fusion.json，不需要手动上传配置。"
            ),
            width="stretch",
        ) or FUSION_MODE_FIXED

    _show_model_health(model_directory)

    engine = None
    threshold_override = None
    selected_metadata = describe_model_directory(model_directory)
    fusion_mode_status = describe_fusion_modes(model_directory)
    selected_mode_status = fusion_mode_status[selected_fusion_mode]
    if selected_mode_status["available"]:
        st.badge("当前策略可用", icon=":material/check:", color="green")
    else:
        st.badge("当前策略不可用", icon=":material/block:", color="red")
    st.caption(selected_mode_status["reason"])
    if selected_metadata["valid_structure"]:
        if not selected_mode_status["available"]:
            st.error(
                f"当前模型不能使用「{selected_mode_status['label']}」："
                f"{selected_mode_status['reason']}"
            )
        else:
            try:
                with st.spinner("正在加载模型并校准类别阈值…"):
                    engine = load_engine(
                        model_directory, device, selected_fusion_mode
                    )
            except (DemoConfigurationError, RuntimeError, ValueError) as exc:
                st.error(f"模型加载或阈值校准失败：{exc}")

    if engine is not None:
        st.markdown(f"**{selected_metadata['category_label']} · 模型已加载**")
        st.metric("自动校准阈值", f"{engine.auto_threshold:.5f}")
        st.caption(f"实际加载策略：{engine.fusion_mode}")
        st.caption(f"阈值来源：{engine.threshold_config['source']}")
        calibration_images = int(engine.threshold_config["calibration_images"])
        if calibration_images:
            st.caption(f"校准样本数：{calibration_images}")
        use_manual_threshold = st.checkbox(
            "手动覆盖自动阈值（实验）",
            value=False,
            help="默认使用当前类别自动校准阈值；只有对比实验时才建议手动覆盖。",
        )
        if use_manual_threshold:
            threshold_override = st.number_input(
                "手动图像级异常阈值（原始分数）",
                min_value=0.0,
                max_value=100.0,
                value=float(engine.auto_threshold),
                step=0.01,
                format="%.5f",
                help="使用图像级融合原始分数判定，不使用单图热力图峰值。",
            )
    else:
        st.caption("自动阈值将在模型成功加载后显示。")
    st.caption(
        "fusion.json 会自动关联；原论文模式不依赖它。"
        "配置缺失或不兼容时会明确提示，不会静默回退。"
    )

selected_metadata = describe_model_directory(model_directory)
if selected_metadata["valid_structure"]:
    with st.container(border=True):
        model_col, upload_hint_col = st.columns([2, 1], vertical_alignment="center")
        with model_col:
            st.badge("类别匹配提示", icon=":material/verified:", color="blue")
            st.markdown(
                f"**{selected_metadata['category_label']}** · "
                f"{selected_metadata['model_family']}"
            )
            st.caption(
                f"{selected_metadata['branch_count']} 个分支 · "
                f"当前策略：{FUSION_MODE_LABELS[selected_fusion_mode]}"
            )
        with upload_hint_col:
            st.caption("建议上传")
            st.markdown(f"**{selected_metadata['category_label']} 产品图片**")
            st.caption("其他类别图片可以运行，但结果不具备可靠意义。")
elif model_directory:
    st.warning(
        "当前模型目录尚未通过结构检查，请先选择包含 FAISS 索引的类别模型目录。",
        icon=":material/folder_off:",
    )

with st.container(border=True):
    st.subheader("上传待检图像", icon=":material/upload_file:")
    st.caption(
        "支持 JPG、JPEG、PNG、BMP、WEBP。fusion.json 会根据当前模型自动加载，"
        "不需要手动上传。"
    )
    uploaded = st.file_uploader(
        "选择一张产品图片",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
        help=(
            "这里只上传要检测的产品图片。图片不会发送到外部服务，"
            "融合配置由程序自动关联。"
        ),
    )

if uploaded is None:
    selected_metadata = describe_model_directory(model_directory)
    with st.container(border=True):
        st.subheader("准备开始", icon=":material/rocket_launch:")
        if selected_metadata["category"]:
            st.info(
                f"当前模型对应 **{selected_metadata['category_label']}**。"
                f"请上传同类产品图片，再查看检测结果与热力图。",
                icon=":material/lightbulb:",
            )
        st.markdown(
            "**操作流程**\n\n"
            "1. 在左侧选择模型与推理模式。\n"
            "2. 确认建议上传的产品类别。\n"
            "3. 上传一张产品图片。\n"
            "4. 查看判定、热力图和分支诊断。"
        )
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
except (OSError, ValueError) as exc:
    st.error(f"无法读取图像：{exc}")
    st.stop()

try:
    if engine is None:
        st.error(
            f"当前推理模式「{FUSION_MODE_LABELS[selected_fusion_mode]}」不可用，"
            "请切换到可用模式；原论文模式需要单记忆库模型，"
            "实验模式需要兼容的 fusion.json。"
        )
        st.stop()
    with st.spinner("正在推理…"):
        result = engine.predict(image, threshold=threshold_override)
except (DemoConfigurationError, RuntimeError, ValueError) as exc:
    st.error(f"模型加载或推理失败：{exc}")
    st.stop()

st.header("检测结果", icon=":material/assessment:")
with st.container(border=True):
    status_col, score_col, map_peak_col, time_col = st.columns(
        4, vertical_alignment="center"
    )
    status_col.metric("判定", result.decision)
    status_col.badge(
        "需要复核" if result.decision == "异常" else "未发现明显异常",
        icon=":material/warning:" if result.decision == "异常" else ":material/check_circle:",
        color="red" if result.decision == "异常" else "green",
    )
    score_col.metric("图像级融合分数", f"{result.fused_image_score:.5f}")
    map_peak_col.metric("热力图显示峰值", f"{result.fused_map_peak:.3f}")
    time_col.metric("单图推理耗时", f"{result.elapsed_seconds:.3f} s")
    st.caption(
        f"{selected_metadata['category_label']} · "
        f"自动阈值 {engine.auto_threshold:.5f} · "
        f"来源：{engine.threshold_config['source']}"
    )
if result.threshold_source == "手动覆盖（实验）":
    st.warning(
        f"本次使用手动阈值 {result.threshold:.5f}，自动校准阈值为 "
        f"{engine.auto_threshold:.5f}。",
        icon=":material/tune:",
    )

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("输入图像", icon=":material/image:")
        st.image(
            result.image,
            width="stretch",
            alt="上传的待检工业图像",
        )
with right:
    with st.container(border=True):
        st.subheader("异常热力图", icon=":material/flare:")
        st.image(
            overlay_anomaly_map(result.image, result.fused_map),
            width="stretch",
            alt="异常热力图叠加结果",
        )

st.header("分支与融合诊断", icon=":material/account_tree:")
st.caption(
    "用表格查看每个模型分支的权重和原始分数。对于现有 MVTec Ensemble 模型，"
    "单个分支可能同时提取 layer2/layer3，不等同于独立的单层权重表。"
)
rows = []
for branch, raw_score, peak in zip(
    result.branches, result.branch_scores, result.branch_map_peaks
):
    rows.append(
        {
            "分支": branch.name,
            "骨干网络": branch.backbone,
            "提取层": ", ".join(branch.layers),
            "图像权重": round(branch.image_weight, 4),
            "像素权重": round(branch.pixel_weight, 4),
            "原始分数": round(raw_score, 5),
            "热力图峰值": round(peak, 4),
        }
    )
st.dataframe(
    rows,
    width="stretch",
    hide_index=True,
    column_config={
        "分支": st.column_config.TextColumn(
            "分支", help="保存模型中的一个独立 PatchCore 分支。"
        ),
        "骨干网络": st.column_config.TextColumn(
            "骨干网络", help="该分支使用的 ImageNet 预训练 backbone。"
        ),
        "提取层": st.column_config.TextColumn(
            "提取层", help="该分支实际提取的网络层；不代表分支就是单独的一层。"
        ),
        "图像权重": st.column_config.NumberColumn(
            "图像权重", format="%.4f", help="融合图像级异常分数时使用的权重。"
        ),
        "像素权重": st.column_config.NumberColumn(
            "像素权重", format="%.4f", help="融合像素级异常图时使用的权重。"
        ),
        "原始分数": st.column_config.NumberColumn(
            "原始分数",
            format="%.5f",
            help="该分支输出的图像级 PatchCore 原始异常分数。",
        ),
        "热力图峰值": st.column_config.NumberColumn(
            "热力图峰值",
            format="%.4f",
            help="单图可视化归一化后的热力图峰值，只用于定位展示。",
        ),
    },
    alt="PatchCore 模型分支、权重和异常分数诊断表",
)
st.caption(f"当前融合配置：{result.fusion_mode}")
fusion_metadata = result.fusion_metadata
with st.expander("融合配置诊断", icon=":material/tune:"):
    st.write(
        "当前模式只使用程序自动找到的预先保存权重配置。"
        "门控结果属于类别/校准集级别，不会根据当前上传图片重新训练或调权。"
    )
    st.json(
        {
            "请求模式": fusion_metadata.get("requested_mode_label"),
            "实际变体": fusion_metadata.get("selected_variant"),
            "配置文件": fusion_metadata.get("config_path"),
            "融合方法": fusion_metadata.get("fusion_method"),
            "门控诊断": fusion_metadata.get("gate"),
        }
    )
for warning in result.warnings:
    if "默认回退" in warning or "等权融合" in warning:
        st.info(warning, icon=":material/info:")
    else:
        st.warning(warning, icon=":material/warning:")

with st.expander("参数说明", icon=":material/help:"):
    st.markdown(
        "- **图像权重**：用于融合各模型分支的整图异常分数，决定整张图的异常判断依据。\n"
        "- **像素权重**：用于融合各模型分支的异常热力图，决定缺陷位置的可视化结果。\n"
        "- **原始分数**：PatchCore 根据查询图像与记忆库近邻距离得到的图像级分数；" 
        "当前 Demo 用融合后的图像级分数与阈值比较。\n"
        "- **热力图峰值**：经过单图显示归一化后的最大值，适合观察定位强度，" 
        "不能直接作为整图判定分数。\n"
        "- **等权回退**：当前模型没有保存 `fusion.json` 时，各模型分支使用相同权重；" 
        "这不是错误，但不等同于你的 layer2/layer3 敏感性融合实验。\n"
        "- **配置自动加载**：优先读取当前模型目录的 `fusion.json`；如果模型是保存在 "
        "`results/demo` 中的 Demo 模型，也会按类别和分支数量自动匹配兼容配置；"
        "不需要上传 `fusion.json`。\n"
        "- **固定融合基线**：优先读取 `fusion.json` 中的 `fixed` 或 `base_*` 权重，"
        "没有配置时才使用等权。\n"
        "- **敏感性收缩融合**：读取预先计算的敏感性收缩权重；它要求模型是独立的 "
        "layer2/layer3 双分支。\n"
        "- **风险受控门控融合**：读取保存的 `gate_selected` 结果；门控通过时使用"
        "敏感性收缩，门控不通过时使用固定像素权重。\n"
        "- **模式专属阈值**：固定、敏感性和门控模式分别校准并缓存阈值，避免不同"
        "融合分数尺度共用一个阈值。"
    )

with st.expander("查看各分支热力图", icon=":material/layers:"):
    branch_columns = st.columns(max(1, len(result.branch_maps)))
    for column, branch, branch_map in zip(
        branch_columns, result.branches, result.branch_maps
    ):
        with column:
            st.image(
                overlay_anomaly_map(result.image, branch_map),
                caption=branch.name,
                width="stretch",
                alt=f"{branch.name} 分支异常热力图",
            )

st.download_button(
    "下载本次 JSON 诊断结果",
    data=_json_download(result),
    file_name="patchcore_prediction.json",
    mime="application/json",
    icon=":material/download:",
)

with st.expander("工程边界与解释"):
    st.markdown(
        "- Demo 用于单图辅助检查，不替代基于验证集校准的生产阈值。\n"
        "- 整图判定使用图像级融合原始分数；热力图峰值经过单图可视化归一化，不能用于整图阈值判定。\n"
        "- 当前界面不根据测试标签或当前图像反向调权。\n"
        "- 程序会自动加载兼容的 `fusion.json`；固定模式没有配置时会使用等权并在结果区提示；敏感性和门控"
        "模式没有兼容配置时会阻止推理，不会静默回退。\n"
        "- `fusion.json` 应与对应模型目录绑定，不能跨类别或跨分支顺序复用。\n"
        "- 现有上游多 backbone 模型不等同于实验中的独立 layer2/layer3 分支；"
        "界面会展示实际加载到的 backbone 与提取层。"
    )

