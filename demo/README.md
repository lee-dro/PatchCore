# PatchCore 工程化 Demo

这是一个本地单图推理 Demo，用于展示当前 PatchCore 项目的工程化闭环：

```text
上传图像 -> 预处理 -> 加载已保存 PatchCore -> 分支推理
        -> 像素热力图融合 -> 判定与诊断 -> JSON 导出
```

## 启动

在项目根目录执行：

```powershell
conda activate patchcore
pip install -r requirements-demo.txt
$env:PYTHONPATH = "src"
streamlit run demo/app.py
```

默认会扫描 `models/` 和 `results/demo/` 下包含 `*nnscorer_search_index.faiss` 的目录。
也可以在侧边栏填写外部模型目录。对已保存的 Demo 模型，`fusion.json` 会随模型自动关联，
不需要手动上传。

模型选择框会显示“产品类别 | 模型版本”，页面同时显示当前类别、分支数量和建议上传的图片类别。
例如选择 `Bottle | IM224_Ensemble_...` 后，应上传 bottle 产品图片；不要用同一个类别模型直接判断
其他产品类别。

侧边栏的“高级选项：加载外部模型目录”默认收起，只有模型不在仓库 `models/` 目录中时才需要填写。

## 推理模式

页面提供四种模式，模式切换只影响已经加载模型的推理策略，不会重新训练 PatchCore：

| 页面模式 | 配置要求 | 行为 |
|---|---|---|
| 原论文 PatchCore（layer2+layer3 单记忆库） | 一个 FAISS 索引；参数文件必须记录 `layer2`、`layer3` | 直接使用一个 PatchCore 模型输出的分数和异常图；不读取 `fusion.json` |
| 固定融合基线 | 无；有 `fusion.json` 时优先读取其中的 `fixed`/`base_*` 权重 | 没有配置时使用等权回退 |
| 敏感性收缩融合（实验） | 当前模型目录必须有兼容的 `fusion.json` | 加载预计算的敏感性收缩权重 |
| 风险受控门控融合（实验） | 当前模型目录必须有兼容的 `fusion.json` 和 `gate_selected` | 加载保存的门控选择结果 |

原论文模式要求当前目录只有一个 `*nnscorer_search_index.faiss`，并且对应 `patchcore_params.pkl` 中的
`layers_to_extract_from` 严格为 `['layer2', 'layer3']`，且没有额外的层内加权。双分支模型即使设置成
1.0/1.0 也不会被识别为原论文模式。

敏感性和门控模式会自动查找与当前模型类别、分支数量兼容的 `fusion.json`，并要求它明确记录两个独立分支，且分支顺序为
`layer2`、`layer3`。如果配置不存在、分支数量/顺序不匹配或变体缺失，页面会阻止推理并说明原因，
不会静默回退成固定模式。

兼容配置由实验脚本在保存模型时生成，理想情况下放在对应模型目录；Demo 也会从
`results/demo` 中自动关联已保存的 Demo 输出：

```text
models/<model-family>/models/<category>/
├── *.faiss
├── *patchcore_params.pkl
└── fusion.json
```

`fusion.json` 至少应包含 `variants.fixed`、实验变体、`branches` 分支顺序以及门控模式需要的
`adaptive_diagnostics.gate`。它不能跨类别、跨模型家族或跨分支顺序复用。

界面中的图片上传框只用于上传待检测图片，不是配置上传框；`fusion.json` 不需要手动上传。

## 输入与输出

- 支持 JPG、JPEG、PNG、BMP、WEBP；
- 默认使用与 MVTec 数据集一致的 Resize(256)、CenterCrop(224) 和 ImageNet 归一化；
- 输出输入图、融合异常热力图、各分支诊断、推理耗时和 JSON；
- 整图判定使用图像级融合原始分数；热力图峰值仅用于定位展示；
- 原论文模式使用一个 PatchCore 单库的原始分数与异常图，不进行分数后融合；
- 固定模式会读取 `fusion.json` 中保存的固定权重；没有配置时显式回退到等权；
- 敏感性和门控模式只读取兼容的预计算配置，缺失时阻止推理而不是静默回退；
- 原论文、固定、敏感性和门控模式分别使用独立的阈值缓存，避免不同分数尺度共用阈值；
- 单图热力图使用显示归一化，热力图峰值不是跨测试集 AUROC 或经过生产校准的概率；
- 默认使用当前类别正常样本图像级融合分数的 99% 分位数作为自动阈值，并保存到模型目录的模式专属阈值文件；
  如果找不到正常校准样本，才回退到 `0.50`，页面会明确提示回退原因。

## 当前边界

现有仓库中的上游模型可能是多 backbone、每个模型内部同时提取多层特征的模型，
不等同于研究实验中的独立 layer2/layer3 分支。Demo 会展示真实的 backbone 和提取层，
不会把它们伪装成独立 layer2/layer3 分数。

敏感性调权和 gate 仍属于实验模式；Demo 默认使用保存模型中的固定融合配置，
避免把类别依赖且计算成本更高的实验策略当作普适生产算法。

