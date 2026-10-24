# Demo 融合配置

Streamlit Demo 不会在用户上传单张图片时重新训练或估计敏感性权重。用户也不需要上传
`fusion.json`：程序会自动查找与当前模型兼容的配置。

## 模式与文件

原论文 PatchCore 模式不需要 `fusion.json`。它只接受一个同时提取 `layer2`、`layer3` 的
PatchCore 单记忆库目录，并使用独立的 `threshold_original.json`。如果目录包含两个独立
FAISS 索引，即使外部权重为 1.0/1.0，也不会被识别为原论文模式。

```text
models/<model-family>/models/<category>/
├── *.faiss
├── *patchcore_params.pkl
└── fusion.json
```

如果当前模型目录没有配置，程序还会在 `results/demo/**/models/<category>/fusion.json`
和同类别的 `<category>_fusion.json` 中查找，并且只接受分支数量兼容的配置。当前模型目录
中的配置拥有最高优先级；存在多个归档候选时选择最近修改的兼容配置，并在诊断中显示实际路径。

- `固定融合基线`：优先使用 `variants.fixed`，其次使用 `base_image_weights`/`base_pixel_weights`；没有配置时使用等权回退。
- `敏感性收缩融合（实验）`：读取 `image_fixed_pixel_sensitivity_shrinkage` 或兼容的敏感性变体。
- `风险受控门控融合（实验）`：读取 `gate_selected`，并显示保存的门控诊断和选择原因。

后两种模式只接受独立的两分支模型，分支顺序必须是：

```text
0: layer2
1: layer3
```

这是为了避免把原论文式的单库 `layer2+layer3` 模型或多 backbone Ensemble 误当作独立层分支。

## 生成方式

使用实验入口保存模型时，传入 `--save_patchcore_model`。在独立 layer2/layer3 双分支配置下，`bin/run_patchcore.py` 会把固定、敏感性和门控变体写入对应模型目录的 `fusion.json`。如果模型保存到了 `results/demo`，Demo 也会自动关联该输出中的配置。

Demo 加载配置后，会根据所选模式分别使用：

```text
原论文模式     -> threshold_original.json
固定模式       -> threshold.json
敏感性模式     -> threshold_sensitivity.json
门控模式       -> threshold_gate.json
```

四种模式分别校准阈值，避免不同分数尺度共用同一个阈值。阈值文件属于运行缓存，已由 `.gitignore` 忽略；`fusion.json` 是方法配置，应与对应模型一起保存。

界面中的图片上传框只用于上传待检测图片，不是配置上传框。若没有找到兼容配置，界面会明确说明原因，并阻止敏感性/门控实验模式，避免静默使用错误配置。
