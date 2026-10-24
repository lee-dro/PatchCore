# PatchCore 工业异常检测与工程化 Demo

本仓库基于 Roth 等人提出的 PatchCore，用于工业图像异常检测与缺陷定位：

- 复现 PatchCore 的记忆库、近邻搜索和异常图生成流程；
- 研究 `layer2` 与 `layer3` 独立分支的图像级/像素级分数融合；
- 对比固定融合、合成缺陷敏感性收缩和风险受控门控策略；
- 提供可直接运行的 Streamlit 单图检测 Demo，展示判定、热力图、分支诊断和 JSON 结果。

## 项目亮点

研究部分在 15 个 MVTec AD 类别和多随机种子上分析了检测与定位之间的权衡、类别依赖性、
随机波动和额外计算成本。当前实验结果显示，门控融合在部分类别上能够带来像素级定位收益，
但提升并非普适，因此项目将其定位为带固定权重回退的实验策略，而不是无条件优于基线的
“通用自适应算法”。

工程化 Demo 支持四种明确模式：

1. 原论文 PatchCore（`layer2+layer3` 单记忆库）；
2. 固定融合基线；
3. 敏感性收缩融合（实验模式）；
4. 风险受控门控融合（实验模式）。

Demo 会根据模型目录自动发现类别、检查模型结构、加载兼容的 `fusion.json` 和模式专属阈值，
用户只需要上传待检测图片，不需要手动上传配置文件。项目同时提供 VS Code 启动配置、
模型结构校验、页面级测试和跨类别推理验证。

原始 PatchCore 论文与上游预训练模型说明见：Roth et al. (2021),
<https://arxiv.org/abs/2106.08265>。

It also provides various pretrained models that can achieve up to _99.6%_ image-level anomaly
detection AUROC, _98.4%_ pixel-level anomaly localization AUROC and _>95%_ PRO score (although the
later metric is not included for license reasons).

![defect_segmentation](images/patchcore_defect_segmentation.png)

_For questions & feedback, please reach out to karsten.rh1@gmail.com!_

---

## Quick Guide

First, clone this repository and set the `PYTHONPATH` environment variable with `env PYTHONPATH=src python bin/run_patchcore.py`.
To train PatchCore on MVTec AD (as described below), run

```
datapath=/path_to_mvtec_folder/mvtec datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))


python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model \
--log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
```

which runs PatchCore on MVTec images of sizes 224x224 using a WideResNet50-backbone pretrained on
ImageNet. For other sample runs with different backbones, larger images or ensembles, see
`sample_training.sh`.

### 本项目：带固定回退的敏感性引导异常分数融合

本项目基于上游 PatchCore，研究 `layer2` 与 `layer3` 的独立异常响应，
并比较检测、定位收益与额外计算成本。当前研究主线恢复为**双分支分数融合**，
不再把单库内的特征加权作为主方案。上方论文成绩属于上游实现，不是本项目改进成绩。

| 方法 | 记忆库 | 图像分数 | 像素异常图 |
|---|---|---|---|
| 原论文 PatchCore | layer2+layer3 共用一个库 | 单模型原始分数 | 单模型原始异常图 |
| 双分支固定分数融合 | 两层分别建库 | 0.9/0.1 | 0.5/0.5 |
| 敏感性 + 收缩 | 两层分别建库 | 固定 0.9/0.1 | 75% 固定权重 + 25% 敏感性权重 |
| 可选门控 | 同上 | 不变 | 满足校准规则用收缩，否则回退 0.5/0.5 |

敏感性来自训练集中留出的正常图及其合成小点、划痕、纹理扰动，
不是从真实测试缺陷标签拟合的权重。门控检查跨合成缺陷组的一致性及分支相对稳定性。
这是类别/校准集级权重，不是每张测试图的自适应网络，也没有形式化风险保证。

VS Code 当前仅保留一套运行配置：`PatchCore Demo - Streamlit`，用于直接启动工程化 Demo。
训练和批量实验仍可通过下方命令及 `bin/` 中的脚本执行，避免把训练数据集和实验参数
硬编码到 Demo 启动配置中。运行生成的模型、阈值和实验结果默认保存在 `results/`，
其中模型文件和过程目录不会作为源代码同步到 GitHub。

完整用法、数据流、参数与局限见 [方法说明](docs/adaptive_score_fusion.md)。
固定分数对照见 [固定融合说明](docs/fixed_weight_fusion.md)。
Demo 的 `fusion.json` 绑定规则见 [Demo 融合配置说明](docs/demo_fusion_config.md)。
批量复核与补跑入口为 `bin/benchmark_fusion.py`，结果索引见 [实验索引](results/README.md)。

### 工程化单图 Demo

仓库现在提供一个独立的 Streamlit 推理 Demo，用于把模型加载、单图推理、
异常热力图、分支诊断、耗时统计和 JSON 导出串成可演示的闭环。Demo 提供原论文 PatchCore
单记忆库、固定融合基线、
敏感性收缩融合和风险受控门控融合四种显式模式；敏感性与门控模式只自动加载与当前模型
绑定且经过结构校验的 `fusion.json`，不会在上传图片时重新计算或静默回退。用户不需要上传
`fusion.json`；程序会优先读取当前模型目录，并自动关联 `results/demo` 中同类别且分支数量兼容的保存配置。

在 `patchcore` 环境中执行：

```powershell
pip install -r requirements-demo.txt
$env:PYTHONPATH = "src"
streamlit run demo/app.py
```

界面支持：

- 上传 JPG、PNG、BMP 或 WEBP 图像；
- 选择仓库内或外部保存的 PatchCore 模型目录；
- 查看输入图、融合异常热力图、各分支原始分数与权重；
- 查看单图推理耗时并下载 JSON 诊断结果。
- 根据当前模型自动判断四种模式是否可用，并显示配置缺失或分支不匹配原因；
- 将 `results/demo` 中保存的 Demo 模型加入模型选择列表，并自动显示实际加载的 `fusion.json` 来源；
- 在诊断结果中显示实际加载的变体、权重、门控状态和配置路径。

原论文模式只接受“一个 FAISS 索引 + `layers_to_extract_from=['layer2', 'layer3']`”的单库模型，
不会把双分支模型设置成 1.0/1.0 后冒充原论文方法。当前仓库内的预训练模型主要是单库
`layer2+layer3` 模型或多 backbone Ensemble，因此原论文和固定模式通常可直接运行；它们没有
被伪装成独立 layer2/layer3 敏感性模型。要启用后两种实验模式，需要使用本项目的独立双分支
实验流程保存带兼容 `fusion.json` 的模型。

实现说明与边界见 [Demo 文档](demo/README.md)。需要注意：单图展示分数使用
可视化归一化，不等同于跨测试集计算的 AUROC，也不能直接当作生产概率阈值。

#### 已核对的历史结果

15 类 × 5 seed 的历史记录中，相对**双分支固定分数融合**的平均 AUROC 差值：

| 方法 | 图像 AUROC | 全像素 AUROC | 异常图像内像素 AUROC |
|---|---:|---:|---:|
| 仅像素敏感性 | 0 | -0.002308 | -0.001736 |
| λ=0.25 收缩 | 0 | +0.000176 | +0.000311 |
| legacy 门控 + 收缩 | 0 | +0.000388 | +0.000501 |

这些数字来自已保存表格的重算，不是本次重新训练 15 类。
门控的全像素 34 胜/29 平/12 负，异常图像像素 35 胜/29 平/11 负。
按类别均值作双侧配对符号置换，p 分别为 0.0746、0.0516；
门控也经过先前测试结果探索，不能据此声称显著、普适提升。
pill、cable、transistor 等仍有失败，历史原始运行配置缺失也限制了精确复现。

单库 0.7/0.3 加权的 40 次负结果保留在归档中，属于另一种方法，
不得与上述分数融合结果混用。下一步应优先评估收益、成本与有效边界，
而不是持续根据同一测试集重新选择门限。

#### 本次冻结协议补跑（2026-10-22）

本次按同一 `80%` 建库、`20%` 校准、`seed=0..4` 完成 bottle、pill、transistor，
共 30 次训练命令；四个双分支方案共享同一次预测。相对双分支固定分数融合的类别等权均值为：

| 方法 | 图像 AUROC | 全像素 AUROC | 异常图像像素 AUROC |
|---|---:|---:|---:|
| 原始单库 | +0.009161 | -0.004087 | -0.003855 |
| 仅像素敏感性 | 0 | -0.014347 | -0.014618 |
| λ=0.25 收缩 | 0 | -0.001866 | -0.002080 |
| legacy 门控 | 0 | -0.000500 | -0.000537 |

这三类不是新的未见类别，结论只支持“门控能减轻敏感性调权的退化”，
不支持“门控已带来跨类别提升”。门控启用次数为 bottle 5/5、pill 4/5、transistor 2/5。
当前 Windows 同环境平均阶段耗时：原始单库 `19.42s`，双分支含校准 `35.07s`；
保存前墙钟时间约为 `25.78s` 与 `51.85s`。旧记录使用的进程设置不同，不能直接作速度归因。

本次完整表格见 `results/restoration_20261022/fusion_restoration_results.xlsx`，
协议、逐 seed 记录、计时和门控诊断同目录保存。该结果把“效果最好阶段”恢复为
双分支敏感性方案的可复核版本，但推荐默认仍是固定分数融合，门控只作为有边界的研究选项。

Given a pretrained PatchCore model (or models for all MVTec AD subdatasets), these can be evaluated using

```shell
datapath=/path_to_mvtec_folder/mvtec
loadpath=/path_to_pretrained_patchcores_models
modelfolder=IM224_WR50_L2-3_P001_D1024-1024_PS-3_AN-1_S0
savefolder=evaluated_results'/'$modelfolder

datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
```

A set of pretrained PatchCores are hosted here: __add link__. To use them (and replicate training),
check out `sample_evaluation.sh` and `sample_training.sh`.

---

## In-Depth Description

### Requirements

Our results were computed using Python 3.8, with packages and respective version noted in
`requirements.txt`. In general, the majority of experiments should not exceed 11GB of GPU memory;
however using significantly large input images will incur higher memory cost.

### Setting up MVTec AD

To set up the main MVTec AD benchmark, download it from here: <https://www.mvtec.com/company/research/datasets/mvtec-ad>.
Place it in some location `datapath`. Make sure that it follows the following data tree:

```shell
mvtec
|-- bottle
|-----|----- ground_truth
|-----|----- test
|-----|--------|------ good
|-----|--------|------ broken_large
|-----|--------|------ ...
|-----|----- train
|-----|--------|------ good
|-- cable
|-- ...
```

containing in total 15 subdatasets: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`,
`leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`.

### "Training" PatchCore

PatchCore extracts a (coreset-subsampled) memory of pretrained, locally aggregated training patch features:

![patchcore_architecture](images/architecture.png)

To do so, we have provided `bin/run_patchcore.py`, which uses `click` to manage and aggregate input
arguments. This looks something like

```shell
python bin/run_patchcore.py \
--gpu <gpu_id> --seed <seed> # Set GPU-id & reproducibility seed.
--save_patchcore_model # If set, saves the patchcore model(s).
--log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results # Logging details: Name of the run & Name of the overall project folder.

patch_core  # We now pass all PatchCore-related parameters.
-b wideresnet50  # Which backbone to use.
-le layer2 -le layer3 # Which layers to extract features from.
--faiss_on_gpu # If similarity-searches should be performed on GPU.
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 # Dimensionality of features extracted from backbone layer(s) and final aggregated PatchCore Dimensionality
--anomaly_scorer_num_nn 1 --patchsize 3 # Num. nearest neighbours to use for anomaly detection & neighbourhoodsize for local aggregation.

sampler # We now pass all the (Coreset-)subsampling parameters.
-p 0.1 approx_greedy_coreset # Subsampling percentage & exact subsampling method.

dataset # We now pass all the Dataset-relevant parameters.
--resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath # Initial resizing shape and final imagesize (centercropped) as well as the MVTec subdatasets to use.
```

`sample_training.sh` contains upstream example training configurations (Bash).
Their comments describe upstream results, not measurements from this checkout.
For the current Windows experiment use `.vscode/launch.json` or `bin/benchmark_fusion.py`.
Saved models can be evaluated with `bin/load_and_evaluate_patchcore.py`.

During (after) training, the following information will be stored:

```shell
|PatchCore model (if --save_patchcore_model is set)
|-- models
|-----|----- mvtec_bottle
|-----|-----------|------- nnscorer_search_index.faiss
|-----|-----------|------- patchcore_params.pkl
|-----|----- mvtec_cable
|-----|----- ...
|-- results.csv # Contains performance for each subdataset.

|Sample_segmentations (if --save_segmentation_images is set)
```

This checkout records results locally. It does not expose the old online-logging flags.

Finally, due to the effectiveness and efficiency of PatchCore, we also incorporate the option to use
an ensemble of backbone networks and network featuremaps. For this, provide the list of backbones to
use (as listed in `src/patchcore/backbones.py`) with `-b <backbone>` and, given their
ordering, denote the layers to extract with `-le idx.<layer_name>`. An example with three different
backbones would look something like

```shell
python bin/run_patchcore.py --gpu <gpu_id> --seed <seed> --save_patchcore_model --log_group <log_name> --log_project <log_project> results \

patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu \

--pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath

```

When using `--save_patchcore_model`, in the case of ensembles, a respective ensemble of PatchCore parameters is stored.

### Evaluating a pretrained PatchCore model

To evaluate a/our pretrained PatchCore model(s), run

```shell
python bin/load_and_evaluate_patchcore.py --gpu <gpu_id> --seed <seed> $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
```

assuming your pretrained model locations to be contained in `model_flags`; one for each subdataset
in `dataset_flags`. Results will then be stored in `savefolder`. Example model & dataset flags:

```shell
model_flags=('-p', 'path_to_mvtec_bottle_patchcore_model', '-p', 'path_to_mvtec_cable_patchcore_model', ...)
dataset_flags=('-d', 'bottle', '-d', 'cable', ...)
```

### Expected performance of pretrained models

While there may be minor changes in performance due to software & hardware differences, the provided
pretrained models should achieve the performances provided in their respective `results.csv`-files.
The mean performance (particularly for the baseline WR50 as well as the larger Ensemble model)
should look something like:

| Model | Mean AUROC | Mean Seg. AUROC | Mean PRO
|---|---|---|---|
| WR50-baseline | 99.2% | 98.1% | 94.4%
| Ensemble | __99.6%__ | __98.2%__ | __94.9%__

### Citing

If you use the code in this repository, please cite

```
@misc{roth2021total,
      title={Towards Total Recall in Industrial Anomaly Detection},
      author={Karsten Roth and Latha Pemula and Joaquin Zepeda and Bernhard Schölkopf and Thomas Brox and Peter Gehler},
      year={2021},
      eprint={2106.08265},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
