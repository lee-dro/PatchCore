# Towards Total Recall in Industrial Anomaly Detection

This repository contains the implementation for `PatchCore` as proposed in Roth et al. (2021), <https://arxiv.org/abs/2106.08265>.

## Project Research Extension

This project extends PatchCore with independent `layer2` and `layer3` branches for industrial anomaly detection. It implements task-specific score fusion, synthetic-defect sensitivity calibration, shrinkage-based pixel weighting, and an optional fallback gate.

Across 15 MVTec AD categories and 75 paired runs, the gated method improved average full-pixel AUROC by `0.00039` and anomaly-pixel AUROC by `0.00050` over the fixed-fusion baseline. The project also analyzes category-dependent failures, detection-localization trade-offs, reproducibility across random seeds, and additional computational cost.

These results indicate a measurable but non-universal improvement. The adaptive strategy is therefore retained as an experimental method with fixed-weight fallback, rather than being presented as universally superior.

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
| 原始特征融合 | layer2+layer3 共用一个库 | 原始分数 | 原始异常图 |
| 双分支固定分数融合 | 两层分别建库 | 0.9/0.1 | 0.5/0.5 |
| 敏感性 + 收缩 | 两层分别建库 | 固定 0.9/0.1 | 75% 固定权重 + 25% 敏感性权重 |
| 可选门控 | 同上 | 不变 | 满足校准规则用收缩，否则回退 0.5/0.5 |

敏感性来自训练集中留出的正常图及其合成小点、划痕、纹理扰动，
不是从真实测试缺陷标签拟合的权重。门控检查跨合成缺陷组的一致性及分支相对稳定性。
这是类别/校准集级权重，不是每张测试图的自适应网络，也没有形式化风险保证。

VS Code 仅保留一套运行配置：`PatchCore - sensitivity shrinkage with optional gate`。
默认单类别 bottle、seed=0、Coreset=0.01、训练/校准=0.8/0.2；
运行一次训练两分支，同时评价固定、敏感性、收缩、门控四个输出。
所有结果放在 `results/`。修改 `-d`、`--seed`、`--log_group` 可换类别和种子，
删除 `--sensitivity_gate` 即关闭门控，不需要修改模型代码。

完整用法、数据流、参数与局限见 [方法说明](docs/adaptive_score_fusion.md)。
固定分数对照见 [固定融合说明](docs/fixed_weight_fusion.md)。
批量复核与补跑入口为 `bin/benchmark_fusion.py`，结果索引见 [实验索引](results/README.md)。

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
