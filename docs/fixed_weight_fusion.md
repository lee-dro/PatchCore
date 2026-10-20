# 固定权重的 PatchCore 分数融合

在 VS Code 的运行与调试菜单选择 `PatchCore - pill image/pixel weighted fusion`。
只保留这一套配置，运行 pill，Coreset 比例为 0.01。
切换类别只需修改这套配置的 `-d` 值，并相应调整 `--log_group` 名称。

在 `.vscode/launch.json` 分别修改检测权重和定位权重：

```json
"--image_fusion_weight", "0.9", "--image_fusion_weight", "0.1",
"--pixel_fusion_weight", "0.5", "--pixel_fusion_weight", "0.5"
```

第一个权重对应 `-b` 的第一个模型及 `0.layer2`；第二个对应第二个模型及
`1.layer3`。每组权重在一次运行内固定。
`image_fusion_weight` 只影响图像级分数和 instance AUROC；
`pixel_fusion_weight` 只影响异常图及两项 pixel AUROC。
这是任务分别调权，不是自适应权重或特征提取方式的改进。
支持非负权重，至少一个大于零，数量必须等于模型数；自动归一化到和为 1。
旧的 `--fusion_weight` 继续可用，作为两个任务的共同默认值。
某个任务提供专用权重时只覆盖该任务；三个选项都不传时，两任务均等权。
新版单一配置只列专用权重，避免混淆。初始权重是探索性选择，不代表最优。

## 数据流与对照

两个独立骨干网络分别提取 layer2、layer3，各自 patchify、建库、预测。
每个分支先在自己的原生 patch 网格计算距离，再将异常图恢复到输入图像尺寸。
分别归一化两分支的图像分数和异常图，最后按权重相加。
图像分数是两个分支图像分数的加权和，不是融合异常图的最大值。

这与原来的 `-b wideresnet50 -le layer2 -le layer3` 不同：后者在 embedding
阶段融合两层特征，只建立一个 Memory Bank。对照表中应分别标注“特征融合”和
“分数融合”，不要都记作 layer2+layer3。

本版沿用仓库的 evaluation-set min-max：图像分数在整个测试集上归一化；
异常图在整个测试集的所有像素上归一化，不是逐图归一化。它不读取测试标签，
但尺度会随测试样本集合变化。常数分支输出零，避免除零。此版是离线基准实验，
尚未加入独立正常校准集或部署用单图阈值。

## 运行记录

日志保持原来的 INFO 格式：每个模型的 `Training models (i/2)` 后显示
`Fit time: ... seconds`，`Embedding test data with models (i/2)` 后显示
`Predict time: ... seconds`，最后显示三项 AUROC。
这里两次训练是同一数据集的两个特征分支，不是运行两个数据集。
所有实验结果统一放在项目 `results/project/<log_group>/` 下。
输出目录中的 `mvtec_pill_fusion.json` 记录归一化后的权重、层、命令参数、
每个 Memory Bank 形状、归一化范围及耗时；`results.csv` 恢复为三项 AUROC。
终端显示各分支耗时，融合配置 JSON 另外记录总耗时。
Fit 总耗时是各分支 fit 之和；Predict 总耗时包括各分支 predict 和融合，
不包括模型初始化、评价指标计算、绘图和模型保存。

保存模型时另写 `models/mvtec_pill/fusion.json`。
`bin/load_and_evaluate_patchcore.py` 自动恢复权重：新版 version=2 的 JSON
分别记录 `image_weights` 和 `pixel_weights`；旧版 version=1 的 `weights`
作为两个任务的共同权重；缺少该文件的旧模型默认等权。
重新评估会按新的评估集合重新计算 min-max，不把此前测试集范围当成部署校准。
`--anomaly_scorer_num_nn` 会传入各分支的实际近邻数，并在保存后重载时恢复。
当前配置为 1；旧代码存在参数拼写不一致的问题，其他取值可能曾被忽略。

双分支会增加特征提取和 Memory Bank 成本。本版零权重分支仍会运行，便于固定
实验流程，因此 `1/0` 与 `0/1` 适合检查融合端点，不适合测单模型速度。
随机采样流程也可能使分支结果与以前单独运行的记录存在差异。

对照共用 0.5/0.5、共用 0.9/0.1，以及图像 0.9/0.1、像素 0.5/0.5。
两个输出独立计算，改变图像权重不应改变异常图，反之亦然。
保持类别、随机种子、采样比例、输入和其他参数一致。
本版没有新增预测缓存功能，每次运行仍重新建库和预测。
查看测试标签反复选出的最佳权重应标注为探索性结果，后续需独立验证，
不能直接宣称对未知异常有提升。
