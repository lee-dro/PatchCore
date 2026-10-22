# 实验结果索引

## 当前主线

`restoration_20261022/` 是本次双分支方案恢复的复核与补跑目录。
`protocol.json` 是事前冻结设置和源码哈希，`runtime.json` 是实际运行环境。
`historical_*` 为旧表重算，`new_*` 为本次真实运行，禁止跨协议混平均。
逐次运行目录已从仓库清理。汇总 CSV、融合配置、协议和 Excel 足以支撑当前分析；
完整逐次运行原始文件的本地备份位于 `E:\Codex\patchcore_local_archives_20261022\`，不属于项目仓库。

- 原始单库：layer2+layer3 在一个 embedding/记忆库中融合。
- 双分支固定：image=0.9/0.1，pixel=0.5/0.5。
- 双分支实验：图像不变，像素敏感性、λ=0.25 收缩、legacy 门控。
- 当前补跑预定 bottle、pill、transistor × seed 0–4，共 30 次训练命令，完成后有 75 条方法指标。
- 四个分数融合方法共享同一次两分支训练/预测，不算四次独立训练。

入口：`python bin/benchmark_fusion.py`。
详细方法见 `docs/adaptive_score_fusion.md`，具体结论以本目录下已生成的完整记录及报告为准。

## 历史证据

`experiment_2_20261022/` 保留此前 Excel 和 15 类统计 CSV，作为只读历史快照。
旧 Excel 包含探索阶段手工记录，不作为本次计时的来源。
`gate_decisions.csv` 保留 75 个 category/seed 的 fixed、shrinkage、gate 指标；
`paired_deltas.csv` 可与 fixed 重建 sensitivity-pixel 指标。
本次已核对门控规则与选中指标，但缺少旧训练划分清单、运行命令及全部校准原始输出。

旧单库加权实验的压缩备份也已移到上述本地目录。
它对应 layer2/layer3 特征权重 0.5/0.5 对 0.7/0.3，不是双分支分数融合。
项目仓库只保留该实验的汇总表和结论，不保留逐次运行原始文件。
负结果必须保留，以说明转回分支方案的依据。

## 哪些应提交

建议保留 README、汇总表、配对指标、协议与有用校准诊断。
默认忽略 `run.log`、模型文件和热力图。不要仅删除失败运行或失败类别。
本次整理未提交或推送 Git；旧暂存内容与本次工作区修改仍需提交前统一审核。
