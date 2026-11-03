# 已拉取源码审计

## Tarantula / Ochiai

- `AFLuent/AFLuent` 提供可核验的标准公式实现，但工具面向 Python/pytest，不能直接运行 CLoc。
- `JoranHonig/tarantula` 可作为 Solidity 覆盖输入参考，仍需把链上交易转换为内部函数覆盖。
- 本适配层直接复现公开公式，并统一候选函数、并列排序和指标协议。

## PRFL

- PRFL 原仓库公开了 PageRank 和 PR-Ochiai 核心，未公开动态覆盖采集器。
- 主实验的 `prfl` 按参考算法实现 PageRank + PR-Ochiai，输入为 CLoc 统一覆盖矩阵与函数调用边。
- `prfl_malo` 仅用于复核 MALO 的简化重实现，不能与 reference PRFL 混报。

## CNNFL

- CNNFL 原作者实现没有公开。
- MALO 公共仓库含第三方 CNNFL 重实现：两层二维卷积、两层池化、两个 1024 维全连接层、20 epoch，并用单位矩阵构造虚拟测试。
- MALO 仓库缺少 `FL4SCDataset` 等依赖；其 `ProgramSpectrum` 还会根据 `FunctionDefinition.y` 把已知故障函数写成 `-1`，把评估标签泄漏到训练输入。
- 本适配保留其网络和训练结构，主结果默认使用纯二值覆盖；泄漏行为只作为显式兼容开关，严禁用于论文主表。

## MALO 公共仓库完整性

该仓库不能原样复现其基线结果：

- `models/Tarantula.py`、`Ochiai.py`、`PRFL.py`、`CNNFL.py` 导入不存在的 `FL4SCDataset`；
- `dataset/nx.py` 导入仓库中不存在的 `downloaders` 包；
- 多个入口依赖未提交的 `settings.py`；
- 仓库含 raw JSON，但不含基线需要的 processed `.pt` 数据。

所以 MALO 中的数字来自研究团队自己的重实现与未完整公开的数据管线，不代表 CNNFL 原作者代码已经公开，也不能视为开箱复现证据。

## Grace

- Grace 原仓库面向 Defects4J/Java，依赖未随仓库提供的预处理 `.pkl`。
- 原实现使用方法、源码行、失败测试、成功测试四类节点，以及方法—行、测试—行覆盖边；本适配将它们对应为 Solidity 函数、源映射行、故障交易和正常交易。
- 原 `run.py` 会在每个 epoch 查看留出案例的真实故障排名并选择最佳 epoch，属于测试标签参与模型选择。CLoc 适配固定训练 15 epoch，执行留一案例训练，不读取留出案例标签来选 epoch。
- 交易标识是哈希，不具备 Java 测试名与方法名的词汇重叠语义，因此该特征按原算法的“零重叠”值固定为 1，并在结果中明确标注为 `grace_cloc_adaptation`，不冒充原始 Defects4J 设置。
