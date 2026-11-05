# CLoc：结合执行证据与大语言模型的区块链应用故障定位

CLoc 是一个面向区块链应用的故障定位框架。它结合交易执行轨迹、正常交易参照和合约源码，利用大语言模型审查交易行为，并对潜在故障位置进行排序，生成附有执行证据和源码解释的分析报告。

目前支持 Ethereum 和 BNB Smart Chain。

## 功能概述

- **执行证据构建**：从交易轨迹中提取调用关系与资产流动，构建动态交易证据图和候选证据链。
- **正常参照审查**：检索可比较的正常交易证据链，由 LLM 比较执行结构和资产行为，判断是否需要继续定位。
- **源码辅助定位**：按需获取相关合约的已验证源码，结合原始执行证据，对候选函数及可疑代码行进行分析。
- **结果报告**：输出 Top-k 候选位置、证据引用和定位解释，支持 JSON 与 Markdown 格式。

## 快速开始

### 1. 安装

需要 Python 3.10 或更高版本。在项目根目录执行：

```powershell
python -m pip install -e .
```

安装后可使用 `cloc` 命令，也可使用 `python -m cloc`。

### 2. 配置服务

CLoc 使用 RPC 获取交易执行轨迹，通过区块浏览器获取已验证源码，并调用 LLM 完成交易审查与故障定位。

在 PowerShell 中设置环境变量，将占位内容替换为自己的配置：

```powershell
$env:ETH_RPC_URL = "<Ethereum RPC 地址>"
$env:ETHERSCAN_API_KEY = "<区块浏览器 API Key>"
$env:CLOC_LLM_API_KEY = "<LLM API Key>"
$env:CLOC_LLM_MODEL = "gpt-4o"
```

分析 BNB Smart Chain 交易时，另行设置：

```powershell
$env:BSC_RPC_URL = "<BNB Smart Chain RPC 地址>"
```

RPC 服务需要支持目标历史交易的 `debug_traceTransaction` 与 `callTracer`。

使用兼容 Chat Completions 的其他模型服务时，可设置：

```powershell
$env:CLOC_LLM_BASE_URL = "<模型服务 API 基础地址>"
$env:CLOC_LLM_MODEL = "<模型名称>"
```

`CLOC_LLM_BASE_URL` 不包含 `/chat/completions` 后缀。个人 RPC 地址和 API Key 通过环境变量提供，请勿写入提交到仓库的文件。

### 3. 构建正常交易参照库

按照 `examples/normal_manifest.example.json` 的格式，准备正常交易轨迹清单，并记录正常性的判断依据。清单中的轨迹路径相对于清单文件所在目录解析。

```powershell
cloc build-reference --manifest data/normal_manifest.json --output data/normal_reference.json
```

参照库按链和业务入口组织。每组需要至少三条可用的参考记录；参考不足或不具备可比性时，交易继续进入源码定位阶段。

### 4. 分析交易

输入交易哈希、所属链和正常交易参照库：

```powershell
cloc analyze --tx-hash "<交易哈希>" --chain eth --baseline data/normal_reference.json --output output/query
```

CLoc 将获取执行轨迹，构建证据链并调用 LLM 进行参照审查。需要进一步分析时，系统获取相关源码，完成候选位置排序并生成报告。

如果已经保存了交易执行轨迹，也可以直接提供 JSON 文件：

```powershell
cloc analyze --trace data/query.trace.json --chain eth --baseline data/normal_reference.json --output output/query
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `--tx-hash` | 待分析交易的哈希，与 `--trace` 二选一 |
| `--trace` | 已保存的 JSON callTracer 轨迹文件 |
| `--chain` | 所属链：`eth` 或 `bsc` |
| `--baseline` | 正常交易参照库路径 |
| `--output` | 分析结果保存目录 |
| `--model` | 指定本次分析使用的模型 |
| `--top-k` | 返回的候选数量，支持 1–5，默认 5 |
| `--source-cache` | 合约源码缓存目录，默认 `data/source_cache` |

例如，分析 BNB Smart Chain 交易并返回前三个候选位置：

```powershell
cloc analyze --tx-hash "<交易哈希>" --chain bsc --baseline data/normal_reference.json --top-k 3 --output output/query
```

完整参数说明：

```powershell
cloc analyze --help
```

## 方法流程

### 阶段一：构建交易执行证据

系统将执行轨迹转换为动态交易证据图，提取调用信息和可观测的资产转移，围绕交易结果组织 CALL 与 ASSET 证据链，为后续分析提供执行上下文。

### 阶段二：正常参照引导的交易审查

系统基于结构距离，为符合条件的查询链检索三条正常参考链。LLM 比较调用结构、资产方向及可观测金额，返回 `NORMAL`、`SUSPICIOUS` 或 `UNCERTAIN` 判定及相应解释。

仅当所有必需的查询链均完成有效审查且全部被判定为 `NORMAL` 时，交易才以正常状态结束分析。其余交易继续进入阶段三。

### 阶段三：源码辅助的候选位置排序

系统根据执行证据中的合约地址获取已验证源码，完成函数匹配和源码切片。LLM 结合原始执行证据与源码，对候选位置进行排序，并说明相关执行行为与代码逻辑之间的联系。

这一阶段使用原始执行证据，不将上一阶段的模型解释作为定位依据。

## 分析结果

结果保存在 `--output` 指定的目录：

| 文件 | 内容 |
|---|---|
| `report.md` | 便于阅读的分析报告 |
| `report.json` | 结构化分析结果 |
| 其他阶段文件 | 执行证据、参考比较、源码获取记录和模型分析结果 |

报告包含交易审查结果；完成源码定位时，还包含候选函数排名、源码位置、证据引用与解释。

模型给出有效的可疑行范围时，报告会进一步展示对应代码行，并区分函数完整范围与预测的可疑范围。证据不足或分析未完成时，报告保留相应状态和原因。

## 代码结构

| 路径 | 功能 |
|---|---|
| `src/cloc/trace/` | 交易轨迹获取与解析 |
| `src/cloc/evidence/` | 动态交易证据图与证据链构建 |
| `src/cloc/reference/` | 正常参照库、结构比较与参考检索 |
| `src/cloc/review/` | LLM 交易审查与流程路由 |
| `src/cloc/source/` | 合约源码获取、索引与切片 |
| `src/cloc/localization/` | 源码候选分析与定位排序 |
| `src/cloc/runtime/` | 模型服务调用与运行配置 |
| `src/cloc/reporting/` | 分析报告生成 |
| `src/cloc/pipeline.py` | 完整分析流程 |
| `src/cloc/cli.py` | 命令行入口 |
