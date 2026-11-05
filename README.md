# CLoc: LLM-Assisted Fault Localization for Blockchain Applications

CLoc is a fault localization framework for blockchain applications. It combines transaction execution traces, normal transaction references, and smart contract source code to support LLM-based transaction review and fault localization. The framework produces ranked candidate locations with supporting execution evidence and source-level explanations.

CLoc currently supports Ethereum and BNB Smart Chain.

## Overview

- **Execution evidence construction:** Extract call relationships and asset transfers from transaction traces to build dynamic transaction evidence graphs and candidate evidence chains.
- **Reference-guided review:** Retrieve comparable normal transaction chains and use an LLM to compare execution structure and asset behavior, determining whether further localization is needed.
- **Source-informed localization:** Retrieve relevant verified contract source code on demand and analyze candidate functions and suspicious lines alongside the original execution evidence.
- **Analysis reports:** Produce Top-k candidate locations, evidence references, and explanations in JSON and Markdown formats.

## Quick Start

### 1. Installation

Python 3.10 or later is required. From the repository root, run:

```powershell
python -m pip install -e .
```

After installation, use either `cloc` or `python -m cloc`.

### 2. Configure Services

CLoc uses an RPC service to retrieve transaction execution traces, a block explorer to retrieve verified source code, and an LLM service for transaction review and fault localization.

Set the following environment variables in PowerShell, replacing the placeholders with your own configuration:

```powershell
$env:ETH_RPC_URL = "<Ethereum RPC URL>"
$env:ETHERSCAN_API_KEY = "<Block explorer API key>"
$env:CLOC_LLM_API_KEY = "<LLM API key>"
$env:CLOC_LLM_MODEL = "gpt-4o"
```

For BNB Smart Chain transactions, also set:

```powershell
$env:BSC_RPC_URL = "<BNB Smart Chain RPC URL>"
```

The RPC service must support `debug_traceTransaction` with `callTracer` for the historical transactions being analyzed.

To use another service compatible with the Chat Completions API, set:

```powershell
$env:CLOC_LLM_BASE_URL = "<Model service API base URL>"
$env:CLOC_LLM_MODEL = "<Model name>"
```

The base URL must omit the `/chat/completions` suffix. Supply personal RPC URLs and API keys through environment variables, and keep them out of files committed to the repository.

### 3. Build a Normal Reference Library

Prepare a manifest of normal transaction traces using the format in `examples/normal_manifest.example.json`. Record the basis for treating each transaction as normal. Trace paths are resolved relative to the manifest file.

```powershell
cloc build-reference --manifest data/normal_manifest.json --output data/normal_reference.json
```

References are grouped by chain and business entry point. Each group requires at least three usable reference records for reference-guided review. Transactions without sufficient comparable references proceed to source-based localization.

### 4. Analyze a Transaction

Provide a transaction hash, its chain, and the normal reference library:

```powershell
cloc analyze --tx-hash "<Transaction hash>" --chain eth --baseline data/normal_reference.json --output output/query
```

CLoc retrieves the execution trace, constructs evidence chains, and invokes the LLM for reference-guided review. When further analysis is needed, it retrieves relevant source code, ranks candidate locations, and generates a report.

You can also provide an existing transaction execution trace as a JSON file:

```powershell
cloc analyze --trace data/query.trace.json --chain eth --baseline data/normal_reference.json --output output/query
```

## Command-Line Options

| Option | Description |
|---|---|
| `--tx-hash` | Transaction hash; mutually exclusive with `--trace` |
| `--trace` | Path to a saved JSON callTracer trace |
| `--chain` | Blockchain: `eth` or `bsc` |
| `--baseline` | Path to the normal reference library |
| `--output` | Directory for analysis results |
| `--model` | Model to use for this analysis |
| `--top-k` | Number of candidates to return: 1-5; default: 5 |
| `--source-cache` | Source code cache directory; default: `data/source_cache` |

For example, analyze a BNB Smart Chain transaction and return the top three candidate locations:

```powershell
cloc analyze --tx-hash "<Transaction hash>" --chain bsc --baseline data/normal_reference.json --top-k 3 --output output/query
```

For the complete list of options, run:

```powershell
cloc analyze --help
```

## Method

### Stage I: Execution Evidence Construction

CLoc converts an execution trace into a dynamic transaction evidence graph, extracting call information and observed asset transfers. It organizes CALL and ASSET evidence chains around transaction outcomes to provide execution context for subsequent analysis.

### Stage II: Reference-Guided Transaction Review

Using structural distance, CLoc retrieves three normal reference chains for each eligible query chain. The LLM compares call structure, asset flow directions, and observed amounts, returning a `NORMAL`, `SUSPICIOUS`, or `UNCERTAIN` verdict with an explanation.

A transaction exits with a normal verdict only when all required query chains have valid review results and every verdict is `NORMAL`. All other transactions proceed to Stage III.

### Stage III: Source-Informed Candidate Ranking

CLoc retrieves verified source code using the contract addresses in the execution evidence, then matches functions and extracts source slices. The LLM combines the original execution evidence with these slices to rank candidate locations and explain the relationship between execution behavior and source code.

This stage uses the original execution evidence. Stage II model explanations are not passed to the localization stage.

## Analysis Results

Results are saved to the directory specified by `--output`:

| File | Contents |
|---|---|
| `report.md` | Human-readable analysis report |
| `report.json` | Structured analysis results |
| Other stage artifacts | Execution evidence, reference comparisons, source acquisition records, and model analysis results |

Reports include the transaction review outcome. When source-based localization completes, they also include candidate function rankings, source locations, evidence references, and explanations.

Valid model-selected suspicious line ranges are reported separately from the full function ranges. When evidence is insufficient or analysis cannot be completed, the report records the corresponding status and reasons.

## Project Structure

| Path | Responsibility |
|---|---|
| `src/cloc/trace/` | Transaction trace acquisition and parsing |
| `src/cloc/evidence/` | Dynamic transaction evidence graphs and evidence chains |
| `src/cloc/reference/` | Normal reference library, structural comparison, and reference retrieval |
| `src/cloc/review/` | LLM transaction review and routing |
| `src/cloc/source/` | Contract source acquisition, indexing, and slicing |
| `src/cloc/localization/` | Source candidate analysis and ranking |
| `src/cloc/runtime/` | Model service clients and runtime configuration |
| `src/cloc/reporting/` | Analysis report generation |
| `src/cloc/pipeline.py` | End-to-end analysis workflow |
| `src/cloc/cli.py` | Command-line interface |
