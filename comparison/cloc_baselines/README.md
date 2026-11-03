# CLoc comparison-baseline adapter

This directory is the reproducibility boundary between the native CLoc dataset and the comparison methods.

## What is fixed

- Dataset: all 40 JSON cases under `dataset/`.
- Fault transactions: one representative incident transaction per CLoc JSON case, locked by the checked-in `data/cache/all_tree` artifact. All original transaction groups remain in the manifest for provenance.
- Passing transactions: every transaction listed under `faultless.transaction_hash`.
- Candidate functions: functions covered by at least one failing transaction.
- Ground truth: a dataset `address#source-file#offset:length` label must be contained by the candidate function's source range.
- Metrics: Recall@1/3/5, MFR and MAR, with stable ordinal ranks.

Run `python scripts/build_cloc_baseline_manifest.py` to regenerate `manifest.json`.

Run `python scripts/audit_cloc_baseline_readiness.py` to verify that transaction sets, chains, coverage artifacts and Grace graphs are complete before any paper table is generated.

On a new server, run `python scripts/baseline_server_preflight.py`. It reports only whether required environment variables are present; it never prints their values.

## Why results are not generated from the existing call-tree cache

Tarantula, Ochiai, PRFL and CNNFL require internal function coverage for every failing and passing transaction. CLoc's current `callTracer` artifacts record external call frames and selectors, but do not record every internal Solidity function reached through jumps. Treating those call frames as function coverage would silently exclude ground-truth functions and make the comparison invalid.

The server-side acquisition stage must therefore collect opcode program counters, associate each counter with its executing bytecode address, and map it through the verified compiler source map. Each finished case is written as `coverage/<case_id>.json` using `cloc_function_coverage_v1`.

The checkpointed collector is `scripts/collect_cloc_function_coverage.py`. It reads `ETH_RPC_URL` and `BSC_RPC_URL` from the environment, caches compressed traces and public Sourcify artifacts, and never writes either RPC URL to an artifact. The same pass emits Solidity line coverage and a validated `cloc_grace_graph_v1` artifact.

## Running completed coverage artifacts

```bash
python scripts/run_cloc_baselines.py \
  --manifest comparison/cloc_baselines/manifest.json \
  --coverage-dir comparison/cloc_baselines/coverage \
  --output-dir comparison/cloc_baselines/results
```

The runner refuses partial data unless `--allow-partial` is explicitly passed. Partial results must not be reported as the 40-case CLoc comparison.

`prfl` follows the reference PRFL implementation (PageRank followed by PR-Ochiai). `prfl_malo` is provided only as a compatibility mode for the simplified public MALO reimplementation; the two must not be mixed in one reported table.

Grace additionally requires a source/line graph and is tracked separately by `grace/<case_id>.json`; a coverage matrix alone is not enough to run it faithfully. Run it after the readiness audit with:

```bash
python scripts/run_cloc_grace.py --seed 0 --epochs 15
```

This is explicitly named `grace_cloc_adaptation`. It uses Solidity function/line nodes and transaction test nodes, leave-one-case-out training, and a fixed epoch count. Unlike the public Grace runner, it never selects an epoch from the held-out case's fault rank.
