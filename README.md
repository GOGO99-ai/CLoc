# CLoc

CLoc is a research prototype for evidence-constrained function-level fault localization in blockchain applications. It constructs transaction graphs and asset evidence, compares behavior with normal references, and constrains an LLM to verified source candidates.

This repository contains project source, comparison adapters, experiment runners and tests. Historical experiment scripts remain separate entry points; there is no claim that one command reproduces every prior result with current defaults.

## Layout

- `src/cloc_v2/`: current graph, asset, endpoint, normal-reference, source binding and LLM components.
- `src/data_collection/`: collection/parsing helpers, including older optional utilities.
- `src/cloc_baselines/`: project-owned comparison adapters.
- `cloc/`, `web3/`: local compatibility modules required by current imports. `web3/` is a minimal legacy shim, not the Web3.py library.
- `scripts/`: standalone build, replay, evaluation and audit entry points. Keep these together because runners import other runners.
- `tests/`: unit tests and data-dependent integration tests.
- `data/runs/eth/`: five reviewed MonoX input artifacts for the small offline example.
- `comparison/cloc_baselines/`: comparison method notes and optional dependency list, without external repositories or run output.

## Local setup

The preparation check uses Python 3.12. The core requirements file currently contains `requests`.

```bash
python -m pip install -r requirements.txt
```

Copy `config.example.py` to `config.py` (`Copy-Item config.example.py config.py` in PowerShell or `cp config.example.py config.py` in a POSIX shell). Set RPC and model credentials using environment variables when external access is needed. `config.py` and `.env` are ignored. `.env.example` documents variable names and is not loaded automatically.

Older collection utilities have optional imports listed in `requirements-legacy-collection.txt`. Learned comparison methods require their separate ML environment and some upstream implementations; see `comparison/cloc_baselines/`. The optional lists are not a new validated dependency lock.

## Offline checks

After creating `config.py`, run:

```bash
python -m unittest tests.cloc_v2.test_result_detection tests.cloc_v2.test_candidate_causal_graph
python -m src.cloc_v2 --help
```

The included MonoX endpoint artifact uses the legacy scoring protocol. Fresh detection defaults to `grouped_mean_v1`, lambda 0.65. See `docs/OUTFLOW_SCORING.md` for a fresh local detection command.

Full regression tests and historical experiment runners need additional datasets, trace/source caches, reference libraries and frozen manifests that are not bundled here. Missing artifacts must not be interpreted as newly measured localization failures or reproduced results.

## Reproducibility status

The repository includes multiple research profiles and historical runners. The 34-transaction Top-5=70.6% result was a development-set offline postprocessing replay before the new endpoint aggregation. The grouped-mean refactor was checked locally on four transactions without new LLM calls. These are different experiments; no new function Recall@K is claimed by the endpoint refactor.

See `docs/ARTIFACTS.md` for the external data required by each workflow. This preparation has not selected a project license or bundled third-party source repositories.
