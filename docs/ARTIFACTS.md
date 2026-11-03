# Data and experiment artifacts

The core upload contains a small offline example, not the entire workspace dataset or historical experiments.

| Workflow | Additional inputs |
|---|---|
| Core endpoint/graph checks | The five included MonoX JSON files and synthetic unit-test inputs |
| Dataset lookup and broad regression | `dataset/` JSON cases, with provenance and release terms reviewed separately |
| Source-level localization | Local traces, verified source cache, proxy/source bindings and normal-reference library |
| Historical 15/19/34-transaction evaluations | Frozen sample manifests, predictions, evaluation labels and dependencies under the original `data/prototypes/` paths |
| Grouped-mean lambda scan | `output/experiments/outflow_lambda_20261015/` final inputs/manifest plus its prechange source snapshot; earlier attempts are debugging artifacts |
| Comparison baselines | Coverage/graph artifacts and, where required, separately obtained upstream implementations |

Keep full trace caches, downloaded contract sources, response caches, prior attempts and duplicated source snapshots outside the primary source upload. Provide an explicit data manifest and checksums when publishing a separate reproducibility bundle.

Some historical tests assert dataset counts 470/151 while the audited local dataset contains 466/148 normal transactions for Ethereum/BSC. That prior mismatch is documented rather than corrected by changing labels or counts for this preparation. The complete test suite also references artifacts beyond this small example.

No network or paid LLM requests are needed for the documented small offline checks. Run budgets, provider/model versions, source/prompt versions and normal-reference/query isolation must be frozen before any new complete evaluation.
