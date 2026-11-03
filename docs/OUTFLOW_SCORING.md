# R_OUTFLOW grouped aggregation

Current defaults use `D=(M+R)/2`, `C=(L+I_root+I_active)/3`, `score=lambda*D+(1-lambda)*C`, lambda 0.65. Configuration stores `grouped_mean_v1`, `runtime_signals_v1`, precision 50 and ROUND_HALF_UP. Confidence and display signals use four decimal places; raw features and group means are retained for auditing. The graph validator reconstructs new outputs from hash-bound inputs. Historical complete five-weight artifacts use the legacy validation branch.

The feature definitions are unchanged: M is per-asset min-max amount normalization; R is FIFO batch retention; L is trace_index divided by node count; I_root compares the receiver with the root callee; I_active tests membership of the payer in the transaction callee set. The last three are not interchangeable with paper descriptions of execution depth, transaction initiator and current callee.

After local setup, generate a fresh endpoint artifact without contacting a provider:

```bash
python scripts/detect_result_candidates.py --graph data/runs/eth/0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299/transaction_graph_v1.json --annotations data/runs/eth/0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299/runtime_annotations_v1.json --ledger data/runs/eth/0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299/observed_asset_ledger_v1.json --continuity data/runs/eth/0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299/asset_continuity_v1.json --outflow-lambda 0.65 --output output/monox_grouped_results.json
```

Lambda zero is a valid value. Endpoint threshold 0.5000, maximum five candidates per asset, deterministic tie breaking and Top-1 fallback retain their original protocols. R_PROFIT has its own unchanged formula. Normal features carry scoring provenance and reject mixing different scoring configurations.

The lambda scan runner also requires the saved prechange source snapshot and experiment manifest; those are external artifacts, not bundled code. The four-transaction local scan is an endpoint/chain check, not a new function-recall experiment.
