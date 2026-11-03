from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validate_normal_baseline_batch import (
    BatchNormalSample,
    NormalBatchManifest,
    build_normal_manifest,
    finalize_normal_baseline,
    main,
    run_chain_batch,
)
from scripts.validate_real_normal_baseline import RpcValidationError
from src.cloc_v2.normal_distance import AssetBehavior, CallStep, ChainFeatureRecord, MainAnchorKey


def tx(number: int) -> str:
    return "0x" + f"{number:064x}"


def addr(character: str) -> str:
    return "0x" + character * 40


def manifest(*samples: BatchNormalSample) -> NormalBatchManifest:
    counts = {
        "eth": sum(item.chain_key == "eth" for item in samples),
        "bsc": sum(item.chain_key == "bsc" for item in samples),
    }
    payload = {
        "chain_counts": {key: counts[key] for key in sorted(counts)},
        "samples": [item.to_dict() for item in samples],
    }
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return NormalBatchManifest(tuple(samples), counts, hashlib.sha256(encoded.encode()).hexdigest())


def sample(chain: str, number: int) -> BatchNormalSample:
    chain_id = 1 if chain == "eth" else 56
    return BatchNormalSample(chain, chain_id, tx(number), f"{chain}_normal_{number:06d}")


def feature(record_id: str, *, chain_id: int = 1, uncertain: bool = False, selector: str = "0x11111111") -> dict:
    return ChainFeatureRecord(
        record_id=record_id,
        main_anchor=MainAnchorKey(chain_id, addr("a"), selector),
        call_steps=(CallStep(addr("a"), selector),),
        asset_behaviors=frozenset({AssetBehavior(addr("b"), "OUTFLOW")}),
        uncertainties=("TEST_UNCERTAINTY",) if uncertain else (),
    ).to_dict()


def processed(sample_value: BatchNormalSample, raw: object, *, records: list[dict] | None = None) -> dict:
    return {
        "stage_counts": {
            "call_node_count": 1,
            "result_candidate_count": len(records or ()),
            "strict_chain_count": len(records or ()),
            "clean_record_count": sum(not item["uncertainties"] for item in (records or ())),
        },
        "reason_codes": [],
        "feature_records": records if records is not None else [feature(f"{sample_value.sample_id}:chain_001", chain_id=sample_value.blockchain_chain_id)],
    }


class FakeClient:
    def __init__(self, chain_id: int, trace_actions: list[object] | None = None) -> None:
        self.chain_id = chain_id
        self.trace_actions = list(trace_actions or ())
        self.request_count = 0
        self.methods: list[str] = []
        self.trace_hashes: list[str] = []

    def call(self, method: str, params: list[object]):
        self.request_count += 1
        self.methods.append(method)
        if method == "eth_chainId":
            return hex(self.chain_id)
        self.trace_hashes.append(str(params[0]))
        action = self.trace_actions.pop(0) if self.trace_actions else {"frame": True}
        if isinstance(action, Exception):
            raise action
        return action


class NormalBaselineBatchValidationTests(unittest.TestCase):
    def test_real_dataset_manifest_is_stable_470_plus_151(self) -> None:
        result = build_normal_manifest()
        self.assertEqual(result.chain_counts, {"eth": 470, "bsc": 151})
        self.assertEqual(len(result.samples), 621)
        self.assertEqual(result.samples, tuple(sorted(result.samples, key=lambda item: (item.blockchain_chain_id, item.tx_hash))))

    def test_multifile_nested_hashes_deduplicate_and_never_output_forbidden_dataset_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "category-one").mkdir()
            (root / "category-two").mkdir()
            (root / "category-one" / "a.json").write_text(
                json.dumps({
                    "platform": "Ethereum",
                    "cause": "PRIVATE_CAUSE",
                    "location": "PRIVATE_LOCATION",
                    "fault": {"transaction_hash": tx(99)},
                    "faultless": {"transaction_hash": [[tx(2)], tx(1)]},
                }),
                encoding="utf-8",
            )
            (root / "category-two" / "b.json").write_text(
                json.dumps({
                    "platform": "ETH",
                    "ground_truth": "PRIVATE_GROUND_TRUTH",
                    "faultless": {"transaction_hash": [tx(1)]},
                }),
                encoding="utf-8",
            )
            result = build_normal_manifest(root, expected_counts={"eth": 2, "bsc": 0})
        rendered = json.dumps(result.to_dict(), sort_keys=True)
        self.assertEqual([item.tx_hash for item in result.samples], [tx(1), tx(2)])
        for forbidden in ("category-one", "category-two", "cause", "location", "fault", "ground_truth", tx(99)):
            self.assertNotIn(forbidden, rendered)

    def test_malformed_unrelated_dataset_field_does_not_block_allowed_field_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.json"
            path.write_text(
                '{"platform":"Ethereum","unrelated":[invalid],'
                f'"faultless":{{"transaction_hash":["{tx(1)}"]}}}}',
                encoding="utf-8",
            )
            result = build_normal_manifest(path.parent, expected_counts={"eth": 1, "bsc": 0})
        self.assertEqual(result.samples[0].tx_hash, tx(1))

    def test_eth_and_bsc_manifests_and_batches_are_strictly_separate(self) -> None:
        samples = (sample("eth", 1), sample("bsc", 2))
        value = manifest(*samples)
        with tempfile.TemporaryDirectory() as temporary:
            client = FakeClient(56)
            result = run_chain_batch(
                value,
                chain_key="bsc",
                output_dir=Path(temporary),
                rpc_client=client,
                processor=processed,
            )
        self.assertEqual(result["succeeded_count"], 1)
        self.assertEqual(client.trace_hashes, [samples[1].tx_hash])

    def test_chain_id_mismatch_stops_after_one_confirmation_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            client = FakeClient(56)
            result = run_chain_batch(
                manifest(sample("eth", 1)),
                chain_key="eth",
                output_dir=Path(temporary),
                rpc_client=client,
                processor=processed,
            )
        self.assertEqual(result["stop_code"], "CHAIN_ID_MISMATCH")
        self.assertEqual(client.request_count, 1)
        self.assertEqual(client.trace_hashes, [])

    def test_default_batch_never_requests_more_than_eight_transactions(self) -> None:
        value = manifest(*(sample("eth", index) for index in range(1, 11)))
        with tempfile.TemporaryDirectory() as temporary:
            client = FakeClient(1)
            result = run_chain_batch(value, chain_key="eth", output_dir=Path(temporary), rpc_client=client, processor=processed)
        self.assertEqual(result["succeeded_count"], 8)
        self.assertEqual(client.request_count, 9)
        self.assertEqual(len(client.trace_hashes), 8)

    def test_batch_size_above_eight_is_rejected_before_rpc(self) -> None:
        client = FakeClient(1)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "BETWEEN_1_AND_8"):
            run_chain_batch(
                manifest(sample("eth", 1)), chain_key="eth", output_dir=Path(temporary),
                rpc_client=client, batch_size=9, processor=processed,
            )
        self.assertEqual(client.request_count, 0)

    def test_each_success_is_checkpointed_and_restart_skips_it(self) -> None:
        value = manifest(sample("eth", 1), sample("eth", 2))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first_client = FakeClient(1)
            first = run_chain_batch(value, chain_key="eth", output_dir=output, rpc_client=first_client, batch_size=1, processor=processed)
            checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
            first_state = checkpoint["samples"][value.samples[0].sample_key]
            self.assertEqual(first_state["status"], "SUCCEEDED")
            self.assertTrue((output / first_state["artifact"]).exists())
            second_client = FakeClient(1)
            second = run_chain_batch(value, chain_key="eth", output_dir=output, rpc_client=second_client, batch_size=1, processor=processed)
        self.assertEqual((first["succeeded_count"], second["succeeded_count"]), (1, 1))
        self.assertEqual(second_client.trace_hashes, [value.samples[1].tx_hash])

    def test_network_error_keeps_prior_success_and_stops_batch(self) -> None:
        value = manifest(sample("eth", 1), sample("eth", 2), sample("eth", 3))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            client = FakeClient(1, [{"ok": True}, RpcValidationError("RPC_NETWORK_ERROR"), {"never": True}])
            result = run_chain_batch(value, chain_key="eth", output_dir=output, rpc_client=client, processor=processed)
            checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "STOPPED")
        self.assertEqual(result["stop_code"], "RPC_NETWORK_ERROR")
        self.assertEqual(client.request_count, 3)
        self.assertEqual(checkpoint["samples"][value.samples[0].sample_key]["status"], "SUCCEEDED")
        self.assertEqual(checkpoint["samples"][value.samples[1].sample_key]["status"], "FAILED")
        self.assertEqual(checkpoint["samples"][value.samples[2].sample_key]["status"], "PENDING")

    def test_nonnetwork_trace_failure_is_sanitized_and_remaining_sample_continues(self) -> None:
        value = manifest(sample("eth", 1), sample("eth", 2))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            client = FakeClient(1, [RpcValidationError("RPC_REMOTE_ERROR"), {"ok": True}])
            result = run_chain_batch(value, chain_key="eth", output_dir=output, rpc_client=client, processor=processed)
            checkpoint_text = (output / "checkpoint.json").read_text(encoding="utf-8")
            checkpoint = json.loads(checkpoint_text)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual((result["failed_count"], result["succeeded_count"]), (1, 1))
        self.assertEqual(checkpoint["samples"][value.samples[0].sample_key]["failure_code"], "RPC_REMOTE_ERROR")
        self.assertNotIn("provider", checkpoint_text.lower())

    def test_failed_sample_is_not_implicitly_retried(self) -> None:
        value = manifest(sample("eth", 1))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first_client = FakeClient(1, [RpcValidationError("RPC_REMOTE_ERROR")])
            run_chain_batch(value, chain_key="eth", output_dir=output, rpc_client=first_client, processor=processed)
            second_client = FakeClient(1)
            result = run_chain_batch(value, chain_key="eth", output_dir=output, rpc_client=second_client, processor=processed)
        self.assertEqual(result["succeeded_count"], 0)
        self.assertEqual(second_client.methods, ["eth_chainId"])

    def test_explicit_retry_requests_failed_sample(self) -> None:
        value = manifest(sample("eth", 1))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_chain_batch(
                value, chain_key="eth", output_dir=output,
                rpc_client=FakeClient(1, [RpcValidationError("RPC_REMOTE_ERROR")]), processor=processed,
            )
            client = FakeClient(1)
            result = run_chain_batch(
                value, chain_key="eth", output_dir=output,
                rpc_client=client, retry_failed=True, processor=processed,
            )
        self.assertEqual(result["succeeded_count"], 1)
        self.assertEqual(len(client.trace_hashes), 1)

    def test_manifest_drift_blocks_before_rpc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_chain_batch(manifest(sample("eth", 1)), chain_key="eth", output_dir=output, rpc_client=FakeClient(1), processor=processed)
            client = FakeClient(1)
            with self.assertRaisesRegex(ValueError, "DRIFT"):
                run_chain_batch(manifest(sample("eth", 2)), chain_key="eth", output_dir=output, rpc_client=client, processor=processed)
        self.assertEqual(client.request_count, 0)

    def test_sample_artifact_rejects_sensitive_or_amount_fields(self) -> None:
        def unsafe(sample_value, raw):
            record = feature(f"{sample_value.sample_id}:chain_001")
            record["input"] = "PRIVATE_CALLDATA"
            return processed(sample_value, raw, records=[record])

        value = manifest(sample("eth", 1))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = run_chain_batch(value, chain_key="eth", output_dir=output, rpc_client=FakeClient(1), processor=unsafe)
            combined = "".join(path.read_text(encoding="utf-8") for path in output.rglob("*.json"))
        self.assertEqual(result["failed_count"], 1)
        self.assertNotIn("PRIVATE_CALLDATA", combined)
        self.assertFalse(any((output / "samples").rglob("*.json")) if (output / "samples").exists() else False)

    def test_feature_uncertainties_are_saved_as_fixed_codes(self) -> None:
        def free_form(sample_value, raw):
            record = feature(f"{sample_value.sample_id}:chain_001", uncertain=True)
            record["uncertainties"] = ["free form private explanation"]
            return processed(sample_value, raw, records=[record])

        value = manifest(sample("eth", 1))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_chain_batch(value, chain_key="eth", output_dir=output, rpc_client=FakeClient(1), processor=free_form)
            artifact = json.loads(next((output / "samples").rglob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(
            artifact["feature_records"][0]["uncertainties"],
            ["NON_MACHINE_READABLE_UNCERTAINTY"],
        )
        self.assertNotIn("private explanation", json.dumps(artifact))

    def test_default_cli_mode_never_loads_config_or_network(self) -> None:
        with patch("scripts.validate_normal_baseline_batch._configured_client", side_effect=AssertionError("config read")):
            self.assertEqual(main(["batch", "--chain", "eth"]), 0)

    def test_finalize_is_deterministic_clean_only_and_excludes_same_transaction(self) -> None:
        samples = tuple(sample("eth", index) for index in range(1, 6))
        value = manifest(*samples)

        def multi_record(sample_value, raw):
            records = [feature(f"{sample_value.sample_id}:clean")]
            if sample_value == samples[0]:
                records.append(feature(f"{sample_value.sample_id}:second-clean"))
                records.append(feature(f"{sample_value.sample_id}:uncertain", uncertain=True))
            return processed(sample_value, raw, records=records)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_chain_batch(
                value, chain_key="eth", output_dir=output,
                rpc_client=FakeClient(1), batch_size=5, processor=multi_record,
            )
            first = finalize_normal_baseline(output, write=False)
            second = finalize_normal_baseline(output, write=False)
        self.assertEqual(first, second)
        self.assertEqual(first["clean_record_count"], 6)
        self.assertEqual([item["calibrated_group_count"] for item in first["settings"]], [1, 1, 1])
        group = first["settings"][0]["groups"][0]
        self.assertEqual((group["clean_chain_count"], group["distinct_normal_transaction_count"]), (6, 5))
        self.assertIsNotNone(group["threshold"])
        for setting in first["leave_one_out"]:
            for query in setting["queries"]:
                query_sample_id = next(
                    item.sample_id
                    for item in samples
                    if item.sample_key == query["query_transaction_key"]
                )
                for match in query["top_matches"]:
                    self.assertFalse(match["normal_record_id"].startswith(query_sample_id + ":"))

    def test_finalize_writes_only_when_explicitly_requested(self) -> None:
        value = manifest(sample("eth", 1), sample("eth", 2))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_chain_batch(value, chain_key="eth", output_dir=output, rpc_client=FakeClient(1), processor=processed)
            finalize_normal_baseline(output, write=False)
            self.assertFalse((output / "formal_normal_baseline.json").exists())
            finalize_normal_baseline(output, write=True)
            self.assertTrue((output / "formal_normal_baseline.json").exists())


if __name__ == "__main__":
    unittest.main()
