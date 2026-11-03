from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.cloc_v2.formal_normal_baseline_loader import (
    load_formal_normal_baseline,
    resolve_formal_normal_transaction_identity,
    serialize_formal_normal_baseline_summary,
    serialize_formal_normal_transaction_identity,
)
from src.cloc_v2.normal_baseline import build_normal_baseline
from src.cloc_v2.normal_distance import AssetBehavior, CallStep, ChainFeatureRecord, MainAnchorKey
from src.cloc_v2.phase1_normality_orchestration import FORMAL_MIN_NORMAL_GROUP_SIZE


ADDRESS = "0x" + "a" * 40
TOKEN = "0x" + "c" * 40
SELECTOR = "0x11111111"


def record(record_id: str, *, chain_id: int = 1, uncertainties=()) -> ChainFeatureRecord:
    return ChainFeatureRecord(
        record_id,
        MainAnchorKey(chain_id, ADDRESS, SELECTOR),
        (CallStep(ADDRESS, SELECTOR),),
        frozenset({AssetBehavior(TOKEN, "OUTFLOW")}),
        tuple(uncertainties),
    )


def digest(samples, counts) -> str:
    payload = {"chain_counts": {key: counts[key] for key in sorted(counts)}, "samples": samples}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class SyntheticWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records = (
            record("opaque:not-an-owner:alpha"),
            record("opaque:not-an-owner:beta"),
            record("opaque:not-an-owner:gamma"),
        )
        self.samples = [
            {
                "chain_key": "eth",
                "blockchain_chain_id": 1,
                "tx_hash": "0x" + f"{index:064x}",
                "sample_id": f"explicit-ethereum-origin-{index}",
            }
            for index in range(1, 4)
        ]
        self.samples.append(
            {
                "chain_key": "bsc",
                "blockchain_chain_id": 56,
                "tx_hash": "0x" + f"{99:064x}",
                "sample_id": "explicit-bsc-origin-1",
            }
        )
        counts = {"eth": 3, "bsc": 1}
        manifest_hash = digest(self.samples, counts)
        self.manifest = {
            "schema_version": "formal_normal_batch_manifest_v1",
            "manifest_sha256": manifest_hash,
            "chain_counts": counts,
            "samples": self.samples,
        }
        self.checkpoint = {
            "schema_version": "formal_normal_batch_checkpoint_v1",
            "manifest_sha256": manifest_hash,
            "rpc_request_count": 3,
            "last_stop_code": None,
            "samples": {},
        }
        origin = {}
        for sample, item in zip(self.samples[:3], self.records, strict=True):
            relative = f"samples/eth/{sample['tx_hash']}.json"
            key = f"eth:{sample['tx_hash']}"
            self.checkpoint["samples"][key] = {
                "status": "SUCCEEDED", "attempt_count": 1, "failure_code": None, "artifact": relative,
            }
            origin[item.record_id] = sample["sample_id"]
            write_json(
                root / relative,
                {
                    "schema_version": "formal_normal_batch_sample_v1",
                    "chain_key": "eth",
                    "blockchain_chain_id": 1,
                    "tx_hash": sample["tx_hash"],
                    "sample_id": sample["sample_id"],
                    "stage_counts": {},
                    "reason_codes": [],
                    "feature_records": [item.to_dict()],
                },
            )
        bsc = self.samples[3]
        self.checkpoint["samples"][f"bsc:{bsc['tx_hash']}"] = {
            "status": "PENDING", "attempt_count": 0, "failure_code": None, "artifact": None,
        }
        settings = []
        for minimum in (2, 3, 5):
            baseline = build_normal_baseline(self.records, min_group_size=minimum)
            groups = [
                {
                    "main_anchor": group.main_anchor.to_dict(),
                    "clean_chain_count": len(group.records),
                    "distinct_normal_transaction_count": len({origin[item.record_id] for item in group.records}),
                    "calibration_status": group.calibration.status,
                    "threshold": group.calibration.to_dict()["threshold"],
                }
                for group in baseline.groups
            ]
            settings.append(
                {
                    "min_group_size": minimum,
                    "calibrated_group_count": baseline.calibrated_group_count,
                    "uncalibrated_group_count": baseline.uncalibrated_group_count,
                    "groups": groups,
                    "baseline": baseline.to_dict(),
                }
            )
        self.final = {
            "schema_version": "formal_normal_batch_finalize_v1",
            "manifest_sha256": manifest_hash,
            "successful_transaction_count": 3,
            "clean_record_count": 3,
            "settings": settings,
            "leave_one_out": [],
        }
        self.flush_top_level()

    def flush_top_level(self) -> None:
        write_json(self.root / "normal_manifest.json", self.manifest)
        write_json(self.root / "checkpoint.json", self.checkpoint)
        write_json(self.root / "formal_normal_baseline.json", self.final)

    def formal_setting(self):
        return next(item for item in self.final["settings"] if item["min_group_size"] == 3)

    def artifact(self, index: int):
        sample = self.samples[index]
        path = self.root / f"samples/{sample['chain_key']}/{sample['tx_hash']}.json"
        return path, json.loads(path.read_text(encoding="utf-8"))


class FormalNormalBaselineLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = SyntheticWorkspace(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_valid_directory_loads(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        self.assertEqual(loaded.baseline.record_count, 3)
        self.assertTrue(loaded.summary.reconstructed_baseline_match)

    def test_selects_formal_min_group_size_three(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        self.assertEqual(loaded.baseline.min_group_size, FORMAL_MIN_NORMAL_GROUP_SIZE)
        self.assertEqual(loaded.summary.formal_min_group_size, 3)

    def test_does_not_select_sensitivity_settings_two_or_five(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        self.assertEqual(loaded.baseline.calibrated_group_count, 1)
        self.assertNotEqual(loaded.baseline.min_group_size, 2)
        self.assertNotEqual(loaded.baseline.min_group_size, 5)

    def test_reconstructed_baseline_matches_embedded_baseline(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        self.assertEqual(loaded.baseline.to_dict(), self.workspace.formal_setting()["baseline"])

    def test_record_origin_uses_explicit_sample_containment(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        self.assertEqual(
            loaded.normal_record_transaction_ids["opaque:not-an-owner:alpha"],
            "explicit-ethereum-origin-1",
        )

    def test_record_id_is_not_parsed_to_guess_owner(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        self.assertNotEqual(
            loaded.normal_record_transaction_ids["opaque:not-an-owner:alpha"],
            "alpha",
        )

    def test_mapping_count_matches_baseline_record_count(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        self.assertEqual(len(loaded.normal_record_transaction_ids), loaded.baseline.record_count)
        self.assertEqual(loaded.summary.record_origin_mapping_count, 3)

    def test_known_normal_transaction_identity_resolves_from_validated_manifest(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        sample = self.workspace.samples[0]
        identity = resolve_formal_normal_transaction_identity(
            loaded, 1, sample["tx_hash"].upper()
        )
        self.assertEqual(identity.status, "KNOWN_NORMAL_ORIGIN")
        self.assertTrue(identity.usable_for_direct_normal_identity)
        self.assertEqual(identity.normal_origin_id, sample["sample_id"])

    def test_legal_unknown_transaction_is_not_in_normal_library(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        identity = resolve_formal_normal_transaction_identity(loaded, 1, "0x" + "f" * 64)
        self.assertEqual(identity.status, "NOT_IN_NORMAL_LIBRARY")
        self.assertFalse(identity.usable_for_direct_normal_identity)
        self.assertIsNone(identity.normal_origin_id)

    def test_invalid_hash_and_chain_mismatch_are_unresolved(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        invalid_hash = resolve_formal_normal_transaction_identity(loaded, 1, "not-a-hash")
        wrong_chain = resolve_formal_normal_transaction_identity(
            loaded, 56, self.workspace.samples[0]["tx_hash"]
        )
        self.assertEqual(invalid_hash.status, "UNRESOLVED")
        self.assertEqual(wrong_chain.status, "UNRESOLVED")
        self.assertNotEqual(invalid_hash.status, "NOT_IN_NORMAL_LIBRARY")

    def test_incomplete_transaction_origin_evidence_is_unresolved(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        incomplete = replace(loaded, normal_transaction_hash_origins={})
        identity = resolve_formal_normal_transaction_identity(
            incomplete, 1, self.workspace.samples[0]["tx_hash"]
        )
        self.assertEqual(identity.status, "UNRESOLVED")
        self.assertIn("NORMAL_TRANSACTION_ORIGIN_EVIDENCE_INCOMPLETE", identity.reason_codes)

    def test_identity_serialization_and_repr_hide_hash_and_origin(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        sample = self.workspace.samples[0]
        identity = resolve_formal_normal_transaction_identity(loaded, 1, sample["tx_hash"])
        text = serialize_formal_normal_transaction_identity(identity)
        self.assertNotIn(sample["tx_hash"], text)
        self.assertNotIn(sample["sample_id"], text)
        self.assertNotIn(sample["tx_hash"], repr(identity))
        self.assertNotIn(sample["sample_id"], repr(identity))
        self.assertEqual(text, serialize_formal_normal_transaction_identity(identity))

    def test_final_schema_error_is_rejected(self) -> None:
        self.workspace.final["schema_version"] = "wrong"
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_manifest_schema_error_is_rejected(self) -> None:
        self.workspace.manifest["schema_version"] = "wrong"
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_checkpoint_schema_error_is_rejected(self) -> None:
        self.workspace.checkpoint["schema_version"] = "wrong"
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_manifest_sha256_mismatch_is_rejected(self) -> None:
        self.workspace.final["manifest_sha256"] = "0" * 64
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_manifest_content_digest_mismatch_is_rejected(self) -> None:
        self.workspace.manifest["samples"][0]["sample_id"] = "tampered"
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_formal_setting_missing_is_rejected(self) -> None:
        self.workspace.final["settings"] = [item for item in self.workspace.final["settings"] if item["min_group_size"] != 3]
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_formal_setting_duplicate_is_rejected(self) -> None:
        self.workspace.final["settings"].append(copy.deepcopy(self.workspace.formal_setting()))
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_successful_checkpoint_without_artifact_is_rejected(self) -> None:
        first = next(iter(self.workspace.checkpoint["samples"].values()))
        first["artifact"] = None
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_artifact_path_escape_is_rejected(self) -> None:
        first = next(iter(self.workspace.checkpoint["samples"].values()))
        first["artifact"] = "../outside.json"
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_missing_artifact_file_is_rejected(self) -> None:
        first = next(iter(self.workspace.checkpoint["samples"].values()))
        first["artifact"] = "samples/eth/missing.json"
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_sample_identity_mismatch_is_rejected(self) -> None:
        path, artifact = self.workspace.artifact(0)
        artifact["sample_id"] = "other"
        write_json(path, artifact)
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_sample_schema_mismatch_is_rejected(self) -> None:
        path, artifact = self.workspace.artifact(0)
        artifact["schema_version"] = "wrong"
        write_json(path, artifact)
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_feature_record_unknown_field_is_rejected(self) -> None:
        path, artifact = self.workspace.artifact(0)
        artifact["feature_records"][0]["unknown"] = True
        write_json(path, artifact)
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_feature_record_amount_is_rejected(self) -> None:
        path, artifact = self.workspace.artifact(0)
        artifact["feature_records"][0]["amount_raw"] = "1"
        write_json(path, artifact)
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_feature_record_label_is_rejected(self) -> None:
        path, artifact = self.workspace.artifact(0)
        artifact["feature_records"][0]["ground_truth"] = "hidden"
        write_json(path, artifact)
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_invalid_asset_direction_is_rejected(self) -> None:
        path, artifact = self.workspace.artifact(0)
        artifact["feature_records"][0]["asset_behaviors"][0]["direction"] = "SIDEWAYS"
        write_json(path, artifact)
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_record_id_duplicate_across_samples_is_rejected(self) -> None:
        first_path, first = self.workspace.artifact(0)
        second_path, second = self.workspace.artifact(1)
        second["feature_records"][0]["record_id"] = first["feature_records"][0]["record_id"]
        write_json(second_path, second)
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_baseline_record_without_source_is_rejected(self) -> None:
        self.workspace.formal_setting()["baseline"]["groups"][0]["records"][0]["record_id"] = "ghost-record"
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_clean_sample_record_missing_from_baseline_is_rejected(self) -> None:
        self.workspace.formal_setting()["baseline"]["groups"][0]["records"].pop()
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_tampered_threshold_is_rejected(self) -> None:
        self.workspace.formal_setting()["baseline"]["groups"][0]["calibration"]["threshold"] = "0.5"
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_tampered_record_order_is_rejected(self) -> None:
        records = self.workspace.formal_setting()["baseline"]["groups"][0]["records"]
        records.reverse()
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_failed_bsc_sample_is_allowed_and_ignored(self) -> None:
        key = next(key for key in self.workspace.checkpoint["samples"] if key.startswith("bsc:"))
        self.workspace.checkpoint["samples"][key]["status"] = "FAILED"
        self.workspace.checkpoint["samples"][key]["failure_code"] = "FIXED_FAILURE"
        self.workspace.flush_top_level()
        loaded = load_formal_normal_baseline(self.root)
        self.assertTrue(loaded.summary.ethereum_only)

    def test_successful_bsc_record_is_rejected(self) -> None:
        sample = self.workspace.samples[3]
        key = f"bsc:{sample['tx_hash']}"
        relative = f"samples/bsc/{sample['tx_hash']}.json"
        self.workspace.checkpoint["samples"][key].update(status="SUCCEEDED", attempt_count=1, artifact=relative)
        write_json(
            self.root / relative,
            {
                "schema_version": "formal_normal_batch_sample_v1",
                **sample,
                "stage_counts": {},
                "reason_codes": [],
                "feature_records": [record("bsc-record", chain_id=56).to_dict()],
            },
        )
        self.workspace.flush_top_level()
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_non_ethereum_anchor_is_rejected(self) -> None:
        path, artifact = self.workspace.artifact(0)
        artifact["feature_records"][0]["main_anchor"]["blockchain_chain_id"] = 56
        write_json(path, artifact)
        with self.assertRaises(ValueError):
            load_formal_normal_baseline(self.root)

    def test_safe_summary_contains_no_evidence_identity(self) -> None:
        summary = serialize_formal_normal_baseline_summary(load_formal_normal_baseline(self.root).summary)
        for forbidden in (ADDRESS, TOKEN, SELECTOR, "opaque:not-an-owner", "explicit-ethereum-origin", "0x" + f"{1:064x}"):
            self.assertNotIn(forbidden, summary)

    def test_summary_serialization_is_stable(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        self.assertEqual(
            serialize_formal_normal_baseline_summary(loaded.summary),
            serialize_formal_normal_baseline_summary(load_formal_normal_baseline(self.root).summary),
        )

    def test_sensitive_loaded_fields_are_hidden_from_repr(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        text = repr(loaded)
        self.assertNotIn("opaque:not-an-owner", text)
        self.assertNotIn("explicit-ethereum-origin", text)
        self.assertNotIn(ADDRESS, text)

    def test_loader_does_not_write_files(self) -> None:
        before = {path.relative_to(self.root): hashlib.sha256(path.read_bytes()).hexdigest() for path in self.root.rglob("*.json")}
        load_formal_normal_baseline(self.root)
        after = {path.relative_to(self.root): hashlib.sha256(path.read_bytes()).hexdigest() for path in self.root.rglob("*.json")}
        self.assertEqual(before, after)

    def test_loader_never_scans_dataset_or_category_directories(self) -> None:
        with patch.object(Path, "rglob", side_effect=AssertionError("dataset scan forbidden")):
            loaded = load_formal_normal_baseline(self.root)
        self.assertEqual(loaded.summary.clean_record_count, 3)

    def test_source_file_count_is_top_level_plus_success_artifacts(self) -> None:
        loaded = load_formal_normal_baseline(self.root)
        self.assertEqual(loaded.summary.source_files_read_count, 6)


if __name__ == "__main__":
    unittest.main()
