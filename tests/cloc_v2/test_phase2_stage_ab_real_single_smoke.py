from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_phase2_stage_ab_real_single_smoke import (
    SAFE_ALIAS,
    LockedStageABSample,
    _base_report,
    _canonical,
    _sha,
    _validate_public,
    select_locked_sample,
    validate_stage_b_alignment,
)


def _history():
    samples = [
        {"sample_alias": "eval_002", "blockchain_chain_id": 1, "order": 2, "tx_hash": "0x" + "2" * 64},
        {"sample_alias": "eval_001", "blockchain_chain_id": 1, "order": 1, "tx_hash": "0x" + "1" * 64},
    ]
    manifest_core = {
        "schema_version": "locked_v1",
        "selection_namespace": "neutral",
        "selection_seed": "fixed",
        "samples": samples,
    }
    manifest = {**manifest_core, "manifest_sha256": _sha(manifest_core)}
    frozen_core = {
        "schema_version": "frozen_v1",
        "kind": "blind",
        "locked_manifest_sha256": manifest["manifest_sha256"],
        "external_call_counts": {"rpc": 0, "source": 0, "deepseek": 0},
        "predictions": [
            {"sample_alias": "eval_002", "stage_a_request_count": 1},
            {"sample_alias": "eval_001", "stage_a_request_count": 1},
        ],
    }
    frozen = {**frozen_core, "payload_sha256": _sha(frozen_core)}
    checkpoint = {
        "status": "COMPLETED",
        "manifest_sha256": manifest["manifest_sha256"],
        "frozen_predictions_sha256": frozen["payload_sha256"],
    }
    return manifest, frozen, checkpoint


class StageABRealSingleSmokeTests(unittest.TestCase):
    def test_selection_uses_first_neutral_alias_with_stage_a_attempt(self):
        manifest, frozen, checkpoint = _history()
        lock = select_locked_sample(manifest, frozen, checkpoint)
        self.assertEqual(lock.sample_alias, SAFE_ALIAS)
        self.assertEqual(lock.tx_hash, "0x" + "1" * 64)
        public = lock.public_manifest()
        self.assertNotIn("tx_hash", _canonical(public))
        self.assertNotIn("eval_001", _canonical(public))

    def test_selection_rejects_sha_drift_and_missing_attempts(self):
        manifest, frozen, checkpoint = _history()
        manifest["samples"][0]["order"] = 9
        with self.assertRaisesRegex(ValueError, "SHA_INVALID"):
            select_locked_sample(manifest, frozen, checkpoint)
        manifest, frozen, checkpoint = _history()
        for row in frozen["predictions"]:
            row["stage_a_request_count"] = 0
        frozen_core = {key: value for key, value in frozen.items() if key != "payload_sha256"}
        frozen["payload_sha256"] = _sha(frozen_core)
        checkpoint["frozen_predictions_sha256"] = frozen["payload_sha256"]
        with self.assertRaisesRegex(ValueError, "NO_HISTORICAL"):
            select_locked_sample(manifest, frozen, checkpoint)

    def test_stage_b_alignment_requires_exact_order_and_no_duplicates(self):
        prep = SimpleNamespace(pre_llm_candidates=(
            {"candidate_alias": "C2", "included_in_prompt": True},
            {"candidate_alias": "C1", "included_in_prompt": True},
        ))
        self.assertEqual(validate_stage_b_alignment(("C2", "C1"), prep), (True, ("C2", "C1")))
        self.assertFalse(validate_stage_b_alignment(("C1", "C2"), prep)[0])
        duplicate = SimpleNamespace(pre_llm_candidates=(
            {"candidate_alias": "C1", "included_in_prompt": True},
            {"candidate_alias": "C1", "included_in_prompt": True},
        ))
        self.assertFalse(validate_stage_b_alignment(("C1", "C1"), duplicate)[0])

    def test_report_is_redacted_and_budgeted(self):
        lock = LockedStageABSample(SAFE_ALIAS, 1, "0x" + "a" * 64, "b" * 64, "c" * 64)
        report = _base_report(lock)
        _validate_public(report)
        rendered = _canonical(report)
        for forbidden in ("0x" + "a" * 64, "rpc_url", "api_key", "provider_response", "reasoning_content", "location", "cause"):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual(report["request_limits"]["rpc"], 2)
        self.assertEqual(report["request_limits"]["source_network"], 3)
        self.assertEqual(report["request_limits"]["stage_a_deepseek"], 2)
        self.assertEqual(report["request_limits"]["stage_b_deepseek"], 2)

    def test_public_validator_rejects_raw_identity_and_forbidden_keys(self):
        with self.assertRaisesRegex(ValueError, "SENSITIVE"):
            _validate_public({"safe": "0x" + "d" * 40})
        with self.assertRaisesRegex(ValueError, "SENSITIVE"):
            _validate_public({"provider_response": "not retained"})

    def test_no_live_mode_has_no_external_effect_and_no_output(self):
        from scripts.run_phase2_stage_ab_real_single_smoke import main
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "out"
            self.assertEqual(main(["--output-dir", str(target)]), 2)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
