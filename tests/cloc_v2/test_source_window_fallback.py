from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.cloc_v2.source_window_fallback import MAX_FILES, MAX_WINDOWS, MODE, build_verified_source_window_fallback


ADDRESS = "0x" + "1" * 40


def _text(lines=220):
    body = ["contract C {", "function run() external {"] + ["uint256 x = 1;" for _ in range(max(0, lines - 4))] + ["}", "}"]
    return "\n".join(body)


def _payload(root: Path, texts):
    records = []
    for index, text in enumerate(texts, start=1):
        path = f"contracts/C{index}.sol"; file = root / "eth" / ADDRESS / path; file.parent.mkdir(parents=True, exist_ok=True); file.write_text(text, encoding="utf-8")
        records.append({"file_path": path, "file_sha256": hashlib.sha256(text.encode()).hexdigest(), "functions": [{"function_name": "run"}]})
    index = {"schema_version": "source_index_v1", "chain": "eth", "addresses": [{"address": ADDRESS, "status": "verified", "verified": True, "proxy_resolution_path": [ADDRESS], "files": records}]}
    graph = {"schema_version": "transaction_graph_v1", "nodes": [{"node_id": "n1", "callee_address": ADDRESS, "function_name": "run"}]}
    exact = {"schema_version": "source_request_slices_v1", "requests": [{"node_id": "n1", "requested_address": ADDRESS, "function_match_status": "FUNCTION_IDENTITY_UNRESOLVED"}], "metadata": {"resolved_count": 0}}
    return index, graph, exact


class SourceWindowFallbackTests(unittest.TestCase):
    def test_exact_slice_is_preserved_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            index, graph, exact = _payload(Path(directory), [_text(20)]); exact["requests"][0]["function_match_status"] = "RESOLVED"
            result = build_verified_source_window_fallback(index, graph, exact, directory)
            self.assertEqual(result["requests"], exact["requests"]); self.assertEqual(result["metadata"]["resolution_mode"], "EXACT_FUNCTION_SLICE")

    def test_verified_source_generates_120_line_overlapping_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            index, graph, exact = _payload(Path(directory), [_text(220)])
            result = build_verified_source_window_fallback(index, graph, exact, directory)
            self.assertEqual([(item["start_line"], item["end_line"]) for item in result["requests"]], [(1, 120), (101, 220)])
            self.assertTrue(all(item["resolution_mode"] == MODE for item in result["requests"]))
            self.assertTrue(all("FUNCTION_IDENTITY_UNRESOLVED" in item["uncertainties"] for item in result["requests"]))

    def test_short_unverified_corrupt_and_unparseable_sources_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); index, graph, exact = _payload(root, [_text(20)])
            result = build_verified_source_window_fallback(index, graph, exact, root); self.assertEqual(len(result["requests"]), 1)
            index["addresses"][0]["verified"] = False; self.assertFalse(build_verified_source_window_fallback(index, graph, exact, root)["requests"])
            index["addresses"][0]["verified"] = True; index["addresses"][0]["files"][0]["file_sha256"] = "0" * 64; self.assertFalse(build_verified_source_window_fallback(index, graph, exact, root)["requests"])

    def test_file_and_window_caps_and_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); index, graph, exact = _payload(root, [_text(220) for _ in range(5)])
            first = build_verified_source_window_fallback(index, graph, exact, root); second = build_verified_source_window_fallback(index, graph, exact, root)
            self.assertEqual(first, second); self.assertLessEqual(first["metadata"]["window_file_count"], MAX_FILES); self.assertLessEqual(first["metadata"]["window_count"], MAX_WINDOWS)


if __name__ == "__main__":
    unittest.main()
