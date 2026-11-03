from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests

from cloc.runtime.execution_policy import ExecutionPolicy
from src.cloc_v2.source_parsing import write_explorer_source_files

try:
    from cloc.data_collection.source_acquisition import (
        SourceAcquisitionBackend,
        SourceArtifact,
        safe_relative_source_path,
        write_temp_sources,
    )
    HAS_SOURCE_ACQUISITION = True
except ImportError:
    HAS_SOURCE_ACQUISITION = False

    @dataclass
    class _CompilerSettings:
        compiler_version: str = ""
        optimization_used: bool = False
        runs: int = 0
        evm_version: str = ""

    @dataclass
    class _Failure:
        code: str = ""
        message: str = ""

    @dataclass
    class SourceArtifact:
        chain: str = ""
        requested_address: str = ""
        resolved_address: str = ""
        verified: bool = False
        source_format: str = "unknown"
        source_file_count: int = 0
        compiler_settings: _CompilerSettings = field(default_factory=_CompilerSettings)
        cache_path: str = ""
        failure: _Failure | None = None
        contract_name: str = "Unknown"
        proxy_metadata: Dict[str, Any] = field(default_factory=dict)
        cache_hit: bool = False
        network_fetch_attempted: bool = False
        network_fetch_succeeded: bool = False

    SourceAcquisitionBackend = None

    def safe_relative_source_path(raw_path: str) -> str | None:
        normalized = str(raw_path).replace("\\", "/").strip().lstrip("/")
        if not normalized or ".." in normalized.split("/"):
            return None
        return normalized

    def write_temp_sources(artifact: SourceArtifact, output_dir: str | Path) -> List[str]:
        return []


LOGGGING_FILE_PATH = "./logs/get_source_code.log"
os.makedirs(os.path.dirname(LOGGGING_FILE_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGGGING_FILE_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

DEFAULT_PROXY_PORT = "7897"


class ContractDownloader:
    def __init__(
        self,
        chain: str,
        address: list,
        chain_config: dict | None = None,
        txn_hash: str | None = None,
        output_dir: str = "./temp_src",
        policy: ExecutionPolicy | None = None,
        force_source_refresh: bool = False,
        disable_legacy_source_fallback: bool = False,
        run_source_root: str | Path | None = None,
        legacy_cache_root: str | Path | None = None,
        use_proxy: bool = False,
        proxy_port: str | int = DEFAULT_PROXY_PORT,
    ):
        self.chain = chain.lower()
        self.address = address
        if chain_config is None:
            try:
                from config import SUPPORTED_CHAINS

                chain_config = SUPPORTED_CHAINS.get(self.chain, {})
            except Exception:
                chain_config = {}
        self.config = chain_config
        self.policy = policy or ExecutionPolicy.current_live_default()
        self.force_source_refresh = force_source_refresh
        self.disable_legacy_source_fallback = disable_legacy_source_fallback
        self.txn_hash = (txn_hash or "default").lower()
        self.base_url = self.config.get("base_url", "https://api.etherscan.io/api")

        if legacy_cache_root is not None:
            self.base_cache_dir = str(Path(legacy_cache_root))
        else:
            self.base_cache_dir = (
                str(Path(run_source_root) / "legacy_contracts")
                if run_source_root is not None
                else "./data/cache/contracts"
            )
        self.txn_root_dir = os.path.join(self.base_cache_dir, self.txn_hash)
        self.dirs = {
            "internal": os.path.join(self.txn_root_dir, "internal_contracts"),
            "external": os.path.join(self.txn_root_dir, "external_contracts"),
            "wallet": os.path.join(self.txn_root_dir, "wallets"),
        }
        for directory in self.dirs.values():
            os.makedirs(directory, exist_ok=True)

        if run_source_root is not None:
            self.output_dir = str(Path(run_source_root))
        else:
            self.output_dir = os.path.join(output_dir, self.txn_hash)
        os.makedirs(self.output_dir, exist_ok=True)

        self.session = requests.Session()
        self.proxies = (
            {
                "http": f"http://127.0.0.1:{proxy_port}",
                "https": f"http://127.0.0.1:{proxy_port}",
            }
            if use_proxy
            else None
        )
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://etherscan.io/",
            "Connection": "keep-alive",
        }
        source_cache_root = (
            Path(run_source_root) / "contracts"
            if run_source_root is not None
            else Path("./data/cache/source_acquisition")
        )
        self.source_backend = None
        if HAS_SOURCE_ACQUISITION and SourceAcquisitionBackend is not None:
            self.source_backend = SourceAcquisitionBackend(
                chain=self.chain,
                chain_config=self.config,
                cache_root=source_cache_root,
                session=self.session,
                proxies=self.proxies,
                policy=self.policy,
            )
        self.acquisition_results: Dict[str, SourceArtifact] = {}
        self.provenance: Dict[str, Any] = {
            "chain": self.chain,
            "transaction_hash": self.txn_hash,
            "force_source_refresh": self.force_source_refresh,
            "disable_legacy_source_fallback": self.disable_legacy_source_fallback,
            "run_source_root": self.output_dir,
            "addresses": [],
        }

    def _classify_from_html(self, address: str) -> Tuple[str, List[str]]:
        """Legacy HTML probe fallback for address-vs-contract classification."""
        if not self.policy.allow_explorer:
            return "external", []
        url = f"https://etherscan.io/address/{address}"
        sol_files: List[str] = []
        category = "external"
        try:
            print(f"   [HTML probe] Inspecting address type: {address}")
            time.sleep(random.uniform(1.5, 3.0))
            response = self.session.get(url, headers=self.headers, proxies=self.proxies, timeout=15)
            if response.status_code != 200:
                return category, sol_files

            html = response.text
            section = re.search(
                r'<section[^>]*class="[^"]*container-xxl[^"]*"[^>]*>(.*?)</section>',
                html,
                re.S | re.I,
            )
            if section:
                content = section.group(1)
                if re.search(r">\s*Address\s*<", content, re.I):
                    category = "wallet"
                elif re.search(r">\s*Contract\s*<", content, re.I):
                    category = "external"

            seen = set()
            for match in re.findall(r"['\"]([^'\"\s]+\.(?:sol|vy|yul|json))['\"]", html, re.I):
                filename = os.path.basename(match)
                if filename not in seen:
                    sol_files.append(filename)
                    seen.add(filename)
        except Exception as exc:
            print(f"   [HTML probe] Failed: {exc}")
        return category, sol_files

    def get_source_code_details(self, address: str, target_dir: str | Path | None = None) -> Dict[str, Any]:
        """Legacy explorer fallback with structured result payload and local artifact writes."""
        if not self.policy.allow_explorer:
            return {
                "address": address.lower(),
                "status": "failed",
                "contract_name": "Unknown",
                "source_format": "unknown",
                "is_proxy": False,
                "implementation": None,
                "saved_dir": str(target_dir) if target_dir is not None else None,
                "sol_files": [],
                "raw_response_path": None,
                "metadata_path": None,
                "artifact_path": None,
                "abi_path": None,
                "error": "network error: explorer access disabled by execution policy",
            }
        api_key = self.config.get("key", "")
        if not api_key:
            return {
                "address": address.lower(),
                "status": "failed",
                "contract_name": "Unknown",
                "source_format": "unknown",
                "is_proxy": False,
                "implementation": None,
                "saved_dir": str(target_dir) if target_dir is not None else None,
                "sol_files": [],
                "raw_response_path": None,
                "metadata_path": None,
                "artifact_path": None,
                "abi_path": None,
                "error": "api key missing",
            }

        output_dir = Path(target_dir or self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}?module=contract&action=getsourcecode&address={address}&apikey={api_key}"
        if "chainid" in self.config:
            url += f"&chainid={self.config['chainid']}"

        try:
            response = self.session.get(url, proxies=self.proxies, timeout=20)
            data = response.json()
        except requests.exceptions.ProxyError:
            return self._failure_result(address, output_dir, "proxy connection failed")
        except requests.exceptions.RequestException as exc:
            return self._failure_result(address, output_dir, f"network error: {exc}")
        except ValueError as exc:
            return self._failure_result(address, output_dir, f"unsupported source format: invalid JSON response ({exc})")

        raw_response_path = output_dir / "raw_response.json"
        raw_response_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        if data.get("status") != "1" or not data.get("result"):
            message = str(data.get("message") or "")
            result_text = str(data.get("result") or "")
            error = self._classify_api_failure(message, result_text)
            return self._failure_result(address, output_dir, error, raw_response_path=raw_response_path)

        result = data["result"][0]
        source = result.get("SourceCode", "") or ""
        name = result.get("ContractName", "Contract") or "Contract"
        abi = result.get("ABI")
        proxy_flag = str(result.get("Proxy", "0")) == "1"
        implementation = result.get("Implementation") or None

        metadata = {
            "contract_name": name,
            "compiler_version": result.get("CompilerVersion"),
            "optimization_used": result.get("OptimizationUsed"),
            "runs": result.get("Runs"),
            "evm_version": result.get("EVMVersion"),
            "constructor_arguments": result.get("ConstructorArguments"),
            "license_type": result.get("LicenseType"),
            "proxy": proxy_flag,
            "implementation": implementation,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        abi_path = None
        if abi and isinstance(abi, str) and abi.strip() and "not verified" not in abi.lower():
            abi_path = output_dir / "abi.json"
            try:
                parsed_abi = json.loads(abi)
                abi_path.write_text(json.dumps(parsed_abi, indent=2, ensure_ascii=False), encoding="utf-8")
            except ValueError:
                abi_path.write_text(abi, encoding="utf-8")

        if not source.strip():
            return self._failure_result(
                address,
                output_dir,
                "unverified contract",
                raw_response_path=raw_response_path,
                metadata_path=metadata_path,
                abi_path=abi_path,
                contract_name=name,
                is_proxy=proxy_flag,
                implementation=implementation,
            )

        try:
            sol_files, source_format = self._write_source_files(source, output_dir, name)
        except Exception as exc:
            return self._failure_result(
                address,
                output_dir,
                f"unsupported source format: {exc}",
                raw_response_path=raw_response_path,
                metadata_path=metadata_path,
                abi_path=abi_path,
                contract_name=name,
                is_proxy=proxy_flag,
                implementation=implementation,
            )

        status = "verified" if sol_files else "failed"
        artifact_payload = {
            "address": address.lower(),
            "status": status,
            "contract_name": name,
            "source_format": source_format,
            "is_proxy": proxy_flag,
            "implementation": implementation,
            "sol_files": sol_files,
            "raw_response_path": str(raw_response_path),
            "metadata_path": str(metadata_path),
            "abi_path": str(abi_path) if abi_path else None,
            "error": None if sol_files else "unsupported source format",
        }
        artifact_path = output_dir / "artifact.json"
        artifact_path.write_text(json.dumps(artifact_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        artifact_payload["artifact_path"] = str(artifact_path)
        artifact_payload["saved_dir"] = str(output_dir)
        return artifact_payload

    def get_source_code(self, address: str) -> Tuple[bool, str, List[str]]:
        """Legacy source fallback; used only when normalized acquisition fails."""
        result = self.get_source_code_details(address, self.output_dir)
        return result["status"] == "verified", result["contract_name"], result.get("sol_files", [])

    def run(self) -> Dict[str, SourceArtifact]:
        """Run cache -> normalized explorer backend -> legacy fallback -> structured failure."""
        for addr in self.address:
            addr_lower = addr.lower()
            print(f"\n[contract download] {addr_lower}")

            use_cache = not self.force_source_refresh
            artifact: SourceArtifact | None = None
            if self.source_backend is not None:
                artifact = self.source_backend.acquire(addr_lower, use_cache=use_cache)
            else:
                artifact = SourceArtifact(
                    chain=self.chain,
                    requested_address=addr_lower,
                    resolved_address=addr_lower,
                    verified=False,
                    failure=_Failure(code="source_acquisition_missing", message="source_acquisition backend not available"),
                )
            self.acquisition_results[addr_lower] = artifact

            sol_files = write_temp_sources(artifact, self.output_dir) if self.source_backend is not None else []
            success_src = bool(getattr(artifact, "verified", False)) and bool(sol_files)
            contract_name = getattr(artifact, "contract_name", None) or "Unknown"

            if (
                not self.disable_legacy_source_fallback
                and not success_src
            ):
                success_src, legacy_name, sol_files = self.get_source_code(addr)
                if legacy_name and legacy_name not in ("Unknown", "Error"):
                    contract_name = legacy_name

            if success_src:
                target_dir = self.dirs["internal"]
                final_type = "Verified_Contract"
                print("   source available")
            else:
                print("   no source from API; falling back to HTML classification")
                category, html_files = self._classify_from_html(addr)
                target_dir = self.dirs[category]
                final_type = "EOA" if category == "wallet" else "Unverified_Contract"
                sol_files = html_files

            final_data = {
                "contract_address": addr_lower,
                "chain": self.chain,
                "address_type": final_type,
                "contract_name": contract_name,
                "sol_files": sol_files,
                "source_acquisition": self._artifact_summary(artifact),
            }
            cache_path = ""
            if self.policy.allow_artifact_cache_write:
                cache_path = self._write_legacy_asset(target_dir, addr_lower, final_data)
            print(f"   legacy asset archived: {cache_path}")
            self._record_provenance(
                addr_lower,
                artifact,
                use_cache=use_cache,
                sol_files=sol_files,
            )

        self._write_provenance()

        return self.acquisition_results

    def _record_provenance(
        self,
        address: str,
        artifact: SourceArtifact | None,
        *,
        use_cache: bool,
        sol_files: List[str],
    ) -> None:
        source_files = []
        for relative in sol_files:
            path = Path(self.output_dir) / relative
            if not path.exists() or path.suffix.lower() != ".sol":
                continue
            source_files.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "created_in_current_run": True,
                }
            )
        cache_hit = bool(getattr(artifact, "cache_hit", False))
        network_attempted = bool(getattr(artifact, "network_fetch_attempted", False))
        failure = getattr(artifact, "failure", None)
        proxy_metadata = getattr(artifact, "proxy_metadata", {}) or {}
        self.provenance["addresses"].append(
            {
                "contract_address": address,
                "discovered_automatically": True,
                "use_cache": use_cache,
                "cache_hit": cache_hit,
                "network_fetch_attempted": network_attempted,
                "network_fetch_succeeded": bool(getattr(artifact, "network_fetch_succeeded", False)),
                "explorer_host": urlparse(str(self.base_url)).netloc,
                "verified": bool(getattr(artifact, "verified", False)),
                "proxy": bool(proxy_metadata.get("is_proxy") or proxy_metadata.get("Proxy")),
                "implementation_address": str(
                    proxy_metadata.get("implementation_address")
                    or proxy_metadata.get("Implementation")
                    or ""
                ),
                "source_files": source_files,
                "failure_code": getattr(failure, "code", "") if failure else "",
            }
        )

    def _write_provenance(self) -> None:
        target = Path(self.output_dir) / "source_acquisition_provenance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_legacy_asset(self, target_dir: str, addr_lower: str, final_data: Dict[str, Any]) -> str:
        nested_path = Path(target_dir) / f"{addr_lower}_{self.chain}.json"
        flat_path = Path(self.base_cache_dir) / f"{addr_lower}_{self.chain}.json"
        for cache_path in (nested_path, flat_path):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(final_data, indent=4, ensure_ascii=False), encoding="utf-8")
        return str(nested_path)

    @staticmethod
    def _artifact_summary(artifact: SourceArtifact | None) -> Dict[str, Any]:
        if artifact is None:
            return {
                "chain": "",
                "requested_address": "",
                "resolved_address": "",
                "verified": False,
                "source_format": "unknown",
                "source_file_count": 0,
                "compiler_version": "",
                "optimizer": False,
                "runs": 0,
                "evm_version": "",
                "cache_path": "",
                "failure": None,
            }
        return {
            "chain": artifact.chain,
            "requested_address": artifact.requested_address,
            "resolved_address": artifact.resolved_address,
            "verified": artifact.verified,
            "source_format": artifact.source_format,
            "source_file_count": artifact.source_file_count,
            "compiler_version": artifact.compiler_settings.compiler_version,
            "optimizer": artifact.compiler_settings.optimization_used,
            "runs": artifact.compiler_settings.runs,
            "evm_version": artifact.compiler_settings.evm_version,
            "cache_path": artifact.cache_path,
            "failure": asdict(artifact.failure) if artifact.failure else None,
        }

    def _write_source_files(self, source: str, output_dir: Path, contract_name: str) -> Tuple[List[str], str]:
        return write_explorer_source_files(source, output_dir, contract_name)

    def _failure_result(
        self,
        address: str,
        output_dir: Path,
        error: str,
        *,
        raw_response_path: Path | None = None,
        metadata_path: Path | None = None,
        abi_path: Path | None = None,
        contract_name: str = "Unknown",
        is_proxy: bool = False,
        implementation: str | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "address": address.lower(),
            "status": "failed" if error != "unverified contract" else "unverified",
            "contract_name": contract_name,
            "source_format": "unknown",
            "is_proxy": is_proxy,
            "implementation": implementation,
            "saved_dir": str(output_dir),
            "sol_files": [],
            "raw_response_path": str(raw_response_path) if raw_response_path else None,
            "metadata_path": str(metadata_path) if metadata_path else None,
            "artifact_path": None,
            "abi_path": str(abi_path) if abi_path else None,
            "error": error,
        }
        artifact_path = output_dir / "artifact.json"
        artifact_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payload["artifact_path"] = str(artifact_path)
        return payload

    @staticmethod
    def _classify_api_failure(message: str, result_text: str) -> str:
        combined = f"{message} {result_text}".lower()
        if "missing or unsupported chainid" in combined or "invalid api key" in combined:
            return "api key missing"
        if "rate limit" in combined or "max rate limit" in combined or "too many invalid api key attempts" in combined:
            return "rate limit"
        if "not verified" in combined:
            return "unverified contract"
        return f"network error: {message or result_text or 'unknown explorer response'}"


def _safe_contract_filename(contract_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", contract_name or "Contract").strip("._")
    return cleaned or "Contract"


def _sha256(path: Path) -> str:
    h = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    pass
