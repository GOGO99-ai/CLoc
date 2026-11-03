import logging
import os
import subprocess
import time

try:
    import requests
except ModuleNotFoundError:
    from pip._vendor import requests

from cloc.runtime.execution_policy import (
    ExecutionPolicy,
    ExternalAccessDenied,
    RequiredArtifactUnavailable,
)


class TraceReplayFailed(RuntimeError):
    def __init__(self, txn_hash: str, cast_error: str, rpc_error: str):
        super().__init__(
            f"Failed to replay transaction {txn_hash}: "
            f"cast_run_error={cast_error}; rpc_fallback_error={rpc_error}"
        )
        self.txn_hash = txn_hash
        self.cast_error = cast_error
        self.rpc_error = rpc_error


LOGGGING_FILE_PATH = "./logs/txn_replayer.log"
os.makedirs(os.path.dirname(LOGGGING_FILE_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOGGGING_FILE_PATH, mode="w", encoding="utf-8")],
)


class TransactionReplayer:
    RPC_FALLBACK_TIMEOUT = 30
    RPC_FALLBACK_RETRIES = 2

    def __init__(
        self,
        rpc_url: str,
        cache_dir="./data/cache/traces",
        chain: str = "",
        policy: ExecutionPolicy | None = None,
    ):
        self.rpc_url = rpc_url
        self.cache_dir = cache_dir
        self.chain = chain.lower()
        self.policy = policy or ExecutionPolicy.current_live_default()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _execute_command(self, command_args: list[str]):
        try:
            process = subprocess.Popen(
                command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logging.error("Command failed (Code %s): %s", process.returncode, stderr)
                return f"Error: {stderr}", process.returncode

            return stdout, process.returncode
        except Exception as exc:
            logging.error("Command execution exception: %s", exc)
            return str(exc), -1

    def _rpc_call(self, method: str, params: list) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        for attempt in range(1, self.RPC_FALLBACK_RETRIES + 1):
            try:
                resp = requests.post(
                    self.rpc_url,
                    json=payload,
                    timeout=self.RPC_FALLBACK_TIMEOUT,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" in data:
                        logging.warning(
                            "RPC %s returned error (attempt %s): %s",
                            method,
                            attempt,
                            data["error"],
                        )
                        return {"error": data["error"]}
                    return data
                logging.warning("RPC %s HTTP %s (attempt %s)", method, resp.status_code, attempt)
            except Exception as exc:
                logging.warning("RPC %s exception (attempt %s): %s", method, attempt, exc)
            if attempt < self.RPC_FALLBACK_RETRIES:
                time.sleep(2)
        return {"error": "RPC request retries exhausted"}

    @classmethod
    def _convert_call_tracer_to_text(cls, trace_result: dict) -> str:
        lines: list[str] = []

        def _fmt_value(value_hex: str) -> str:
            if not value_hex or value_hex in {"0x0", "0x00"}:
                return ""
            try:
                return f"{{value: {int(value_hex, 16)}}}"
            except ValueError:
                return ""

        def _walk(call: dict, depth: int, ancestor_continues: list[bool]):
            to_addr = call.get("to", "0x0000000000000000000000000000000000000000")
            input_hex = call.get("input", "0x")
            gas_hex = call.get("gas", "0x0")
            gas_used_hex = call.get("gasUsed", "0x0")
            call_type = call.get("type", "CALL")
            value_hex = call.get("value", "0x0")

            try:
                gas = str(int(gas_used_hex or gas_hex, 16))
            except ValueError:
                gas = "0"

            selector = input_hex[:10] if len(input_hex) >= 10 else input_hex
            func_part = selector
            value_str = _fmt_value(value_hex)
            if value_str:
                func_part += value_str

            prefix_parts = []
            for i in range(depth - 1):
                prefix_parts.append("|  " if ancestor_continues[i] else "   ")
            if depth > 0:
                prefix_parts.append("|-- " if (not ancestor_continues or ancestor_continues[-1]) else "`-- ")

            lines.append(f"{''.join(prefix_parts)}[{gas}] {to_addr}::{func_part}({input_hex}) [{call_type}]")

            children = call.get("calls", [])
            for i, child in enumerate(children):
                _walk(child, depth + 1, ancestor_continues + [i != len(children) - 1])

        root_to = trace_result.get("to", "0x0000000000000000000000000000000000000000")
        root_input = trace_result.get("input", "0x")
        root_gas = trace_result.get("gas", "0x0")
        root_gas_used = trace_result.get("gasUsed", "0x0")
        root_call_type = trace_result.get("type", "CALL")
        root_value = trace_result.get("value", "0x0")

        try:
            gas = str(int(root_gas_used or root_gas, 16))
        except ValueError:
            gas = "0"

        selector = root_input[:10] if len(root_input) >= 10 else root_input
        func_part = selector
        value_str = _fmt_value(root_value)
        if value_str:
            func_part += value_str

        lines.append(f"[{gas}] {root_to}::{func_part}({root_input}) [{root_call_type}]")
        for i, child in enumerate(trace_result.get("calls", [])):
            _walk(child, 1, [i != len(trace_result.get('calls', [])) - 1])

        return "Traces:\n" + "\n".join(lines) + "\n"

    def _fetch_trace_via_rpc(self, txn_hash: str) -> str:
        print("[REPLAY] cast run failed; falling back to debug_traceTransaction RPC...")
        result = self._rpc_call("debug_traceTransaction", [txn_hash, {"tracer": "callTracer"}])
        if "error" in result:
            raise RuntimeError(f"debug_traceTransaction RPC fallback failed: {result['error']}")

        trace_result = result.get("result", {})
        if not trace_result:
            raise RuntimeError("debug_traceTransaction returned an empty result")

        text = self._convert_call_tracer_to_text(trace_result)
        print(f"[REPLAY] RPC trace acquired; generated {len(text)} chars of trace text")
        return text

    def run(self, txn_hash: str):
        cache_file = os.path.join(
            self.cache_dir,
            f"{self.chain}_{txn_hash.lower()}.txt" if self.chain else f"{txn_hash.lower()}.txt",
        )
        if os.path.exists(cache_file):
            print("[REPLAY] Using cached trace.")
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        if not self.policy.allow_foundry and not self.policy.allow_rpc:
            raise RequiredArtifactUnavailable("trace", os.path.basename(cache_file), "trace_cache_miss")

        logging.info("Starting replay for transaction %s", txn_hash)
        print(f"[REPLAY] Replaying transaction {txn_hash}...")

        if not self.policy.allow_foundry:
            if not self.policy.allow_rpc:
                raise ExternalAccessDenied("foundry", "allow_foundry", "trace")
            try:
                text = self._fetch_trace_via_rpc(txn_hash)
                if self.policy.allow_artifact_cache_write:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(text)
                return text
            except Exception as exc:
                logging.error("RPC trace fetch failed: %s", exc)
                raise TraceReplayFailed(txn_hash, "foundry disabled by execution policy", str(exc)) from exc

        command = ["cast", "run", txn_hash, "-vvvv", "--rpc-url", self.rpc_url]
        stdout, code = self._execute_command(command)

        if code == 0:
            print("[REPLAY] Replay succeeded via cast run")
            if self.policy.allow_artifact_cache_write:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(stdout)
            return stdout

        cast_error = stdout.replace("Error: ", "").strip()[:300]
        logging.warning("cast run failed: %s", cast_error)
        print(f"[REPLAY] cast run failed: {cast_error[:120]}...")

        if not self.policy.allow_rpc:
            raise ExternalAccessDenied("rpc", "allow_rpc", "trace")

        try:
            text = self._fetch_trace_via_rpc(txn_hash)
            if self.policy.allow_artifact_cache_write:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(text)
            return text
        except Exception as exc:
            logging.error("RPC fallback failed: %s", exc)
            print(f"[REPLAY] RPC fallback also failed: {exc}")
            raise TraceReplayFailed(txn_hash, cast_error, str(exc)) from exc


if __name__ == "__main__":
    test_rpc = "https://eth.llamarpc.com"
    test_hash = "0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299"
    replayer = TransactionReplayer(rpc_url=test_rpc)
    result = replayer.run(test_hash)
    print("\n--- Trace Preview ---")
    print(result[:500] + "...")
