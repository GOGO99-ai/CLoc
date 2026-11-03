import json
import logging
import os

from cloc.data_collection.trace_parser import TraceParser
from cloc.data_collection.txn_replayer import TransactionReplayer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TransactionSequencer:
    def __init__(
        self,
        rpc_url: str,
        contract_cache_dir="./data/cache/contracts",
        forensics_dir="./data/cache/forensics",
    ):
        self.replayer = TransactionReplayer(rpc_url=rpc_url)
        self.parser = TraceParser()
        self.cache_dir = contract_cache_dir
        self.forensics_dir = forensics_dir
        os.makedirs(self.forensics_dir, exist_ok=True)

    def run(self, txn_hash: str, rpc_url: str | None = None):
        logging.info("Analyzing transaction %s...", txn_hash[:12])

        trace_text = self.replayer.run(txn_hash)
        if not trace_text:
            logging.error("Unable to fetch raw trace text.")
            return None

        # Emit a raw call tree only. No ABI/source assets are loaded or applied.
        analysis_result = self.parser.parse(trace_text)

        if isinstance(analysis_result, dict) and "call_tree" in analysis_result:
            if analysis_result["call_tree"] and hasattr(analysis_result["call_tree"], "to_dict"):
                analysis_result["call_tree"] = analysis_result["call_tree"].to_dict()
        elif hasattr(analysis_result, "to_dict"):
            analysis_result = analysis_result.to_dict()

        output_path = os.path.join(self.forensics_dir, f"{txn_hash.lower()}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)

        logging.info("Saved raw call tree to %s", output_path)
        return output_path


if __name__ == "__main__":
    sequencer = TransactionSequencer(
        rpc_url=os.environ["ETH_RPC_URL"]
    )
    target_tx = "0x9f14d093a2349de08f02fc0fb018dadb449351d0cdb7d0738ff69cc6fef5f299"

    try:
        sequencer.run(target_tx)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
