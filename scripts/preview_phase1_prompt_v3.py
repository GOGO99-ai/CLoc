"""Render an offline, role-separated phase-1 request preview; never calls an LLM."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.cloc_v2.phase1_prompting import build_preview, preview_metrics, prompt_document, write_preview

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--batch-id", default="phase1_batch_b001")
    parser.add_argument("--preview-output", required=True, type=Path)
    parser.add_argument("--prompt-output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.packets.resolve() in {args.preview_output.resolve(), args.prompt_output.resolve()}:
        raise SystemExit("outputs must differ from packets input")
    packets = json.loads(args.packets.read_text(encoding="utf-8"))
    preview = build_preview(packets, args.batch_id)
    write_preview(preview, args.preview_output)
    from src.cloc_v2.phase1_scoring import write_text
    write_text(prompt_document(), args.prompt_output)
    print(json.dumps({"ok": True, "preview": str(args.preview_output), "prompt": str(args.prompt_output), **preview_metrics(preview)}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
