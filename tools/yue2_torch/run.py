"""CLI entry for the YuE2 torch worker (argparse only at module top).

Fast ``--help``: this module imports no torch/heavy packages; the pipeline
is imported lazily inside :func:`main`. Process death is the cancel
mechanism (the NRT supervisor terminates the child on cancel/timeout).

Exit codes: 0 ok / 2 usage / 3 model-missing / 4 CUDA OOM / 1 other.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Linux-only allocator tuning (unsupported + warning-spam on Windows):
# lets the CUDA allocator split large freed blocks, which avoids failure
# on marginal cards when stage peaks interleave.
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")


def build_parser():
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Self-contained PyTorch YuE2 song generation "
                    "(spec.json in, song.wav + provenance out).")
    p.add_argument("--spec", required=True,
                   help="Worker spec JSON path (lyrics/style by file).")
    p.add_argument("--out", required=True, help="Output song.wav path.")
    p.add_argument("--out-dir", required=True,
                   help="Run directory (score.abc, semantic.json).")
    p.add_argument("--metrics", action="store_true",
                   help="Print metrics.<k>=<float> lines on stdout.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override the spec seed (debug only).")
    return p


def _fail_usage(message):
    print(f"run.py: error: {message}", file=sys.stderr)
    return 2


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    except OSError as e:
        print(f"run.py: cannot read spec: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"run.py: invalid spec JSON: {e}", file=sys.stderr)
        return 2
    if not isinstance(spec, dict):
        print("run.py: spec must be a JSON object", file=sys.stderr)
        return 2
    if args.seed is not None:
        spec["seed"] = args.seed

    # Validate before touching torch (usage errors stay fast + clear).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from yue2_torch.cli import validate_spec
    except ImportError:
        from cli import validate_spec
    try:
        validate_spec(spec)
    except ValueError as e:
        print(f"run.py: invalid spec: {e}", file=sys.stderr)
        return 2

    model_dir = Path(spec["model_dir"])
    if not (model_dir / "yue2_3b_bf16.safetensors").exists():
        print(f"run.py: torch weights not found in {model_dir} "
              "(run tools/audiocpp/fetch_yue2_torch.py)", file=sys.stderr)
        return 3

    try:
        try:
            from yue2_torch.pipeline import run_pipeline
        except ImportError:
            from pipeline import run_pipeline
        result = run_pipeline(spec, args.out, args.out_dir)
    except FileNotFoundError as e:
        print(f"run.py: missing file: {e}", file=sys.stderr)
        return 3
    except MemoryError:
        print("run.py: CUDA out of memory", file=sys.stderr)
        return 4
    except ValueError as e:
        # Input/fit errors (bad lyrics, oversize prompt+budget for the
        # card): clean message, no traceback.
        print(f"run.py: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"run.py: CUDA out of memory: {e}", file=sys.stderr)
            return 4
        import traceback
        traceback.print_exc()
        return 1

    if args.metrics:
        for key, value in sorted(result["metrics"].items()):
            try:
                print(f"metrics.{key}={float(value)}", flush=True)
            except (TypeError, ValueError):
                # Non-numeric provenance (e.g. the sampler name); the node's
                # _parse_metricspicks these up as strings.
                print(f"metrics.{key}={value}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
