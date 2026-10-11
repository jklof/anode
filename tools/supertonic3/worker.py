#!/usr/bin/env python3
"""Supertonic 3 synthesis child worker (child-only, CPU inference).

Reads a ``--request`` JSON file written by the NRT parent, synthesizes
one utterance with the pinned Supertonic SDK, and writes ``speech.wav``
plus ``metrics.json`` into the run directory. Stdlib + numpy +
soundfile + supertonic only; never imports Qt, torch, or ANode nodes.

Exit codes: 0 ok / 2 usage / 3 assets-missing / 1 other.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# The worker may be launched from any working directory: make the repo
# root importable so ``supertonic3_backend`` and ``audio_io`` resolve
# (established sidecar pattern).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fail_usage(message):
    print(f"worker.py: error: {message}", file=sys.stderr)
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="worker.py",
        description="Supertonic 3 child synthesis (request JSON in, "
                    "speech.wav + metrics.json out).")
    parser.add_argument("--request", required=True,
                        help="Absolute request JSON path (in the run dir).")
    args = parser.parse_args(argv)

    request_path = Path(args.request)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except OSError as e:
        print(f"worker.py: cannot read request: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"worker.py: invalid request JSON: {e}", file=sys.stderr)
        return 2
    if not isinstance(request, dict):
        print("worker.py: request must be a JSON object", file=sys.stderr)
        return 2

    run_dir = request_path.parent
    t0 = time.monotonic()
    try:
        from supertonic3_backend import (
            SUPERTONIC3_MODEL_PIN,
            SUPERTONIC3_SDK_PIN,
            check_assets,
            load_and_validate_profile_file,
            read_bounded_text_file,
            scalar_duration,
            validate_tts_settings,
            validate_waveform,
        )
    except ImportError as e:
        print(f"worker.py: cannot import backend helpers: {e}",
              file=sys.stderr)
        return 1

    try:
        text_kind = request.get("text_kind")
        text_value = request.get("text_value", "")
        style_kind = request.get("style_kind")
        style_value = request.get("style_value", "")
        model_dir = Path(request.get("model_dir", ""))
        language = request.get("language", "en")
        speed = request.get("speed", 1.05)
        total_steps = request.get("total_steps", 8)
        silence_duration = request.get("silence_duration", 0.3)
        threads = request.get("threads", 2)
        request_id = request.get("request_id", "")

        # Bounded UTF-8 text resolution; expression tags are preserved
        # verbatim (no stripping or normalization).
        validate_tts_settings(language=language, speed=speed,
                              total_steps=total_steps,
                              silence_duration=silence_duration,
                              threads=threads)
        if text_kind == "uri":
            if not text_value:
                raise ValueError("wired text snapshot is empty")
            text = read_bounded_text_file(text_value)
        elif text_kind == "inline":
            text = str(text_value or "")
            if not text.strip():
                raise ValueError("text is empty")
            from supertonic3_backend import MAX_TEXT_LENGTH
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"text has {len(text)} characters "
                    f"(cap {MAX_TEXT_LENGTH})")
        else:
            raise ValueError(f"bad text_kind {text_kind!r}")

        ok, missing = check_assets(model_dir)
        if not ok:
            print(f"worker.py: model assets missing: {', '.join(missing)} "
                  f"(run tools/supertonic3/fetch.py)", file=sys.stderr)
            return 3

        try:
            from supertonic import TTS
        except ImportError:
            print("worker.py: the 'supertonic' SDK is not installed "
                  "(python -m pip install "
                  "\"git+https://github.com/supertone-oss-archive/"
                  "supertonic-py.git@"
                  f"{SUPERTONIC3_SDK_PIN}\")",
                  file=sys.stderr)
            return 3

        tts = TTS(model="supertonic-3", model_dir=str(model_dir),
                  auto_download=False,
                  intra_op_num_threads=int(threads),
                  inter_op_num_threads=1)

        # Resolve the voice profile: a preset name, or a validated
        # imported/kept style JSON path.
        profile_hash = ""
        if style_kind == "preset":
            style = tts.get_voice_style(str(style_value))
            try:
                preset_path = model_dir / "voice_styles" / f"{style_value}.json"
                profile_hash = hashlib.sha256(
                    preset_path.read_bytes()).hexdigest()
            except OSError:
                profile_hash = ""
        elif style_kind in ("file", "uri"):
            if not style_value:
                raise ValueError("voice style snapshot is empty")
            _obj, profile_hash, _validation = load_and_validate_profile_file(
                style_value)
            style = tts.get_voice_style_from_path(str(style_value))
        else:
            raise ValueError(f"bad style_kind {style_kind!r}")

        wav, duration = tts.synthesize(
            text, voice_style=style, lang=str(language),
            total_steps=int(total_steps), speed=float(speed),
            silence_duration=float(silence_duration), verbose=False)

        import numpy as np
        wav = np.asarray(wav, dtype=np.float32)
        # The SDK returns (1, N) mono; duration may be an array.
        file_duration = validate_waveform(wav, int(tts.sample_rate))
        try:
            _sdk_duration = scalar_duration(
                duration, wav.shape[1], int(tts.sample_rate))
        except ValueError:
            _sdk_duration = file_duration

        from audio_io import write_wav_file
        out_wav = run_dir / "speech.wav"
        write_wav_file(out_wav, wav, int(tts.sample_rate))
        output_hash = hashlib.sha256(out_wav.read_bytes()).hexdigest()

        elapsed = time.monotonic() - t0
        metrics = {
            "sample_rate": int(tts.sample_rate),
            "frames": int(wav.shape[1]),
            "duration": file_duration,
            "elapsed": elapsed,
            "settings": {"language": str(language),
                         "speed": float(speed),
                         "total_steps": int(total_steps),
                         "silence_duration": float(silence_duration),
                         "threads": int(threads)},
            "sdk_pin": SUPERTONIC3_SDK_PIN,
            "model_pin": SUPERTONIC3_MODEL_PIN,
            "profile_hash": profile_hash,
            "output_hash": output_hash,
            "request_id": str(request_id),
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        # Bounded stdout: one metrics line for the supervisor tail.
        print(f"metrics.duration={file_duration:.3f}")
        print(f"metrics.elapsed={elapsed:.3f}")
        return 0
    except ValueError as e:
        print(f"worker.py: invalid request: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"worker.py: missing file: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        # Bounded error output (no traceback spam, no per-chunk logs).
        print(f"worker.py: synthesis failed: {str(e)[:500]}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
