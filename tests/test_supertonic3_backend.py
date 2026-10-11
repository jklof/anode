"""Tests for supertonic3_backend: pins, fetch, validation, requests.

All tests run offline with fake files: no model/network downloads.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import plugin_system
import supertonic3_backend as st3
from supertonic3_backend import (
    LANGUAGES,
    STYLE_DP_DIMS,
    STYLE_TTL_DIMS,
    VOICES,
)


def _make_profile(ttl_val=0.1, dp_val=0.2, extra=None):
    import itertools
    ttl_n = 1 * 50 * 256
    dp_n = 1 * 8 * 16
    # Nested to match the archived layout: [1][50][256] and [1][8][16].
    ttl_inner = [[float(ttl_val)] * 256 for _ in range(50)]
    dp_inner = [[float(dp_val)] * 16 for _ in range(8)]
    obj = {"style_ttl": {"dims": [1, 50, 256], "data": [ttl_inner]},
           "style_dp": {"dims": [1, 8, 16], "data": [dp_inner]}}
    if extra:
        obj.update(extra)
    return obj


# ----------------------------------------------------------------------
# pins / fetch manifest
# ----------------------------------------------------------------------
def test_pins_are_archive_revisions():
    assert st3.SUPERTONIC3_SDK_PIN == "df0f9686dac7fbbde391b759e2ee5286a3737622"
    assert st3.SUPERTONIC3_MODEL_PIN == "aafc6e32416a594460b32413efc49d7fe4ce6d46"
    assert st3.SUPERTONIC3_MODEL_PIN in st3.SUPERTONIC3_HF_BASE
    assert "supertone-oss-archive/supertonic-3" in st3.SUPERTONIC3_HF_BASE


def test_fetch_specs_are_model_only_with_pinned_hashes(tmp_path):
    specs = st3.supertonic3_fetch_specs(tmp_path / "models" / "Supertonic3")
    rels = sorted(str(s.dest).replace("\\", "/").split("Supertonic3/")[-1]
                  for s in specs)
    assert "onnx/duration_predictor.onnx" in rels
    assert "onnx/text_encoder.onnx" in rels
    assert "onnx/vector_estimator.onnx" in rels
    assert "onnx/vocoder.onnx" in rels
    assert "onnx/tts.json" in rels
    assert "onnx/unicode_indexer.json" in rels
    for v in VOICES:
        assert f"voice_styles/{v}.json" in rels
    # URLs pin the archive revision.
    for s in specs:
        assert st3.SUPERTONIC3_MODEL_PIN in s.url
    # ONNX carry upstream LFS SHA-256; small files are size-only.
    by_rel = {}
    for s in specs:
        rel = str(s.dest).replace("\\", "/").split("Supertonic3/")[-1]
        by_rel[rel] = s
    assert by_rel["onnx/duration_predictor.onnx"].size == 3700147
    assert by_rel["onnx/duration_predictor.onnx"].sha256 == (
        "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db")
    assert by_rel["onnx/text_encoder.onnx"].size == 36416150
    assert by_rel["onnx/vector_estimator.onnx"].size == 256534781
    assert by_rel["onnx/vocoder.onnx"].size == 101424195
    assert by_rel["onnx/tts.json"].sha256 == (
        "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09")
    assert by_rel["onnx/tts.json"].size == 8253
    assert by_rel["voice_styles/M1.json"].sha256 == (
        "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b")
    assert by_rel["LICENSE"].sha256 == (
        "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f")


def test_check_assets_reports_missing(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    ok, missing = st3.check_assets(model_dir)
    assert not ok and len(missing) == len(st3.SUPERTONIC3_FILES)
    # Materialize exact bytes for one hashed file: still missing the rest.
    import hashlib as _hashlib
    rel, size, sha = st3.SUPERTONIC3_FILES[4]  # onnx/tts.json
    dest = model_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Content differs from the pinned hash, so it still reports missing
    # (same-size corruption is detected).
    with open(dest, "wb") as f:
        f.truncate(size)
    ok, missing = st3.check_assets(model_dir)
    assert not ok
    assert rel in missing


def test_missing_sdk_does_not_break_discovery():
    plugin_system.load_plugins("plugins")
    assert "Supertonic3VoiceDesigner" in plugin_system.NODE_REGISTRY
    assert "Supertonic3TTS" in plugin_system.NODE_REGISTRY
    assert sys.modules.get("supertonic") is None


# ----------------------------------------------------------------------
# profile validation
# ----------------------------------------------------------------------
def test_valid_profile_accepted_with_honest_provenance():
    obj = _make_profile()
    out = st3.validate_profile_dict(obj)
    assert out["ttl_count"] == 1 * 50 * 256
    assert out["dp_count"] == 1 * 8 * 16
    assert out["provenance"]["verified_v3"] is False
    assert "user" in out["provenance"]["note"]


def test_profile_rejects_shape_mismatch():
    obj = _make_profile()
    obj["style_ttl"]["dims"] = [1, 50, 128]
    with pytest.raises(ValueError, match="pinned v3"):
        st3.validate_profile_dict(obj)


def test_profile_rejects_short_data():
    obj = _make_profile()
    obj["style_ttl"]["data"] = [[[0.1] * 256 for _ in range(49)]]
    with pytest.raises(ValueError, match="elements"):
        st3.validate_profile_dict(obj)


def test_profile_rejects_non_finite_and_bools():
    for bad in (float("inf"), float("nan"), True, "x", None):
        obj = _make_profile()
        obj["style_dp"]["data"][0][0][0] = bad
        with pytest.raises(ValueError):
            st3.validate_profile_dict(obj)


def test_profile_rejects_oversized_dims_before_allocation():
    obj = {"style_ttl": {"dims": [1, 5000, 5000], "data": []},
           "style_dp": {"dims": [1, 8, 16], "data": []}}
    with pytest.raises(ValueError, match="sanity cap|extent"):
        st3.validate_profile_dict(obj)


def test_profile_rejects_contradictory_version():
    for extra in ({"version": "2"}, {"version": 2},
                  {"model": "supertonic-2"},
                  {"metadata": {"version": "v2"}}):
        obj = _make_profile(extra=extra)
        with pytest.raises(ValueError, match="not.*3|v2|targets"):
            st3.validate_profile_dict(obj)


def test_profile_preserves_values_and_unique_paths(tmp_path):
    obj = _make_profile(ttl_val=0.123, dp_val=-0.456)
    src = tmp_path / "import.json"
    src.write_text(json.dumps(obj), encoding="utf-8")
    loaded, source_hash, _v = st3.load_and_validate_profile_file(src)
    assert source_hash == hashlib.sha256(src.read_bytes()).hexdigest()
    kept1 = st3.prepare_profile(
        profile_obj=loaded, source=f"file:{src}", source_hash=source_hash,
        preset_name="", display_name="My Voice",
        profiles_dir=tmp_path / "profiles")
    kept2 = st3.prepare_profile(
        profile_obj=loaded, source=f"file:{src}", source_hash=source_hash,
        preset_name="", display_name="My Voice",
        profiles_dir=tmp_path / "profiles")
    # Overwriting the same path is prohibited: each call is unique.
    assert kept1["profile_path"] != kept2["profile_path"]
    for kept in (kept1, kept2):
        reloaded = json.loads(Path(kept["profile_path"]).read_text(
            encoding="utf-8"))
        assert reloaded == loaded
        prov = json.loads(Path(kept["provenance_path"]).read_text(
            encoding="utf-8"))
        assert prov["source_hash"] == source_hash
        assert prov["output_hash"] == kept["output_hash"]
        assert prov["sdk"]["pin"] == st3.SUPERTONIC3_SDK_PIN
        assert prov["model"]["pin"] == st3.SUPERTONIC3_MODEL_PIN


def test_profile_file_size_cap(tmp_path):
    big = tmp_path / "big.json"
    big.write_bytes(b"x" * (st3.MAX_PROFILE_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        st3.load_and_validate_profile_file(big)


# ----------------------------------------------------------------------
# text / settings / waveform
# ----------------------------------------------------------------------
def test_language_menu_order_pinned():
    assert len(LANGUAGES) == 32  # 31 codes + na
    assert LANGUAGES[0] == "en" and LANGUAGES[-1] == "na"
    assert "fi" in LANGUAGES
    items = st3.language_menu_items()
    assert len(items) == 32 and items[0].endswith("(en)")


def test_voice_order_pinned():
    assert VOICES == ("M1", "M2", "M3", "M4", "M5",
                      "F1", "F2", "F3", "F4", "F5")


def test_pick_text_source_precedence():
    kind, val = st3.pick_text_source(
        inline_text="hi", uri_connected=True, uri_snapshot="/t.txt")
    assert (kind, val) == ("uri", "/t.txt")
    with pytest.raises(ValueError, match="empty"):
        st3.pick_text_source(inline_text="hi", uri_connected=True,
                             uri_snapshot="")
    kind, val = st3.pick_text_source(
        inline_text="hi", uri_connected=False, uri_snapshot="")
    assert (kind, val) == ("inline", "hi")
    with pytest.raises(ValueError, match="empty"):
        st3.pick_text_source(inline_text="  ", uri_connected=False,
                             uri_snapshot="")


def test_pick_profile_source_precedence():
    kind, val = st3.pick_profile_source(
        preset_idx=0, file_param="/f.json",
        uri_connected=True, uri_snapshot="/w.json")
    assert (kind, val) == ("uri", "/w.json")
    with pytest.raises(ValueError, match="empty"):
        st3.pick_profile_source(preset_idx=0, file_param="/f.json",
                                uri_connected=True, uri_snapshot="")
    kind, val = st3.pick_profile_source(
        preset_idx=0, file_param="/f.json",
        uri_connected=False, uri_snapshot="")
    assert (kind, val) == ("file", "/f.json")
    kind, val = st3.pick_profile_source(
        preset_idx=3, file_param="",
        uri_connected=False, uri_snapshot="")
    assert (kind, val) == ("preset", "M4")


def test_validate_tts_settings_bounds():
    st3.validate_tts_settings(language="en", speed=1.05, total_steps=8,
                              silence_duration=0.3, threads=2)
    with pytest.raises(ValueError):
        st3.validate_tts_settings(language="xx", speed=1.05, total_steps=8,
                                  silence_duration=0.3, threads=2)
    with pytest.raises(ValueError):
        st3.validate_tts_settings(language="en", speed=5.0, total_steps=8,
                                  silence_duration=0.3, threads=2)
    with pytest.raises(ValueError):
        st3.validate_tts_settings(language="en", speed=1.05, total_steps=0,
                                  silence_duration=0.3, threads=2)
    with pytest.raises(ValueError):
        st3.validate_tts_settings(language="en", speed=1.05, total_steps=8,
                                  silence_duration=9.0, threads=2)
    with pytest.raises(ValueError):
        st3.validate_tts_settings(language="en", speed=1.05, total_steps=8,
                                  silence_duration=0.3, threads=99)


def test_build_tts_request_preserves_expression_tags():
    text = "Hello <laugh> world <breath>!"
    spec = st3.build_tts_request(
        text_kind="inline", text_value=text, style_kind="preset",
        style_value="M1", model_dir="models/Supertonic3", language="en",
        speed=1.05, total_steps=8, silence_duration=0.3, threads=2,
        request_id="r1")
    assert spec["text_value"] == text  # verbatim, never stripped
    assert spec["request_id"] == "r1"


def test_build_worker_argv_uses_list_and_absolute_paths(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("x")
    request = tmp_path / "request.json"
    request.write_text("{}")
    argv = st3.build_worker_argv(str(worker), str(request))
    assert argv[0] == sys.executable
    assert argv[1] == str(worker.resolve())
    assert argv[2] == "--request"
    assert argv[3] == str(request.resolve())
    assert isinstance(argv, list)


def test_read_bounded_text_file_caps(tmp_path):
    good = tmp_path / "t.txt"
    good.write_text("hello <laugh>\n", encoding="utf-8")
    assert st3.read_bounded_text_file(good).strip() == "hello <laugh>"
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (st3.MAX_TEXT_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        st3.read_bounded_text_file(big)
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        st3.read_bounded_text_file(empty)


def test_validate_waveform_and_scalar_duration():
    wav = np.zeros((1, 4410), dtype=np.float32)
    assert st3.validate_waveform(wav, 44100) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        st3.validate_waveform(np.zeros((2, 100), dtype=np.float32), 44100)
    with pytest.raises(ValueError):
        st3.validate_waveform(np.zeros((1, 0), dtype=np.float32), 44100)
    bad = np.zeros((1, 10), dtype=np.float32)
    bad[0, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        st3.validate_waveform(bad, 44100)
    # Duration arrays (SDK) collapse to the file duration.
    assert st3.scalar_duration(np.array([2.5]), 4410, 44100) == pytest.approx(
        0.1)


def test_fetch_lock_is_per_destination(tmp_path):
    a = tmp_path / "A"
    b = tmp_path / "B"
    assert st3.fetch_lock_for(a) is st3.fetch_lock_for(a)
    assert st3.fetch_lock_for(a) is not st3.fetch_lock_for(b)


def test_acquire_fetch_lock_cancel_responsive():
    import threading
    lock = threading.Lock()
    lock.acquire()
    cancel = threading.Event()
    cancel.set()
    from download_util import DownloadCancelled
    with pytest.raises(DownloadCancelled):
        st3.acquire_fetch_lock(lock, cancel, timeout_s=0.01)
    lock.release()


def test_result_request_id_from_dict_and_exception():
    assert st3.result_request_id({"request_id": "r1"}) == "r1"
    assert st3.result_request_id(RuntimeError("x")) is None
    err = RuntimeError("boom")
    st3.tag_exception_request_id(err, "r9")
    assert st3.result_request_id(err) == "r9"


def test_bounded_read_enforced_without_unbounded_alloc(tmp_path, monkeypatch):
    # A file that grows between stat and read still cannot exceed cap+1.
    import supertonic3_backend as _b
    p = tmp_path / "grow.txt"
    p.write_bytes(b"hi")
    real_open = open

    def fake_open(path, mode="r", *a, **k):
        f = real_open(path, mode, *a, **k)
        if "b" in mode:
            orig_read = f.read

            def capped(n=-1):
                # Pretend the source now yields more than the cap.
                return b"x" * (_b.MAX_TEXT_BYTES + 5)
            f.read = capped
        return f

    import builtins
    monkeypatch.setattr(builtins, "open", fake_open)
    with pytest.raises(ValueError, match="too large"):
        st3.read_bounded_text_file(p)
