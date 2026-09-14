"""
DeEsser — split-band dynamic sibilance attenuator (Effects).

Thin FFINode wrapper over libdeesser (cpp/deesser.cpp). A Linkwitz-Riley 4
crossover splits the signal at `frequency`; a linked peak detector on an
independent bandpass (Q ~= 1) drives HF-only gain reduction (fixed 3:1
ratio, clamped to `depth`), so vowel formants and low harmonics pass
untouched while ess energy is ducked. All per-sample ballistics run
natively; the Python side only marshals pointers and pushes staged
parameters. `get_gr_db` is an extended export with explicit ctypes
annotations (cf. plugins/envelope.py).
"""

import ctypes
import logging

import numpy as np
import torch

from ffi_base import FFINode
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QComboBox,
        QPushButton, QProgressBar,
    )
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class DeEsser(FFINode):
    category = "Effects"
    label = "De-Esser"
    description = (
        "Split-band de-esser: attenuates only the high-frequency band above "
        "'frequency' when sibilant energy exceeds 'threshold' (up to 'depth' "
        "dB), leaving vowel body untouched. Detection is linked across "
        "channels. 'listen' auditions the tuning signal: Delta plays only the "
        "removed sibilance (silence when idle); Sibilant Band solos the HF "
        "crossover band. 'auto_threshold' makes reduction level-independent "
        "by comparing sibilant energy against a vowel-body reference. Zero "
        "added latency; mix blends dry/wet without comb filtering."
    )

    LIB_NAME = "deesser"
    # Matches cpp/deesser.cpp ParamId
    PARAM_MAP = {
        "frequency": 0,
        "threshold": 1,
        "depth": 2,
        "attack_ms": 3,
        "release_ms": 4,
        "listen": 5,
        "mix": 6,
        "listen_mode": 7,
        "auto_threshold": 8,
    }

    def __init__(self, name=""):
        super().__init__(name)

        self.inp = self.add_input(
            "in", help="Signal to de-ess; mono inputs are duplicated to stereo.")
        self.out = self.add_output(
            "out", channels=CHANNELS, help="De-essed stereo output.")

        self.add_float_param("frequency", 6500.0, 3000.0, 9000.0, unit="Hz",
                             help="Crossover/detector center frequency for the sibilant band.")
        self.add_float_param("threshold", -18.0, -40.0, 0.0, unit="dB",
                             help="Sibilant-band level above which reduction engages.")
        self.add_float_param("depth", 6.0, 0.0, 12.0, unit="dB",
                             help="Maximum HF attenuation applied to hot sibilants.")
        self.add_float_param("attack_ms", 1.0, 0.1, 10.0, unit="ms",
                             help="Detector attack time (catches ess onsets).")
        self.add_float_param("release_ms", 60.0, 10.0, 300.0, unit="ms",
                             help="Detector release time (smooth recovery).")
        self.add_bool_param("listen", False,
                            help="Audition the tuning signal (see listen_mode).")
        self.add_menu_param(
            "listen_mode", ["Delta (Removed)", "Sibilant Band"], initial_idx=0,
            help="Audition mode while Listen is active. Delta plays only the "
                 "removed sibilance (silence when idle); Sibilant Band solos "
                 "the HF crossover band.")
        self.add_bool_param(
            "auto_threshold", False,
            help="Level-independent auto-threshold: compares sibilant energy "
                 "against a vowel-body reference (threshold acts as a contrast "
                 "offset around parity at the default -18 dB).")
        self.add_bool_param(
            "auto_learn", False,
            help="Momentary: capture 2.5 s of input and auto-set frequency "
                 "and threshold from the sibilant spectrum. Not saved with "
                 "the patch.")
        self.add_float_param("mix", 1.0, 0.0, 1.0,
                             help="Dry/wet balance (0.0 = bit-exact bypass). Intermediate "
                                  "values blend inside the crossover's phase-consistent "
                                  "domain, so no comb filtering occurs. Note: moving off "
                                  "0.0 engages the crossover phase and fresh detector "
                                  "state (clean engagement, no ghost of bypassed audio).")

        # --- Auto-Learn calibration state (NRT) ---
        # Two pre-allocated capture buffers: the audio thread writes into the
        # active one and swaps ownership with the NRT pool at completion, so
        # no allocation or copy happens on the audio thread.
        self._learn_capacity_samples = int(2.5 * SAMPLE_RATE)
        self._learn_capacity_blocks = max(
            1, self._learn_capacity_samples // BLOCK_SIZE)
        self._learn_bufs = [
            torch.zeros(self._learn_capacity_blocks * BLOCK_SIZE, dtype=torch.float32)
            for _ in range(2)
        ]
        self._learn_active = 0
        self._learn_block_idx = 0
        self._learning = False   # audio thread only sets/clears this flag
        self._analyzing = False  # set/cleared on the engine/control thread
        # Status text: only written on engine/control threads (never in
        # process()); the audio thread just flips _learning/_analyzing.
        self._learn_status = "Ready"

    def _bind_functions(self):
        super()._bind_functions()
        if hasattr(self.lib, "get_gr_db"):
            self.lib.get_gr_db.restype = ctypes.c_float
            self.lib.get_gr_db.argtypes = [ctypes.c_void_p]

    def process(self):
        # Standard FFINode dispatch with mono->stereo duplication and
        # anti-ghosting when the backend is unavailable.
        if not self.lib or not self.dsp_handle:
            out_slot = self.outputs.get("out")
            if out_slot:
                out_slot.buffer.zero_()
            return

        self._sync_params_to_cpp()

        raw_tensor = self.inp.get_tensor()
        processed_tensor = self._preprocess_input(raw_tensor, self._ffi_in_buffer)
        in_channels = processed_tensor.shape[0]

        out_slot = self.outputs.get("out")
        if not out_slot:
            return
        out_tensor = out_slot.buffer
        out_channels = out_tensor.shape[0]

        if not out_tensor.is_contiguous():
            raise RuntimeError(f"Output tensor is not contiguous. Node: {self.name}")

        if processed_tensor.device.type != "cpu":
            processed_tensor = processed_tensor.cpu()

        if processed_tensor.is_contiguous():
            processing_tensor = processed_tensor
        else:
            self._ffi_in_buffer.copy_(processed_tensor)
            processing_tensor = self._ffi_in_buffer

        if in_channels == 1 and out_channels == 2:
            self._ffi_in_buffer[0].copy_(processed_tensor[0])
            self._ffi_in_buffer[1].copy_(processed_tensor[0])
            processing_tensor = self._ffi_in_buffer
            process_channels = 2
        else:
            process_channels = min(in_channels, out_channels)
            if process_channels < out_channels:
                out_tensor[process_channels:].zero_()

        in_ptr = ctypes.cast(processing_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_ptr = ctypes.cast(out_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        self.lib.process(self.dsp_handle, in_ptr, out_ptr, process_channels, BLOCK_SIZE)

        # --- Auto-Learn capture (zero-allocation RT path) ---
        # Writes channel 0 into the pre-allocated capture buffer with a
        # no-alloc torch copy; the filled buffer is handed to the NRT pool by
        # reference (buffer swap), never copied here. Only flags/indices are
        # mutated — status text is composed on the control side.
        if self._learning:
            buf = self._learn_bufs[self._learn_active]
            start = self._learn_block_idx * BLOCK_SIZE
            buf[start:start + BLOCK_SIZE].copy_(processing_tensor[0])
            self._learn_block_idx += 1
            if self._learn_block_idx >= self._learn_capacity_blocks:
                self._learning = False
                self._learn_block_idx = 0
                engine = getattr(getattr(self, "graph", None), "engine", None)
                if engine is not None:
                    self._analyzing = True
                    self.submit_nrt(self._analyze_profile_nrt, buf,
                                    tag="auto_profile")
                    # Swap ownership: next capture uses the other buffer.
                    self._learn_active = 1 - self._learn_active

    def get_telemetry(self) -> dict:
        gr = 0.0
        if self.lib and self.dsp_handle and hasattr(self.lib, "get_gr_db"):
            try:
                gr = float(self.lib.get_gr_db(self.dsp_handle))
            except Exception:
                pass
        return {
            "gr_db": round(gr, 2),
            "latency_samples": 0,
            "latency_ms": 0.0,
            "status": self._learn_status_text(),
        }

    # ------------------------------------------------------------------
    # Auto-Learn (NRT calibration)
    # ------------------------------------------------------------------

    def _learn_status_text(self) -> str:
        """Compose the user-facing status on the control/UI side only."""
        if self._learning:
            return "Listening (2.5s)..."
        if self._analyzing:
            return "Analyzing..."
        return self._learn_status

    def on_ui_param_change(self, param_name: str):
        super().on_ui_param_change(param_name)
        if param_name == "auto_learn":
            # The engine has set()+sync()ed the value already (AGENTS.md §5).
            if self.params["auto_learn"].value:
                if self._learning or self._analyzing:
                    # Already busy: immediately cancel the momentary request
                    # so the checkbox doesn't stick.
                    engine = getattr(getattr(self, "graph", None), "engine", None)
                    if engine is not None:
                        engine.push_command(
                            ("param", self.id, "auto_learn", False))
                else:
                    self._learn_block_idx = 0
                    self._learning = True
            elif not self._learning:
                # Toggled off mid-capture (the NRT completion also resets it).
                pass

    def _analyze_profile_nrt(self, audio_buf):
        """Runs on the NRT pool. Analyzes the captured phrase and returns
        (frequency_hz, threshold_db) or None when no usable sibilance was
        detected (silence, no speech, ...)."""
        data = audio_buf.numpy()
        n_fft = 1024
        hop = 256
        if len(data) <= n_fft:
            return None

        # Reject silence: below ~-60 dBFS peak there is nothing to learn from.
        if float(np.max(np.abs(data))) < 1.0e-3:
            return None

        window = np.hanning(n_fft).astype(np.float32)
        num_frames = (len(data) - n_fft) // hop
        frames = np.stack([
            data[i * hop:i * hop + n_fft] * window for i in range(num_frames)
        ])
        specs = np.abs(np.fft.rfft(frames, axis=1))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)

        mid_mask = (freqs >= 200.0) & (freqs <= 3000.0)
        hf_mask = (freqs >= 4000.0) & (freqs <= 9000.0)
        e_mid = np.sum(specs[:, mid_mask] ** 2, axis=1) + 1e-9
        e_hf = np.sum(specs[:, hf_mask] ** 2, axis=1) + 1e-9
        ratios = e_hf / e_mid

        # Top 5% of frames by HF/mid ratio are the sibilants.
        thresh_ratio = np.percentile(ratios, 95)
        selected = ratios >= thresh_ratio
        if not np.any(selected):
            return None

        # Sibilant center: spectral peak of the average profile in 4.5-8.5 kHz.
        profile = np.mean(specs[selected], axis=0)
        search = (freqs >= 4500.0) & (freqs <= 8500.0)
        search_idx = np.where(search)[0]
        peak_idx = search_idx[np.argmax(profile[search])]
        learned_freq = float(round(freqs[peak_idx] / 100.0) * 100.0)

        # Threshold: the native detector is a peak follower, so derive the
        # level from the time-domain peak of the sibilant frames (dBFS), not
        # from FFT bin power. Place it 4 dB under peak sibilance.
        peak = float(np.max(np.abs(frames[selected])))
        learned_thresh = float(np.clip(20.0 * np.log10(peak + 1e-9) - 4.0,
                                       -36.0, -10.0))
        return learned_freq, learned_thresh

    def on_nrt_complete(self, tag, ok, result):
        if tag != "auto_profile":
            return
        self._analyzing = False
        engine = getattr(getattr(self, "graph", None), "engine", None)
        if ok and isinstance(result, tuple) and engine is not None:
            freq, thresh = result
            engine.push_command(("param", self.id, "frequency", freq))
            engine.push_command(("param", self.id, "threshold", thresh))
            engine.push_command(("param", self.id, "auto_learn", False))
            self._learn_status = f"Tuned: {freq:.0f} Hz, {thresh:.1f} dB"
        else:
            if engine is not None:
                engine.push_command(("param", self.id, "auto_learn", False))
            self._learn_status = ("Calibration Failed" if not ok
                                  else "Calibration Failed: no sibilance found")

    def on_nrt_discarded(self, tag, ok, payload):
        if tag == "auto_profile":
            # Node deleted / superseded while analysis was in flight: the
            # capture buffer is self-owned (nothing to release), just clear
            # the transient flag.
            self._analyzing = False

    # ------------------------------------------------------------------
    # Serialization: auto_learn is a momentary action, never persisted.
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["params"].pop("auto_learn", None)
        return d

    def load_state(self, data: dict):
        data = dict(data)
        if "params" in data and isinstance(data["params"], dict):
            data["params"] = {k: v for k, v in data["params"].items()
                              if k != "auto_learn"}
        super().load_state(data)
        self._learning = False
        self._analyzing = False
        self._learn_block_idx = 0


# ==============================================================================
# 2. UI Widget: hardware-style de-esser panel
# ==============================================================================
if GUI_AVAILABLE:

    class DeEsserWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "DeEsser"

        def __init__(self, proxy):
            super().__init__()
            self.proxy = proxy
            self.setMinimumSize(320, 260)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(4)

            # Generic smart parameter widgets (sliders/dials). A custom UI
            # REPLACES the node's auto-generated parameter panel, so the
            # continuous controls must be embedded here explicitly.
            self.param_widgets = {}  # Retain references for backend updates
            for pname in ("frequency", "threshold", "depth",
                          "attack_ms", "release_ms", "mix"):
                try:
                    w = proxy.create_param_widget(pname)
                    self.param_widgets[pname] = w
                    layout.addWidget(w)
                except Exception:
                    logging.exception(f"param widget '{pname}' failed")

            # Row 1: Listen toggle + audition mode selector
            row1 = QHBoxLayout()
            self.cb_listen = QCheckBox("LISTEN")
            self.cb_mode = QComboBox()
            self.cb_mode.addItems(["Delta (Removed)", "Sibilant Band"])
            row1.addWidget(self.cb_listen)
            row1.addWidget(self.cb_mode)
            row1.addStretch(1)
            layout.addLayout(row1)

            # Row 2: auto-threshold + one-click learn
            row2 = QHBoxLayout()
            self.cb_auto = QCheckBox("AUTO-THRESHOLD")
            self.btn_learn = QPushButton("LEARN (2.5s)")
            row2.addWidget(self.cb_auto)
            row2.addWidget(self.btn_learn)
            row2.addStretch(1)
            layout.addLayout(row2)

            # Row 3: live GR meter
            self.gr_bar = QProgressBar()
            self.gr_bar.setRange(0, 12)
            self.gr_bar.setFormat("Reduction: %v dB")
            self.gr_bar.setTextVisible(True)
            layout.addWidget(self.gr_bar)

            # Row 4: learn status
            self.lbl_status = QLabel("Ready")
            layout.addWidget(self.lbl_status)

            # Route control changes to the engine command path.
            self.cb_listen.stateChanged.connect(self._on_listen)
            self.cb_mode.currentIndexChanged.connect(self._on_mode)
            self.cb_auto.stateChanged.connect(self._on_auto)
            self.btn_learn.clicked.connect(self._on_learn)

        # -- UI -> engine ------------------------------------------------
        def _on_listen(self, state):
            # stateChanged delivers a Qt.CheckState in PySide6 6.x; read the
            # boolean straight off the control instead of bitmasking.
            self.proxy.set_parameter("listen", self.cb_listen.isChecked())

        def _on_mode(self, idx):
            self.proxy.set_parameter("listen_mode", idx)

        def _on_auto(self, state):
            self.proxy.set_parameter("auto_threshold",
                                     self.cb_auto.isChecked())

        def _on_learn(self):
            self.proxy.set_parameter("auto_learn", True)

        # -- engine -> UI ------------------------------------------------
        def update_from_params(self, simple_params: dict):
            self._updating = True
            try:
                # 1. Update embedded slider widgets (frequency, threshold,
                #    depth, ...) so Auto-Learn tuning moves the controls.
                for pname, val in simple_params.items():
                    widget = self.param_widgets.get(pname)
                    if widget is not None and hasattr(widget, "update_from_backend"):
                        widget.update_from_backend(val)

                # 2. Update toggles and modes
                if "listen" in simple_params:
                    self.cb_listen.setChecked(bool(simple_params["listen"]))
                if "listen_mode" in simple_params:
                    idx = int(simple_params["listen_mode"])
                    if 0 <= idx < self.cb_mode.count():
                        self.cb_mode.setCurrentIndex(idx)
                if "auto_threshold" in simple_params:
                    self.cb_auto.setChecked(bool(simple_params["auto_threshold"]))
                if "auto_learn" in simple_params:
                    self.btn_learn.setEnabled(not bool(simple_params["auto_learn"]))
            finally:
                self._updating = False

        def on_telemetry(self, data: dict):
            if not isinstance(data, dict):
                return
            if "gr_db" in data:
                gr = data["gr_db"]
                self.gr_bar.setValue(int(min(12, max(0, round(-gr)))))
                self.gr_bar.setFormat(f"Reduction: {gr:.1f} dB")
            status = data.get("status")
            if status:
                self.lbl_status.setText(str(status))
