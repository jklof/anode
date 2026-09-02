import torch
import torch.fft
import numpy as np
import os
import logging
import soundfile as sf
import resampy
from dataclasses import dataclass
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtCore import Qt

from base import Node, BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE

logger = logging.getLogger(__name__)

# Constants for Partitioned Convolution
PARTITION_SIZE = BLOCK_SIZE
FFT_SIZE = 2 * PARTITION_SIZE


@dataclass
class PreparedReverbState:
    """Fully prepared convolution reverb state for zero-allocation RT processing."""
    ir_ffts: torch.Tensor                    # (num_partitions, ir_channels, num_bins) complex64
    num_partitions: int
    ir_channels: int
    input_history: torch.Tensor              # (num_partitions, proc_channels, num_bins) complex64
    overlap_buffer: torch.Tensor             # (proc_channels, PARTITION_SIZE) float32
    padding_buffer: torch.Tensor             # (proc_channels, FFT_SIZE) float32
    product_buffer: torch.Tensor             # (num_partitions, proc_channels, num_bins) complex64
    accum_fft_buffer: torch.Tensor           # (proc_channels, num_bins) complex64
    result_buffer: torch.Tensor              # (proc_channels, PARTITION_SIZE) float32
    partition_indices: torch.Tensor          # (num_partitions,) long
    wrap_indices: torch.Tensor               # (num_partitions,) long
    ordered_input: torch.Tensor              # (num_partitions, proc_channels, num_bins) complex64
    dry_buffer: torch.Tensor                 # (proc_channels, PARTITION_SIZE) float32
    wet_buffer: torch.Tensor                 # (proc_channels, PARTITION_SIZE) float32
    log_indices: torch.Tensor                # (DISPLAY_BINS,) long
    history_ptr: int = 0


# ==============================================================================
# UI Class
# ==============================================================================


class ReverbWidget(QWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "ConvolutionReverb"

    def __init__(self, node_proxy):
        super().__init__()
        self.proxy = node_proxy
        self.setMinimumWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # --- Section 1: Parameters (File & Mix) ---
        # Unified File Parameter Widget
        self.file_widget = self.proxy.create_param_widget("ir_path")
        layout.addWidget(self.file_widget)

        self.mix_widget = self.proxy.create_param_widget("mix")
        layout.addWidget(self.mix_widget)

        # --- Section 2: Status ---
        # We keep status labels to show async loader state (Success/Error)
        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #666; margin-top: 5px;")
        layout.addWidget(self.lbl_status)

        self.lbl_file = QLabel("No IR Loaded")
        self.lbl_file.setStyleSheet("color: #aaa; font-size: 10px;")
        self.lbl_file.setWordWrap(True)
        layout.addWidget(self.lbl_file)

    def on_telemetry(self, data: dict):
        if "status" in data:
            self.lbl_status.setText(data["status"])
            style = "color: #00FF00" if data["status"] == "Ready" else "color: #FFaa00"
            self.lbl_status.setStyleSheet(style)
        if "filename" in data:
            self.lbl_file.setText(data["filename"])

    def update_from_params(self, params):
        # Update smart widgets
        if "mix" in params:
            self.mix_widget.update_from_backend(params["mix"])
        if "ir_path" in params:
            self.file_widget.update_from_backend(params["ir_path"])


class ConvolutionReverb(Node):
    category = "Effects"
    label = "Convolution Reverb"
    description = (
        "Partitioned frequency-domain convolution reverb. Impulse responses are "
        "decoded, resampled, and FFT-partitioned on a background NRT worker; the "
        "audio thread runs a zero-allocation uniform-partitioned overlap-add "
        "convolver. Before an IR is ready the dry signal is passed through "
        "scaled by (1 - mix)."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("in", help="Signal to reverberate; mono inputs are duplicated for stereo IRs.")
        self.add_output("out", help="Dry/wet mixed output (wet only after an IR is loaded).")

        self.add_float_param("mix", 0.5, 0.0, 1.0,
                             help="Dry/wet balance: 0 = dry only, 1 = wet only.")
        self.add_file_param("ir_path", "", filter="Audio Files (*.wav *.flac *.mp3)",
                            help="Impulse response file; prepared on a background worker.")

        self.loading = False
        self.current_ir_path = ""

        # Status tracking for telemetry
        self._status = "Idle"
        self._current_filename = "No IR Loaded"

        # DSP State - PreparedReverbState or None
        self._prepared_state: Optional[PreparedReverbState] = None

        # Log-frequency mapping (shared, computed once)
        self._fft_bins = FFT_SIZE // 2 + 1
        self._log_indices = self._compute_log_indices()

    def _compute_log_indices(self):
        """Compute log-frequency mapping indices (constant, can be shared)."""
        import numpy as np
        MIN_FREQ = 20.0
        MAX_FREQ = 20000.0
        DISPLAY_BINS = 128  # Not used but kept for compatibility
        bin_freqs = torch.linspace(0, SAMPLE_RATE / 2.0, steps=self._fft_bins)
        targets = torch.tensor(
            np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), num=128),  # Use a default
            dtype=DTYPE,
        )
        return torch.searchsorted(bin_freqs, targets).clamp_(0, self._fft_bins - 1)

    def on_ui_param_change(self, param_name):
        if param_name == "ir_path":
            path = self.params["ir_path"].get_staging_safe()
            if path and path != self.current_ir_path:
                self._start_loading(path)

    def load_state(self, data):
        super().load_state(data)
        if "ir_path" in self.params:
            path = self.params["ir_path"].value
            if path:
                self._start_loading(path)

    def _start_loading(self, path):
        self.loading = True
        self.current_ir_path = path
        self._status = "Loading..."
        self._current_filename = os.path.basename(path)
        self.submit_nrt(self._load_ir_blocking, path)

    def _load_ir_blocking(self, path):
        """Runs on an NRT pool thread. Uses soundfile/resampy (0% PyTorch) for
        file I/O and resampling to avoid OpenMP deadlocks with the audio thread.
        Returns a fully constructed PreparedReverbState."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # 1. Load WAV file into numpy array using soundfile (Pure C/Numpy, 0% PyTorch)
        data, sr = sf.read(path, dtype='float32')
        if len(data.shape) == 1:
            waveform = data[np.newaxis, :]  # Mono: Shape (1, samples)
        else:
            waveform = data.T  # Stereo/Multi: Shape (channels, samples)

        # 2. Resample using resampy if sample rate doesn't match
        if sr != SAMPLE_RATE:
            waveform = resampy.resample(waveform, sr, SAMPLE_RATE, axis=-1)

        # 3. Limit to max 2 channels
        if waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        # 4. Normalize
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform /= max_val
        waveform *= 0.2

        # 5. Partition and Pad
        num_samples = waveform.shape[1]
        num_partitions = max(1, int(np.ceil(num_samples / PARTITION_SIZE)))
        pad_len = num_partitions * PARTITION_SIZE - num_samples
        if pad_len > 0:
            waveform = np.pad(waveform, ((0, 0), (0, pad_len)))

        num_ir_channels = waveform.shape[0]
        num_bins = FFT_SIZE // 2 + 1
        proc_channels = max(num_ir_channels, CHANNELS)

        # Convert to PyTorch Tensor right before FFT
        waveform_tensor = torch.from_numpy(waveform)

        ir_ffts = torch.zeros((num_partitions, num_ir_channels, num_bins), dtype=torch.complex64)
        for i in range(num_partitions):
            start = i * PARTITION_SIZE
            chunk = waveform_tensor[:, start:start + PARTITION_SIZE]
            chunk_padded = torch.nn.functional.pad(chunk, (0, PARTITION_SIZE))
            ir_ffts[i] = torch.fft.rfft(chunk_padded, n=FFT_SIZE, dim=1)

        # 6. Pre-allocate ALL runtime buffers (zero-allocation RT path)
        input_history = torch.zeros((num_partitions, proc_channels, num_bins), dtype=torch.complex64)
        overlap_buffer = torch.zeros((proc_channels, PARTITION_SIZE), dtype=DTYPE)
        padding_buffer = torch.zeros((proc_channels, FFT_SIZE), dtype=DTYPE)
        product_buffer = torch.zeros((num_partitions, proc_channels, num_bins), dtype=torch.complex64)
        accum_fft_buffer = torch.zeros((proc_channels, num_bins), dtype=torch.complex64)
        result_buffer = torch.zeros((proc_channels, PARTITION_SIZE), dtype=DTYPE)

        # Circular-history ordering: precomputed index buffers
        partition_indices = torch.arange(num_partitions)
        wrap_indices = torch.empty(num_partitions, dtype=torch.long)
        ordered_input = torch.zeros((num_partitions, proc_channels, num_bins), dtype=torch.complex64)

        dry_buffer = torch.zeros((proc_channels, PARTITION_SIZE), dtype=DTYPE)
        wet_buffer = torch.zeros((proc_channels, PARTITION_SIZE), dtype=DTYPE)

        return PreparedReverbState(
            ir_ffts=ir_ffts,
            num_partitions=num_partitions,
            ir_channels=num_ir_channels,
            input_history=input_history,
            overlap_buffer=overlap_buffer,
            padding_buffer=padding_buffer,
            product_buffer=product_buffer,
            accum_fft_buffer=accum_fft_buffer,
            result_buffer=result_buffer,
            partition_indices=partition_indices,
            wrap_indices=wrap_indices,
            ordered_input=ordered_input,
            dry_buffer=dry_buffer,
            wet_buffer=wet_buffer,
            log_indices=self._log_indices,
            history_ptr=0,
        )

    def on_nrt_complete(self, tag, ok, result):
        if ok and isinstance(result, PreparedReverbState):
            # Atomic state swap on engine thread
            self._prepared_state = result
            # current_ir_path and _current_filename already set in _start_loading;
            # no need to reassign here.
            self.loading = False
            self._status = "Ready"
            self._current_filename = os.path.basename(self.current_ir_path)
        else:
            self.loading = False
            self._status = "Error"
            self._current_filename = "Load Failed"

    def start(self):
        # Transport restart must not replay the previous session: clear the
        # convolution history, overlap-add carry and the retained last input
        # block so no stale reverb tail leaks into the restarted stream
        # (AGENTS.md §7 reset contract).
        if self._prepared_state is not None:
            self._prepared_state.input_history.zero_()
            self._prepared_state.overlap_buffer.zero_()
            self._prepared_state.padding_buffer.zero_()
            self._prepared_state.history_ptr = 0

    def get_telemetry(self) -> dict:
        return {"status": self._status, "filename": self._current_filename}

    def process(self):
        input_tensor = self.inputs["in"].get_tensor()

        # 1. Get mix parameter
        mix_val = self.params["mix"].value

        # 2. Bypass / Not Ready
        if self._prepared_state is None:
            # Output input signal scaled by (1.0 - mix)
            self.outputs["out"].buffer.copy_(input_tensor)
            self.outputs["out"].buffer.mul_(1.0 - mix_val)
            return

        state = self._prepared_state
        in_channels = input_tensor.shape[0]
        ir_channels = state.ir_channels
        out_channels = max(in_channels, ir_channels)
        proc_channels = state.input_history.shape[1]

        # 3. DSP (Convolution) - Zero allocation using pre-allocated buffers
        state.padding_buffer.zero_()
        state.padding_buffer[:in_channels, :PARTITION_SIZE].copy_(input_tensor)
        if in_channels == 1 and out_channels == 2:
            state.padding_buffer[1, :PARTITION_SIZE].copy_(input_tensor[0])

        current_fft = torch.fft.rfft(state.padding_buffer[:proc_channels], n=FFT_SIZE, dim=1)

        state.history_ptr = (state.history_ptr - 1) % state.num_partitions
        state.input_history[state.history_ptr] = current_fft

        # Time-aligned ordering of the circular history. Zero-allocation:
        torch.add(state.partition_indices, state.history_ptr, out=state.wrap_indices)
        torch.remainder(state.wrap_indices, state.num_partitions, out=state.wrap_indices)
        torch.index_select(state.input_history, 0, state.wrap_indices, out=state.ordered_input)
        ordered_input = state.ordered_input

        ir_working = state.ir_ffts
        if ir_channels == 1 and proc_channels == 2:
            ir_working = ir_working.expand(-1, 2, -1)

        # Use pre-allocated buffers with out= to avoid heap allocations in RT
        torch.mul(ordered_input, ir_working, out=state.product_buffer)
        torch.sum(state.product_buffer, dim=0, out=state.accum_fft_buffer)

        # torch.fft has no out= variant; this one small allocation per block is
        # unavoidable with the public API.
        time_domain = torch.fft.irfft(state.accum_fft_buffer, n=FFT_SIZE, dim=1)

        # result = time_domain[:, :PARTITION_SIZE] + overlap_buffer
        state.result_buffer.copy_(time_domain[:, :PARTITION_SIZE])
        state.result_buffer.add_(state.overlap_buffer)

        # Update overlap buffer (view swap, no allocation)
        state.overlap_buffer.copy_(time_domain[:, PARTITION_SIZE:])

        # 4. Mix (Zero allocation using pre-allocated dry/wet buffers)
        torch.mul(input_tensor, 1.0 - mix_val, out=state.dry_buffer[:in_channels])
        if in_channels == 1 and out_channels == 2:
            state.dry_buffer[1].copy_(state.dry_buffer[0])

        torch.mul(state.result_buffer[:out_channels], mix_val, out=state.wet_buffer[:out_channels])

        target_buff = self.outputs["out"].buffer
        target_buff.zero_()
        copy_ch = min(target_buff.shape[0], out_channels)

        # output = dry_signal + wet_signal
        target_buff[:copy_ch].copy_(state.dry_buffer[:copy_ch])
        target_buff[:copy_ch].add_(state.wet_buffer[:copy_ch])
