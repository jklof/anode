import torch
import torch.fft
import torchaudio
import numpy as np
import threading
import queue
import os
import logging

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSlider,
)
from PySide6.QtCore import Qt, QTimer, QSignalBlocker

from base import Node, BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE

logger = logging.getLogger(__name__)

# Constants for Partitioned Convolution
PARTITION_SIZE = BLOCK_SIZE
FFT_SIZE = 2 * PARTITION_SIZE


class IrLoaderThread(threading.Thread):
    def __init__(self, path, result_queue):
        super().__init__(daemon=True)
        self.path = path
        self.result_queue = result_queue

    def run(self):
        try:
            if not os.path.exists(self.path):
                self.result_queue.put(("error", f"File not found: {self.path}"))
                return

            try:
                waveform, sr = torchaudio.load(self.path)
            except Exception as e:
                self.result_queue.put(("error", f"Load Failed: {e}"))
                return

            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)

            if waveform.shape[0] > 2:
                waveform = waveform[:2, :]

            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform /= max_val

            # Compensation for convolution energy gain
            waveform *= 0.2

            num_samples = waveform.shape[1]
            num_partitions = int(np.ceil(num_samples / PARTITION_SIZE))
            if num_partitions == 0:
                num_partitions = 1

            pad_len = num_partitions * PARTITION_SIZE - num_samples
            if pad_len > 0:
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))

            num_ir_channels = waveform.shape[0]
            num_bins = FFT_SIZE // 2 + 1
            complex_dtype = torch.complex64

            ir_ffts = torch.zeros((num_partitions, num_ir_channels, num_bins), dtype=complex_dtype)

            for i in range(num_partitions):
                start = i * PARTITION_SIZE
                end = start + PARTITION_SIZE
                chunk = waveform[:, start:end]
                chunk_padded = torch.nn.functional.pad(chunk, (0, PARTITION_SIZE))
                fft_chunk = torch.fft.rfft(chunk_padded, n=FFT_SIZE, dim=1)
                ir_ffts[i] = fft_chunk

            self.result_queue.put(
                (
                    "success",
                    {
                        "ir_ffts": ir_ffts,
                        "num_partitions": num_partitions,
                        "channels": num_ir_channels,
                        "path": self.path,
                    },
                )
            )

        except Exception as e:
            logger.error(f"IR Load Error: {e}")
            self.result_queue.put(("error", str(e)))


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

        # --- Section 1: File Loader ---
        self.lbl_file = QLabel("No IR Loaded")
        self.lbl_file.setStyleSheet("color: #aaa; font-size: 10px; margin-bottom: 2px;")
        self.lbl_file.setWordWrap(True)
        layout.addWidget(self.lbl_file)

        btn_load = QPushButton("Load IR File")
        btn_load.clicked.connect(self.browse)
        layout.addWidget(btn_load)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #666; margin-bottom: 5px;")
        layout.addWidget(self.lbl_status)

        # --- Section 2: Parameters ---
        self.mix_widget = self.proxy.create_param_widget("mix")
        layout.addWidget(self.mix_widget)


    def browse(self):
        f, _ = QFileDialog.getOpenFileName(None, "Open Impulse Response", "", "Audio Files (*.wav *.flac *.mp3)")
        if f:
            self.lbl_status.setText("Requesting...")
            self.proxy.set_parameter("ir_path", f)

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


class ConvolutionReverb(Node):
    category = "Effects"
    label = "Convolution Reverb"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("in")
        self.add_output("out")

        self.add_float_param("mix", 0.5, 0.0, 1.0)
        self.add_string_param("ir_path", "")

        self.loader_queue = queue.Queue()
        self.loading = False
        self.current_ir_path = ""

        # Status tracking for telemetry
        self._status = "Idle"
        self._current_filename = "No IR Loaded"

        # DSP State
        self.ir_ffts = None
        self.input_history = None
        self.history_ptr = 0
        self.overlap_buffer = None
        self.dsp_ready = False

    def on_ui_param_change(self, param_name):
        self.params[param_name].sync()
        if param_name == "ir_path":
            path = self.params["ir_path"].value
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
        t = IrLoaderThread(path, self.loader_queue)
        t.start()

    def get_telemetry(self) -> dict:
        return {"status": self._status, "filename": self._current_filename}

    def _init_buffers(self, num_partitions, ir_channels, audio_channels):
        num_bins = FFT_SIZE // 2 + 1
        proc_channels = max(ir_channels, audio_channels)
        self.input_history = torch.zeros((num_partitions, proc_channels, num_bins), dtype=torch.complex64)
        self.overlap_buffer = torch.zeros((proc_channels, PARTITION_SIZE), dtype=DTYPE)
        self.history_ptr = 0
        self.dsp_ready = True

    def process(self):
        input_tensor = self.inputs["in"].get_tensor()

        # 1. Check for Load
        try:
            msg = self.loader_queue.get_nowait()
            if msg[0] == "success":
                data = msg[1]
                self.ir_ffts = data["ir_ffts"]
                self.dsp_ready = False
                self.current_ir_path = data["path"]
                self.loading = False
                self._status = "Ready"
                self._current_filename = os.path.basename(data["path"])
            elif msg[0] == "error":
                self.loading = False
                self._status = "Error"
                self._current_filename = "Load Failed"
        except queue.Empty:
            pass

        # 2. Get mix parameter
        mix_val = self.params["mix"].value

        # 3. Bypass / Not Ready
        if not self.dsp_ready and self.ir_ffts is None:
            # Output input signal scaled by (1.0 - mix)
            self.outputs["out"].buffer.copy_(input_tensor)
            self.outputs["out"].buffer.mul_(1.0 - mix_val)
            return

        # 4. Initialize buffers
        in_channels = input_tensor.shape[0]
        ir_channels = self.ir_ffts.shape[1]
        out_channels = max(in_channels, ir_channels)

        if (
            not self.dsp_ready
            or self.input_history is None
            or self.input_history.shape[1] != out_channels
            or self.input_history.shape[0] != self.ir_ffts.shape[0]
        ):
            self._init_buffers(self.ir_ffts.shape[0], ir_channels, in_channels)

        # 5. DSP (Convolution)
        padded_input = torch.nn.functional.pad(input_tensor, (0, PARTITION_SIZE))
        if in_channels == 1 and out_channels == 2:
            padded_input = padded_input.expand(2, -1)

        current_fft = torch.fft.rfft(padded_input, n=FFT_SIZE, dim=1)

        self.history_ptr = (self.history_ptr - 1) % self.input_history.shape[0]
        self.input_history[self.history_ptr] = current_fft

        indices = torch.arange(self.history_ptr, self.history_ptr + self.input_history.shape[0])
        indices %= self.input_history.shape[0]
        ordered_input = self.input_history[indices]

        ir_working = self.ir_ffts
        if ir_channels == 1 and out_channels == 2:
            ir_working = ir_working.expand(-1, 2, -1)

        product = ordered_input * ir_working
        accum_fft = torch.sum(product, dim=0)

        time_domain = torch.fft.irfft(accum_fft, n=FFT_SIZE, dim=1)

        result = time_domain[:, :PARTITION_SIZE] + self.overlap_buffer
        self.overlap_buffer = time_domain[:, PARTITION_SIZE:]

        # 6. Mix
        dry_signal = input_tensor * (1.0 - mix_val)
        wet_signal = result * mix_val

        # Handle channel mismatches for dry_signal
        if in_channels == 1 and out_channels == 2:
            dry_signal = dry_signal.expand(2, -1)
        elif in_channels == 2 and out_channels == 1:
            dry_signal = dry_signal[:1, :]

        target_buff = self.outputs["out"].buffer
        target_buff.zero_()
        copy_ch = min(target_buff.shape[0], out_channels)

        # output = dry_signal + wet_signal
        target_buff[:copy_ch].copy_(dry_signal[:copy_ch])
        target_buff[:copy_ch].add_(wet_signal[:copy_ch])
