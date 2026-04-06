import threading
import time
import logging
import queue
import numpy as np
import torch
import os

# --- Node System Imports ---
from base import Node, BLOCK_SIZE, SAMPLE_RATE, DTYPE

# --- Qt Imports ---
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
)
from PySide6.QtCore import Qt

# --- Media Dependencies ---
try:
    import av
    import yt_dlp

    MEDIA_DEPS_AVAILABLE = True
except ImportError:
    MEDIA_DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==============================================================================
# Worker Thread
# ==============================================================================


class MediaStreamWorker(threading.Thread):
    def __init__(self, source: str, output_queue: queue.Queue, event_callback, start_time=0.0):
        super().__init__(daemon=True)
        self.source = source
        self.output_queue = output_queue
        self.event_callback = event_callback
        self.stop_event = threading.Event()
        self.seek_request = -1.0
        self.start_offset = start_time

    def run(self):
        container = None
        try:
            url = self.source
            title = os.path.basename(self.source)

            # --- 1. URL Resolution ---
            if self.source.startswith("http") or "www." in self.source:
                self.event_callback("status", "Resolving...")
                ydl_opts = {"format": "bestaudio/best", "quiet": True, "noplaylist": True}
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(self.source, download=False)
                        url = info["url"]
                        title = info.get("title", title)
                except Exception as e:
                    logger.error(f"YTDL Error: {e}")
                    self.event_callback("status", "URL Error")
                    return

            self.event_callback("meta", {"title": title})

            # --- 2. Open Stream ---
            self.event_callback("status", "Opening...")
            # Reconnect options help with network streams stopping randomly
            options = {"reconnect": "1", "reconnect_streamed": "1", "reconnect_delay_max": "10"}

            try:
                container = av.open(url, options=options)
            except Exception as e:
                logger.error(f"AV Open Error: {e}")
                self.event_callback("status", "Open Failed")
                return

            if not container.streams.audio:
                self.event_callback("status", "No Audio")
                return

            stream = container.streams.audio[0]
            duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
            self.event_callback("meta", {"duration": duration})

            # --- Handle Initial Seek ---
            if self.start_offset > 0:
                try:
                    target_ts = int(self.start_offset / stream.time_base)
                    container.seek(target_ts, stream=stream)
                except:
                    pass

            # --- 3. Configure Resampler ---
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=int(SAMPLE_RATE))
            self.event_callback("status", "Buffering...")

            buffer_accum = np.zeros((2, 0), dtype=np.float32)

            # --- 4. Decode Loop ---
            for frame in container.decode(stream):
                if self.stop_event.is_set():
                    break

                # Handle Seek Request
                if self.seek_request >= 0:
                    self.event_callback("status", "Seeking...")
                    try:
                        timestamp = int(self.seek_request / stream.time_base)
                        container.seek(timestamp, stream=stream)
                        # Clear accumulator and queue
                        buffer_accum = np.zeros((2, 0), dtype=np.float32)
                        with self.output_queue.mutex:
                            self.output_queue.queue.clear()
                        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=int(SAMPLE_RATE))
                    except Exception as e:
                        logger.error(f"Seek Error: {e}")

                    self.seek_request = -1.0
                    self.event_callback("status", "Buffering...")
                    continue

                # Resample
                try:
                    resampled_frames = resampler.resample(frame)
                except Exception:
                    continue

                if not resampled_frames:
                    continue

                # Convert to numpy and stack
                # AV returns list of frames (usually 1, but can be more)
                for r_frame in resampled_frames:
                    np_frame = r_frame.to_ndarray()  # Shape (Channels, Samples)

                    # Force Stereo
                    if np_frame.shape[0] == 1:
                        np_frame = np.vstack([np_frame, np_frame])
                    elif np_frame.shape[0] > 2:
                        np_frame = np_frame[:2, :]

                    buffer_accum = np.hstack([buffer_accum, np_frame])

                # Push blocks to queue
                while buffer_accum.shape[1] >= BLOCK_SIZE:
                    # Extract one block
                    block = buffer_accum[:, :BLOCK_SIZE]
                    buffer_accum = buffer_accum[:, BLOCK_SIZE:]

                    tensor_block = torch.from_numpy(block.copy())

                    # Blocking Put with timeout to allow checking stop_event
                    inserted = False
                    while not inserted and not self.stop_event.is_set() and self.seek_request < 0:
                        try:
                            self.output_queue.put(tensor_block, timeout=0.1)
                            inserted = True
                            self.event_callback("status", "Playing")
                        except queue.Full:
                            # If queue is full, just wait and try again
                            # This throttles the decoding to the playback speed
                            continue

            self.event_callback("status", "Finished")
            self.event_callback("eof", True)

        except Exception as e:
            logger.error(f"Worker Crash: {e}")
            self.event_callback("status", "Error")
        finally:
            if container:
                try:
                    container.close()
                except:
                    pass

    def seek(self, time_sec):
        self.seek_request = time_sec

    def stop(self):
        self.stop_event.set()


# ==============================================================================
# UI Class
# ==============================================================================


class MediaPlayerWidget(QWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "MediaPlayerNode"

    def __init__(self, node_proxy):
        super().__init__()
        self.proxy = node_proxy
        self.stored_title = "No Media"

        self.setMinimumSize(450, 150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        if not MEDIA_DEPS_AVAILABLE:
            layout.addWidget(QLabel("Missing deps: av, yt-dlp"))
            return

        # Row 1: Unified File Parameter
        self.file_widget = self.proxy.create_param_widget("file_path")
        layout.addWidget(self.file_widget)

        # Row 2: Metadata
        self.lbl_title = QLabel(self.stored_title)
        self.lbl_title.setStyleSheet("color: #ccc; font-weight: bold; font-size: 11pt;")
        self.lbl_title.setWordWrap(True)
        layout.addWidget(self.lbl_title)

        # Row 3: Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self.on_slider_release)
        layout.addWidget(self.slider)

        # Row 4: Controls & Status
        r5 = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setFixedSize(60, 30)
        self.btn_play.clicked.connect(self.toggle_play)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setStyleSheet("color: #888;")

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_time.setStyleSheet("font-family: monospace;")

        r5.addWidget(self.btn_play)
        r5.addWidget(self.lbl_status)
        r5.addStretch()
        r5.addWidget(self.lbl_time)
        layout.addLayout(r5)

    def toggle_play(self):
        playing = self.btn_play.isChecked()
        self.btn_play.setText("Pause" if playing else "Play")
        self.proxy.set_parameter("playing", playing)

    def on_slider_release(self):
        val = self.slider.value() / 1000.0
        self.proxy.set_parameter("seek_ratio", val)

    def on_telemetry(self, data: dict):
        if "status" in data:
            self.lbl_status.setText(data["status"])
        if "title" in data:
            self.stored_title = data["title"]
            self.lbl_title.setText(data["title"])
        if "time_str" in data:
            self.lbl_time.setText(data["time_str"])
        if "progress" in data and not self.slider.isSliderDown():
            self.slider.setEnabled(True)
            self.slider.setValue(int(data["progress"] * 1000))
        if "playing_state" in data:
            is_playing = data["playing_state"]
            self.btn_play.setChecked(is_playing)
            self.btn_play.setText("Pause" if is_playing else "Play")
            if not is_playing and self.slider.value() > 950:
                self.slider.setValue(1000)

    def update_from_params(self, params):
        if "file_path" in params:
            self.file_widget.update_from_backend(params["file_path"])
        if "playing" in params:
            p = bool(params["playing"])
            self.btn_play.setChecked(p)
            self.btn_play.setText("Pause" if p else "Play")


# ==============================================================================
# Logic Class
# ==============================================================================


class MediaPlayerNode(Node):
    category = "I/O"
    label = "Media Player"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_file_param("file_path", "", filter="Audio Files (*.mp3 *.wav *.flac *.m4a);;All (*.*)")
        self.add_bool_param("playing", False)
        self.add_float_param("seek_ratio", -1.0)
        self.add_output("out")

        # Increase Queue size to prevent buffer underruns
        self.queue = queue.Queue(maxsize=500)
        self.worker = None

        self.current_path = ""
        self.playback_frames = 0
        self.total_duration = 0.0
        self.current_title = "No Media"
        self.status_msg = "Idle"
        self.eof = False

    def load_state(self, data: dict):
        """
        Override to trigger worker start on load.
        """
        super().load_state(data)

        # Restore metadata
        if "meta" in data:
            self.current_title = data["meta"].get("title", "No Media")
            self.total_duration = data["meta"].get("duration", 0.0)
            self.current_path = data["meta"].get("path", "")

        # Trigger explicit load if path exists
        if "file_path" in self.params:
            path = self.params["file_path"].value
            if path:
                self.current_path = path
                self._restart_worker(path)

    def on_ui_param_change(self, param_name):
        if param_name in self.params:
            self.params[param_name].sync()

        if param_name == "file_path":
            path = self.params["file_path"].value
            if path:
                self.current_path = path
                self._restart_worker(path)

        elif param_name == "playing":
            should_play = self.params["playing"].value
            if should_play and (self.eof or self.worker is None) and self.current_path:
                self._restart_worker(self.current_path)

        elif param_name == "seek_ratio":
            val = self.params["seek_ratio"].value
            if val >= 0:
                target_time = val * self.total_duration
                if self.eof or self.worker is None:
                    if self.current_path:
                        self._restart_worker(self.current_path, start_time=target_time)
                elif self.worker:
                    self.worker.seek(target_time)

                self.playback_frames = int(target_time * SAMPLE_RATE)
                self.eof = False
                self.params["seek_ratio"].set(-1.0)
                self.params["seek_ratio"].sync()

    def _restart_worker(self, path, start_time=0.0):
        if self.worker:
            self.worker.stop()
            self.worker = None

        # Create NEW queue object instead of clearing
        self.queue = queue.Queue(maxsize=500)  # ✅ Thread-safe

        self.playback_frames = int(start_time * SAMPLE_RATE)
        self.total_duration = 0.0
        self.eof = False

        if MEDIA_DEPS_AVAILABLE:
            self.worker = MediaStreamWorker(
                path, self.queue, lambda type, data: self._handle_worker_event(type, data), start_time=start_time
            )
            self.worker.start()
        else:
            self.status_msg = "Dependencies Missing"

    def _handle_worker_event(self, type, data):
        if type == "meta":
            if "duration" in data:
                self.total_duration = data["duration"]
            if "title" in data:
                self.current_title = data["title"]
        elif type == "status":
            self.status_msg = data
        elif type == "eof":
            self.eof = True
            self.status_msg = "Finished"

    def process(self):
        # If play param is False, we just output silence.
        # But we keep worker alive (it pauses on full queue).
        if not self.params["playing"].value:
            self.outputs["out"].buffer.zero_()
            return

        try:
            data = self.queue.get_nowait()
            self.outputs["out"].buffer.copy_(data)
            self.playback_frames += BLOCK_SIZE
        except queue.Empty:
            # Buffer Underrun
            self.outputs["out"].buffer.zero_()
            if self.worker and not self.eof:
                if self.status_msg != "Buffering...":
                    self.status_msg = "Buffering..."
            elif self.eof:
                # Actual end of song
                if self.params["playing"].value:
                    self.params["playing"].set(False)

    def get_telemetry(self) -> dict:
        sec = self.playback_frames / SAMPLE_RATE
        dur_str = f"{int(self.total_duration//60):02}:{int(self.total_duration%60):02}"
        time_str = f"{int(sec//60):02}:{int(sec%60):02} / {dur_str}"

        progress = 0.0
        if self.total_duration > 0:
            progress = np.clip(sec / self.total_duration, 0.0, 1.0)

        return {
            "status": self.status_msg,
            "title": self.current_title,
            "time_str": time_str,
            "progress": progress,
            "playing_state": self.params["playing"].value,
        }

    def stop(self):
        """Called when audio transport stops. Pause logical playback."""
        self.params["playing"].set(False)
        self.params["playing"].sync()

    def remove(self):
        """Called when node is deleted. Cleanup threads."""
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=1.0)
            self.worker = None

    def to_dict(self):
        d = super().to_dict()
        d["meta"] = {"title": self.current_title, "duration": self.total_duration, "path": self.current_path}
        return d
