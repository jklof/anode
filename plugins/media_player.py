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
    def __init__(self, source: str, output_queue: queue.Queue, event_queue: queue.SimpleQueue, looping: bool = False, start_time: float = 0.0):
        super().__init__(daemon=True)
        self.source = source
        self.output_queue = output_queue
        self.event_queue = event_queue
        self.looping = looping
        self.stop_event = threading.Event()
        self.seek_queue = queue.Queue()
        self.start_offset = start_time

    def run(self):
        container = None
        try:
            url = self.source
            title = os.path.basename(self.source)

            # --- 1. URL Resolution ---
            if self.source.startswith("http") or "www." in self.source:
                self.event_queue.put(("status", "Resolving..."))
                ydl_opts = {"format": "bestaudio/best", "quiet": True, "noplaylist": True}
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(self.source, download=False)
                        url = info["url"]
                        title = info.get("title", title)
                except Exception as e:
                    logger.error(f"YTDL Error: {e}")
                    self.event_queue.put(("status", "URL Error"))
                    return

            self.event_queue.put(("meta", {"title": title}))

            # --- 2. Open Stream ---
            self.event_queue.put(("status", "Opening..."))
            # Reconnect options help with network streams stopping randomly
            options = {"reconnect": "1", "reconnect_streamed": "1", "reconnect_delay_max": "10"}

            try:
                container = av.open(url, options=options)
            except Exception as e:
                logger.error(f"AV Open Error: {e}")
                self.event_queue.put(("status", "Open Failed"))
                return

            if not container.streams.audio:
                self.event_queue.put(("status", "No Audio"))
                return

            stream = container.streams.audio[0]
            duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
            self.event_queue.put(("meta", {"duration": duration}))

            # --- Handle Initial Seek ---
            if self.start_offset > 0:
                try:
                    target_ts = int(self.start_offset / stream.time_base)
                    container.seek(target_ts, stream=stream)
                except:
                    pass

            # --- 3. Configure Resampler ---
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=int(SAMPLE_RATE))
            self.event_queue.put(("status", "Buffering..."))

            buffer_accum = np.zeros((2, 0), dtype=np.float32)

            # --- 4. Decode Loop ---
            while not self.stop_event.is_set():
                for frame in container.decode(stream):
                    if self.stop_event.is_set():
                        break

                    # Handle Seek Request
                    if not self.seek_queue.empty():
                        self.event_queue.put(("status", "Seeking..."))
                        try:
                            target_ts = self.seek_queue.get_nowait()
                            timestamp = int(target_ts / stream.time_base)
                            container.seek(timestamp, stream=stream)
                            # Clear accumulator and queue
                            buffer_accum = np.zeros((2, 0), dtype=np.float32)
                            try:
                                while not self.output_queue.empty():
                                    self.output_queue.get_nowait()
                            except queue.Empty:
                                pass
                            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=int(SAMPLE_RATE))
                            self.event_queue.put(("seeked", target_ts))
                        except Exception as e:
                            logger.error(f"Seek Error: {e}")

                        self.event_queue.put(("status", "Buffering..."))
                        continue

                    # Resample
                    try:
                        resampled_frames = resampler.resample(frame)
                    except Exception:
                        continue

                    if not resampled_frames:
                        continue

                    # Convert to numpy and concatenate in one pass per decode
                    # batch (a rolling np.hstack per frame is O(n^2) in copies).
                    # AV returns list of frames (usually 1, but can be more)
                    chunks = [buffer_accum] if buffer_accum.shape[1] > 0 else []
                    for r_frame in resampled_frames:
                        np_frame = r_frame.to_ndarray()  # Shape (Channels, Samples)

                        # Force Stereo
                        if np_frame.shape[0] == 1:
                            np_frame = np.vstack([np_frame, np_frame])
                        elif np_frame.shape[0] > 2:
                            np_frame = np_frame[:2, :]

                        chunks.append(np_frame)
                    buffer_accum = (
                        np.concatenate(chunks, axis=1)
                        if chunks
                        else np.zeros((2, 0), dtype=np.float32)
                    )

                    # Push blocks to queue
                    while buffer_accum.shape[1] >= BLOCK_SIZE:
                        # Extract one block
                        block = buffer_accum[:, :BLOCK_SIZE]
                        buffer_accum = buffer_accum[:, BLOCK_SIZE:]

                        tensor_block = torch.from_numpy(block.copy())

                        # Blocking Put with timeout to allow checking stop_event
                        inserted = False
                        while not inserted and not self.stop_event.is_set() and self.seek_queue.empty():
                            try:
                                self.output_queue.put(tensor_block, timeout=0.1)
                                inserted = True
                                self.event_queue.put(("status", "Playing"))
                            except queue.Full:
                                # If queue is full, just wait and try again
                                # This throttles the decoding to the playback speed
                                continue

                if self.stop_event.is_set():
                    break

                # We hit EOF
                if self.looping:
                    self.event_queue.put(("status", "Looping..."))
                    try:
                        container.seek(0, stream=stream)
                        buffer_accum = np.zeros((2, 0), dtype=np.float32)
                        try:
                            while not self.output_queue.empty():
                                self.output_queue.get_nowait()
                        except queue.Empty:
                            pass
                        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=int(SAMPLE_RATE))
                        self.event_queue.put(("seeked", 0.0))
                        self.event_queue.put(("status", "Playing"))
                        continue
                    except Exception as e:
                        logger.error(f"Looping Seek Error: {e}")
                        break
                else:
                    break

            if not self.stop_event.is_set() and not self.looping:
                self.event_queue.put(("status", "Finished"))
                self.event_queue.put(("eof", True))

        except Exception as e:
            logger.error(f"Worker Crash: {e}")
            self.event_queue.put(("status", "Error"))
        finally:
            if container:
                try:
                    container.close()
                except:
                    pass

    def seek(self, time_sec):
        try:
            while not self.seek_queue.empty():
                self.seek_queue.get_nowait()
        except queue.Empty:
            pass
        self.seek_queue.put(time_sec)

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

        self.btn_loop = QPushButton("Once")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setFixedSize(60, 30)
        self.btn_loop.clicked.connect(self.toggle_loop)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setStyleSheet("color: #888;")

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_time.setStyleSheet("font-family: monospace;")

        r5.addWidget(self.btn_play)
        r5.addWidget(self.btn_loop)
        r5.addWidget(self.lbl_status)
        r5.addStretch()
        r5.addWidget(self.lbl_time)
        layout.addLayout(r5)

    def toggle_play(self):
        playing = self.btn_play.isChecked()
        self.btn_play.setText("Pause" if playing else "Play")
        self.proxy.set_parameter("playing", playing)

    def toggle_loop(self):
        is_loop = self.btn_loop.isChecked()
        self.btn_loop.setText("Loop" if is_loop else "Once")
        self.proxy.set_parameter("looping", is_loop)

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
        if "looping" in params:
            l = bool(params["looping"])
            self.btn_loop.setChecked(l)
            self.btn_loop.setText("Loop" if l else "Once")
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
    description = (
        "Streams local audio files or remote URLs (via yt-dlp resolution) "
        "through a background worker thread with reconnecting ffmpeg demuxing. "
        "The audio thread pulls pre-converted blocks from a bounded queue "
        "without blocking; underruns output silence. Supports play/pause, "
        "looping, and seeking from its custom UI."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_file_param("file_path", "", filter="Audio Files (*.mp3 *.wav *.flac *.m4a);;All (*.*)",
                            help="Media file (or URL) to play; loading happens on a background worker.")
        self.add_bool_param("playing", True,
                            help="Play/pause; when off the node outputs silence but keeps buffering.")
        self.add_bool_param("looping", False,
                            help="Restart from the beginning when playback reaches the end.")
        self.add_float_param("seek_ratio", -1.0, min_v=-1.0, max_v=1.0,
                             help="Write-only seek trigger: set to a 0..1 position to seek; -1 = idle.")
        self.add_output("out", help="Decoded stereo audio (silence while paused or buffering).")

        # Increase Queue size to prevent buffer underruns
        self.queue = queue.Queue(maxsize=500)
        self._event_queue = queue.SimpleQueue()
        self.worker = None

        # Worker restart machinery. The NRT restart path owns worker lifecycle
        # (serialized by _restart_lock); completed bundles are installed on
        # the engine thread in on_nrt_complete so the audio path never sees a
        # half-configured worker/queue pair.
        self._restart_lock = threading.Lock()
        self._pending_bundle = None

        self.current_path = ""
        self.playback_frames = 0
        self.total_duration = 0.0
        self.current_title = "No Media"
        self.status_msg = "Idle"
        self.eof = False

    def _drain_events(self):
        if self._event_queue is None:
            return
        while True:
            try:
                ev_type, ev_data = self._event_queue.get_nowait()
                self._handle_worker_event(ev_type, ev_data)
            except queue.Empty:
                break

    def _request_restart(self, path, start_time=0.0):
        self.submit_nrt(self._do_restart_worker, path, start_time, tag="restart")

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
                self._request_restart(path)

    def on_ui_param_change(self, param_name):
        # The engine set()+sync()ed the parameter before invoking this callback
        # (AGENTS.md §5), so params[param_name].value is already committed.
        if param_name == "file_path":
            path = self.params["file_path"].value
            if path:
                self.current_path = path
                self._request_restart(path)

        elif param_name == "playing":
            should_play = self.params["playing"].value
            if should_play and (self.eof or self.worker is None) and self.current_path:
                self._request_restart(self.current_path)

        elif param_name == "looping":
            if self.worker:
                self.worker.looping = bool(self.params["looping"].value)

        elif param_name == "seek_ratio":
            val = self.params["seek_ratio"].value
            if val >= 0:
                target_time = val * self.total_duration
                if self.eof or self.worker is None:
                    if self.current_path:
                        self._request_restart(self.current_path, start_time=target_time)
                elif self.worker:
                    self.worker.seek(target_time)

                self.playback_frames = int(target_time * SAMPLE_RATE)
                self.eof = False
                self.params["seek_ratio"].set(-1.0)
                self.params["seek_ratio"].sync()

    def _do_restart_worker(self, path, start_time=0.0):
        """NRT thread. Owns the full worker lifecycle: retires the currently
        installed worker and any superseded pending worker, then builds a
        complete replacement bundle. The bundle is installed on the engine
        thread (on_nrt_complete), so shared node state is only mutated at a
        safe boundary — never concurrently with audio processing."""
        with self._restart_lock:
            prev = self._pending_bundle
            if prev is not None and prev.get("worker") is not None:
                prev["worker"].stop()
                prev["worker"].join(timeout=2.0)

            installed = self.worker
            if installed is not None and (prev is None or installed is not prev.get("worker")):
                installed.stop()
                installed.join(timeout=2.0)
                if installed.is_alive():
                    logging.warning(
                        f"MediaPlayerNode {self.id}: previous worker did not "
                        f"exit in time; abandoning it."
                    )

            bundle = {
                "worker": None,
                "queue": None,
                "event_queue": None,
                "start_frames": int(start_time * SAMPLE_RATE),
                "deps_missing": False,
            }
            if MEDIA_DEPS_AVAILABLE:
                q = queue.Queue(maxsize=500)
                ev = queue.SimpleQueue()
                w = MediaStreamWorker(
                    path,
                    q,
                    ev,
                    looping=bool(self.params["looping"].value),
                    start_time=start_time,
                )
                w.start()
                bundle.update(worker=w, queue=q, event_queue=ev)
            else:
                bundle["deps_missing"] = True
            self._pending_bundle = bundle
            return bundle

    def on_nrt_complete(self, tag, ok, result):
        if tag != "restart":
            return
        if not ok:
            self.status_msg = "Playback Error"
            return
        # Install the prepared bundle atomically on the engine thread.
        with self._restart_lock:
            self._pending_bundle = None
        self.queue = result["queue"] if result["queue"] is not None else queue.Queue(maxsize=500)
        self._event_queue = result["event_queue"] if result["event_queue"] is not None else queue.SimpleQueue()
        self.worker = result["worker"]
        self.playback_frames = result["start_frames"]
        self.total_duration = 0.0
        self.eof = False
        if result["deps_missing"]:
            self.status_msg = "Dependencies Missing"

    def on_nrt_discarded(self, tag, ok, result):
        if tag == "restart" and ok and isinstance(result, dict):
            # Superseded restart: the bundle was never installed, so its
            # worker thread must be retired here (the next restart's NRT job
            # only knows about _pending_bundle / installed workers).
            w = result.get("worker")
            if w is not None:
                w.stop()
                w.join(timeout=2.0)

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
        elif type == "seeked":
            self.playback_frames = int(data * SAMPLE_RATE)
            self.eof = False

    def process(self):
        self._drain_events()
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
        self._drain_events()
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

    def remove(self):
        """Called when node is deleted. Non-blocking via NRT executor."""
        if self.worker:
            self.graph.engine.nrt.stop_stream(self, self.worker.stop, self.worker)
        pending = self._pending_bundle
        if pending is not None and pending.get("worker") is not None and pending["worker"] is not self.worker:
            self.graph.engine.nrt.stop_stream(self, pending["worker"].stop, pending["worker"])
        self._pending_bundle = None

    def to_dict(self):
        self._drain_events()
        d = super().to_dict()
        d["meta"] = {"title": self.current_title, "duration": self.total_duration, "path": self.current_path}
        return d