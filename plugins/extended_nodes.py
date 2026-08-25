import logging
import queue
import threading
import wave

import torch
import numpy as np
from base import Node, IClockProvider, BLOCK_SIZE, DTYPE, SAMPLE_RATE, CHANNELS

logger = logging.getLogger(__name__)


class Note(Node):
    category = "Visual"
    label = "Comment / Note"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_string_param("text", "Hello World")

    def process(self):
        pass


class Noise(Node):
    category = "Sources"
    label = "White Noise"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_bool_param("enabled", True)
        self.add_float_param("amp", 0.1)
        self.out = self.add_output("out")

    def process(self):
        if self.params["enabled"].value:
            torch.rand(self.out.buffer.shape, out=self.out.buffer)
            self.out.buffer.mul_(2.0).sub_(1.0)
            self.out.buffer.mul_(self.params["amp"].value)
        else:
            self.out.buffer.zero_()


class Selector(Node):
    category = "Utilities"
    label = "A/B Selector"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_menu_param("source", ["Input A", "Input B"])
        self.in_a = self.add_input("A")
        self.in_b = self.add_input("B")
        self.out = self.add_output("out")

    def process(self):
        idx = int(self.params["source"].value)
        if idx == 0:
            self.out.buffer.copy_(self.in_a.get_tensor())
        else:
            self.out.buffer.copy_(self.in_b.get_tensor())


class FileRecorder(Node):
    category = "I/O"
    label = "File Recorder"

    # Blocks of audio the writer may fall behind before RT drops frames
    # (128 blocks ≈ 1.4 s cushion at 48 kHz / 512).
    QUEUE_CAPACITY = 128

    def __init__(self, name=""):
        super().__init__(name)
        self.add_file_param("filename", "output.wav", filter="WAV Files (*.wav)", mode="save")
        self.add_bool_param("record", False)
        self.inp = self.add_input("in")

        # RT-side gate only; flipped on engine/UI threads, read in process().
        self._recording = False

        # All disk I/O and the wave handle live on the writer thread.
        self._file = None                # writer thread only
        self._frames_written = 0         # writer thread only
        self._write_queue = queue.Queue(maxsize=self.QUEUE_CAPACITY)
        self._writer_thread = None

        # Pre-allocated per-block conversion buffers (RT-safe, no allocs)
        self._block_f32 = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)  # interleaved
        self._block_i16 = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.int16)

    # ------------------------------------------------------------------
    # Writer thread — owns the wave file handle
    # ------------------------------------------------------------------

    def _writer_loop(self):
        while True:
            kind, payload = self._write_queue.get()
            if kind == "open":
                self._writer_open(payload)
            elif kind == "frames":
                if self._file is not None:
                    try:
                        self._file.writeframes(payload)
                        self._frames_written += BLOCK_SIZE
                    except Exception as e:
                        logger.error(f"Recorder write failed: {e}")
                        self._writer_close()
                        self._recording = False
                        if "record" in self.params:
                            self.params["record"].set(False)
            elif kind == "close":
                self._writer_close()
                break

    def _writer_open(self, filename):
        self._writer_close()  # safety: never hold two handles
        try:
            self._file = wave.open(filename, "wb")
            self._file.setnchannels(CHANNELS)
            self._file.setsampwidth(2)  # 16-bit
            self._file.setframerate(SAMPLE_RATE)
            self._frames_written = 0
            logger.info(f"Recorder: Opened {filename}")
        except Exception as e:
            logger.error(f"Recorder open failed: {e}")
            self._file = None
            self._recording = False

    def _writer_close(self):
        if self._file is not None:
            try:
                self._file.close()
            finally:
                logger.info(f"Recorder: Saved {self._frames_written} frames")
                self._file = None

    # ------------------------------------------------------------------
    # Control paths (engine thread when running, UI thread when stopped)
    # ------------------------------------------------------------------

    def _engine_running(self):
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None)
        return bool(engine and engine.running)

    def _start_recording(self):
        filename = self.params["filename"].get_staging_safe()
        if not filename:
            logger.warning("Recorder: no filename set, ignoring record start")
            return
        # Queue first so the thread finds the open command immediately.
        self._write_queue.put(("open", filename))
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name=f"anode-recorder-{self.id[:8]}"
        )
        self._writer_thread.start()
        self._recording = True

    def _stop_recording(self):
        was_recording = self._recording
        self._recording = False
        if not was_recording:
            return
        self._write_queue.put(("close", None))
        writer = self._writer_thread
        self._writer_thread = None
        # Deterministic flush only when no real-time loop depends on us;
        # while the engine runs we never block here.
        if writer is not None and not self._engine_running():
            writer.join(timeout=2.0)

    def on_ui_param_change(self, param_name):
        if param_name == "record":
            if self.params["record"].get_staging_safe():
                if not self._recording:
                    self._start_recording()
            else:
                self._stop_recording()

    def process(self):
        tensor = self.inp.get_tensor()
        if self._recording:
            # Convert into pre-allocated buffers; only the final byte string
            # (handed to the writer thread) is newly allocated.
            np.copyto(self._block_f32[:, 0], tensor[0])
            np.copyto(self._block_f32[:, 1], tensor[1] if tensor.shape[0] > 1 else tensor[0])
            np.clip(self._block_f32, -1.0, 1.0, out=self._block_f32)
            np.multiply(self._block_f32, 32767, out=self._block_i16, casting="unsafe")
            payload = self._block_i16.tobytes()
            try:
                self._write_queue.put_nowait(("frames", payload))
            except queue.Full:
                pass  # drop this block rather than stall the audio thread

    def stop(self):
        self._stop_recording()

    def remove(self):
        self._stop_recording()
