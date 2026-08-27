import logging
import queue
import threading
import wave
import ctypes

import torch
import numpy as np
from base import Node, IClockProvider, BLOCK_SIZE, DTYPE, SAMPLE_RATE, CHANNELS, SPSCRingBuffer

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

        # Lock-free SPSC ring buffer for block indices (audio thread -> writer thread)
        self._index_queue = SPSCRingBuffer(capacity=self.QUEUE_CAPACITY)

        # Pre-allocated block pool: [QUEUE_CAPACITY, BLOCK_SIZE, CHANNELS] int16
        self._block_pool = np.zeros((self.QUEUE_CAPACITY, BLOCK_SIZE, CHANNELS), dtype=np.int16)
        self._write_index = 0  # audio thread writes to this pool index

        # Pre-allocated float32 temp buffer for clipping (RT-safe, no allocs)
        self._temp_f32 = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)

        # Writer thread management
        self._writer_thread = None
        self._shutdown_event = threading.Event()

    # ------------------------------------------------------------------
    # Writer thread — owns the wave file handle
    # ------------------------------------------------------------------

    def _writer_loop(self):
        while not self._shutdown_event.is_set():
            # Non-blocking pop with small sleep to avoid busy-wait
            index, ok = self._index_queue.try_pop()
            if not ok:
                # Queue empty, brief sleep
                self._shutdown_event.wait(timeout=0.001)
                continue

            kind = index[0] if isinstance(index, tuple) else index
            if kind == "open":
                filename = index[1]
                self._writer_open(filename)
            elif kind == "frames":
                block_idx = index[1]
                if self._file is not None:
                    try:
                        # Write pre-converted int16 block directly from pool
                        self._file.writeframes(self._block_pool[block_idx].tobytes())
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

        # Reset writer thread state
        self._shutdown_event.clear()
        self._write_index = 0
        # Clear index queue
        while self._index_queue.try_pop()[1]:
            pass

        # Queue open command
        self._index_queue.try_push(("open", filename))

        # Start writer thread (created off RT path via NRT or here if stopped)
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

        # Signal writer thread to close
        self._shutdown_event.set()
        self._index_queue.try_push(("close", None))

        writer = self._writer_thread
        self._writer_thread = None

        # Deterministic flush only when no real-time loop depends on us
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
            # Convert into pre-allocated pool slot; NO .tobytes() on audio thread
            block_idx = self._write_index
            pool_slot = self._block_pool[block_idx]

            # Convert float32 [-1, 1] -> int16 directly into pool slot
            # Use float32 temp buffer for clipping, then convert to int16 in pool
            # Channel 0
            np.copyto(self._temp_f32[:, 0], tensor[0].cpu().numpy())
            np.clip(self._temp_f32[:, 0], -1.0, 1.0, out=self._temp_f32[:, 0])
            np.multiply(self._temp_f32[:, 0], 32767, out=self._temp_f32[:, 0], casting="unsafe")
            pool_slot[:, 0] = self._temp_f32[:, 0].astype(np.int16, copy=False)

            # Channel 1 (or duplicate channel 0 if mono)
            if tensor.shape[0] > 1:
                np.copyto(self._temp_f32[:, 1], tensor[1].cpu().numpy())
                np.clip(self._temp_f32[:, 1], -1.0, 1.0, out=self._temp_f32[:, 1])
                np.multiply(self._temp_f32[:, 1], 32767, out=self._temp_f32[:, 1], casting="unsafe")
                pool_slot[:, 1] = self._temp_f32[:, 1].astype(np.int16, copy=False)
            else:
                pool_slot[:, 1] = pool_slot[:, 0]

            # Try to push block index to writer thread
            if not self._index_queue.try_push(("frames", block_idx)):
                # Pool exhausted - drop frame (per spec)
                pass
            else:
                # Advance to next pool slot (ring)
                self._write_index = (self._write_index + 1) % self.QUEUE_CAPACITY

    def stop(self):
        self._stop_recording()

    def remove(self):
        self._stop_recording()
