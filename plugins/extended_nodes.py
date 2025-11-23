import torch
import wave
import numpy as np
from base import Node, IClockProvider, BLOCK_SIZE, DTYPE, SAMPLE_RATE, CHANNELS


class Note(Node):
    def __init__(self, name="Note"):
        super().__init__(name)
        self.add_string_param("text", "Hello World")

    def process(self):
        pass


class Noise(Node):
    def __init__(self, name="Noise"):
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
    def __init__(self, name="Sel"):
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


class FileRecorder(Node, IClockProvider):
    def __init__(self, name="Recorder"):
        Node.__init__(self, name)
        IClockProvider.__init__(self)
        self.add_string_param("filename", "output.wav")
        self.add_bool_param("record", False)
        self.inp = self.add_input("in")

        self._file = None
        self._recording = False
        self._frames_written = 0

    def start_clock(self):
        pass

    def stop_clock(self):
        pass

    def wait_for_sync(self):
        # Offline/Fast mode: return immediately
        pass

    def _open_file(self, filename):
        self._close_file()
        try:
            self._file = wave.open(filename, "wb")
            self._file.setnchannels(CHANNELS)
            self._file.setsampwidth(2)  # 16-bit
            self._file.setframerate(SAMPLE_RATE)
            print(f"Recorder: Opened {filename}")
        except Exception as e:
            print(f"Recorder Error: {e}")
            self._file = None

    def _close_file(self):
        if self._file:
            self._file.close()
            self._file = None
            print(f"Recorder: Saved {self._frames_written} frames")

    def on_ui_param_change(self, param_name):
        if param_name == "record":
            should_record = self.params["record"].value
            if should_record and not self._recording:
                fn = self.params["filename"].value
                self._open_file(fn)
                self._recording = True
                self._frames_written = 0
            elif not should_record and self._recording:
                self._recording = False
                self._close_file()

    def process(self):
        tensor = self.inp.get_tensor()
        if self._recording and self._file:
            # FIX: Ensure we have stereo data for the stereo WAV file
            if tensor.shape[0] == 1:
                # Expand (1, N) -> (2, N) for the file writer
                export_tensor = tensor.expand(2, -1)
            else:
                export_tensor = tensor

            data = export_tensor.t().cpu().numpy()
            data = np.clip(data, -1.0, 1.0)
            int_data = (data * 32767).astype(np.int16)
            try:
                self._file.writeframes(int_data.tobytes())
                self._frames_written += BLOCK_SIZE
            except Exception as e:
                print(f"Write Error: {e}")
                self._recording = False
                self.params["record"].value = False
                self._close_file()

    def stop(self):
        if self._recording:
            self._recording = False
            self._close_file()
