import torch
import numpy as np
import sounddevice as sd
import time
from base import Node, IClockProvider, BLOCK_SIZE, DTYPE, SAMPLE_RATE, CHANNELS


class RingBuffer:
    def __init__(self, buffer_size_blocks=8, block_size=512, channels=2):
        self.buffer_size_blocks = buffer_size_blocks
        self.block_size = block_size
        self.channels = channels
        self.capacity_blocks = buffer_size_blocks
        self.total_frames = buffer_size_blocks * block_size
        self.storage = np.zeros((self.total_frames, channels), dtype=np.float32)
        self.write_count = 0
        self.read_count = 0

    def write(self, tensor_data):
        if (self.write_count - self.read_count) >= self.capacity_blocks:
            return False
        # Convert tensor to numpy
        np_data = tensor_data.detach().cpu().numpy()
        # Calculate start index
        start_idx = (self.write_count % self.capacity_blocks) * self.block_size
        # Ensure we don't write beyond capacity
        if np_data.ndim == 2:
            # Handle stereo or multi-channel
            np_data = np_data[: self.channels].T  # Transpose to (frames, channels)
        else:
            # If mono, duplicate to stereo
            mono = np_data[0].reshape(-1, 1)
            np_data = np.tile(mono, (1, self.channels))
        self.storage[start_idx : start_idx + self.block_size, :] = np_data
        self.write_count += 1
        return True

    def read(self, outdata):
        if (self.write_count - self.read_count) == 0:
            return False
        # Calculate start index
        start_idx = (self.read_count % self.capacity_blocks) * self.block_size
        # Copy data to outdata
        block_data = self.storage[start_idx : start_idx + self.block_size, :]
        outdata[:] = block_data
        self.read_count += 1
        return True


class SineOscillator(Node):
    def __init__(self, name="Sine"):
        super().__init__(name)
        self.add_float_param("freq", 440.0, 20.0, 20000.0)
        self.add_float_param("amp", 0.5, 0.0, 1.0)
        self.in_freq = self.add_input("freq_in", "freq")
        self.in_amp = self.add_input("amp_in", "amp")
        self.out_sig = self.add_output("signal", channels=1)
        self.two_pi = 2 * np.pi
        self.sr_recip = 1.0 / SAMPLE_RATE
        self.phase = 0.0
        self._phase_buffer = torch.zeros(BLOCK_SIZE, dtype=DTYPE)

    def process(self):
        freq_sig = self.in_freq.get_tensor()[0]
        amp_sig = self.in_amp.get_tensor()[0]
        torch.mul(freq_sig, self.two_pi * self.sr_recip, out=self._phase_buffer)
        self._phase_buffer.cumsum_(dim=0)
        self._phase_buffer.add_(self.phase)
        torch.sin(self._phase_buffer, out=self.out_sig.buffer[0])
        self.out_sig.buffer[0].mul_(amp_sig)
        self.phase = self._phase_buffer[-1].item() % self.two_pi


class StereoToMono(Node):
    def __init__(self, name="To Mono"):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=1)

    def process(self):
        t = self.inp.get_tensor()

        # FIX: Ensure output buffer is clean.
        # Since we only write to buffer[0], buffer[1] (if it exists) would retain stale data.
        self.out.buffer.zero_()

        if t.shape[0] == 1:
            self.out.buffer[0].copy_(t[0])
        else:
            torch.add(t[0], t[1], out=self.out.buffer[0])
            self.out.buffer[0].mul_(0.5)


class MonoToStereo(Node):
    def __init__(self, name="To Stereo"):
        super().__init__(name)
        self.add_float_param("pan", 0.0, -1.0, 1.0)
        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=2)

    def process(self):
        t = self.inp.get_tensor()
        pan = self.params["pan"].value
        left_gain = (1 - pan) / 2
        right_gain = (1 + pan) / 2
        torch.mul(t[0], left_gain, out=self.out.buffer[0])
        torch.mul(t[0], right_gain, out=self.out.buffer[1])


class Gain(Node):
    def __init__(self, name="Gain"):
        super().__init__(name)
        self.add_float_param("vol", 1.0, 0.0, 2.0)
        self.inp = self.add_input("in")
        self.gain_mod = self.add_input("mod", "vol")
        self.out = self.add_output("out")

    def process(self):
        t = self.inp.get_tensor()
        mod = self.gain_mod.get_tensor()
        torch.mul(t, mod, out=self.out.buffer)


class AudioOutput(Node, IClockProvider):
    def __init__(self, name="Speakers", device=None):
        Node.__init__(self, name)
        IClockProvider.__init__(self)
        self.in_audio = self.add_input("audio_in")
        self.device = device
        self.ring_buffer = RingBuffer(buffer_size_blocks=8, block_size=BLOCK_SIZE, channels=CHANNELS)
        self.stream = None
        self._active = False

    def start_clock(self):
        pass

    def stop_clock(self):
        pass

    def wait_for_sync(self):
        if self.is_master and self._active:
            while (
                self.ring_buffer.write_count - self.ring_buffer.read_count
            ) >= self.ring_buffer.capacity_blocks and self._active:
                time.sleep(0.005)

    def start(self):
        self._active = True
        try:
            self.stream = sd.OutputStream(
                device=self.device, channels=2, blocksize=BLOCK_SIZE, samplerate=SAMPLE_RATE, callback=self._callback
            )
            self.stream.start()
        except Exception as e:
            print(f"Audio Error: {e}")

    def stop(self):
        self._active = False
        if self.stream:
            self.stream.abort()
            self.stream.close()
            self.stream = None
        # Reset ring buffer
        self.ring_buffer.write_count = 0
        self.ring_buffer.read_count = 0

    def _callback(self, outdata, frames, time, status):
        if not self._active:
            outdata.fill(0)
            return
        if not self.ring_buffer.read(outdata):
            outdata.fill(0)

    def process(self):
        audio_data = self.in_audio.get_tensor()
        start_time = time.time()
        timeout = 0.010  # 10ms timeout to prevent deadlocks
        while not self.ring_buffer.write(audio_data) and self._active:
            time.sleep(0.001)
            if time.time() - start_time > timeout:
                break  # Prevent deadlock
        # Reset counts on underrun/overrun handling might be added later if needed
