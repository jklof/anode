import torch
import numpy as np
import sounddevice as sd
import queue
import time
from core import Node, IClockProvider, register_node, BLOCK_SIZE, SAMPLE_RATE, DTYPE

@register_node
class SineOscillator(Node):
    def __init__(self, name="Sine"):
        super().__init__(name)
        self.add_float_param('freq', 440.0, 20.0, 20000.0)
        self.add_float_param('amp', 0.5, 0.0, 1.0)
        self.in_freq = self.add_input("freq_in", "freq")
        self.in_amp = self.add_input("amp_in", "amp")
        self.out_sig = self.add_output("signal")
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
        self.out_sig.buffer[1].zero_()
        self.phase = self._phase_buffer[-1].item() % self.two_pi

@register_node
class MonoToStereo(Node):
    def __init__(self, name="MonoToStereo"):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out")
    def process(self):
        t = self.inp.get_tensor()
        self.out.buffer[0].copy_(t[0])
        self.out.buffer[1].copy_(t[0])

@register_node
class Gain(Node):
    def __init__(self, name="Gain"):
        super().__init__(name)
        self.add_float_param('vol', 1.0, 0.0, 2.0)
        self.inp = self.add_input("in")
        self.gain_mod = self.add_input("mod", "vol")
        self.out = self.add_output("out")
    def process(self):
        t = self.inp.get_tensor()
        mod = self.gain_mod.get_tensor()
        torch.mul(t, mod, out=self.out.buffer)

@register_node
class AudioOutput(Node, IClockProvider):
    def __init__(self, name="Speakers", device=None):
        Node.__init__(self, name)
        IClockProvider.__init__(self)
        self.in_audio = self.add_input("audio_in")
        self.device = device
        self.queue = queue.Queue(maxsize=4) 
        self.stream = None
        self._active = False

    def start_clock(self): pass 
    def stop_clock(self): pass

    def wait_for_sync(self):
        if self.is_master and self._active:
            while self.queue.full() and self._active:
                time.sleep(0.002)

    def start(self):
        self._active = True
        try:
            self.stream = sd.OutputStream(
                device=self.device, channels=2, blocksize=BLOCK_SIZE, 
                samplerate=SAMPLE_RATE, callback=self._callback
            )
            self.stream.start()
        except Exception as e:
            print(f"Audio Error: {e}")

    def stop(self):
        self._active = False
        if self.stream:
            self.stream.abort(); self.stream.close(); self.stream = None
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except: break

    def _callback(self, outdata, frames, time, status):
        if not self._active:
            outdata.fill(0); return
        if status: print(f"{self.name}: {status}")
        try:
            data = self.queue.get_nowait()
            outdata[:] = data.t().numpy()
        except queue.Empty:
            if self.is_master and self._active: print("U", end="", flush=True)
            outdata.fill(0)

    def process(self):
        audio_data = self.in_audio.get_tensor()
        if self.is_master:
            self.queue.put(audio_data.clone(), block=True)
        else:
            try: self.queue.put_nowait(audio_data.clone())
            except queue.Full: pass