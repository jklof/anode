import torch
import numpy as np
from base import Node, BLOCK_SIZE, DTYPE, SAMPLE_RATE, CHANNELS


class SineOscillator(Node):
    category = "Sources"
    label = "Sine Oscillator"

    def __init__(self, name=""):
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
        self._phase_buffer.remainder_(self.two_pi)
        torch.sin(self._phase_buffer, out=self.out_sig.buffer[0])
        self.out_sig.buffer[0].mul_(amp_sig)
        self.phase = self._phase_buffer[-1].item() % self.two_pi


class StereoToMono(Node):
    category = "Utilities"
    label = "Stereo to Mono"

    def __init__(self, name=""):
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
    category = "Utilities"
    label = "Mono to Stereo"

    def __init__(self, name=""):
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
    category = "Utilities"
    label = "Gain"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_float_param("vol", 1.0, 0.0, 2.0)
        self.inp = self.add_input("in")
        self.gain_mod = self.add_input("mod", "vol")
        self.out = self.add_output("out")

    def process(self):
        t = self.inp.get_tensor()
        mod = self.gain_mod.get_tensor()
        torch.mul(t, mod, out=self.out.buffer)


class ChannelSplitter(Node):
    category = "Utilities"
    label = "Channel Splitter"

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in")
        # Create outputs and store in a list for loop-based processing
        self.outputs_list = [self.add_output("left", channels=1), self.add_output("right", channels=1)]

    def process(self):
        t = self.inp.get_tensor()
        in_channels = t.shape[0]

        # Loop-based logic (Future-proof for N channels)
        for i, out_slot in enumerate(self.outputs_list):
            out_slot.buffer.zero_()
            if i < in_channels:
                # Copy input channel 'i' to output buffer channel 0 (since output is mono)
                out_slot.buffer[0].copy_(t[i])


class ChannelJoiner(Node):
    category = "Utilities"
    label = "Channel Joiner"

    def __init__(self, name=""):
        super().__init__(name)
        # Create inputs and store in a list
        self.inputs_list = [self.add_input("left"), self.add_input("right")]
        self.out = self.add_output("out", channels=2)

    def process(self):
        out_buffer = self.out.buffer
        out_buffer.zero_()
        max_out_channels = out_buffer.shape[0]

        # Loop-based logic
        for i, inp_slot in enumerate(self.inputs_list):
            if i < max_out_channels:
                sig = inp_slot.get_tensor()
                # Copy 1st channel of mono source to ith channel of output
                if sig.shape[0] > 0:
                    out_buffer[i].copy_(sig[0])


# ==============================================================================
# Dial Node (Constant Signal Generator)
# ==============================================================================


class DialNode(Node):
    category = "Sources"
    label = "Dial"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_float_param("value", 0.5, 0.0, 1.0)
        self.out = self.add_output("out", channels=CHANNELS)

    def process(self):
        val = self.params["value"].value
        self.out.buffer.fill_(val)


try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDial
    from PySide6.QtCore import Qt, QSignalBlocker

    class DialNodeWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "DialNode"

        def __init__(self, node_proxy):
            super().__init__()
            self.proxy = node_proxy
            self.setMinimumSize(100, 120)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            self.label = QLabel("Value: 0.50")
            self.label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.label)

            self.dial = QDial()
            self.dial.setRange(0, 1000)
            self.dial.setNotchesVisible(True)
            self.dial.setWrapping(False)

            # Set initial value
            init_val = self.proxy.node_item.params["value"]["value"]
            self.dial.setValue(int(init_val * 1000))
            self.label.setText(f"Value: {init_val:.2f}")

            self.dial.valueChanged.connect(self.on_dial_changed)
            layout.addWidget(self.dial)

        def on_dial_changed(self, val):
            f_val = val / 1000.0
            self.proxy.set_parameter("value", f_val)
            self.label.setText(f"Value: {f_val:.2f}")

        def update_from_params(self, params):
            if "value" in params:
                val = params["value"]
                self.label.setText(f"Value: {val:.2f}")
                if not self.dial.isSliderDown():
                    with QSignalBlocker(self.dial):
                        self.dial.setValue(int(val * 1000))

except ImportError:
    pass
