import torch
import numpy as np
from base import Node, BLOCK_SIZE, DTYPE, SAMPLE_RATE, CHANNELS


class SineOscillator(Node):
    category = "Sources"
    label = "Sine Oscillator"
    description = (
        "Pure sine wave generator: phase accumulates at the incoming frequency, "
        "wrapped to [0, 2π), and sine() is evaluated per sample. The frequency and "
        "amplitude inputs are parameter-bound modulation inputs, so an unconnected "
        "slot falls back to the constant parameter value. Mono output."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_float_param("freq", 440.0, 20.0, 20000.0, unit="Hz",
                             help="Oscillator frequency used when no modulation signal is connected.")
        self.add_float_param("amp", 0.5, 0.0, 1.0,
                             help="Output peak amplitude used when no modulation signal is connected.")
        self.in_freq = self.add_input("freq_in", "freq",
                                      help="Audio-rate frequency modulation input (Hz). Unconnected: uses 'freq' parameter.")
        self.in_amp = self.add_input("amp_in", "amp",
                                     help="Audio-rate amplitude modulation input (linear gain). Unconnected: uses 'amp' parameter.")
        self.out_sig = self.add_output("signal", channels=1,
                                       help="Mono sine output in [-amp, +amp].")
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
    description = (
        "Downmixes a stereo input to mono by averaging the left and right channels "
        "((L + R) / 2). A mono input is passed through unchanged. No parameters; "
        "pure stateless math with zero latency."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in", help="Stereo (or mono) signal to downmix.")
        self.out = self.add_output("out", channels=1, help="Mono average of the input channels.")

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
    description = (
        "Upmixes mono (or stereo) input to stereo with a linear pan law: "
        "L = in * (1 - pan) / 2 and R = in * (1 + pan) / 2. At pan = 0 both "
        "channels receive equal, attenuated copies of the input."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_float_param("pan", 0.0, -1.0, 1.0,
                             help="Pan position: -1 hard left, 0 center, +1 hard right.")
        self.inp = self.add_input("in", help="Mono signal to pan into stereo.")
        self.out = self.add_output("out", channels=2, help="Panned stereo output.")

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
    description = (
        "Simple linear gain stage: output = input * gain. The 'mod' input is "
        "parameter-bound to the volume parameter, providing audio-rate amplitude "
        "modulation when connected (unconnected slots use the constant parameter "
        "value). Zero latency, no state."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_float_param("vol", 1.0, 0.0, 2.0, unit="x",
                             help="Linear gain multiplier used when no modulation signal is connected.")
        self.inp = self.add_input("in", help="Signal to amplify.")
        self.gain_mod = self.add_input("mod", "vol",
                                       help="Audio-rate gain modulation input (linear multiplier). Unconnected: uses 'vol' parameter.")
        self.out = self.add_output("out", help="Amplified signal, same channel count as the input.")

    def process(self):
        t = self.inp.get_tensor()
        mod = self.gain_mod.get_tensor()
        # In-place ops only: functional torch.mul(..., out=buf) would RESIZE
        # buf to the broadcast shape (e.g. (1, BLOCK) for mono inputs),
        # shrinking the pre-allocated stereo output buffer. copy_ broadcasts
        # without resizing, and mul_ broadcasts its operand in place.
        self.out.buffer.copy_(t)
        self.out.buffer.mul_(mod)


class ChannelSplitter(Node):
    category = "Utilities"
    label = "Channel Splitter"
    description = (
        "Splits the channels of an input signal into separate mono outputs: "
        "channel 0 goes to 'left', channel 1 to 'right'. Missing input channels "
        "produce silence on the corresponding output. Zero latency, no state."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in", help="Stereo (or wider) signal to split.")
        # Create outputs and store in a list for loop-based processing
        self.outputs_list = [
            self.add_output("left", channels=1, help="Mono copy of input channel 0."),
            self.add_output("right", channels=1, help="Mono copy of input channel 1 (silence if mono input)."),
        ]

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
    description = (
        "Joins two mono signals into a single stereo signal: 'left' becomes "
        "channel 0 and 'right' becomes channel 1. Zero latency, no state."
    )

    def __init__(self, name=""):
        super().__init__(name)
        # Create inputs and store in a list
        self.inputs_list = [
            self.add_input("left", help="Mono signal placed on the left channel."),
            self.add_input("right", help="Mono signal placed on the right channel."),
        ]
        self.out = self.add_output("out", channels=2, help="Stereo output combining both inputs.")

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
    description = (
        "Constant CV source: outputs the 'value' parameter as a static signal on "
        "every channel. Intended as a modulation source for parameter-bound "
        "inputs of other nodes. Includes a rotary-dial custom UI."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_float_param("value", 0.5, 0.0, 1.0,
                             help="Constant output value produced on every channel.")
        self.out = self.add_output("out", channels=CHANNELS,
                                   help="Stereo constant signal at the dial value.")

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
