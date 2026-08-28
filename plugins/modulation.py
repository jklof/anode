"""
Modulation and Control Voltage (CV) generator nodes.

ADSRNode (Utilities): Sample-accurate multi-stage envelope generator.
LFONode (Sources): Multi-waveform low-frequency oscillator with sync.
GateButtonNode (Sources): Interactive manual gate trigger with custom Qt UI.

Real-time notes:
- All steady-state buffers and masks are pre-allocated in ``__init__``; the
  audio processing path performs zero heap allocation.
- CV signals are mono (``channels=1``).
"""

import math

import numpy as np
import torch

from base import Node, BLOCK_SIZE, SAMPLE_RATE, DTYPE


# ==============================================================================
# ADSRNode — sample-accurate multi-stage envelope generator (Utilities)
# ==============================================================================
class ADSRNode(Node):
    category = "Utilities"
    label = "ADSR Envelope"
    description = (
        "Multi-stage attack-decay-sustain-release envelope generator driven by "
        "gate inputs. Computes sample-accurate exponential release and linear "
        "attack/decay transitions with zero runtime allocation."
    )

    STATE_IDLE = 0
    STATE_ATTACK = 1
    STATE_DECAY = 2
    STATE_SUSTAIN = 3
    STATE_RELEASE = 4

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("gate",
                       help="Gate signal (> 0 starts Attack, <= 0 triggers Release).")
        self.add_input("attack_in", "attack", help="Attack duration in seconds (bound to 'attack').")
        self.add_input("decay_in", "decay", help="Decay duration in seconds (bound to 'decay').")
        self.add_input("sustain_in", "sustain", help="Sustain level in [0, 1] (bound to 'sustain').")
        self.add_input("release_in", "release", help="Release duration in seconds (bound to 'release').")
        self.out = self.add_output("out", channels=1,
                                   help="Mono CV envelope in [0.0, 1.0].")

        self.add_float_param("attack", 0.01, 0.001, 10.0, unit="s",
                             help="Attack duration to reach peak 1.0.")
        self.add_float_param("decay", 0.1, 0.001, 10.0, unit="s",
                             help="Decay duration from 1.0 to sustain level.")
        self.add_float_param("sustain", 0.7, 0.0, 1.0, unit="",
                             help="Sustain level held while gate is high.")
        self.add_float_param("release", 0.3, 0.001, 10.0, unit="s",
                             help="Release duration to drop from current level to 0.0.")

        # Numpy view directly backed by the mono output buffer: the scalar loop
        # below writes straight into the torch output, so we never allocate a
        # temporary tensor per block (and never need a per-sample copy).
        self._out_np = self.out.buffer[0].numpy()
        self._level = 0.0
        self._state = self.STATE_IDLE
        self._prev_gate = 0.0
        self._level_at_release = 0.0

    def start(self):
        self._state = self.STATE_IDLE
        self._level = 0.0
        self._prev_gate = 0.0
        self._level_at_release = 0.0
        self._out_np.fill(0.0)

    def process(self):
        gate = self.inputs["gate"].get_tensor()
        gate_arr = gate[0].numpy()

        att_s = max(0.001, float(self.inputs["attack_in"].get_tensor()[0, 0].item()))
        dec_s = max(0.001, float(self.inputs["decay_in"].get_tensor()[0, 0].item()))
        sus = float(np.clip(
            float(self.inputs["sustain_in"].get_tensor()[0, 0].item()), 0.0, 1.0))
        rel_s = max(0.001, float(self.inputs["release_in"].get_tensor()[0, 0].item()))

        att_step = 1.0 / (att_s * SAMPLE_RATE)
        dec_step = (1.0 - sus) / (dec_s * SAMPLE_RATE) if sus < 1.0 else 0.0
        rel_coeff = math.exp(-1.0 / (rel_s * SAMPLE_RATE))

        level = self._level
        state = self._state
        prev_gate = self._prev_gate
        out = self._out_np

        for i in range(BLOCK_SIZE):
            g = gate_arr[i]
            if g > 0.0 and prev_gate <= 0.0:
                # Rising gate edge: start Attack from the current level.
                state = self.STATE_ATTACK
            elif g <= 0.0 and prev_gate > 0.0:
                # Falling gate edge: begin Release from the current level.
                self._level_at_release = level
                state = self.STATE_RELEASE

            if state == self.STATE_ATTACK:
                level += att_step
                if level >= 1.0:
                    level = 1.0
                    state = self.STATE_DECAY
            elif state == self.STATE_DECAY:
                level -= dec_step
                if level <= sus:
                    level = sus
                    state = self.STATE_SUSTAIN
            elif state == self.STATE_SUSTAIN:
                level = sus
            elif state == self.STATE_RELEASE:
                level *= rel_coeff
                if level < 1e-5:
                    level = 0.0
                    state = self.STATE_IDLE
            else:  # STATE_IDLE
                level = 0.0

            prev_gate = g
            out[i] = level

        self._level = level
        self._state = state
        self._prev_gate = prev_gate


# ==============================================================================
# LFONode — multi-waveform low-frequency oscillator with sync (Sources)
# ==============================================================================
class LFONode(Node):
    category = "Sources"
    label = "LFO"
    description = (
        "Multi-waveform low-frequency oscillator providing simultaneous Sine, "
        "Triangle, Sawtooth, and Square outputs with sync reset and "
        "unipolar/bipolar modes."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("freq_in", "freq",
                       help="Frequency modulation in Hz. Unconnected: uses 'freq' parameter.")
        self.add_input("sync",
                       help="Hard sync input; rising edge (> 0) resets phase to 0.0.")

        self.out_sine = self.add_output("sine", channels=1, help="Mono Sine LFO output.")
        self.out_triangle = self.add_output("triangle", channels=1, help="Mono Triangle LFO output.")
        self.out_saw = self.add_output("saw", channels=1, help="Mono Sawtooth LFO output.")
        self.out_square = self.add_output("square", channels=1, help="Mono Square LFO output.")

        self.add_float_param("freq", 1.0, 0.01, 50.0, unit="Hz",
                             help="Base LFO frequency (can be modulated higher via freq_in).")
        self.add_bool_param("bipolar", True,
                            help="When true, output range is [-1, +1]; when false, unipolar [0, 1].")

        self.phase = 0.0
        self._prev_sync = 0.0

        # Pre-allocated scratch buffers (zero runtime allocation).
        self._dt = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._phase_buf = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._temp = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._mask = torch.zeros(BLOCK_SIZE, dtype=torch.bool)
        self._edge_mask = torch.zeros(BLOCK_SIZE, dtype=torch.bool)

    def start(self):
        self.phase = 0.0
        self._prev_sync = 0.0

    def process(self):
        freq_sig = self.inputs["freq_in"].get_tensor()[0]

        torch.mul(freq_sig, 1.0 / SAMPLE_RATE, out=self._dt)
        self._dt.clamp_(0.0, 0.49)

        # --- Sync edge detection (only if the sync slot is connected) ---
        if self.inputs["sync"].connected_outputs:
            sync_tensor = self.inputs["sync"].get_tensor()[0]
            torch.gt(sync_tensor, 0.0, out=self._mask)
            self._edge_mask[0] = bool(self._mask[0] and self._prev_sync <= 0.0)
            self._edge_mask[1:] = self._mask[1:] & (~self._mask[:-1])
            self._prev_sync = float(sync_tensor[-1].item())

            if bool(self._edge_mask.any().item()):
                # Hard sync: reset phase to 0.0 at the first rising edge index k.
                k = int(torch.argmax(self._edge_mask.to(torch.int8)).item())
                self._phase_buf[:k].copy_(self._dt[:k]).cumsum_(0).add_(self.phase).remainder_(1.0)
                self._phase_buf[k:].copy_(self._dt[k:]).cumsum_(0).remainder_(1.0)
            else:
                self._phase_buf.copy_(self._dt).cumsum_(0).add_(self.phase).remainder_(1.0)
        else:
            self._phase_buf.copy_(self._dt).cumsum_(0).add_(self.phase).remainder_(1.0)

        self.phase = float(self._phase_buf[-1].item())

        bipolar = bool(self.params["bipolar"].value)

        # Sine
        torch.mul(self._phase_buf, 2.0 * math.pi, out=self._temp)
        torch.sin(self._temp, out=self.out_sine.buffer[0])

        # Triangle: |2p - 1| * 2 - 1
        torch.mul(self._phase_buf, 2.0, out=self._temp)
        self._temp.sub_(1.0).abs_().mul_(2.0).sub_(1.0)
        self.out_triangle.buffer[0].copy_(self._temp)

        # Sawtooth: -2p + 1
        torch.mul(self._phase_buf, -2.0, out=self.out_saw.buffer[0]).add_(1.0)

        # Square: +1 for phase < 0.5, -1 otherwise
        torch.lt(self._phase_buf, 0.5, out=self._mask)
        self.out_square.buffer[0].copy_(self._mask).mul_(2.0).sub_(1.0)

        if not bipolar:
            for name in ("sine", "triangle", "saw", "square"):
                self.outputs[name].buffer[0].add_(1.0).mul_(0.5)
# ==============================================================================
# GateButtonNode — interactive manual gate trigger (Sources)
# ==============================================================================
class GateButtonNode(Node):
    category = "Sources"
    label = "Gate Button"
    description = (
        "Interactive manual gate trigger for pulsing or latching control signals."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.out = self.add_output("out", channels=1,
                                   help="Mono gate signal (1.0 active, 0.0 inactive).")

        self.add_bool_param("state", False,
                            help="Current gate state (True = 1.0, False = 0.0).")
        self.add_menu_param("mode", ["Momentary", "Toggle"], 0,
                            help="Button behavior: momentary hold or toggle latch.")

    def process(self):
        val = 1.0 if self.params["state"].value else 0.0
        self.out.buffer[0].fill_(val)


# ==============================================================================
# Qt custom UI for GateButtonNode
# ==============================================================================
try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
    from PySide6.QtCore import QTimer

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


if GUI_AVAILABLE:

    class GateButtonWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "GateButtonNode"

        def __init__(self, proxy):
            super().__init__()
            self.proxy = proxy
            self.setMinimumSize(120, 50)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            self.button = QPushButton("GATE")
            layout.addWidget(self.button)

            self._release_timer = QTimer(self)
            self._release_timer.setSingleShot(True)
            self._release_timer.timeout.connect(self._on_delayed_release)

            # Local toggle latch. We intentionally do NOT derive the next
            # toggle value from proxy.node_item.params: that dict is the UI-side
            # snapshot and only refreshes asynchronously. Deriving from it would
            # make the second click in toggle mode read a stale value and re-set
            # the same state, leaving the button "stuck on". The local latch is
            # authoritative for clicks and is reconciled with authoritative
            # params via update_from_params().
            self._toggle_state = bool(self._get_state())
            self._set_button_visual(self._toggle_state)

            # Rely on Qt's own button signals instead of overriding
            # mousePressEvent/mouseReleaseEvent on the container: QPushButton
            # grabs the mouse on press and ALWAYS sees the release, so a click
            # can never be "stuck on" by a swallowed release event. (This is the
            # same reliable pattern used by the DialNode widget.)
            self.button.pressed.connect(self._on_pressed)
            self.button.released.connect(self._on_released)

        def _get_mode(self):
            node_item = self.proxy.node_item
            if node_item is None or "mode" not in node_item.params:
                return 0
            return int(node_item.params["mode"]["value"])

        def _get_state(self):
            node_item = self.proxy.node_item
            if node_item is None or "state" not in node_item.params:
                return False
            return node_item.params["state"]["value"]

        def _on_pressed(self):
            if self._get_mode() == 0:  # Momentary: engage on press.
                self._release_timer.stop()
                self.proxy.set_parameter("state", True)
                self._set_button_visual(True)
            # In Toggle mode we act on release (a press is not a click).

        def _on_released(self):
            if self._get_mode() == 0:
                # Momentary: ensure >= 50 ms pulse so the ~30 ms debounce loop
                # flushes the True before we release.
                self._release_timer.start(50)
            else:  # Toggle: flip the local latch on every completed click.
                self._toggle_state = not self._toggle_state
                self.proxy.set_parameter("state", self._toggle_state)
                self._set_button_visual(self._toggle_state)

        def _on_delayed_release(self):
            self.proxy.set_parameter("state", False)
            self._set_button_visual(False)

        def _set_button_visual(self, active: bool):
            if active:
                self.button.setStyleSheet(
                    "background-color: #00ccff; color: black; font-weight: bold;"
                )
            else:
                self.button.setStyleSheet("")

        def update_from_params(self, params):
            if "state" in params:
                # Reconcile the local toggle latch with the authoritative value
                # pushed down by the engine (load/undo/other control paths).
                self._toggle_state = bool(params["state"])
                self._set_button_visual(self._toggle_state)