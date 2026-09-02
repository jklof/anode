"""
Hardware / virtual MIDI device input and output nodes.

Devices are opened on NRT pool threads (never the audio path) and lifecycle is
governed by an epoch counter so stale open requests are discarded. The audio
path only ever touches a lock-free SPSC ring buffer.

- MIDIInputNode: a background ``mido`` callback thread pushes ``(0, Message)``
  tuples into an SPSC queue; ``process()`` drains it into the MIDI output packet.
- MIDIOutputNode: ``process()`` pushes messages into a bounded writer queue; a
  dedicated writer thread sends them to the hardware port (drop-on-overflow).
"""

import logging
import threading
from base import Node, SPSCRingBuffer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    mido = None
    MIDO_AVAILABLE = False


class MIDIDeviceManager:
    @staticmethod
    def get_input_names():
        if not MIDO_AVAILABLE:
            return []
        try:
            return mido.get_input_names()
        except Exception as e:
            logger.error(f"Failed to query MIDI inputs: {e}")
            return []

    @staticmethod
    def get_output_names():
        if not MIDO_AVAILABLE:
            return []
        try:
            return mido.get_output_names()
        except Exception as e:
            logger.error(f"Failed to query MIDI outputs: {e}")
            return []


class MIDIInputNode(Node):
    category = "I/O"
    label = "MIDI Device Input"
    description = (
        "Receives real-time MIDI messages from a hardware port on a background "
        "thread, queuing them into a lock-free SPSC buffer. Port lifecycle runs via NRT."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.error_msg = None
        self.msg_out = self.add_midi_output("msg_out", help="Decoded MIDI stream from hardware port.")
        self.add_string_param("device_name", "", help="Target MIDI hardware input device name.")

        self._queue = SPSCRingBuffer(capacity=256)
        self._inport = None
        self._device_epoch = 0
        self._status = "Idle"

    def on_ui_param_change(self, param_name):
        if param_name == "device_name":
            # The engine committed the staged value (set()+sync()) before
            # invoking this callback (AGENTS.md §5); no re-sync needed here.
            self._request_port_restart()

    def _request_port_restart(self):
        port_name = self.params["device_name"].value
        self._device_epoch += 1
        self._status = "Opening..."

        if getattr(self, "graph", None) and getattr(self.graph, "engine", None):
            self.submit_nrt(self._open_port_nrt, port_name, self._device_epoch, tag="open_input")
        else:
            self._status = "No Device"

    def _open_port_nrt(self, port_name, epoch):
        if not port_name or not MIDO_AVAILABLE:
            return None, "No Device", epoch
        try:
            port = mido.open_input(port_name, callback=self._midi_callback)
            return port, f"Active: {port_name[:20]}", epoch
        except Exception as e:
            raise RuntimeError(f"Failed to open '{port_name}': {e}")

    def on_nrt_complete(self, tag, ok, result):
        if tag == "open_input":
            if ok:
                new_port, status_str, epoch = result
                if epoch != self._device_epoch:
                    if new_port:
                        try:
                            new_port.close()
                        except Exception:
                            pass
                    return
                self._close_port_sync()
                self._inport = new_port
                self._status = status_str
                self.error_msg = None
            else:
                self._close_port_sync()
                self._status = "Error"
                self.error_msg = str(result)

    def on_nrt_discarded(self, tag, ok, result):
        if tag == "open_input" and ok and result:
            # Superseded open: the port was created but never installed, so
            # close it here (engine thread) to avoid leaking the device.
            new_port = result[0] if isinstance(result, tuple) else None
            if new_port:
                try:
                    new_port.close()
                except Exception:
                    pass

    def _midi_callback(self, message):
        self._queue.try_push((0, message))

    def _close_port_sync(self):
        if self._inport is not None:
            try:
                self._inport.close()
            except Exception:
                pass
            self._inport = None

    def start(self):
        self._request_port_restart()

    def stop(self):
        self._device_epoch += 1
        self._close_port_sync()
        self._status = "Stopped"

    def remove(self):
        self._device_epoch += 1
        self._close_port_sync()

    def process(self):
        self.msg_out.packet.messages.clear()
        while True:
            item, ok = self._queue.try_pop()
            if not ok:
                break
            self.msg_out.packet.messages.append(item)

    def get_telemetry(self) -> dict:
        return {"status": self._status}


class MIDIOutputNode(Node):
    category = "I/O"
    label = "MIDI Device Output"
    description = (
        "Sends MIDI messages to an external hardware or virtual MIDI port via a "
        "bounded background writer queue (drop-on-overflow)."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.error_msg = None
        self.midi_in = self.add_midi_input("midi_in", help="MIDI stream to send to hardware.")
        self.add_string_param("device_name", "", help="Target MIDI hardware output device name.")

        self._send_queue = SPSCRingBuffer(capacity=256)
        self._outport = None
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._device_epoch = 0
        self._status = "Idle"

    def on_ui_param_change(self, param_name):
        if param_name == "device_name":
            # The engine committed the staged value (set()+sync()) before
            # invoking this callback (AGENTS.md §5); no re-sync needed here.
            self._request_port_restart()

    def _request_port_restart(self):
        port_name = self.params["device_name"].value
        self._device_epoch += 1
        self._status = "Opening..."

        if getattr(self, "graph", None) and getattr(self.graph, "engine", None):
            self.submit_nrt(self._open_port_nrt, port_name, self._device_epoch, tag="open_output")
        else:
            self._status = "No Device"

    def _open_port_nrt(self, port_name, epoch):
        if not port_name or not MIDO_AVAILABLE:
            return None, "No Device", epoch
        try:
            port = mido.open_output(port_name)
            return port, f"Active: {port_name[:20]}", epoch
        except Exception as e:
            raise RuntimeError(f"Failed to open '{port_name}': {e}")

    def on_nrt_complete(self, tag, ok, result):
        if tag == "open_output":
            if ok:
                new_port, status_str, epoch = result
                if epoch != self._device_epoch:
                    if new_port:
                        try:
                            new_port.close()
                        except Exception:
                            pass
                    return
                self._close_port_sync()
                self._outport = new_port
                self._status = status_str
                self.error_msg = None

                # Dedicated daemon writer thread (off the audio thread)
                if self._outport is not None:
                    self._stop_event.clear()
                    self._worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
                    self._worker_thread.start()
            else:
                self._close_port_sync()
                self._status = "Error"
                self.error_msg = str(result)

    def on_nrt_discarded(self, tag, ok, result):
        if tag == "open_output" and ok and result:
            # Superseded open: the port was created but never installed (and
            # no writer thread was started for it), so close it here to avoid
            # leaking the device.
            new_port = result[0] if isinstance(result, tuple) else None
            if new_port:
                try:
                    new_port.close()
                except Exception:
                    pass

    def _writer_loop(self):
        while not self._stop_event.is_set():
            msg, ok = self._send_queue.try_pop()
            if ok and self._outport is not None:
                try:
                    self._outport.send(msg)
                except Exception as e:
                    logger.error(f"MIDI send error: {e}")
            else:
                self._stop_event.wait(0.001)

    def _close_port_sync(self):
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=0.5)
            self._worker_thread = None
        if self._outport is not None:
            try:
                self._outport.close()
            except Exception:
                pass
            self._outport = None

    def start(self):
        self._request_port_restart()

    def stop(self):
        self._device_epoch += 1
        self._close_port_sync()
        self._status = "Stopped"

    def remove(self):
        self._device_epoch += 1
        self._close_port_sync()

    def process(self):
        packet = self.midi_in.get_packet()
        for offset, msg in packet.messages:
            self._send_queue.try_push(msg)

    def get_telemetry(self) -> dict:
        return {"status": self._status}


# --- UI Custom Widgets ---

class MIDIInputWidget(QWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "MIDIInputNode"

    def __init__(self, proxy):
        super().__init__()
        self.proxy = proxy
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        self.combo = QComboBox()
        self.btn_refresh = QPushButton("⟳")
        self.btn_refresh.setFixedWidth(25)
        row.addWidget(self.combo)
        row.addWidget(self.btn_refresh)
        layout.addLayout(row)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.lbl_status)

        self.btn_refresh.clicked.connect(self._refresh_ports)
        self.combo.activated.connect(self._on_user_selection)
        self._refresh_ports()

    def _refresh_ports(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("None", "")
        for name in MIDIDeviceManager.get_input_names():
            self.combo.addItem(name, name)
        current_name = self.proxy.node_item.params.get("device_name", {}).get("value", "")
        idx = self.combo.findData(current_name)
        self.combo.setCurrentIndex(max(0, idx))
        self.combo.blockSignals(False)

    def _on_user_selection(self, index):
        name = self.combo.currentData()
        self.proxy.set_parameter("device_name", name)

    def on_telemetry(self, data):
        if "status" in data:
            self.lbl_status.setText(data["status"])


class MIDIOutputWidget(QWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "MIDIOutputNode"

    def __init__(self, proxy):
        super().__init__()
        self.proxy = proxy
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        self.combo = QComboBox()
        self.btn_refresh = QPushButton("⟳")
        self.btn_refresh.setFixedWidth(25)
        row.addWidget(self.combo)
        row.addWidget(self.btn_refresh)
        layout.addLayout(row)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.lbl_status)

        self.btn_refresh.clicked.connect(self._refresh_ports)
        self.combo.activated.connect(self._on_user_selection)
        self._refresh_ports()

    def _refresh_ports(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("None", "")
        for name in MIDIDeviceManager.get_output_names():
            self.combo.addItem(name, name)
        current_name = self.proxy.node_item.params.get("device_name", {}).get("value", "")
        idx = self.combo.findData(current_name)
        self.combo.setCurrentIndex(max(0, idx))
        self.combo.blockSignals(False)

    def _on_user_selection(self, index):
        name = self.combo.currentData()
        self.proxy.set_parameter("device_name", name)

    def on_telemetry(self, data):
        if "status" in data:
            self.lbl_status.setText(data["status"])