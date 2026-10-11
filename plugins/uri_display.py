"""
UriDisplay — URI monitor tap (Visual & Analysis).

Shows the file path carried by a wired URI output and offers it for
one-click copying to the clipboard. Insert it anywhere a URI flows
(e.g. between a generator's ``audio`` output and SamplePlayer's
``uri_in``) to inspect the path without opening patch files.

Ports:

* ``uri_in`` (URI in) — watched file reference; never read as file data.
* ``trigger_in`` (audio in) — a rising edge snapshots ``uri_in``.
* ``uri_out`` (URI out) — the last snapshotted path ("" until the first
  trigger edge). Wire onward wherever the original URI would go.
* ``trigger_out`` (audio out, 1 ch) — transparent copy of ``trigger_in``,
  so the node can sit inside a trigger chain without changing its timing.

Only the snapshot travels: the node performs no file I/O on any thread
(it never opens, hashes, or decodes the referenced file) and the audio
path stays allocation-free. The latched path is runtime display state:
it is not saved with the patch — after reload, pulse the trigger again
to re-snapshot.
"""

import logging

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from base import Node

logger = logging.getLogger(__name__)

_EMPTY_TEXT = "—"


class UriDisplayWidget(QWidget):
    """Single-button display: the button text IS the URI; clicking it
    copies that path to the clipboard."""

    IS_NODE_UI = True
    NODE_CLASS_NAME = "UriDisplay"

    def __init__(self, node_proxy):
        super().__init__()
        self.proxy = node_proxy
        self._uri = ""
        self.setMinimumWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.btn_copy = QPushButton(_EMPTY_TEXT)
        self.btn_copy.setToolTip("No URI yet — wire uri_in and pulse trigger_in.")
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(self._on_copy_pressed)
        layout.addWidget(self.btn_copy)

    def on_telemetry(self, data: dict):
        uri = data.get("uri", "") or ""
        live = data.get("live_uri", "") or ""
        # Latched snapshot wins; before the first trigger edge, preview
        # the live wire value so the button is useful immediately.
        self._uri = uri or live
        if self._uri:
            self.btn_copy.setText(self._uri)
            self.btn_copy.setToolTip(self._uri)
            self.btn_copy.setEnabled(True)
        else:
            self.btn_copy.setText(_EMPTY_TEXT)
            self.btn_copy.setToolTip(
                "No URI yet — wire uri_in and pulse trigger_in.")
            self.btn_copy.setEnabled(False)

    def update_from_params(self, params):
        # No control params to show; display is telemetry-driven.
        pass

    def _on_copy_pressed(self):
        if not self._uri:
            return
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        clipboard.setText(self._uri)
        self.btn_copy.setText("Copied!")
        QTimer.singleShot(1500, self._restore_text)

    def _restore_text(self):
        try:
            self.btn_copy.setText(self._uri if self._uri else _EMPTY_TEXT)
        except RuntimeError:
            pass  # widget destroyed before the timer fired


class UriDisplay(Node):
    category = "Visual & Analysis"
    label = "URI Display"
    description = (
        "URI monitor tap. Shows the file path carried by the wired uri_in "
        "input with a one-click Copy button; a rising edge on trigger_in "
        "snapshots the path onto the uri_out output for onward wiring, "
        "while trigger_out passes the trigger signal through unchanged. "
        "Performs no file I/O — it only displays and forwards the path "
        "string. The snapshot is runtime state and is not saved; re-trigger "
        "after patch reload."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_uri_input("uri_in",
                           help="Watched file reference (e.g. from a generator "
                                "node). Only the path string is read, never "
                                "the file itself.")
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge snapshots the "
                            "current uri_in path onto uri_out.")
        self.add_uri_output("uri_out",
                            help="Last snapshotted path (empty until the first "
                                 "trigger edge). Wire onward like the original URI.")
        self.trigger_out = self.add_output("trigger_out", channels=1,
                                           help="Transparent copy of trigger_in.")

        self._latched_uri = ""
        self._last_trig = 0.0
        self._snapshots = 0

    def start(self):
        # Fresh edge state so a sustained high re-fires; the latched path
        # survives transport restarts like a loaded sample does.
        self._last_trig = 0.0

    def process(self):
        trig = self.inputs["trigger_in"].get_tensor()
        out = self.trigger_out.buffer
        if trig.shape[0] >= out.shape[0]:
            out.copy_(trig[:out.shape[0]])
        else:
            # Mono source into a wider output: copy_ broadcasts without
            # resizing the pre-allocated destination (AGENTS.md §2).
            out.copy_(trig)

        ch = trig[0]
        t_max = float(ch.max().item())
        if self._last_trig <= 0.0 and t_max > 0.0:
            # Sanctioned cross-thread read: immutable-string attribute swap,
            # no I/O with the result here (AGENTS.md §2).
            uri = self.inputs["uri_in"].get_uri()
            self._latched_uri = uri
            self.outputs["uri_out"].uri = uri
            self._snapshots += 1
        self._last_trig = float(ch[-1].item())

    def get_telemetry(self) -> dict:
        try:
            live = self.inputs["uri_in"].get_uri()
        except Exception:
            live = ""
        return {"status": "Ready" if self._latched_uri else "Idle",
                "uri": self._latched_uri,
                "live_uri": live,
                "snapshots": self._snapshots}
