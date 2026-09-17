"""Shared custom UI for offline (NRT) job nodes.

``OfflineJobWidget`` builds the standard layout every audio.cpp-backed node
needs: one row per control param, a Download button (resumable background
fetch), a run button (Generate/Transcribe trigger) and a Cancel button,
plus status/detail labels fed by ``on_telemetry``.

Concrete widgets subclass this with only configuration (no layout code):

* ``PARAM_KEYS`` — control params to create rows for, in order,
* ``ACTION_LABEL`` / ``ACTION_PARAM`` — e.g. ``("Generate song", "generate")``,
* ``DOWNLOAD_LABEL`` — e.g. ``"Download runtime + model (~3.8 GB)"``,

plus optional overrides of ``_build_after_param`` (extra rows at a
position, e.g. a Randomize-seed button after ``seed``) and
``_on_action_pressed`` (e.g. commit an editor before triggering).

This module defines no ``Node`` subclass and no ``IS_NODE_UI`` class, so
plugin discovery ignores it; the concrete widgets keep living in their
node modules (``NODE_CLASS_NAME`` registration is unchanged).
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class OfflineJobWidget(QWidget):
    PARAM_KEYS = ()
    ACTION_LABEL = "Run"
    ACTION_PARAM = "generate"
    DOWNLOAD_LABEL = "Download"

    def __init__(self, node_proxy):
        super().__init__()
        self.proxy = node_proxy
        self.setMinimumWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.param_widgets = {}
        for key in self.PARAM_KEYS:
            widget = self.proxy.create_param_widget(key)
            self.param_widgets[key] = widget
            layout.addWidget(widget)
            self._build_after_param(layout, key)

        self.btn_download = QPushButton(self.DOWNLOAD_LABEL)
        self.btn_download.clicked.connect(
            lambda: self.proxy.set_parameter("download", True))
        layout.addWidget(self.btn_download)

        self.btn_action = QPushButton(self.ACTION_LABEL)
        self.btn_action.clicked.connect(self._on_action_pressed)
        layout.addWidget(self.btn_action)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(
            lambda: self.proxy.set_parameter("cancel", True))
        layout.addWidget(self.btn_cancel)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)

        self.lbl_detail = QLabel("")
        self.lbl_detail.setStyleSheet("color: #aaa; font-size: 10px;")
        self.lbl_detail.setWordWrap(True)
        layout.addWidget(self.lbl_detail)

    def _build_after_param(self, layout, key):
        """Extra-row hook called after each param row. Base: nothing."""

    def _on_action_pressed(self):
        self.proxy.set_parameter(self.ACTION_PARAM, True)

    def on_telemetry(self, data: dict):
        if "status" in data:
            self.lbl_status.setText(data["status"])
        if "audio" in data:
            self.lbl_detail.setText(data["audio"])

    def update_from_params(self, params):
        for key, widget in self.param_widgets.items():
            if key in params:
                widget.update_from_backend(params[key])
