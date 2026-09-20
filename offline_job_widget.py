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

Score viewer: subclasses with ``SHOW_SCORE = True`` get a read-only score
browser plus a clickable section list, fed by the ``score_text`` telemetry
key (full ABC text; the widget re-renders only when it changes). Read-only
by design — editing is a later phase; the kept score file stays the single
source of truth.

This module defines no ``Node`` subclass and no ``IS_NODE_UI`` class, so
plugin discovery ignores it; the concrete widgets keep living in their
node modules (``NODE_CLASS_NAME`` registration is unchanged).
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from abc_score import parse_score


class OfflineJobWidget(QWidget):
    PARAM_KEYS = ()
    ACTION_LABEL = "Run"
    ACTION_PARAM = "generate"
    DOWNLOAD_LABEL = "Download"
    SHOW_SCORE = False

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

        self._score_text = None
        if self.SHOW_SCORE:
            self.section_list = QListWidget()
            self.section_list.setMaximumHeight(64)
            self.section_list.itemClicked.connect(self._on_section_clicked)
            layout.addWidget(self.section_list)
            self.score_browser = QTextBrowser()
            self.score_browser.setFont(QFont("Monospace", 8))
            self.score_browser.setReadOnly(True)
            self.score_browser.setMinimumHeight(120)
            layout.addWidget(self.score_browser)

    def _on_section_clicked(self, item):
        """Scroll the browser to the clicked section's % line.

        Tolerant of ``%name`` (no space) headers: tries ``"% " + name``
        first, then ``"%" + name``. The "—" empty-section placeholder
        (or an empty name) is a safe no-op — there is no header to find.
        """
        if not self.SHOW_SCORE:
            return
        name = item.text()
        if not name or name == "—":
            return
        cursor = self.score_browser.document().find(f"% {name}")
        if cursor.isNull():
            cursor = self.score_browser.document().find(f"%{name}")
        if not cursor.isNull():
            self.score_browser.setTextCursor(cursor)
            self.score_browser.ensureCursorVisible()

    def _build_after_param(self, layout, key):
        """Extra-row hook called after each param row. Base: nothing."""

    def _on_action_pressed(self):
        self.proxy.set_parameter(self.ACTION_PARAM, True)

    def on_telemetry(self, data: dict):
        if "status" in data:
            self.lbl_status.setText(data["status"])
        if "audio" in data:
            self.lbl_detail.setText(data["audio"])
        if self.SHOW_SCORE and "score_text" in data:
            self._refresh_score(data["score_text"] or "")

    def _refresh_score(self, text):
        """Re-render the browser + section list when the score changed."""
        if text == self._score_text:
            return
        self._score_text = text
        self.score_browser.setPlainText(text)
        self.section_list.clear()
        for section in parse_score(text).sections:
            self.section_list.addItem(section.name or "—")

    def update_from_params(self, params):
        for key, widget in self.param_widgets.items():
            if key in params:
                widget.update_from_backend(params[key])
