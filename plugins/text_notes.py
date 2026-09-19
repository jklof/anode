"""
TextNote + ABCNote — resizable editable text / ABC score notes (Offline).

Single-node, no edit/display toggle: the editor IS the display. Typing never
fights the backend — remote text (a wired load, undo, patch load) is applied
to the editor only when it does not have focus (same focus-guard idiom as
StringParamWidget and ScriptNodeWidget.update_from_params); Apply commits.

Both nodes keep text inline (a multiline string param, patch-portable) AND
file-backed (a content-addressed kept file — SHA-256 of the content as the
file name under outputs/text_notes/ or outputs/abc_notes/ — published on a
URI output for wiring into LyricFitter / YuE2SongGenerator / ABCPlayer, plus
a one-block done pulse for trigger chaining). They follow the offline visual language (is_offline, dashed
URI wires, done pulse) and never do file I/O on the audio thread: Apply and
trigger edges only stage NRT work; the worker writes/reads, on_nrt_complete
installs the result between blocks.

Ports (both nodes):
  trigger_in (audio) — rising edge re-publishes (no wire: kept copy of the
    current text) or loads (wired text_uri/abc_uri: snapshot + read).
  text_uri / abc_uri (uri in) — wired source file, snapshotted on trigger.
  text / abc (uri out) — published file path.
  done (audio, 1 ch) — one-block 1.0 pulse when a new file is ready.

ABCNote additionally validates with abc_score.parse_score (lenient, never
raises) in the worker and shows sections in the widget's section list (parsed
locally in the UI for instant feedback; the engine telemetry carries the
note/section/beat summary and any error).

TextFileSource / ABCFileSource — minimal URI sources: a file picker whose
selected path is published directly on the URI output (no reading, no worker,
no kept copy). TextFileSource is the generic file node (any file kind —
lyrics, audio, …) for handing an existing file to a consumer without an
editable note in the middle; ABCFileSource is its ABC-filtered sibling.
The TextFileSource type name and its "text" output are kept for saved-patch
compatibility (patches wire that output into any uri input).
"""

import hashlib
import logging
import os
import re
from pathlib import Path

from base import Node

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TEXT = """# Text Note

Type or paste anything here (lyrics, prompts, notes), then Apply.
Apply publishes a kept file on the text output for wiring into
lyric/song nodes. Wire a file into text_uri and press Load (or pulse
trigger_in) to replace this text with that file's content.
"""

DEFAULT_ABC = """X:1
T:Untitled
M:4/4
L:1/8
Q:120
K:C
% Verse
C D E F|G4 z2|
"""

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QPushButton,
        QLabel, QListWidget, QSizePolicy,
    )
    from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
    from PySide6.QtCore import Qt, QSignalBlocker, QEvent

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


if GUI_AVAILABLE:

    class ABCSyntaxHighlighter(QSyntaxHighlighter):
        """Minimal ABC highlighting: % comments, H: headers, "chords"."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.highlighting_rules = []

            comment_format = QTextCharFormat()
            comment_format.setForeground(QColor("#6272a4"))
            self.highlighting_rules.append((re.compile(r"%[^\n]*"), comment_format))

            header_format = QTextCharFormat()
            header_format.setForeground(QColor("#8be9fd"))
            self.highlighting_rules.append((re.compile(r"^[A-Za-z]:", re.MULTILINE), header_format))

            chord_format = QTextCharFormat()
            chord_format.setForeground(QColor("#f1fa8c"))
            self.highlighting_rules.append((re.compile(r'"[^"\n]*"'), chord_format))

            lyric_format = QTextCharFormat()
            lyric_format.setForeground(QColor("#50fa7b"))
            self.highlighting_rules.append((re.compile(r"^w:.*", re.MULTILINE), lyric_format))

        def highlightBlock(self, text):
            for pattern, fmt in self.highlighting_rules:
                for match in pattern.finditer(text):
                    start, end = match.span()
                    self.setFormat(start, end - start, fmt)


class _NoteNodeBase(Node):
    """Shared engine logic for TextNote / ABCNote. Subclasses set:

    TEXT_PARAM  — name of the multiline string param ("text" / "abc")
    URI_IN      — uri input name ("text_uri" / "abc_uri")
    URI_OUT     — uri output name ("text" / "abc")
    KEEP_DIR    — outputs/ subdir for kept publishes
    KEEP_SUFFIX — kept file suffix (".txt" / ".abc")
    KIND        — human kind for status messages ("text" / "ABC")
    """

    is_abstract = True
    is_offline = True

    TEXT_PARAM = "text"
    URI_IN = "text_uri"
    URI_OUT = "text"
    KEEP_DIR = "text_notes"
    KEEP_SUFFIX = ".txt"
    KIND = "text"
    DEFAULT_CONTENT = DEFAULT_TEXT

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge re-publishes the current text "
                            "(or loads the wired URI input) on a background worker.")
        self.add_uri_input(self.URI_IN,
                           help="Wired source file. While connected and non-empty it is "
                                "snapshotted on trigger/Load instead of re-publishing.")
        self.add_uri_output(self.URI_OUT,
                            help="Published file path (kept copy on Apply, source path on load).")
        self.done = self.add_output("done", channels=1,
                                    help="One-block 1.0 pulse when a new file is ready.")
        self.add_string_param(self.TEXT_PARAM, self.DEFAULT_CONTENT, multiline=True,
                              help="Note content (inline, patch-portable). Apply publishes it.")
        self.add_bool_param("refresh", False,
                            help="Transient trigger: snapshot the wired URI (or re-publish) "
                                 "on a background worker, then resets itself.")
        self.add_string_param("last_path", "",
                              help="Path of the last published file; re-linked on patch load.")

        self._done_pulse = False
        self._last_trig = 0.0
        self._io_epoch = 0
        self._status = "Idle"
        self._status_detail = "No publish yet"

    # ------------------------------------------------------------------
    # engine/control thread
    # ------------------------------------------------------------------
    def start(self):
        self._done_pulse = False
        self._last_trig = 0.0

    def process(self):
        """Emit the one-block done pulse; a trigger rising edge stages NRT
        work via the engine command queue (never I/O here)."""
        buf = self.done.buffer
        buf.zero_()  # anti-ghost: a stale pulse must never retrigger downstream
        if self._done_pulse:
            self._done_pulse = False
            buf.fill_(1.0)
        trig = self.inputs["trigger_in"].get_tensor()[0]
        t_max = float(trig.max().item())
        if self._last_trig <= 0.0 and t_max > 0.0:
            self._request_refresh()
        self._last_trig = float(trig[-1].item())

    def _request_refresh(self):
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None) if graph is not None else None
        if engine is None:
            return
        engine.push_command(("param", self.id, "refresh", True))

    def _restage(self, name, value):
        self.params[name].set(value)
        self.params[name].sync()

    def _wired_source(self):
        """Snapshot of the wired URI input, or None when unconnected/empty."""
        uri_in = self.inputs.get(self.URI_IN)
        if uri_in is not None and uri_in.connected_outputs:
            wired = uri_in.get_uri()
            if wired:
                return wired
        return None

    def validate_text(self, text):
        """Hook: raise ValueError to reject content (ABC syntax check)."""
        return None

    def on_ui_param_change(self, param_name: str):
        if param_name == self.TEXT_PARAM:
            # Committed text auto-publishes a kept copy (NRT). Rapid Applies
            # collapse via the shared NRT epoch + intra-tag _io_epoch.
            self._status = "Publishing"
            self._io_epoch += 1
            spec = {"op": "publish", self.TEXT_PARAM: self.params[self.TEXT_PARAM].value,
                    "epoch": self._io_epoch}
            self.submit_nrt(self._io_nrt, spec, tag="io")
        elif param_name == "refresh" and self.params["refresh"].value:
            # Deliberate re-stage of the transient trigger (AGENTS.md §5).
            self._restage("refresh", False)
            if getattr(self, "graph", None) is None or self.graph.engine is None:
                self._fail("Node is not attached to an engine.")
                return
            self._io_epoch += 1
            wired = self._wired_source()
            if wired is not None:
                if not Path(wired).exists():
                    self._fail(f"Wired file not found: {wired}")
                    return
                self._status = "Loading"
                spec = {"op": "load", "path": wired, "epoch": self._io_epoch}
            else:
                self._status = "Publishing"
                spec = {"op": "publish", self.TEXT_PARAM: self.params[self.TEXT_PARAM].value,
                        "epoch": self._io_epoch}
            self.submit_nrt(self._io_nrt, spec, tag="io")

    def _fail(self, message):
        self._status = "Error"
        self._status_detail = message
        self.error_msg = message
        logger.error(f"{self.__class__.__name__} {self.name}: {message}")

    # ------------------------------------------------------------------
    # NRT worker (file I/O only here, never on the audio thread)
    # ------------------------------------------------------------------
    def _cas_path(self, text):
        """Content-addressed kept path: one flat dir, SHA-256 of the UTF-8
        content as the file name. Identical content always maps to the same
        path, so re-Applies are byte-stable and naturally deduplicated."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        keep_dir = REPO_ROOT / "outputs" / self.KEEP_DIR
        keep_dir.mkdir(parents=True, exist_ok=True)
        return keep_dir / f"{digest}{self.KEEP_SUFFIX}"

    @staticmethod
    def _write_cas(dest, text):
        """Write text to its CAS path unless already stored. Temp + rename
        keeps the store atomic under concurrent identical publishes (same
        bytes either way, so races are harmless)."""
        if dest.exists():
            return
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, dest)

    def _io_nrt(self, spec):
        op = spec.get("op")
        if op == "load":
            # Normalize wired loads into the CAS too: every published URI is
            # then an immutable content-addressed path, never a mutable
            # source file that could change under a consumer.
            text = Path(spec["path"]).read_text(encoding="utf-8")
            summary = self.validate_text(text)
            dest = self._cas_path(text)
            self._write_cas(dest, text)
            return {"op": "load", "epoch": spec["epoch"], "path": str(dest),
                    "text": text, "summary": summary}
        text = spec.get(self.TEXT_PARAM, "")
        summary = self.validate_text(text)
        dest = self._cas_path(text)
        self._write_cas(dest, text)
        return {"op": "publish", "epoch": spec["epoch"], "path": str(dest),
                "summary": summary}

    def on_nrt_complete(self, tag, ok, result):
        if tag != "io":
            return
        if not ok:
            self._fail(f"I/O failed: {result}")
            return
        if result.get("epoch") != self._io_epoch:
            return  # superseded intra-tag submit; newest wins
        if result.get("op") == "load":
            # Install loaded text as the new committed value (direct set+sync
            # here does NOT re-fire on_ui_param_change, so no publish loop;
            # the widget picks it up via the param_update side-channel when
            # the editor is not focused).
            self.params[self.TEXT_PARAM].set(result.get("text", ""))
            self.params[self.TEXT_PARAM].sync()
        self.outputs[self.URI_OUT].uri = result["path"]
        self._done_pulse = True
        self.error_msg = None
        self._status = "Ready"
        self._status_detail = result.get("summary") or Path(result["path"]).name
        self.params["last_path"].set(result["path"])
        self.params["last_path"].sync()
        self._refresh_ui()

    def _refresh_ui(self):
        """Push telemetry + snapshot now so a trigger-staged load updates the
        editor immediately. The direct set()+sync() above emits neither the
        param_update side-channel nor a snapshot, so without this the widget
        text would refresh only on the next unrelated graph change (same
        rationale as AudioCppJob._refresh_ui; completions are rare)."""
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None) if graph is not None else None
        if engine is None:
            return
        try:
            engine.output_queue.put_nowait(
                {"type": "telemetry", "node_data": {self.id: self.get_telemetry()}})
        except Exception:
            pass
        emit = getattr(engine, "_emit_snapshot", None)
        if callable(emit):
            try:
                emit()
            except Exception:
                pass

    def load_state(self, data: dict):
        super().load_state(data)
        path = self.params["last_path"].value if "last_path" in self.params else ""
        if path and Path(path).exists():
            self.outputs[self.URI_OUT].uri = path
            self._status = "Ready"
            self._status_detail = f"Re-linked {Path(path).name}"
            self.error_msg = None
        elif path:
            self.outputs[self.URI_OUT].uri = ""
            self._status = "Idle"
            self._status_detail = "Previous file is missing; Apply again"

    def get_telemetry(self) -> dict:
        return {"status": self._status, "audio": self._status_detail}


class TextNote(_NoteNodeBase):
    category = "Offline"
    label = "Text Note"
    is_abstract = False
    description = (
        "Resizable editable text note (lyrics, prompts, scratch). The editor is "
        "the display — no mode toggle. Apply publishes a kept text file on the "
        "text URI output with a done pulse (wire it to lyric/song nodes); a "
        "wired text_uri plus Load (or a trigger_in pulse) replaces the text "
        "with that file's content. Inline text is patch-portable; file I/O "
        "runs on a background NRT worker."
    )

    TEXT_PARAM = "text"
    URI_IN = "text_uri"
    URI_OUT = "text"
    KEEP_DIR = "text_notes"
    KEEP_SUFFIX = ".txt"
    KIND = "text"
    DEFAULT_CONTENT = DEFAULT_TEXT


class ABCNote(_NoteNodeBase):
    category = "Offline"
    label = "ABC Note"
    is_abstract = False
    description = (
        "Resizable editable ABC score note. Same publish/load contract as Text "
        "Note (abc URI output, done pulse, trigger_in) plus ABC validation via "
        "abc_score and a section list parsed live in the widget. Publish is "
        "rejected when the score has no playable notes."
    )

    TEXT_PARAM = "abc"
    URI_IN = "abc_uri"
    URI_OUT = "abc"
    KEEP_DIR = "abc_notes"
    KEEP_SUFFIX = ".abc"
    KIND = "ABC"
    DEFAULT_CONTENT = DEFAULT_ABC

    def validate_text(self, text):
        from abc_score import parse_score
        score = parse_score(text or "")
        if not score.notes:
            raise ValueError("score has no playable notes")
        n_sections = len(score.sections)
        sec = f", {n_sections} section(s)" if n_sections else ""
        return f"{len(score.notes)} notes, {score.total_beats:.0f} beats @ {score.bpm:g} BPM{sec}"


class FileSource(Node):
    # Generic file node.
    # (lyrics, audio, …) can be published and wired into any uri input.
    category = "Offline"
    label = "File"
    is_abstract = False
    description = (
        "Publishes any picked file directly on the file URI output "
        "(no editing, no copy — the path itself is the payload). Picking, the "
        "Publish button, or a trigger_in pulse re-publishes with a done pulse "
        "for chaining into lyric/song/score/audio nodes."
    )

    URI_OUT = "file"
    FILE_FILTER = "All Files (*.*)"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge re-publishes the selected "
                            "file and emits the done pulse (e.g. wired from a done pulse).")
        self.add_uri_output(self.URI_OUT,
                            help="Selected file path (labeled 'file' for wiring); "
                                 "published on pick, on trigger, "
                                 "and on patch-load relink.")
        self.done = self.add_output("done", channels=1,
                                    help="One-block 1.0 pulse when the file is (re-)published.")
        self.add_file_param("file", "", filter=self.FILE_FILTER,
                            help="File to publish on the URI output.")
        self.add_bool_param("refresh", False,
                            help="Transient trigger: re-publish the selected file, then resets itself.")

        self._done_pulse = False
        self._last_trig = 0.0
        self._status = "Idle"
        self._status_detail = "No file selected"

    def start(self):
        self._done_pulse = False
        self._last_trig = 0.0

    def process(self):
        buf = self.done.buffer
        buf.zero_()  # anti-ghost: a stale pulse must never retrigger downstream
        if self._done_pulse:
            self._done_pulse = False
            buf.fill_(1.0)
        trig = self.inputs["trigger_in"].get_tensor()[0]
        t_max = float(trig.max().item())
        if self._last_trig <= 0.0 and t_max > 0.0:
            graph = getattr(self, "graph", None)
            engine = getattr(graph, "engine", None) if graph is not None else None
            if engine is not None:
                engine.push_command(("param", self.id, "refresh", True))
        self._last_trig = float(trig[-1].item())

    def _publish(self, pulse=True):
        """Install the committed file path on the URI output. Synchronous —
        no I/O, just string assignment. Existence is reported in telemetry;
        consumers validate at use time with their own clear messages."""
        path = self.params["file"].value
        if not path:
            self.outputs[self.URI_OUT].uri = ""
            self._status = "Idle"
            self._status_detail = "No file selected"
            self.error_msg = None
            return
        self.outputs[self.URI_OUT].uri = path
        if pulse:
            self._done_pulse = True
        self.error_msg = None
        if Path(path).exists():
            self._status = "Ready"
            self._status_detail = Path(path).name
        else:
            self._status = "Warning"
            self._status_detail = f"Missing: {path}"

    def on_ui_param_change(self, param_name: str):
        if param_name == "file":
            self._publish()
        elif param_name == "refresh" and self.params["refresh"].value:
            # Deliberate re-stage of the transient trigger (AGENTS.md §5).
            self.params["refresh"].set(False)
            self.params["refresh"].sync()
            self._publish()

    def load_state(self, data: dict):
        super().load_state(data)
        # Re-link without pulsing: a restored file is not newly ready. A
        # missing file clears (like the note nodes) instead of warning.
        path = self.params["file"].value
        if path and Path(path).exists():
            self.outputs[self.URI_OUT].uri = path
            self._status = "Ready"
            self._status_detail = f"Re-linked {Path(path).name}"
            self.error_msg = None
        else:
            self.outputs[self.URI_OUT].uri = ""
            self._status = "Idle"
            self._status_detail = "Previous file is missing; pick again" if path else "No file selected"

    def get_telemetry(self) -> dict:
        return {"status": self._status, "audio": self._status_detail}



if GUI_AVAILABLE:

    class _NoteWidgetBase(QWidget):
        """Resizable editor widget shared by TextNoteWidget / ABCNoteWidget.

        Subclasses set NODE_CLASS_NAME, TEXT_PARAM, SHOW_SECTIONS and optionally
        enable the ABC highlighter. No edit/display toggle by design: the
        QPlainTextEdit is always editable; backend text only replaces the
        editor when it is not focused.
        """

        NODE_CLASS_NAME = ""
        TEXT_PARAM = "text"
        SHOW_SECTIONS = False
        USE_ABC_HIGHLIGHT = False
        IS_RESIZABLE = True
        MIN_SIZE = (300, 200)
        DEFAULT_SIZE = (440, 320)

        def __init__(self, node_proxy):
            super().__init__()
            self.proxy = node_proxy
            self.setMinimumSize(*self.MIN_SIZE)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            self.editor = QPlainTextEdit()
            self.editor.setFont(QFont("Courier New", 10))
            self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
            self.editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # Pending backend text (see update_from_params): applied on
            # focus-out so remote loads never clobber in-progress typing.
            self._pending_text = None
            self.editor.installEventFilter(self)
            if self.USE_ABC_HIGHLIGHT:
                self.highlighter = ABCSyntaxHighlighter(self.editor.document())
            init_text = self.proxy.node_item.params.get(self.TEXT_PARAM, {}).get("value", "")
            self.editor.setPlainText(init_text)
            layout.addWidget(self.editor)

            self.section_list = None
            if self.SHOW_SECTIONS:
                self.section_list = QListWidget()
                self.section_list.setMaximumHeight(64)
                self.section_list.itemClicked.connect(self._on_section_clicked)
                layout.addWidget(self.section_list)
                self.editor.textChanged.connect(self._refresh_sections)
                self._refresh_sections()

            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(0, 0, 0, 0)
            self.btn_apply = QPushButton("Apply")
            self.btn_apply.setFixedHeight(30)
            self.btn_apply.clicked.connect(self.on_apply)
            self.btn_load = QPushButton("Load from wire")
            self.btn_load.setFixedHeight(30)
            self.btn_load.setToolTip("Snapshot the wired URI input (or re-publish current text when unwired).")
            self.btn_load.clicked.connect(lambda: self.proxy.set_parameter("refresh", True))
            btn_row.addWidget(self.btn_apply)
            btn_row.addWidget(self.btn_load)
            layout.addLayout(btn_row)

            self.lbl_status = QLabel("Status: Idle")
            self.lbl_status.setStyleSheet("color: #00FF00; font-size: 10px; font-weight: bold;")
            layout.addWidget(self.lbl_status)

        # -- actions ---------------------------------------------------
        def on_apply(self):
            self.proxy.set_parameter(self.TEXT_PARAM, self.editor.toPlainText())

        def _on_section_clicked(self, item):
            if self.section_list is None:
                return
            cursor = self.editor.document().find(f"% {item.text()}")
            if not cursor.isNull():
                self.editor.setTextCursor(cursor)
                self.editor.ensureCursorVisible()

        def _refresh_sections(self):
            if self.section_list is None:
                return
            try:
                from abc_score import parse_score
                sections = parse_score(self.editor.toPlainText()).sections
            except Exception:
                sections = ()
            current = [self.section_list.item(i).text() for i in range(self.section_list.count())]
            names = [s.name or "—" for s in sections]
            if current != names:
                with QSignalBlocker(self.section_list):
                    self.section_list.clear()
                    for name in names:
                        self.section_list.addItem(name)

        # -- backend sync ----------------------------------------------
        def on_telemetry(self, data: dict):
            status = data.get("status", "Idle")
            detail = data.get("audio", "")
            label = f"{status}: {detail}" if detail else status
            if status == "Error":
                self.lbl_status.setText(f"Error: {detail}")
                self.lbl_status.setStyleSheet("color: #ff5555; font-size: 10px; font-weight: bold;")
            else:
                self.lbl_status.setText(f"Status: {label}")
                self.lbl_status.setStyleSheet("color: #00FF00; font-size: 10px; font-weight: bold;")

        def update_from_params(self, params):
            if self.TEXT_PARAM in params:
                val = params[self.TEXT_PARAM]
                if self.editor.toPlainText() != val:
                    if self.editor.hasFocus():
                        # Typing in progress: hold for focus-out instead of
                        # clobbering (or silently dropping) the backend text.
                        self._pending_text = val
                    else:
                        self._pending_text = None
                        with QSignalBlocker(self.editor):
                            self.editor.setPlainText(val)
                        self._refresh_sections()

        def eventFilter(self, obj, event):
            if obj is self.editor and event.type() == QEvent.FocusOut:
                self._apply_pending_text()
            return super().eventFilter(obj, event)

        def _apply_pending_text(self):
            if self._pending_text is not None:
                if self.editor.toPlainText() != self._pending_text:
                    with QSignalBlocker(self.editor):
                        self.editor.setPlainText(self._pending_text)
                    self._refresh_sections()
                self._pending_text = None


    class TextNoteWidget(_NoteWidgetBase):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "TextNote"
        TEXT_PARAM = "text"


    class ABCNoteWidget(_NoteWidgetBase):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "ABCNote"
        TEXT_PARAM = "abc"
        SHOW_SECTIONS = True
        USE_ABC_HIGHLIGHT = True


    class FileSourceWidget(QWidget):
        """File picker + Publish button + status for the file source nodes.
        Small fixed widget (not resizable): the file row is the whole UI."""

        IS_NODE_UI = True
        NODE_CLASS_NAME = "FileSource"

        def __init__(self, node_proxy):
            super().__init__()
            self.proxy = node_proxy
            self.setMinimumWidth(240)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(6)

            self.file_widget = self.proxy.create_param_widget("file")
            layout.addWidget(self.file_widget)

            self.btn_publish = QPushButton("Publish")
            self.btn_publish.setFixedHeight(30)
            self.btn_publish.setToolTip("Re-publish the selected file with a done pulse.")
            self.btn_publish.clicked.connect(
                lambda: self.proxy.set_parameter("refresh", True))
            layout.addWidget(self.btn_publish)

            self.lbl_status = QLabel("Status: Idle")
            self.lbl_status.setStyleSheet("color: #00FF00; font-size: 10px; font-weight: bold;")
            layout.addWidget(self.lbl_status)

        def on_telemetry(self, data: dict):
            status = data.get("status", "Idle")
            detail = data.get("audio", "")
            label = f"{status}: {detail}" if detail else status
            if status == "Warning":
                self.lbl_status.setText(label)
                self.lbl_status.setStyleSheet("color: #ffcc66; font-size: 10px; font-weight: bold;")
            else:
                self.lbl_status.setText(f"Status: {label}")
                self.lbl_status.setStyleSheet("color: #00FF00; font-size: 10px; font-weight: bold;")

        def update_from_params(self, params):
            if "file" in params:
                self.file_widget.update_from_backend(params["file"])

