"""
LyricFitter — fit any lyrics onto any ABC score's phrase structure (Utilities).

Takes words (lyrics_file, or a wired lyrics_uri for future ASR nodes) plus
a score (abc_file or wired abc_uri, e.g. from SheetSage2Transcriber) and
produces YuE2-ready lyrics: same tags in score order, lines distributed
across sung phrases, hook repeats written out, instrumental stretches
left empty — with a fit report saying what was repeated, dropped, or left
short. Works for same-song covers and for fitting one song's words onto
another song's melody.

The work is milliseconds of text processing, but file reads still ride an
NRT worker (never the audio thread). Results publish as a fitted-lyrics
URI output plus a one-block pulse on done, following the offline visual
language (is_offline, dashed URI wires); there is no sidecar, no download,
and no GPU involved.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from base import Node

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
LYRICS_FILTER = "Text Files (*.txt *.md);;All Files (*.*)"
ABC_FILTER = "ABC Files (*.abc);;All Files (*.*)"


class LyricFitterWidget(QWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "LyricFitter"

    PARAM_KEYS = ("lyrics_file", "abc_file")

    def __init__(self, node_proxy):
        super().__init__()
        self.proxy = node_proxy
        self.setMinimumWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.param_widgets = {}
        for key in self.PARAM_KEYS:
            widget = self.proxy.create_param_widget(key)
            self.param_widgets[key] = widget
            layout.addWidget(widget)

        self.btn_fit = QPushButton("Fit lyrics to score")
        self.btn_fit.clicked.connect(
            lambda: self.proxy.set_parameter("fit", True))
        layout.addWidget(self.btn_fit)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)

        self.report_browser = QTextBrowser()
        self.report_browser.setFont(QFont("Monospace", 8))
        self.report_browser.setReadOnly(True)
        self.report_browser.setMinimumHeight(80)
        layout.addWidget(self.report_browser)

    def on_telemetry(self, data: dict):
        if "status" in data:
            self.lbl_status.setText(data["status"])
        if "report" in data:
            text = data["report"] or ""
            if text != self.report_browser.toPlainText():
                self.report_browser.setPlainText(text)

    def update_from_params(self, params):
        for key, widget in self.param_widgets.items():
            if key in params:
                widget.update_from_backend(params[key])


class LyricFitter(Node):
    category = "Offline"
    label = "Lyric Fitter"
    description = (
        "Fits any lyrics onto any ABC score's phrase structure for YuE2 "
        "covers: same tags in score order, hook repeats written out, "
        "instrumental stretches left empty, plus a fit report. Words come "
        "from lyrics_file (or a wired lyrics_uri); the score from abc_file "
        "or a wired abc_uri (e.g. SheetSage2Transcriber). Publishes the "
        "fitted file on the lyrics URI output with a one-block pulse on "
        "done — wire it to YuE2SongGenerator's lyrics side. Fitting starts "
        "from the Fit button or a rising edge on trigger_in (e.g. wired "
        "from a done pulse). Text-only, no GPU; file reads run on a "
        "background NRT worker."
    )
    is_offline = True

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge stages a background fit "
                            "(same as the Fit button), e.g. wired from a done pulse.")
        self.add_uri_input("abc_uri",
                           help="Wired score path (e.g. from SheetSage2Transcriber). While connected "
                                "and non-empty it overrides abc_file; snapshots at Fit time.")
        self.add_uri_input("lyrics_uri",
                           help="Wired lyrics path (future ASR nodes). While connected and non-empty "
                                "it overrides lyrics_file; snapshots at Fit time.")
        self.add_uri_output("lyrics",
                            help="Kept fitted-lyrics path; published on completion and on patch-load relink.")
        self.done = self.add_output("done", channels=1,
                                    help="One-block 1.0 pulse when new fitted lyrics are ready.")
        self.add_file_param("lyrics_file", "", filter=LYRICS_FILTER,
                            help="Lyric words to fit (any tagged or plain text file).")
        self.add_file_param("abc_file", "", filter=ABC_FILTER,
                            help="ABC score whose phrase structure the words must ride. "
                                 "A connected abc_uri input overrides this.")
        self.add_bool_param("fit", False,
                            help="Transient trigger: fits lyrics to the score on a background worker, "
                                 "then resets itself.")
        self.add_string_param("last_fit", "",
                              help="Path of the last fitted lyrics; re-linked on patch load.")

        self._done_pulse = False  # emitted as a one-block pulse on done
        self._last_trig = 0.0
        self._status = "Idle"
        self._status_detail = "No fit yet"
        self._report = ""

    # ------------------------------------------------------------------
    # engine/control thread
    # ------------------------------------------------------------------
    def start(self):
        self._done_pulse = False
        self._last_trig = 0.0

    def process(self):
        """Emit the one-block done pulse; a trigger rising edge stages a
        background fit. Otherwise nothing per block."""
        buf = self.done.buffer
        buf.zero_()  # anti-ghost: a stale pulse must never retrigger downstream
        if self._done_pulse:
            self._done_pulse = False
            buf.fill_(1.0)
        trig = self.inputs["trigger_in"].get_tensor()[0]
        t_max = float(trig.max().item())
        if self._last_trig <= 0.0 and t_max > 0.0:
            self._request_fit()
        self._last_trig = float(trig[-1].item())

    def _request_fit(self):
        """Ask the engine thread to stage a fit (one-shot per edge).

        The audio thread must never build the job spec (file existence
        checks) or touch params directly (AGENTS.md section 5); instead it
        queues a ("param", ...) command — the unbounded command queue never
        blocks — and the engine thread runs the normal Fit path, including
        validation and instant telemetry.
        """
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None) if graph is not None else None
        if engine is None:
            return
        engine.push_command(("param", self.id, "fit", True))

    def _restage(self, name, value):
        self.params[name].set(value)
        self.params[name].sync()

    def _snapshot_source(self, uri_name, param_name, what):
        """Wired URI wins over the file param (both snapshot at Fit time).
        Raises ValueError with a clear message."""
        uri_in = self.inputs.get(uri_name)
        if uri_in is not None and uri_in.connected_outputs:
            wired = uri_in.get_uri()
            if not wired:
                raise ValueError(f"Wired {what} is empty — produce it first, then fit.")
            if not Path(wired).exists():
                raise ValueError(f"Wired {what} file not found: {wired}")
            return wired
        path = self.params[param_name].value
        if not path or not Path(path).exists():
            raise ValueError(f"Pick {what} first ({param_name} is empty or missing).")
        return path

    def _build_spec(self):
        """Snapshot committed params into a worker spec. Raises ValueError."""
        return {
            "lyrics_file": self._snapshot_source("lyrics_uri", "lyrics_file", "lyrics"),
            "abc_file": self._snapshot_source("abc_uri", "abc_file", "score"),
        }

    def on_ui_param_change(self, param_name: str):
        if param_name != "fit" or not self.params["fit"].value:
            return
        # Deliberate re-stage of the transient trigger (AGENTS.md section 5).
        self._restage("fit", False)
        if getattr(self, "graph", None) is None or self.graph.engine is None:
            self._fail("Node is not attached to an engine.")
            return
        try:
            spec = self._build_spec()
        except ValueError as e:
            self._fail(str(e))
            return
        self.error_msg = None
        self._status = "Fitting"
        self._status_detail = (f"{Path(spec['lyrics_file']).name} → "
                               f"{Path(spec['abc_file']).name}…")
        self.submit_nrt(self._fit_nrt, spec, tag="fit")

    def _fail(self, message):
        self._status = "Error"
        self._status_detail = message
        self.error_msg = message
        logger.error(f"LyricFitter {self.name}: {message}")

    # ------------------------------------------------------------------
    # NRT worker
    # ------------------------------------------------------------------
    def _fit_nrt(self, spec):
        """Read inputs, align, keep fitted file + report. Fast (ms), but
        file I/O keeps it on the worker per AGENTS.md section 11."""
        from abc_score import load_score
        from lyric_fit import fit_lyrics_to_score
        lyrics_text = Path(spec["lyrics_file"]).read_text(encoding="utf-8")
        score = load_score(spec["abc_file"])
        result = fit_lyrics_to_score(lyrics_text, score)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = Path(spec["lyrics_file"]).stem
        keep = REPO_ROOT / "outputs" / "lyric_fits" / f"{stamp}_{stem}"
        keep.mkdir(parents=True, exist_ok=True)
        (keep / "fitted.txt").write_text(result.text, encoding="utf-8")
        (keep / "report.txt").write_text(result.report, encoding="utf-8")
        (keep / "request.json").write_text(
            json.dumps({
                "lyrics_file": spec["lyrics_file"],
                "lyrics_text": lyrics_text,
                "abc_file": spec["abc_file"],
                "coverage": result.coverage,
                "repeated": result.repeated,
                "dropped": list(result.dropped),
                "warnings": list(result.warnings),
            }, indent=2) + "\n",
            encoding="utf-8")
        return {"path": str(keep / "fitted.txt"), "report": result.report,
                "coverage": result.coverage}

    def on_nrt_complete(self, tag, ok, result):
        if tag != "fit":
            return
        if not ok:
            self._fail(f"Fit failed: {result}")
            return
        self.outputs["lyrics"].uri = result["path"]
        self._done_pulse = True
        self.error_msg = None
        self._report = result.get("report", "")
        self._status = "Ready"
        self._status_detail = f"{result['coverage']:.0%} phrases covered"
        self.params["last_fit"].set(result["path"])
        self.params["last_fit"].sync()

    def load_state(self, data: dict):
        super().load_state(data)
        # Fitted lyrics are small text: re-link synchronously on the control
        # thread. No done pulse: a re-linked fit is not newly ready.
        path = self.params["last_fit"].value if "last_fit" in self.params else ""
        if path and Path(path).exists():
            try:
                self._report = (Path(path).parent / "report.txt").read_text(encoding="utf-8")
            except OSError:
                self._report = ""
            self.outputs["lyrics"].uri = path
            self._status = "Ready"
            self._status_detail = f"Re-linked {Path(path).name}"
            self.error_msg = None
        elif path:
            self.outputs["lyrics"].uri = ""
            self._status = "Idle"
            self._status_detail = "Previous fit is missing; fit again"

    def get_telemetry(self) -> dict:
        return {"status": self._status, "audio": self._status_detail,
                "report": self._report,
                "busy": self._status == "Fitting"}
