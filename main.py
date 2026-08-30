import sys
import logging
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QToolBar,
    QWidget,
    QLabel,
    QFileDialog,
    QSizePolicy,
    QDockWidget,
)
from PySide6.QtGui import QAction, QKeySequence, QPalette, QColor
from PySide6.QtCore import Qt
from controller import AppController
from ui_system import GraphScene, GraphView, NodeHelpWidget, NodeItem
import plugin_system

from ui_icons import create_icon

import torch

torch.set_float32_matmul_precision("high")
torch.set_num_threads(1)  # Prevent OpenMP deadlocks on CPU between audio and NRT threads


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ANode - Audio Node Processor")
        self.resize(1200, 800)

        plugin_system.load_plugins()
        self.controller = AppController()
        self.controller.graphUpdated.connect(self.on_graph_update)

        self.scene = GraphScene(self.controller)
        self.view = GraphView(self.scene)
        self.setCentralWidget(self.view)

        # --- Node Reference dock panel ---
        self.dock_help = QDockWidget("Node Reference", self)
        self.dock_help.setObjectName("DockNodeReference")
        self.help_widget = NodeHelpWidget()
        self.dock_help.setWidget(self.help_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_help)

        self.scene.selectionChanged.connect(self._on_selection_changed)
        self.scene.nodeHelpRequested.connect(self._on_node_help_requested)

        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self.create_default_patch()

    def _on_selection_changed(self):
        # Qt can emit selectionChanged while the scene's C++ object is already
        # torn down (e.g. during widget destruction at shutdown); ignore it.
        try:
            selected_nodes = [item for item in self.scene.selectedItems() if isinstance(item, NodeItem)]
        except RuntimeError:
            return
        if len(selected_nodes) == 1:
            self.help_widget.display_node_type(selected_nodes[0].node_type)
        elif len(selected_nodes) == 0:
            self.help_widget.display_welcome()

    def _on_node_help_requested(self, node_type: str):
        self.help_widget.display_node_type(node_type)
        self.dock_help.show()
        self.dock_help.raise_()

    def create_default_patch(self):
        id_osc = self.controller.add_node("SineOscillator", (-275, 0))
        id_conv = self.controller.add_node("MonoToStereo", (-75, 0))
        id_out = self.controller.add_node("AudioDeviceOutput", (125, 0))
        if id_osc and id_conv and id_out:
            self.controller.connect_nodes(id_osc, "signal", id_conv, "in")
            self.controller.connect_nodes(id_conv, "out", id_out, "audio_in")

    def _create_actions(self):
        self.act_new = QAction("&New", self)
        self.act_new.setShortcut(QKeySequence.New)
        self.act_new.triggered.connect(self.controller.clear)
        self.act_new.setIcon(create_icon("new_file"))

        self.act_open = QAction("&Open...", self)
        self.act_open.setShortcut(QKeySequence.Open)
        self.act_open.triggered.connect(self.handle_load)
        self.act_open.setIcon(create_icon("open_folder"))

        self.act_save = QAction("&Save...", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.triggered.connect(self.handle_save)
        self.act_save.setIcon(create_icon("save"))

        self.act_exit = QAction("&Exit", self)
        self.act_exit.setShortcut(QKeySequence.Quit)
        self.act_exit.triggered.connect(self.close)

        self.act_start = QAction("&Start Audio", self)
        self.act_start.setShortcut("F5")
        self.act_start.triggered.connect(self.start_audio_action)
        self.act_start.setIcon(create_icon("play"))

        self.act_stop = QAction("&Stop Audio", self)
        self.act_stop.setShortcut("F6")
        self.act_stop.triggered.connect(self.stop_audio_action)
        self.act_stop.setEnabled(False)
        self.act_stop.setIcon(create_icon("stop"))

        self.act_zoom_in = QAction("Zoom &In", self)
        self.act_zoom_in.setShortcut(QKeySequence.ZoomIn)
        self.act_zoom_in.triggered.connect(self.view.zoom_in)
        self.act_zoom_in.setIcon(create_icon("zoom_in"))

        self.act_zoom_out = QAction("Zoom &Out", self)
        self.act_zoom_out.setShortcut(QKeySequence.ZoomOut)
        self.act_zoom_out.triggered.connect(self.view.zoom_out)
        self.act_zoom_out.setIcon(create_icon("zoom_out"))

        self.act_zoom_fit = QAction("Zoom to &Fit", self)
        self.act_zoom_fit.setShortcut("Ctrl+0")
        self.act_zoom_fit.triggered.connect(self.view.zoom_to_fit)
        self.act_zoom_fit.setIcon(create_icon("zoom_fit"))

        self.act_show_load = QAction("Show Processing &Load", self)
        self.act_show_load.setCheckable(True)
        self.act_show_load.triggered.connect(self.scene.toggle_load_view)

        self.act_reload = QAction("&Reload Plugins", self)
        self.act_reload.triggered.connect(self.controller.reload_plugins)

        self.act_copy = QAction("&Copy", self)
        self.act_copy.setShortcut(QKeySequence.Copy)
        self.act_copy.triggered.connect(self.view.scene().copy_selection)

        self.act_paste = QAction("&Paste", self)
        self.act_paste.setShortcut(QKeySequence.Paste)
        self.act_paste.triggered.connect(self.view.scene().paste_selection)

        self.act_select_all = QAction("Select &All", self)
        self.act_select_all.setShortcut(QKeySequence.SelectAll)
        self.act_select_all.triggered.connect(self.view.scene().selectAll)

        self.act_undo = QAction("&Undo", self)
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.act_undo.triggered.connect(self.controller.undo)
        self.act_undo.setIcon(create_icon("undo"))

        self.act_redo = QAction("&Redo", self)
        self.act_redo.setShortcut(QKeySequence.Redo)
        self.act_redo.triggered.connect(self.controller.redo)
        self.act_redo.setIcon(create_icon("redo"))

    def _create_menus(self):
        menubar = self.menuBar()
        file = menubar.addMenu("&File")
        file.addAction(self.act_new)
        file.addAction(self.act_open)
        file.addAction(self.act_save)
        file.addSeparator()
        file.addAction(self.act_exit)

        edit = menubar.addMenu("&Edit")
        edit.addAction(self.act_undo)
        edit.addAction(self.act_redo)
        edit.addSeparator()
        edit.addAction(self.act_copy)
        edit.addAction(self.act_paste)
        edit.addAction(self.act_select_all)

        process = menubar.addMenu("&Process")
        process.addAction(self.act_start)
        process.addAction(self.act_stop)

        view = menubar.addMenu("&View")
        view.addAction(self.act_zoom_in)
        view.addAction(self.act_zoom_out)
        view.addAction(self.act_zoom_fit)

        dev = menubar.addMenu("&Developer")
        dev.addAction(self.act_show_load)
        dev.addAction(self.act_reload)

        help_menu = menubar.addMenu("&Help")
        act_toggle_doc = self.dock_help.toggleViewAction()
        act_toggle_doc.setText("&Node Reference Panel")
        help_menu.addAction(act_toggle_doc)

        act_shortcuts = QAction("&Keyboard Shortcuts", self)
        act_shortcuts.triggered.connect(lambda: self.help_widget.display_welcome() or self.dock_help.show())
        help_menu.addAction(act_shortcuts)

    def _create_toolbar(self):
        t = QToolBar("Main Toolbar", self)
        t.setMovable(False)
        self.addToolBar(t)
        t.addAction(self.act_new)
        t.addAction(self.act_open)
        t.addAction(self.act_save)
        t.addSeparator()
        t.addAction(self.act_undo)
        t.addAction(self.act_redo)
        t.addSeparator()
        t.addAction(self.act_start)
        t.addAction(self.act_stop)
        t.addSeparator()
        t.addAction(self.act_zoom_in)
        t.addAction(self.act_zoom_out)
        t.addAction(self.act_zoom_fit)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        t.addWidget(spacer)
        self.lbl_status = QLabel("Clock: None  ")
        t.addWidget(self.lbl_status)

    def handle_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Patch", "", "JSON Files (*.json)")
        if fn:
            self.controller.save(fn)

    def handle_load(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load Patch", "", "JSON Files (*.json)")
        if fn:
            self.controller.load(fn)

    def start_audio_action(self):
        self.controller.start_audio()

    def stop_audio_action(self):
        self.controller.stop_audio()

    def on_graph_update(self, snapshot):
        is_running = snapshot.get("is_running", False)
        self.act_start.setEnabled(not is_running)
        self.act_stop.setEnabled(is_running)
        clk_id = snapshot.get("clock_id")
        if clk_id:
            name = "Unknown"
            for n in snapshot["nodes"]:
                if n["id"] == clk_id:
                    name = n["name"]
                    break
            self.lbl_status.setText(f"Clock: {name}  ")
            self.lbl_status.setStyleSheet("color: #00FF00; font-weight: bold;")
        else:
            self.lbl_status.setText("Clock: NONE  ")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")

    def closeEvent(self, event):
        # Detach scene callbacks before Qt tears the C++ scene down; late
        # selectionChanged emissions during destruction would otherwise hit a
        # deleted object.
        try:
            self.scene.selectionChanged.disconnect(self._on_selection_changed)
            self.scene.nodeHelpRequested.disconnect(self._on_node_help_requested)
        except (RuntimeError, TypeError):
            pass
        self.controller.stop_audio()
        # The engine is stopped (audio thread joined), so this is a control-
        # thread operation: give nodes a chance to release native resources
        # (native DSP handles, streams, workers) deterministically instead of
        # letting them die at interpreter shutdown, where extension-module
        # teardown order can report leaked handles (e.g. nanobind/pylibrb).
        engine = getattr(self.controller, "engine", None)
        graph = getattr(engine, "graph", None)
        if graph is not None:
            for node in list(graph.nodes):
                try:
                    node.remove()
                except Exception:
                    logging.exception(f"Node cleanup failed on shutdown: {node}")
        super().closeEvent(event)


def set_dark_theme(app):
    """
    Apply a consistent dark theme using the Fusion style and a custom palette.
    This ensures Linux/Windows/Mac all look the same.
    """
    app.setStyle("Fusion")

    palette = QPalette()
    dark_gray = QColor(53, 53, 53)
    gray = QColor(128, 128, 128)
    black = QColor(25, 25, 25)
    blue = QColor(42, 130, 218)

    palette.setColor(QPalette.Window, dark_gray)
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, black)
    palette.setColor(QPalette.AlternateBase, dark_gray)
    palette.setColor(QPalette.ToolTipBase, black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, dark_gray)
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, blue)
    palette.setColor(QPalette.Highlight, blue)
    palette.setColor(QPalette.HighlightedText, Qt.black)

    # Disabled colors
    palette.setColor(QPalette.Disabled, QPalette.WindowText, gray)
    palette.setColor(QPalette.Disabled, QPalette.Text, gray)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, gray)
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, gray)

    app.setStyleSheet("""
        QToolTip {
            color: #ffffff;
            background-color: #2b2b2b;
            border: 1px solid #767676;
            padding: 2px;
        }
    """)

    app.setPalette(palette)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    set_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
