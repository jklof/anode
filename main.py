import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QToolBar,
    QWidget,
    QMenu,
    QToolButton,
    QLabel,
    QFileDialog,
    QSizePolicy,
    QStyle,
)
from PySide6.QtGui import QAction, QKeySequence
from controller import AppController
from ui_system import GraphScene, GraphView
import plugin_system


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bare Core V5 (Professional)")
        self.resize(1200, 800)

        plugin_system.load_plugins()
        self.controller = AppController()
        self.controller.graphUpdated.connect(self.on_graph_update)

        self.scene = GraphScene(self.controller)
        self.view = GraphView(self.scene)
        self.setCentralWidget(self.view)

        # Setup UI elements
        self._create_actions()
        self._create_menus()
        self._create_toolbar()

        # Create default patch
        self.create_default_patch()

    def create_default_patch(self):
        id_osc = self.controller.add_node("SineOscillator", (100, 100))
        id_conv = self.controller.add_node("MonoToStereo", (300, 100))
        id_out = self.controller.add_node("AudioOutput", (500, 100))
        if id_osc and id_conv and id_out:
            self.controller.connect_nodes(id_osc, "signal", id_conv, "in")
            self.controller.connect_nodes(id_conv, "out", id_out, "audio_in")

    def _create_actions(self):
        # File Actions
        self.act_new = QAction("&New", self)
        self.act_new.setShortcut(QKeySequence.New)
        self.act_new.triggered.connect(self.controller.clear)

        self.act_open = QAction("&Open...", self)
        self.act_open.setShortcut(QKeySequence.Open)
        self.act_open.triggered.connect(self.handle_load)

        self.act_save = QAction("&Save...", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.triggered.connect(self.handle_save)

        self.act_exit = QAction("&Exit", self)
        self.act_exit.setShortcut(QKeySequence.Quit)
        self.act_exit.triggered.connect(self.close)

        # Process Actions
        self.act_start = QAction("&Start Audio", self)
        self.act_start.setShortcut("F5")
        self.act_start.triggered.connect(self.start_audio_action)

        self.act_stop = QAction("&Stop Audio", self)
        self.act_stop.setShortcut("F6")
        self.act_stop.triggered.connect(self.stop_audio_action)
        self.act_stop.setEnabled(False)

        # View Actions
        self.act_zoom_in = QAction("Zoom &In", self)
        self.act_zoom_in.setShortcut(QKeySequence.ZoomIn)
        self.act_zoom_in.triggered.connect(self.view.zoom_in)

        self.act_zoom_out = QAction("Zoom &Out", self)
        self.act_zoom_out.setShortcut(QKeySequence.ZoomOut)
        self.act_zoom_out.triggered.connect(self.view.zoom_out)

        self.act_zoom_fit = QAction("Zoom to &Fit", self)
        self.act_zoom_fit.setShortcut("Ctrl+0")
        self.act_zoom_fit.triggered.connect(self.view.zoom_to_fit)

        # Dev Actions
        self.act_reload = QAction("&Reload Plugins", self)
        self.act_reload.triggered.connect(lambda: plugin_system.load_plugins())

    def _create_menus(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.act_new)
        file_menu.addAction(self.act_open)
        file_menu.addAction(self.act_save)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)

        # Process Menu
        process_menu = menubar.addMenu("&Process")
        process_menu.addAction(self.act_start)
        process_menu.addAction(self.act_stop)

        # View Menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.act_zoom_in)
        view_menu.addAction(self.act_zoom_out)
        view_menu.addAction(self.act_zoom_fit)

        # Developer Menu
        dev_menu = menubar.addMenu("&Developer")
        dev_menu.addAction(self.act_reload)

    def _create_toolbar(self):
        t = QToolBar("Main Toolbar", self)
        t.setMovable(False)
        self.addToolBar(t)

        # Removed "Add Node" button as requested

        # 1. File Operations
        t.addAction(self.act_new)
        t.addAction(self.act_open)
        t.addAction(self.act_save)

        t.addSeparator()

        # 2. Processing
        t.addAction(self.act_start)
        t.addAction(self.act_stop)

        t.addSeparator()

        # 3. View
        t.addAction(self.act_zoom_in)
        t.addAction(self.act_zoom_out)
        t.addAction(self.act_zoom_fit)

        # 4. Spacer & Status
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
        # 1. Update Play/Stop Button State based on actual engine state
        is_running = snapshot.get("is_running", False)
        self.act_start.setEnabled(not is_running)
        self.act_stop.setEnabled(is_running)

        # 2. Update clock status
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
        self.controller.stop_audio()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
