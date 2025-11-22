import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QMenu, QToolButton, QLabel, QFileDialog
from controller import AppController
from ui_system import GraphScene, GraphView
import plugin_system


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bare Core V5 (Async I/O)")
        self.resize(1000, 700)

        plugin_system.load_plugins()
        self.controller = AppController()
        self.controller.graphUpdated.connect(self.on_graph_update)

        self.scene = GraphScene(self.controller)
        self.view = GraphView(self.scene)
        self.setCentralWidget(self.view)

        self.create_toolbar()

        # Default Patch
        self.create_default_patch()

    def create_default_patch(self):
        id_osc = self.controller.add_node("SineOscillator", (100, 100))
        id_conv = self.controller.add_node("MonoToStereo", (300, 100))
        id_out = self.controller.add_node("AudioOutput", (500, 100))
        if id_osc and id_conv and id_out:
            self.controller.connect_nodes(id_osc, "signal", id_conv, "in")
            self.controller.connect_nodes(id_conv, "out", id_out, "audio_in")

    def create_toolbar(self):
        t = QToolBar()
        self.addToolBar(t)

        btn_add = QToolButton()
        btn_add.setText("Add Node")
        btn_add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(btn_add)
        for name in plugin_system.NODE_REGISTRY.keys():
            a = menu.addAction(name)
            a.triggered.connect(lambda c=False, n=name: self.controller.add_node(n, (200, 200)))
        btn_add.setMenu(menu)
        t.addWidget(btn_add)

        t.addSeparator()
        self.act_start = t.addAction("Start Audio", self.toggle_audio)

        t.addSeparator()
        t.addAction("Clear", self.controller.clear)
        t.addAction("Save Patch", self.handle_save)
        t.addAction("Load Patch", self.handle_load)

        t.addSeparator()
        self.lbl_status = QLabel("Clock: None")
        t.addWidget(self.lbl_status)

    def handle_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Patch", "", "JSON Files (*.json)")
        if fn:
            self.controller.save(fn)

    def handle_load(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load Patch", "", "JSON Files (*.json)")
        if fn:
            self.controller.load(fn)

    def on_graph_update(self, snapshot):
        clk_id = snapshot.get("clock_id")
        if clk_id:
            name = "Unknown"
            for n in snapshot["nodes"]:
                if n["id"] == clk_id:
                    name = n["name"]
                    break
            self.lbl_status.setText(f"Clock: {name}")
            self.lbl_status.setStyleSheet("color: #00FF00; font-weight: bold;")
        else:
            self.lbl_status.setText("Clock: NONE")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")

    def toggle_audio(self):
        if self.controller.engine.running:
            self.controller.stop_audio()
            self.act_start.setText("Start Audio")
        else:
            self.controller.start_audio()
            self.act_start.setText("Stop Audio")

    def closeEvent(self, event):
        self.controller.stop_audio()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
