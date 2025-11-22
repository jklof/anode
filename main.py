import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QMenu, QToolButton, QLabel
from PySide6.QtGui import QAction
from controller import AppController
from ui_system import GraphScene, GraphView
import plugin_system

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bare Core V5")
        self.resize(1000, 700)
        
        plugin_system.load_plugins()
        self.controller = AppController()
        self.controller.graphUpdated.connect(self.update_status_label)
        
        self.scene = GraphScene(self.controller)
        self.view = GraphView(self.scene)
        self.setCentralWidget(self.view)
        
        self.create_toolbar()
        
        # Default Patch
        osc = self.controller.add_node("SineOscillator", (100, 100))
        conv = self.controller.add_node("MonoToStereo", (300, 100))
        out = self.controller.add_node("AudioOutput", (500, 100))
        if osc and conv and out:
            self.controller.connect_nodes(osc.id, "signal", conv.id, "in")
            self.controller.connect_nodes(conv.id, "out", out.id, "audio_in")
        
        self.update_status_label()

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
        t.addAction("Save", lambda: self.controller.save("patch.json"))
        t.addAction("Load", lambda: self.controller.load("patch.json"))
        t.addAction("Clear", self.controller.clear)
        
        t.addSeparator()
        self.lbl_status = QLabel("Clock: None")
        t.addWidget(self.lbl_status)

    def update_status_label(self):
        clk = self.controller.graph.clock_source
        if clk:
            self.lbl_status.setText(f"Clock: {clk.name}")
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