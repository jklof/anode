import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QToolBar
from PySide6.QtGui import QAction, QPainter
from PySide6.QtCore import Qt, QTimer

from core import Graph, Engine, register_node
from nodes import SineOscillator, AudioOutput, MonoToStereo
from ui_framework import NodeItem, ConnectionItem


class GraphScene(QGraphicsScene):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

        # The View State
        self.node_items = {}  # node_id -> NodeItem
        self.wire_items = {}  # (src_id, src_p, dst_id, dst_p) -> ConnectionItem

        self.temp_wire = None
        self.drag_start_socket = None

    def reconcile(self):
        """
        The Core Algorithm: Make the UI match the Graph.
        Safe to call 1 time or 100 times per second.
        """

        # --- 1. Reconcile Nodes ---

        # A. Identify Logic vs UI
        logic_ids = {n.id for n in self.graph.nodes}
        current_ui_ids = set(self.node_items.keys())

        # B. Remove Dead Nodes
        for node_id in current_ui_ids - logic_ids:
            item = self.node_items.pop(node_id)
            self.removeItem(item)

        # C. Add New Nodes
        for node_id in logic_ids - current_ui_ids:
            node = self.graph.node_map[node_id]
            item = NodeItem(node)

            # Set initial position from logic
            pos = getattr(node, "pos", (0, 0))
            item.setPos(*pos)

            # Bind position changes back to logic
            # (Python closure capture trick)
            def sync_pos(n=node, i=item):
                n.pos = (i.x(), i.y())

            item.positionChanged.connect(sync_pos)

            self.addItem(item)
            self.node_items[node_id] = item

        # --- 2. Reconcile Connections ---

        # A. Build Set of Logical Connections
        # Key format: (src_id, src_port, dst_id, dst_port)
        logic_connections = set()
        for dst_node in self.graph.nodes:
            for dst_port, inp in dst_node.inputs.items():
                if inp.connected_output:
                    out = inp.connected_output
                    # We need to find the source node ID.
                    # In Core v4, OutputSlot holds .parent
                    src_node = out.parent
                    key = (src_node.id, out.name, dst_node.id, dst_port)
                    logic_connections.add(key)

        current_ui_keys = set(self.wire_items.keys())

        # B. Remove Dead Wires
        for key in current_ui_keys - logic_connections:
            wire = self.wire_items.pop(key)
            self.removeItem(wire)

        # C. Add New Wires
        for key in logic_connections - current_ui_keys:
            src_id, src_port, dst_id, dst_port = key

            # Ensure both nodes exist in UI (should be true due to step 1)
            if src_id in self.node_items and dst_id in self.node_items:
                src_item = self.node_items[src_id]
                dst_item = self.node_items[dst_id]

                # Find the specific socket graphics items
                # (ui_framework.NodeItem creates .output_items and .input_items dicts)
                src_socket_item = src_item.output_items.get(src_port)
                dst_socket_item = dst_item.input_items.get(dst_port)

                if src_socket_item and dst_socket_item:
                    wire = ConnectionItem(src_socket_item.scenePos(), dst_socket_item.scenePos())
                    self.addItem(wire)
                    self.wire_items[key] = wire

                    # Dynamic updates when nodes move
                    # We use a closure to bind specific wire/sockets
                    def update_wire(w=wire, s=src_socket_item, d=dst_socket_item):
                        w.start_pos = s.scenePos()
                        w.end_pos = d.scenePos()
                        w.update_path()

                    src_socket_item.parentItem().positionChanged.connect(update_wire)
                    dst_socket_item.parentItem().positionChanged.connect(update_wire)


class EditorView(QGraphicsView):
    def __init__(self, scene, graph):
        super().__init__(scene)
        self.graph = graph
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        view_pos = event.position().toPoint()
        item = self.itemAt(view_pos)
        if hasattr(item, "is_input"):
            self.scene().drag_start_socket = item
            self.setDragMode(QGraphicsView.NoDrag)
            self.scene().temp_wire = ConnectionItem(item.scenePos(), self.mapToScene(view_pos))
            self.scene().addItem(self.scene().temp_wire)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.scene().temp_wire:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.scene().temp_wire.end_pos = scene_pos
            self.scene().temp_wire.update_path()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.scene().temp_wire:
            view_pos = event.position().toPoint()
            end_item = self.itemAt(view_pos)
            start_item = self.scene().drag_start_socket

            if hasattr(end_item, "is_input") and start_item:
                if start_item.is_input != end_item.is_input and start_item.parentItem() != end_item.parentItem():
                    src = start_item if not start_item.is_input else end_item
                    dst = end_item if end_item.is_input else start_item

                    # 1. Mutate Data
                    self.graph.connect(src.slot_ref.parent, src.slot_ref.name, dst.slot_ref.parent, dst.slot_ref.name)

                    # 2. Reconcile UI
                    self.scene().reconcile()

            self.scene().removeItem(self.scene().temp_wire)
            self.scene().temp_wire = None
            self.scene().drag_start_socket = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)

        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bare Core Audio - Reconciled")
        self.resize(800, 600)

        self.graph = Graph()

        # GUI Setup
        self.scene = GraphScene(self.graph)
        self.view = EditorView(self.scene, self.graph)
        self.setCentralWidget(self.view)

        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # Actions
        toolbar.addAction("Add Sine", self.add_sine)
        toolbar.addAction("Add Split", self.add_split)
        toolbar.addAction("Add Speaker", self.add_speaker)
        toolbar.addSeparator()
        toolbar.addAction("Save", self.save_graph)
        toolbar.addAction("Load", self.load_graph)
        toolbar.addAction("Clear", self.clear_graph)
        toolbar.addSeparator()

        self.act_start = QAction("Start", self)
        self.act_start.triggered.connect(self.toggle_audio)
        toolbar.addAction(self.act_start)

        self.engine = Engine(self.graph)

        # Initial Populate
        osc = self.add_sine()
        conv = self.add_split()
        out = self.add_speaker()
        self.graph.connect(osc, "signal", conv, "in")
        self.graph.connect(conv, "out", out, "audio_in")
        self.graph.set_master_clock(out)
        self.scene.reconcile()

    def add_sine(self):
        node = SineOscillator("Osc")
        node.pos = (100, 100)
        self.graph.add_node(node)
        self.scene.reconcile()
        return node

    def add_split(self):
        node = MonoToStereo("Split")
        node.pos = (300, 100)
        self.graph.add_node(node)
        self.scene.reconcile()
        return node

    def add_speaker(self):
        node = AudioOutput("Speakers")
        node.pos = (500, 100)
        self.graph.add_node(node)
        self.graph.set_master_clock(node)
        self.scene.reconcile()
        return node

    def clear_graph(self):
        self.engine.stop()
        self.graph = Graph()  # New instance
        # Update references
        self.engine.graph = self.graph
        self.scene.graph = self.graph
        self.view.graph = self.graph

        self.scene.reconcile()  # Will wipe the screen

    def save_graph(self):
        with open("patch.json", "w") as f:
            f.write(self.graph.to_json())
        print("Saved to patch.json")

    def load_graph(self):
        self.engine.stop()
        try:
            with open("patch.json", "r") as f:
                # Load Logic
                self.graph = Graph.from_json(f.read())

                # Update references
                self.engine.graph = self.graph
                self.scene.graph = self.graph
                self.view.graph = self.graph

                # Rebuild UI
                self.scene.reconcile()
                print("Loaded patch.json")
        except FileNotFoundError:
            print("No save file found.")

    def toggle_audio(self):
        if self.engine.running:
            self.engine.stop()
            self.act_start.setText("Start")
        else:
            self.engine.start()
            self.act_start.setText("Stop")

    def closeEvent(self, event):
        self.engine.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
