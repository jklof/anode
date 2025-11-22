from PySide6.QtCore import QObject, Signal, QTimer
import uuid
from core import Engine


class AppController(QObject):
    graphUpdated = Signal(dict)

    def __init__(self):
        super().__init__()
        self.engine = Engine()

        self.poll_timer = QTimer()
        self.poll_timer.interval = 30
        self.poll_timer.timeout.connect(self.check_engine_messages)
        self.poll_timer.start()

    def check_engine_messages(self):
        try:
            snap = None
            while not self.engine.output_queue.empty():
                snap = self.engine.output_queue.get_nowait()
            if snap:
                self.graphUpdated.emit(snap)
        except:
            pass

    def start_audio(self):
        self.engine.start()

    def stop_audio(self):
        self.engine.stop()

    def add_node(self, node_type_name, pos=(0, 0)):
        new_id = str(uuid.uuid4())
        self.engine.push_command(("add", node_type_name, new_id, pos))
        return new_id

    def delete_node(self, node_id):
        self.engine.push_command(("del", node_id))

    def connect_nodes(self, src_id, src_port, dst_id, dst_port):
        self.engine.push_command(("conn", src_id, src_port, dst_id, dst_port))

    def disconnect_nodes(self, dst_id, dst_port):
        self.engine.push_command(("disconn", dst_id, dst_port))

    def set_master_clock(self, node_id):
        self.engine.push_command(("clock", node_id))

    def set_parameter(self, node_id, param_name, value):
        self.engine.push_command(("param", node_id, param_name, value))

    def move_node(self, node_id, pos):
        self.engine.push_command(("move", node_id, pos[0], pos[1]))

    def save(self, filename):
        if not filename:
            return
        self.engine.push_command(("save", filename))

    def load(self, filename):
        if not filename:
            return
        try:
            with open(filename, "r") as f:
                json_str = f.read()
                self.engine.push_command(("load", json_str))
        except Exception as e:
            print(f"Controller Load Error: {e}")

    def clear(self):
        self.engine.push_command(("clear",))
