from PySide6.QtCore import QObject, Signal, QTimer
import uuid
import logging
from core import Engine


class AppController(QObject):
    graphUpdated = Signal(dict)
    telemetryUpdated = Signal(dict)
    parameterUpdated = Signal(dict)

    def __init__(self):
        super().__init__()
        self.engine = Engine()

        self.poll_timer = QTimer()
        self.poll_timer.interval = 30
        self.poll_timer.timeout.connect(self.check_engine_messages)
        self.poll_timer.start()

    def check_engine_messages(self):
        while not self.engine.output_queue.empty():
            try:
                msg = self.engine.output_queue.get_nowait()
                if msg.get("type") == "telemetry":
                    self.telemetryUpdated.emit(msg)
                elif msg.get("type") == "param_update":
                    self.parameterUpdated.emit(msg)
                else:
                    self.graphUpdated.emit(msg)
            except Exception:
                logging.exception("Error processing engine message")

    def start_audio(self):
        self.engine.start()

    def stop_audio(self):
        self.engine.stop()

    def add_node(self, node_type_name, pos=(0, 0)):
        new_id = str(uuid.uuid4())
        self.engine.push_command(("add", node_type_name, new_id, pos, None))
        return new_id

    def add_node_with_id(self, node_type_name, pos, node_id, params=None):
        """
        Add a node with a specific ID and initial parameters.
        This is used for clipboard paste functionality.

        Parameters must be in dictionary format: {"param_name": {"value": actual_value, "type": ..., "meta": ...}}
        """
        self.engine.push_command(("add", node_type_name, node_id, pos, params))
        return node_id

    def delete_node(self, node_id):
        self.engine.push_command(("del", node_id))

    def connect_nodes(self, src_id, src_port, dst_id, dst_port):
        self.engine.push_command(("conn", src_id, src_port, dst_id, dst_port))

    def disconnect_nodes(self, src_id, src_port, dst_id, dst_port):
        self.engine.push_command(("disconn", src_id, src_port, dst_id, dst_port))

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

    def reload_plugins(self):
        self.engine.push_command(("reload",))
