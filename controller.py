from PySide6.QtCore import QObject, Signal
from core import Graph, Engine
from plugin_system import NODE_REGISTRY

class AppController(QObject):
    graphUpdated = Signal() 
    
    def __init__(self):
        super().__init__()
        self.graph = Graph()
        self.engine = Engine(self.graph)

    def start_audio(self): self.engine.start()
    def stop_audio(self): self.engine.stop()

    def add_node(self, node_type_name, pos=(0,0)):
        cls = NODE_REGISTRY.get(node_type_name)
        if not cls: return None
        
        try:
            node = cls(name=node_type_name)
            node.pos = pos
        except Exception as e:
            print(f"Node Creation Failed: {e}")
            return None
        
        def mutation():
            self.graph.add_node(node)
            if self.engine.running: node.start()
            self.graphUpdated.emit()

        if self.engine.running: self.engine.push_command(mutation)
        else: mutation()
        return node

    def delete_node(self, node_id):
        def mutation():
            if self.engine.running:
                n = self.graph.node_map.get(node_id)
                if n: n.stop()
            self.graph.remove_node(node_id)
            self.graphUpdated.emit()

        if self.engine.running: self.engine.push_command(mutation)
        else: mutation()

    def connect_nodes(self, src_id, src_port, dst_id, dst_port):
        def mutation():
            self.graph.connect(src_id, src_port, dst_id, dst_port)
            self.graphUpdated.emit()

        if self.engine.running: self.engine.push_command(mutation)
        else: mutation()

    def disconnect_nodes(self, dst_id, dst_port):
        def mutation():
            self.graph.disconnect(dst_id, dst_port)
            self.graphUpdated.emit()

        if self.engine.running: self.engine.push_command(mutation)
        else: mutation()

    def set_master_clock(self, node_id):
        def mutation():
            node = self.graph.node_map.get(node_id)
            if node:
                try:
                    self.graph.set_master_clock(node)
                    self.graphUpdated.emit()
                except ValueError as e: print(e)
        if self.engine.running: self.engine.push_command(mutation)
        else: mutation()

    def set_parameter(self, node_id, param_name, value):
        node = self.graph.node_map.get(node_id)
        if node and param_name in node.params:
            node.params[param_name].set(value)
            try: node.on_ui_param_change(param_name)
            except Exception as e: print(f"Param Error: {e}")

    def save(self, filename):
        with self.engine.lock: 
            data = self.graph.to_json()
        with open(filename, 'w') as f: f.write(data)

    def load(self, filename):
        was_running = self.engine.running
        if was_running: self.engine.stop()
        try:
            with open(filename, 'r') as f:
                import json
                data = json.loads(f.read())
                new_graph = Graph()
                for n_data in data['nodes']:
                    cls = NODE_REGISTRY.get(n_data['type'])
                    if cls:
                        node = cls(n_data['name'])
                        node.id = n_data['id']; node.load_state(n_data); new_graph.add_node(node)
                for c in data['connections']:
                    new_graph.connect(c['src_id'], c['src_port'], c['dst_id'], c['dst_port'])
                if data.get('clock_id'):
                    clk = new_graph.node_map.get(data['clock_id'])
                    if clk: new_graph.set_master_clock(clk)
                self.graph = new_graph; self.engine.graph = new_graph
                self.graphUpdated.emit()
                if was_running: self.engine.start()
        except Exception as e: print(f"Load Error: {e}")

    def clear(self):
        was_running = self.engine.running
        if was_running: self.engine.stop()
        self.graph = Graph()
        self.engine.graph = self.graph
        self.graphUpdated.emit()