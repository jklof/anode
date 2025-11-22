from PySide6.QtWidgets import (QGraphicsItem, QGraphicsObject, QGraphicsPathItem, QMenu,
    QGraphicsProxyWidget, QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout, QGraphicsScene, QGraphicsView,
    QLineEdit, QCheckBox, QComboBox, QDoubleSpinBox)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QLinearGradient, QCursor
import plugin_system
from core import IClockProvider 

NODE_WIDTH = 160
HEADER_HEIGHT = 30
SOCKET_RADIUS = 6

class SocketItem(QGraphicsItem):
    def __init__(self, parent, name, is_input, slot_ref):
        super().__init__(parent)
        self.name = name
        self.is_input = is_input
        self.slot_ref = slot_ref
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        self.setCursor(QCursor(Qt.CrossCursor))
        self._base_color = QColor("#ff9900") if is_input else QColor("#00ccff")
        self._current_color = self._base_color
    def boundingRect(self):
        pad = 4 
        return QRectF(-SOCKET_RADIUS-pad, -SOCKET_RADIUS-pad, (SOCKET_RADIUS+pad)*2, (SOCKET_RADIUS+pad)*2)
    def paint(self, painter, option, widget):
        painter.setBrush(self._current_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(0,0), SOCKET_RADIUS, SOCKET_RADIUS)
        painter.setPen(QColor("white"))
        text_rect = QRectF(15 if self.is_input else -105, -10, 90, 20)
        align = Qt.AlignLeft if self.is_input else Qt.AlignRight
        painter.drawText(text_rect, align | Qt.AlignVCenter, self.name)
    def set_highlight(self, active):
        self._current_color = QColor("white") if active else self._base_color
        self.update()
    def hoverEnterEvent(self, e): self.set_highlight(True); super().hoverEnterEvent(e)
    def hoverLeaveEvent(self, e): self.set_highlight(False); super().hoverLeaveEvent(e)

class ConnectionItem(QGraphicsPathItem):
    def __init__(self, start_item, end_item, logic_key=None):
        super().__init__()
        self.setZValue(-1)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.start_item = start_item; self.end_item = end_item; self.logic_key = logic_key
        self.update_path()
    def update_path(self):
        if isinstance(self.start_item, QGraphicsItem) and self.start_item.scene(): p1 = self.start_item.scenePos(); start_is_input = getattr(self.start_item, 'is_input', False)
        elif isinstance(self.start_item, (QPointF, list, tuple)): p1 = self.start_item; start_is_input = False
        else: return
        if isinstance(self.end_item, QGraphicsItem) and self.end_item.scene(): p2 = self.end_item.scenePos(); end_is_input = getattr(self.end_item, 'is_input', True)
        elif isinstance(self.end_item, (QPointF, list, tuple)): p2 = self.end_item; end_is_input = not start_is_input
        else: return
        path = QPainterPath(); path.moveTo(p1)
        dist = max(abs(p1.x() - p2.x()) * 0.5, 50.0)
        cp1 = -dist if start_is_input else dist; cp2 = -dist if end_is_input else dist
        path.cubicTo(QPointF(p1.x() + cp1, p1.y()), QPointF(p2.x() + cp2, p2.y()), p2)
        self.setPath(path)
    def paint(self, p, o, w):
        p.setPen(QPen(QColor("yellow") if self.isSelected() else QColor("white"), 3 if self.isSelected() else 2)); p.drawPath(self.path())

class NodeItem(QGraphicsObject):
    positionChanged = Signal(str, object) 

    def __init__(self, node_logic, controller):
        super().__init__()
        self.node = node_logic
        self.controller = controller
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setCursor(QCursor(Qt.SizeAllCursor))
        
        self.width = NODE_WIDTH
        self.input_items = {}
        self.output_items = {}
        
        y = HEADER_HEIGHT + 10
        for name, slot in self.node.inputs.items():
            item = SocketItem(self, name, True, slot)
            item.setPos(0, y)
            self.input_items[name] = item
            y += 20
            
        y_out = HEADER_HEIGHT + 10
        for name, slot in self.node.outputs.items():
            item = SocketItem(self, name, False, slot)
            item.setPos(self.width, y_out)
            self.output_items[name] = item
            y_out += 20
            
        self.height = max(y, y_out) + 10

        self.proxy = QGraphicsProxyWidget(self)
        self.widget = None
        CustomUIClass = plugin_system.get_ui_class(self.node.__class__.__name__)
        if CustomUIClass:
            try: self.widget = CustomUIClass(self.node)
            except Exception as e: print(f"UI Error: {e}")

        if not self.widget and self.node.params:
            self.widget = QWidget()
            self.layout = QVBoxLayout()
            self.widget.setLayout(self.layout)
            self.widget.setStyleSheet("background-color: transparent; color: white;")
            for p_name, param in self.node.params.items(): self._create_widget(p_name, param)

        if self.widget:
            self.proxy.setWidget(self.widget)
            self.proxy.setPos(10, self.height)
            w_width = max(self.width - 20, self.widget.minimumSize().width())
            w_height = self.widget.minimumSize().height() if CustomUIClass else self.widget.sizeHint().height()
            self.proxy.resize(w_width, w_height)
            self.width = w_width + 20
            self.height += w_height + 10
            for item in self.output_items.values(): item.setX(self.width)

    def _create_widget(self, name, param):
        c = QWidget(); l = QHBoxLayout(c); l.setContentsMargins(0,0,0,0)
        if param.type == 'float':
            lbl = QLabel(f"{name}: {param.value:.2f}")
            slider = QSlider(Qt.Horizontal); slider.setRange(0, 1000)
            norm = (param.value - param.meta['min']) / (param.meta['max'] - param.meta['min'])
            slider.setValue(int(norm * 1000))
            def on_slide(val):
                f = param.meta['min'] + (val/1000.0)*(param.meta['max'] - param.meta['min'])
                self.controller.set_parameter(self.node.id, name, f)
                lbl.setText(f"{name}: {f:.2f}")
            slider.valueChanged.connect(on_slide)
            l.addWidget(lbl); self.layout.addWidget(c); self.layout.addWidget(slider)
        elif param.type == 'bool':
            chk = QCheckBox(name); chk.setChecked(param.value)
            chk.toggled.connect(lambda v: self.controller.set_parameter(self.node.id, name, v))
            self.layout.addWidget(chk)
        elif param.type == 'string':
            lbl = QLabel(name); line = QLineEdit(str(param.value))
            line.editingFinished.connect(lambda: self.controller.set_parameter(self.node.id, name, line.text()))
            self.layout.addWidget(lbl); self.layout.addWidget(line)
        elif param.type == 'menu':
            lbl = QLabel(name); combo = QComboBox(); combo.addItems(param.meta.get('items', []))
            combo.setCurrentIndex(int(param.value))
            combo.currentIndexChanged.connect(lambda idx: self.controller.set_parameter(self.node.id, name, idx))
            self.layout.addWidget(lbl); self.layout.addWidget(combo)

    def boundingRect(self): return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        painter.setBrush(QColor(40, 40, 40))
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.drawRoundedRect(self.boundingRect(), 5, 5)
        grad = QLinearGradient(0, 0, 0, HEADER_HEIGHT)
        grad.setColorAt(0, QColor(60, 60, 60)); grad.setColorAt(1, QColor(50, 50, 50))
        painter.setBrush(grad)
        painter.drawRoundedRect(0, 0, self.width, HEADER_HEIGHT, 5, 5)
        painter.setPen(QColor("white"))
        title = self.node.name
        if isinstance(self.node, IClockProvider) and self.node.is_master:
            title += " [MASTER]"
        painter.drawText(QRectF(0, 0, self.width, HEADER_HEIGHT), Qt.AlignCenter, title)

    def contextMenuEvent(self, event):
        menu = QMenu()
        if isinstance(self.node, IClockProvider):
            act_clock = menu.addAction("Set as Master Clock")
            act_clock.setCheckable(True)
            act_clock.setChecked(self.node.is_master)
            act_clock.triggered.connect(lambda: self.controller.set_master_clock(self.node.id))
            menu.addSeparator()
        act_del = menu.addAction("Delete")
        act_del.triggered.connect(lambda: self.controller.delete_node(self.node.id))
        menu.exec(event.screenPos())

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node.pos = (value.x(), value.y())
            self.positionChanged.emit(self.node.id, value)
        return super().itemChange(change, value)

class GraphScene(QGraphicsScene):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.node_items = {} 
        self.wire_items = {}
        self.drag_start = None
        self.temp_wire = None
        self.controller.graphUpdated.connect(self.reconcile, Qt.QueuedConnection)

    def reconcile(self):
        graph = self.controller.graph
        logic_ids = {n.id for n in graph.nodes}
        ui_ids = set(self.node_items.keys())
        for nid in ui_ids - logic_ids: self.removeItem(self.node_items.pop(nid))
        for nid in logic_ids - ui_ids:
            node = graph.node_map[nid]
            item = NodeItem(node, self.controller)
            item.setPos(*node.pos)
            self.addItem(item)
            self.node_items[nid] = item

        logic_conns = set()
        for dst in graph.nodes:
            for d_p, inp in dst.inputs.items():
                if inp.connected_output:
                    out = inp.connected_output
                    logic_conns.add((out.parent.id, out.name, dst.id, d_p))
        
        ui_keys = set(self.wire_items.keys())
        for k in ui_keys - logic_conns: self.removeItem(self.wire_items.pop(k))
        for k in logic_conns - ui_keys:
            sid, sp, did, dp = k
            if sid in self.node_items and did in self.node_items:
                s_item, d_item = self.node_items[sid], self.node_items[did]
                if sp in s_item.output_items and dp in d_item.input_items:
                    s_sock = s_item.output_items[sp]
                    d_sock = d_item.input_items[dp]
                    wire = ConnectionItem(s_sock, d_sock, k)
                    self.addItem(wire)
                    self.wire_items[k] = wire
                    def upd(*args, w=wire): w.update_path()
                    s_item.positionChanged.connect(upd)
                    d_item.positionChanged.connect(upd)
        
        self.update()

class GraphView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        item = self.itemAt(pos)
        if isinstance(item, SocketItem):
            self.scene().drag_start = item
            self.setDragMode(QGraphicsView.NoDrag)
            self.scene().temp_wire = ConnectionItem(item, self.mapToScene(pos))
            self.scene().addItem(self.scene().temp_wire)
        else: super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.scene().temp_wire:
            self.scene().temp_wire.end_item = self.mapToScene(event.position().toPoint())
            self.scene().temp_wire.update_path()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.scene().temp_wire:
            end = self.itemAt(event.position().toPoint())
            start = self.scene().drag_start
            if isinstance(end, SocketItem) and start:
                if start.is_input != end.is_input and start.parentItem() != end.parentItem():
                    src = start if not start.is_input else end
                    dst = end if end.is_input else start
                    self.scene().controller.connect_nodes(
                        src.slot_ref.parent.id, src.slot_ref.name,
                        dst.slot_ref.parent.id, dst.slot_ref.name
                    )
            self.scene().removeItem(self.scene().temp_wire)
            self.scene().temp_wire = None
            self.scene().drag_start = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            for item in self.scene().selectedItems():
                if isinstance(item, NodeItem):
                    self.scene().controller.delete_node(item.node.id)
                elif isinstance(item, ConnectionItem) and item.logic_key:
                    sid, sp, did, dp = item.logic_key
                    self.scene().controller.disconnect_nodes(did, dp)
        super().keyPressEvent(event)