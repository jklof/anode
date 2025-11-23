from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QMenu,
    QGraphicsProxyWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
    QSlider,
    QHBoxLayout,
    QGraphicsScene,
    QGraphicsView,
    QCheckBox,
    QComboBox,
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QSignalBlocker, Slot
from PySide6.QtGui import (
    QPainter,
    QPen,
    QColor,
    QPainterPath,
    QLinearGradient,
    QCursor,
    QTransform,
    QPainterPathStroker,
)
import plugin_system

NODE_WIDTH = 160
HEADER_HEIGHT = 30
SOCKET_RADIUS = 6


class NodeProxy:
    """
    Helper object passed to Custom UI Widgets.
    """

    def __init__(self, node_id, controller, monitor_queue):
        self.node_id = node_id
        self.controller = controller
        self.monitor_queue = monitor_queue

    def set_parameter(self, name, value):
        self.controller.set_parameter(self.node_id, name, value)

    def update_queue(self, new_queue):
        self.monitor_queue = new_queue


class SocketItem(QGraphicsItem):
    def __init__(self, parent, name, is_input, node_id):
        super().__init__(parent)
        self.name = name
        self.is_input = is_input
        self.node_id = node_id
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        self.setCursor(QCursor(Qt.CrossCursor))
        self._base_color = QColor("#ff9900") if is_input else QColor("#00ccff")
        self._current_color = self._base_color

    def boundingRect(self):
        pad = 4
        return QRectF(-SOCKET_RADIUS - pad, -SOCKET_RADIUS - pad, (SOCKET_RADIUS + pad) * 2, (SOCKET_RADIUS + pad) * 2)

    def paint(self, painter, option, widget):
        painter.setBrush(self._current_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(0, 0), SOCKET_RADIUS, SOCKET_RADIUS)
        painter.setPen(QColor("white"))
        text_rect = QRectF(15 if self.is_input else -105, -10, 90, 20)
        align = Qt.AlignLeft if self.is_input else Qt.AlignRight
        painter.drawText(text_rect, align | Qt.AlignVCenter, self.name)

    def hoverEnterEvent(self, e):
        self._current_color = QColor("white")
        self.update()

    def hoverLeaveEvent(self, e):
        self._current_color = self._base_color
        self.update()


class ConnectionItem(QGraphicsPathItem):
    def __init__(self, start_item, end_item, logic_key=None):
        super().__init__()
        self.setZValue(-1)
        self.setAcceptedMouseButtons(Qt.RightButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.start_item = start_item
        self.end_item = end_item
        self.logic_key = logic_key

        # Store references to parents to manage signals
        self.start_node = start_item.parentItem() if isinstance(start_item, QGraphicsItem) else None
        self.end_node = end_item.parentItem() if isinstance(end_item, QGraphicsItem) else None

        # Connect signals locally
        if self.start_node and hasattr(self.start_node, "positionChanged"):
            self.start_node.positionChanged.connect(self.update_path)
        if self.end_node and hasattr(self.end_node, "positionChanged"):
            self.end_node.positionChanged.connect(self.update_path)

        self.update_path()

    def detach(self):
        """Clean up signal connections before deletion."""
        try:
            if self.start_node and hasattr(self.start_node, "positionChanged"):
                self.start_node.positionChanged.disconnect(self.update_path)
        except (RuntimeError, TypeError):
            pass

        try:
            if self.end_node and hasattr(self.end_node, "positionChanged"):
                self.end_node.positionChanged.disconnect(self.update_path)
        except (RuntimeError, TypeError):
            pass

    def update_path(self):
        if isinstance(self.start_item, QGraphicsItem):
            if not self.start_item.scene():
                return
            p1 = self.start_item.scenePos()
            start_is_input = getattr(self.start_item, "is_input", False)
        elif isinstance(self.start_item, (QPointF, list, tuple)):
            p1 = self.start_item
            start_is_input = False
        else:
            return

        if isinstance(self.end_item, QGraphicsItem):
            if not self.end_item.scene():
                return
            p2 = self.end_item.scenePos()
            end_is_input = getattr(self.end_item, "is_input", True)
        elif isinstance(self.end_item, (QPointF, list, tuple)):
            p2 = self.end_item
            end_is_input = not start_is_input
        else:
            return

        path = QPainterPath()
        path.moveTo(p1)
        dist = max(abs(p1.x() - p2.x()) * 0.5, 50.0)
        cp1 = -dist if start_is_input else dist
        cp2 = -dist if end_is_input else dist
        path.cubicTo(QPointF(p1.x() + cp1, p1.y()), QPointF(p2.x() + cp2, p2.y()), p2)
        self.setPath(path)

    def paint(self, p, o, w):
        p.setPen(QPen(QColor("yellow") if self.isSelected() else QColor("white"), 3 if self.isSelected() else 2))
        p.drawPath(self.path())

    def contextMenuEvent(self, event):
        menu = QMenu()
        action_del = menu.addAction("Delete Connection")
        if self.logic_key:
            sid, sp, did, dp = self.logic_key
            action_del.triggered.connect(lambda: self.scene().controller.disconnect_nodes(sid, sp, did, dp))
        menu.exec(event.screenPos())
        event.accept()

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(10)
        return stroker.createStroke(self.path())


class NodeItem(QGraphicsObject):
    positionChanged = Signal()

    def __init__(self, node_data, controller):
        super().__init__()
        self.error_msg = None
        self.nid = node_data["id"]
        self.node_type = node_data["type"]
        self.node_name = node_data["name"]
        self.params = node_data["params"]
        self.monitor_queue = node_data["monitor_queue"]
        self.controller = controller

        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.param_controls = {}
        self._processing_load = 0.0
        self._show_load = False
        self.proxy = None

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setCursor(QCursor(Qt.SizeAllCursor))

        self.width = NODE_WIDTH
        self.input_items = {}
        self.output_items = {}

        y = HEADER_HEIGHT + 10
        for name in node_data["inputs"]:
            item = SocketItem(self, name, True, self.nid)
            item.setPos(0, y)
            self.input_items[name] = item
            y += 20

        y_out = HEADER_HEIGHT + 10
        for name in node_data["outputs"]:
            item = SocketItem(self, name, False, self.nid)
            item.setPos(self.width, y_out)
            self.output_items[name] = item
            y_out += 20

        self.height = max(y, y_out) + 10

        self.proxy = QGraphicsProxyWidget(self)
        self.widget = None
        CustomUIClass = plugin_system.get_ui_class(self.node_type)

        if CustomUIClass:
            self.proxy_obj = NodeProxy(self.nid, self.controller, self.monitor_queue)
            self.widget = CustomUIClass(self.proxy_obj)

        if not self.widget and self.params:
            self.widget = QWidget()
            self.layout = QVBoxLayout()
            self.widget.setLayout(self.layout)
            self.widget.setStyleSheet("background-color: transparent; color: white;")
            for p_name, p_data in self.params.items():
                self._create_param_widget(p_name, p_data)

        if self.widget:
            self.proxy.setWidget(self.widget)
            self.proxy.setPos(10, self.height)
            w_width = max(self.width - 20, self.widget.minimumSize().width())
            w_height = self.widget.minimumSize().height() if CustomUIClass else self.widget.sizeHint().height()
            self.proxy.resize(w_width, w_height)
            self.width = w_width + 20
            self.height += w_height + 10
            for item in self.output_items.values():
                item.setX(self.width)

    def _create_param_widget(self, name, p_data):
        c = QWidget()
        l = QHBoxLayout(c)
        l.setContentsMargins(0, 0, 0, 0)
        ptype = p_data["type"]
        meta = p_data["meta"]
        val = p_data["value"]
        control_ref = {"type": ptype, "meta": meta}

        if ptype == "float":
            lbl = QLabel(f"{name}: {val:.2f}")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 1000)
            norm = (val - meta["min"]) / (meta["max"] - meta["min"])
            slider.setValue(int(norm * 1000))

            def on_slide(v):
                f = meta["min"] + (v / 1000.0) * (meta["max"] - meta["min"])
                self.controller.set_parameter(self.nid, name, f)
                lbl.setText(f"{name}: {f:.2f}")

            slider.valueChanged.connect(on_slide)
            l.addWidget(lbl)
            self.layout.addWidget(c)
            self.layout.addWidget(slider)
            control_ref["widget"] = slider
            control_ref["label"] = lbl

        elif ptype == "bool":
            chk = QCheckBox(name)
            chk.setChecked(val)
            chk.toggled.connect(lambda v: self.controller.set_parameter(self.nid, name, v))
            self.layout.addWidget(chk)
            control_ref["widget"] = chk

        elif ptype == "menu":
            lbl = QLabel(name)
            combo = QComboBox()
            combo.addItems(meta.get("items", []))
            combo.setCurrentIndex(int(val))
            combo.currentIndexChanged.connect(lambda idx: self.controller.set_parameter(self.nid, name, idx))
            self.layout.addWidget(lbl)
            self.layout.addWidget(combo)
            control_ref["widget"] = combo

        self.param_controls[name] = control_ref

    def update_from_snapshot(self, node_data):
        new_pos = QPointF(*node_data["pos"])
        if self.pos() != new_pos:
            self.setPos(new_pos)
            self.update()

        # CRITICAL FIX: Check if monitor_queue object changed (due to reload/load)
        new_queue = node_data.get("monitor_queue")
        if new_queue is not None and self.monitor_queue is not new_queue:
            self.monitor_queue = new_queue
            if hasattr(self, "proxy_obj") and self.proxy_obj:
                self.proxy_obj.update_queue(new_queue)

        new_params = node_data["params"]

        # Standard Widgets
        for name, control in self.param_controls.items():
            if name in new_params:
                new_val = new_params[name]["value"]
                widget = control["widget"]
                with QSignalBlocker(widget):
                    if control["type"] == "float":
                        meta = control["meta"]
                        norm = (new_val - meta["min"]) / (meta["max"] - meta["min"])
                        if not widget.isSliderDown():
                            widget.setValue(int(norm * 1000))
                            if "label" in control:
                                control["label"].setText(f"{name}: {new_val:.2f}")
                    elif control["type"] == "bool":
                        widget.setChecked(bool(new_val))
                    elif control["type"] == "menu":
                        if not widget.view().isVisible():
                            widget.setCurrentIndex(int(new_val))

        # Custom Widgets
        if self.widget and hasattr(self.widget, "update_from_params"):
            simple_params = {k: v["value"] for k, v in new_params.items()}
            self.widget.update_from_params(simple_params)

        self.error_msg = node_data.get("error")
        self.setToolTip(self.error_msg if self.error_msg else self.node_name)

        self.update()

    def set_processing_load(self, pct):
        self._processing_load = pct
        self.update()

    def set_show_load(self, show):
        self._show_load = show
        self.update()

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        painter.setBrush(QColor(40, 40, 40))
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.drawRoundedRect(self.boundingRect(), 5, 5)
        grad = QLinearGradient(0, 0, 0, HEADER_HEIGHT)
        if self.error_msg:
            grad.setColorAt(0, QColor(100, 60, 60))
            grad.setColorAt(1, QColor(80, 50, 50))
        else:
            grad.setColorAt(0, QColor(60, 60, 60))
            grad.setColorAt(1, QColor(50, 50, 50))
        painter.setBrush(grad)
        painter.drawRoundedRect(0, 0, self.width, HEADER_HEIGHT, 5, 5)

        if self._show_load and self._processing_load > 0:
            pct = min(self._processing_load, 100.0)
            bar_width = self.width * (pct / 100.0)
            color = QColor(0, 255, 0, 100)
            if pct > 50:
                color = QColor(255, 255, 0, 100)
            if pct > 85:
                color = QColor(255, 0, 0, 120)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRect(0, 0, bar_width, HEADER_HEIGHT)

        painter.setPen(QColor("white"))
        painter.drawText(QRectF(0, 0, self.width, HEADER_HEIGHT), Qt.AlignCenter, self.node_name)

        if self.error_msg:
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.boundingRect(), 5, 5)

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.addAction("Set Master Clock", lambda: self.controller.set_master_clock(self.nid))
        menu.addAction("Delete", lambda: self.controller.delete_node(self.nid))
        menu.exec(event.screenPos())

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.controller.move_node(self.nid, (value.x(), value.y()))
            self.positionChanged.emit()
        return super().itemChange(change, value)


class GraphScene(QGraphicsScene):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.node_items = {}
        self.wire_items = {}
        self._last_reload_version = 0
        self.drag_start = None
        self.temp_wire = None
        self._show_load = False
        self.controller.graphUpdated.connect(self.reconcile)
        self.controller.statsUpdated.connect(self.on_stats_updated)

    def reconcile(self, snapshot: dict):
        reload_version = snapshot.get("reload_version", 0)
        if reload_version != self._last_reload_version:
            self._last_reload_version = reload_version
            for item in list(self.wire_items.values()):
                item.detach()
                self.removeItem(item)
            self.wire_items.clear()
            for item in list(self.node_items.values()):
                self.removeItem(item)
            self.node_items.clear()

        snap_nodes = snapshot["nodes"]
        snap_ids = {n["id"] for n in snap_nodes}
        snap_conns = set()
        for c in snapshot["connections"]:
            snap_conns.add((c["src_id"], c["src_port"], c["dst_id"], c["dst_port"]))

        ui_keys = set(self.wire_items.keys())
        for k in ui_keys - snap_conns:
            wire = self.wire_items.pop(k)
            wire.detach()
            self.removeItem(wire)

        ui_ids = set(self.node_items.keys())
        for nid in ui_ids - snap_ids:
            self.removeItem(self.node_items.pop(nid))

        for n_data in snap_nodes:
            nid = n_data["id"]
            if nid not in self.node_items:
                item = NodeItem(n_data, self.controller)
                item.setPos(*n_data["pos"])
                item.set_show_load(self._show_load)
                self.addItem(item)
                self.node_items[nid] = item
            else:
                self.node_items[nid].update_from_snapshot(n_data)

        ui_keys = set(self.wire_items.keys())
        for k in snap_conns - ui_keys:
            sid, sp, did, dp = k
            if sid in self.node_items and did in self.node_items:
                s_item = self.node_items[sid]
                d_item = self.node_items[did]
                if sp in s_item.output_items and dp in d_item.input_items:
                    wire = ConnectionItem(s_item.output_items[sp], d_item.input_items[dp], k)
                    self.addItem(wire)
                    self.wire_items[k] = wire
        self.update()

    def on_stats_updated(self, stats):
        for nid, load in stats.items():
            if nid in self.node_items:
                self.node_items[nid].set_processing_load(load)

    def toggle_load_view(self, show):
        self._show_load = show
        for item in self.node_items.values():
            item.set_show_load(show)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.scenePos(), QTransform())
        if item:
            super().contextMenuEvent(event)
            return
        menu = QMenu()
        add_menu = menu.addMenu("Add Node")
        click_pos = event.scenePos()
        for name in sorted(plugin_system.NODE_REGISTRY.keys()):
            action = add_menu.addAction(name)
            action.triggered.connect(
                lambda c=False, n=name, p=(click_pos.x(), click_pos.y()): self.controller.add_node(n, p)
            )
        menu.exec(event.screenPos())
        event.accept()


class GraphView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            super().wheelEvent(event)
        else:
            zoom = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(zoom, zoom)
            event.accept()

    @Slot()
    def zoom_in(self):
        self.scale(1.2, 1.2)

    @Slot()
    def zoom_out(self):
        self.scale(1 / 1.2, 1 / 1.2)

    @Slot()
    def zoom_to_fit(self):
        if self.scene().items():
            self.fitInView(self.scene().itemsBoundingRect().adjusted(-50, -50, 50, 50), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        item = self.itemAt(pos)
        if isinstance(item, SocketItem):
            self.scene().drag_start = item
            self.setDragMode(QGraphicsView.NoDrag)
            self.scene().temp_wire = ConnectionItem(item, self.mapToScene(pos))
            self.scene().addItem(self.scene().temp_wire)
        else:
            super().mousePressEvent(event)

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
                if start.is_input != end.is_input and start.node_id != end.node_id:
                    src = start if not start.is_input else end
                    dst = end if end.is_input else start
                    self.scene().controller.connect_nodes(src.node_id, src.name, dst.node_id, dst.name)
            self.scene().removeItem(self.scene().temp_wire)
            self.scene().temp_wire = None
            self.scene().drag_start = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            for item in self.scene().selectedItems():
                if isinstance(item, NodeItem):
                    self.scene().controller.delete_node(item.nid)
                elif isinstance(item, ConnectionItem) and item.logic_key:
                    sid, sp, did, dp = item.logic_key
                    self.scene().controller.disconnect_nodes(sid, sp, did, dp)
        super().keyPressEvent(event)
