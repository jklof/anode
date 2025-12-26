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
    QLineEdit,
    QSpinBox,
    QPushButton,
    QFileDialog,
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QSignalBlocker, Slot, QLineF, QCoreApplication
from PySide6.QtGui import (
    QPainter,
    QPen,
    QColor,
    QPainterPath,
    QLinearGradient,
    QCursor,
    QTransform,
    QPainterPathStroker,
    QMouseEvent,
    QPixmap,
    QBrush,
    QKeySequence,
)
from PySide6.QtSvg import QSvgRenderer
from ui_icons import create_colored_logo
from theme import Theme

import plugin_system
import os
import math
import json
import uuid
import base


class NodeProxy:
    """
    Helper object passed to Custom UI Widgets.
    """

    def __init__(self, node_id, controller, monitor_queue, node_item):
        self.node_id = node_id
        self.controller = controller
        self.monitor_queue = monitor_queue
        self.node_item = node_item

    def set_parameter(self, name, value):
        self.controller.set_parameter(self.node_id, name, value)

    def update_queue(self, new_queue):
        self.monitor_queue = new_queue

    def create_param_widget(self, param_name):
        """
        Create a smart parameter widget for the given parameter name.

        Args:
            param_name: Name of the parameter to create a widget for

        Returns:
            QWidget: The appropriate smart widget instance
        """
        if param_name not in self.node_item.params:
            raise ValueError(f"Parameter '{param_name}' not found in node parameters")

        p_data = self.node_item.params[param_name]
        ptype = p_data["type"]
        meta = p_data["meta"]
        val = p_data["value"]

        def callback(new_value):
            self.controller.set_parameter(self.node_id, param_name, new_value)

        return ParamWidgetFactory.create(param_name, ptype, meta, val, callback)


class SocketItem(QGraphicsItem):
    def __init__(self, parent, name, is_input, node_id):
        super().__init__(parent)
        self.name = name
        self.is_input = is_input
        self.node_id = node_id
        self.setAcceptHoverEvents(True)
        self.setZValue(Theme.Z_LAYERS["socket"])
        self.setCursor(QCursor(Qt.CrossCursor))
        self._base_color = Theme.COLORS["socket_input"] if is_input else Theme.COLORS["socket_output"]
        self._hovered = False

    def boundingRect(self):
        pad = 10
        return QRectF(
            -Theme.DIMENSIONS["socket_radius"] - pad,
            -Theme.DIMENSIONS["socket_radius"] - pad,
            (Theme.DIMENSIONS["socket_radius"] + pad) * 2,
            (Theme.DIMENSIONS["socket_radius"] + pad) * 2,
        )

    def paint(self, painter, option, widget):
        if self._hovered:
            painter.save()
            painter.scale(1.3, 1.3)
            painter.setBrush(self._base_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(0, 0), Theme.DIMENSIONS["socket_radius"], Theme.DIMENSIONS["socket_radius"])
            painter.restore()
        else:
            painter.setBrush(self._base_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(0, 0), Theme.DIMENSIONS["socket_radius"], Theme.DIMENSIONS["socket_radius"])
        painter.setPen(Theme.COLORS["text_normal"])
        text_rect = QRectF(15 if self.is_input else -105, -10, 90, 20)
        align = Qt.AlignLeft if self.is_input else Qt.AlignRight
        painter.drawText(text_rect, align | Qt.AlignVCenter, self.name)

    def hoverEnterEvent(self, e):
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, e):
        self._hovered = False
        self.update()


class ConnectionItem(QGraphicsPathItem):
    def __init__(self, start_item, end_item, logic_key=None):
        super().__init__()
        self.setZValue(Theme.Z_LAYERS["wire"])
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.hovered = False
        self.start_item = start_item
        self.end_item = end_item
        self.logic_key = logic_key
        self.temp_mode = False
        self.temp_color = None

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

        # Store positions for gradient in paint method
        self.p1 = p1
        self.p2 = p2

        path = QPainterPath()
        path.moveTo(p1)
        # Implement Tangent approach with dynamic curvature
        curvature = abs(p2.x() - p1.x()) * 0.3
        cp1 = QPointF(p1.x() + curvature, p1.y())
        cp2 = QPointF(p2.x() - curvature, p2.y())
        path.cubicTo(cp1, cp2, p2)
        self.setPath(path)

    def paint(self, p, o, w):
        if self.temp_mode:
            if self.temp_color == Theme.COLORS["wire_temp_white"]:
                pen = QPen(Theme.COLORS["wire_temp_white"], 2, Qt.DashLine)
            elif self.temp_color == Theme.COLORS["wire_temp_red"]:
                pen = QPen(Theme.COLORS["wire_temp_red"], 3)
            elif self.temp_color == Theme.COLORS["wire_temp_green"]:
                pen = QPen(Theme.COLORS["wire_temp_green"], 3)
            else:
                pen = QPen(Theme.COLORS["wire_temp_white"], 2, Qt.DashLine)
        elif self.isSelected():
            pen = QPen(Theme.COLORS["wire_selected"], 3)
        elif self.hovered:
            pen = QPen(Theme.COLORS["wire_hovered"], 4)
        else:
            # Use gradient for normal wires to visualize signal flow
            gradient = QLinearGradient(self.p1, self.p2)
            gradient.setColorAt(0, Theme.COLORS["socket_output"])  # Output color at start
            gradient.setColorAt(1, Theme.COLORS["socket_input"])  # Input color at end
            pen = QPen()
            pen.setBrush(gradient)
            pen.setWidth(2)
        p.setPen(pen)
        p.drawPath(self.path())

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()

    def mouseDoubleClickEvent(self, event):
        if self.logic_key:
            sid, sp, did, dp = self.logic_key
            self.scene().controller.disconnect_nodes(sid, sp, did, dp)

    def contextMenuEvent(self, event):
        menu = QMenu()
        action_del = menu.addAction("Delete Connection")

        def delete_action():
            if self.isSelected():
                self.scene().delete_selection()
            elif self.logic_key:
                sid, sp, did, dp = self.logic_key
                self.scene().controller.disconnect_nodes(sid, sp, did, dp)

        if self.logic_key:
            action_del.triggered.connect(delete_action)
        menu.exec(event.screenPos())
        event.accept()

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(20)
        return stroker.createStroke(self.path())


class FloatParamWidget(QWidget):
    """Smart widget for float parameters with slider and label."""

    def __init__(self, param_name, metadata, current_value, callback):
        super().__init__()
        self.param_name = param_name
        self.metadata = metadata
        self.current_value = current_value
        self.callback = callback

        # Layout setup
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # Label
        self.label = QLabel(f"{param_name}: {current_value:.2f}")
        self.layout.addWidget(self.label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self._update_slider_from_value(current_value)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.layout.addWidget(self.slider)

    def _update_slider_from_value(self, value):
        """Update slider position based on float value."""
        norm = (value - self.metadata["min"]) / (self.metadata["max"] - self.metadata["min"])
        self.slider.setValue(int(norm * 1000))

    def _on_slider_changed(self, value):
        """Handle slider value changes."""
        f = self.metadata["min"] + (value / 1000.0) * (self.metadata["max"] - self.metadata["min"])
        self.callback(f)
        self.label.setText(f"{self.param_name}: {f:.2f}")
        self.current_value = f

    def update_from_backend(self, new_value):
        """Update widget from backend value changes."""
        if abs(self.current_value - new_value) < 1e-6:  # Check if value actually changed
            return

        self.current_value = new_value

        # Check if slider is being dragged to prevent fighting the user
        if not self.slider.isSliderDown():
            with QSignalBlocker(self.slider):
                self._update_slider_from_value(new_value)
                self.label.setText(f"{self.param_name}: {new_value:.2f}")


class BoolParamWidget(QWidget):
    """Smart widget for boolean parameters with checkbox."""

    def __init__(self, param_name, metadata, current_value, callback):
        super().__init__()
        self.param_name = param_name
        self.metadata = metadata
        self.current_value = bool(current_value)
        self.callback = callback

        # Layout setup
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # Checkbox
        self.checkbox = QCheckBox(param_name)
        self.checkbox.setChecked(self.current_value)
        self.checkbox.toggled.connect(self._on_checkbox_toggled)
        self.layout.addWidget(self.checkbox)

    def _on_checkbox_toggled(self, checked):
        """Handle checkbox state changes."""
        self.callback(checked)

    def update_from_backend(self, new_value):
        """Update widget from backend value changes."""
        new_value = bool(new_value)
        if self.current_value == new_value:
            return

        self.current_value = new_value

        with QSignalBlocker(self.checkbox):
            self.checkbox.setChecked(new_value)


class MenuParamWidget(QWidget):
    """Smart widget for menu parameters with combo box."""

    def __init__(self, param_name, metadata, current_value, callback):
        super().__init__()
        self.param_name = param_name
        self.metadata = metadata
        self.current_value = int(current_value)
        self.callback = callback

        # Layout setup
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # Label
        self.label = QLabel(param_name)
        self.layout.addWidget(self.label)

        # Combo box
        self.combo = QComboBox()
        self.combo.addItems(metadata.get("items", []))
        self.combo.setCurrentIndex(self.current_value)
        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        self.layout.addWidget(self.combo)

    def _on_combo_changed(self, index):
        """Handle combo box index changes."""
        self.callback(index)

    def update_from_backend(self, new_value):
        """Update widget from backend value changes."""
        new_value = int(new_value)
        if self.current_value == new_value:
            return

        self.current_value = new_value

        # Check if combo box dropdown is visible to prevent fighting the user
        if not self.combo.view().isVisible():
            with QSignalBlocker(self.combo):
                self.combo.setCurrentIndex(new_value)


class FileParamWidget(QWidget):
    """Smart widget for file parameters with line edit and browse button."""

    def __init__(self, param_name, metadata, current_value, callback):
        super().__init__()
        self.param_name = param_name
        self.metadata = metadata
        self.current_value = str(current_value)
        self.callback = callback

        # Main layout (vertical)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        self.setLayout(self.layout)

        # Label
        self.label = QLabel(param_name)
        self.label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.layout.addWidget(self.label)

        # Container for line edit and button
        self.container = QWidget()
        self.hbox = QHBoxLayout(self.container)
        self.hbox.setContentsMargins(0, 0, 0, 0)
        self.hbox.setSpacing(2)

        # Line edit
        self.line_edit = QLineEdit(self.current_value)
        self.line_edit.setMinimumWidth(200)
        self.line_edit.setToolTip(self.current_value)
        self.line_edit.editingFinished.connect(self._on_editing_finished)
        self.hbox.addWidget(self.line_edit)

        # Browse button
        self.button = QPushButton("...")
        self.button.setFixedWidth(25)
        self.button.setFixedHeight(22)
        self.button.clicked.connect(self._on_browse_clicked)
        self.hbox.addWidget(self.button)

        self.layout.addWidget(self.container)

    def _on_editing_finished(self):
        """Handle line edit text changes."""
        text = self.line_edit.text()
        self.callback(text)

    def _on_browse_clicked(self):
        """Handle browse button click."""
        start = ""
        curr = self.line_edit.text()
        if curr and os.path.exists(curr):
            start = curr if os.path.isdir(curr) else os.path.dirname(curr)

        filt = self.metadata.get("filter", "All Files (*.*)")
        if self.metadata.get("mode") == "save":
            path, _ = QFileDialog.getSaveFileName(None, f"Save {self.param_name}", start, filt)
        else:
            path, _ = QFileDialog.getOpenFileName(None, f"Open {self.param_name}", start, filt)

        if path:
            self.line_edit.setText(path)
            self.line_edit.setToolTip(path)
            self.line_edit.setCursorPosition(len(path))
            self.callback(path)

    def update_from_backend(self, new_value):
        """Update widget from backend value changes."""
        new_value = str(new_value)
        if self.current_value == new_value:
            return

        self.current_value = new_value

        # Check if line edit has focus to prevent fighting the user
        if not self.line_edit.hasFocus():
            with QSignalBlocker(self.line_edit):
                self.line_edit.setText(new_value)
                self.line_edit.setToolTip(new_value)
                self.line_edit.setCursorPosition(len(new_value))


class StringParamWidget(QWidget):
    """Smart widget for string parameters with line edit."""

    def __init__(self, param_name, metadata, current_value, callback):
        super().__init__()
        self.param_name = param_name
        self.metadata = metadata
        self.current_value = str(current_value)
        self.callback = callback

        # Layout setup
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # Label
        self.label = QLabel(param_name)
        self.layout.addWidget(self.label)

        # Line edit
        self.line_edit = QLineEdit(self.current_value)
        self.line_edit.returnPressed.connect(self._on_return_pressed)
        self.layout.addWidget(self.line_edit)

    def _on_return_pressed(self):
        """Handle line edit return key press."""
        text = self.line_edit.text()
        self.callback(text)

    def update_from_backend(self, new_value):
        """Update widget from backend value changes."""
        new_value = str(new_value)
        if self.current_value == new_value:
            return

        self.current_value = new_value

        # Check if line edit has focus to prevent fighting the user
        if not self.line_edit.hasFocus():
            with QSignalBlocker(self.line_edit):
                self.line_edit.setText(new_value)


class IntParamWidget(QWidget):
    """Smart widget for integer parameters with spin box."""

    def __init__(self, param_name, metadata, current_value, callback):
        super().__init__()
        self.param_name = param_name
        self.metadata = metadata
        self.current_value = int(current_value)
        self.callback = callback

        # Layout setup
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # Label
        self.label = QLabel(param_name)
        self.layout.addWidget(self.label)

        # Spin box
        self.spin_box = QSpinBox()
        self.spin_box.setRange(metadata.get("min", 0), metadata.get("max", 100))
        self.spin_box.setValue(self.current_value)
        self.spin_box.valueChanged.connect(self._on_value_changed)
        self.layout.addWidget(self.spin_box)

    def _on_value_changed(self, value):
        """Handle spin box value changes."""
        self.callback(value)

    def update_from_backend(self, new_value):
        """Update widget from backend value changes."""
        new_value = int(new_value)
        if self.current_value == new_value:
            return

        self.current_value = new_value

        with QSignalBlocker(self.spin_box):
            self.spin_box.setValue(new_value)


class ParamWidgetFactory:
    """Factory for creating smart parameter widgets."""

    @staticmethod
    def create(param_name, param_type, metadata, current_value, callback):
        """
        Create the appropriate smart widget for the given parameter.

        Args:
            param_name: Name of the parameter
            param_type: Type of the parameter (e.g., "float", "bool", "menu", etc.)
            metadata: Metadata dictionary for the parameter
            current_value: Current value of the parameter
            callback: Function to call when the parameter value changes

        Returns:
            QWidget: The appropriate smart widget instance
        """
        if param_type == "float":
            return FloatParamWidget(param_name, metadata, current_value, callback)
        elif param_type == "bool":
            return BoolParamWidget(param_name, metadata, current_value, callback)
        elif param_type == "menu":
            return MenuParamWidget(param_name, metadata, current_value, callback)
        elif param_type == "file":
            return FileParamWidget(param_name, metadata, current_value, callback)
        elif param_type == "string":
            return StringParamWidget(param_name, metadata, current_value, callback)
        elif param_type == "int":
            return IntParamWidget(param_name, metadata, current_value, callback)
        else:
            # Fallback for unsupported types
            widget = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(QLabel(f"Unsupported parameter type: {param_type}"))
            widget.setLayout(layout)
            return widget


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
        self.can_be_master = node_data.get("can_be_master", False)
        self.is_master = node_data.get("is_master", False)

        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.param_controls = {}
        self._processing_load = 0.0
        self._show_load = False
        self.proxy = None
        self.widget = None

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setCursor(QCursor(Qt.SizeAllCursor))

        # Initialize position cache
        self._last_committed_pos = self.pos()

        self.width = Theme.DIMENSIONS["node_width"]
        self.input_items = {}
        self.output_items = {}

        # 1. Create Sockets Immediately
        y = Theme.DIMENSIONS["header_height"] + 10
        for name in node_data["inputs"]:
            item = SocketItem(self, name, True, self.nid)
            item.setPos(0, y)
            self.input_items[name] = item
            y += 20

        y_out = Theme.DIMENSIONS["header_height"] + 10
        for name in node_data["outputs"]:
            item = SocketItem(self, name, False, self.nid)
            item.setPos(self.width, y_out)
            self.output_items[name] = item
            y_out += 20

        # Calculate initial height based on sockets
        self.height = max(y, y_out) + 10

        # UI Construction (Proxy/Widget) is DEFERRED to build_ui()
        # This ensures the NodeItem is in the scene before QComboBox/Complex widgets init.

    def build_ui(self):
        """
        Called by GraphScene AFTER adding this item to the scene.
        """
        self.proxy = QGraphicsProxyWidget(self)
        self.widget = None
        CustomUIClass = plugin_system.get_ui_class(self.node_type)

        if CustomUIClass:
            self.proxy_obj = NodeProxy(self.nid, self.controller, self.monitor_queue, self)
            self.widget = CustomUIClass(self.proxy_obj)

        if not self.widget and self.params:
            self.widget = QWidget()
            self.widget.setObjectName("genericNodeContainer")  # Give it an ID
            self.layout = QVBoxLayout()
            self.widget.setLayout(self.layout)
            # NEW LINE: Target ID for transparency, use specific CSS for text color to avoid breaking complex widgets
            self.widget.setStyleSheet(Theme.STYLES["generic_node_container"])
            for p_name, p_data in self.params.items():
                ptype = p_data["type"]
                meta = p_data["meta"]
                val = p_data["value"]

                def callback(new_value, name=p_name):
                    self.controller.set_parameter(self.nid, name, new_value)

                widget = ParamWidgetFactory.create(p_name, ptype, meta, val, callback)
                self.layout.addWidget(widget)
                self.param_controls[p_name] = {"widget": widget, "type": ptype}

        if self.widget:
            self.proxy.setWidget(self.widget)
            self.proxy.setPos(10, self.height)
            w_width = max(self.width - 20, self.widget.minimumSize().width())
            w_height = self.widget.minimumSize().height() if CustomUIClass else self.widget.sizeHint().height()
            self.proxy.resize(w_width, w_height)

            # Expand Node if UI is wider/taller
            self.width = w_width + 20
            self.height += w_height + 10

            # 2. Relayout Output Sockets if width changed
            for item in self.output_items.values():
                item.setX(self.width)

    def update_from_snapshot(self, node_data):
        new_pos = QPointF(*node_data["pos"])
        if self.pos() != new_pos:
            self.setPos(new_pos)
            self.update()

        self.can_be_master = node_data.get("can_be_master", False)
        prev_master = self.is_master
        self.is_master = node_data.get("is_master", False)
        if prev_master != self.is_master:
            self.update()

        # CRITICAL FIX: Check if monitor_queue object changed (due to reload/load)
        new_queue = node_data.get("monitor_queue")
        if new_queue is not None and self.monitor_queue is not new_queue:
            self.monitor_queue = new_queue
            if hasattr(self, "proxy_obj") and self.proxy_obj:
                self.proxy_obj.update_queue(new_queue)

        new_params = node_data["params"]

        # Update Smart Widgets - only call widget.update_from_backend if value actually changed
        for name, control in self.param_controls.items():
            if name in new_params:
                new_val = new_params[name]["value"]
                # Compare with current value before calling widget update
                if name in self.params and self.params[name]["value"] != new_val:
                    widget = control["widget"]
                    if hasattr(widget, "update_from_backend"):
                        widget.update_from_backend(new_val)

        # Custom Widgets
        if self.widget and hasattr(self.widget, "update_from_params"):
            simple_params = {k: v["value"] for k, v in new_params.items()}
            self.widget.update_from_params(simple_params)

        # Cache the latest params data for thread-safe access by clipboard operations
        self.params = new_params

        self.error_msg = node_data.get("error")
        self.setToolTip(self.error_msg if self.error_msg else self.node_name)

        self.update()

        # Update position cache after setting position
        self._last_committed_pos = self.pos()

    def set_processing_load(self, pct):
        self._processing_load = pct
        self.update()

    def set_show_load(self, show):
        self._show_load = show
        self.update()

    def propagate_telemetry(self, data: dict):
        if "cpu_load" in data:
            self.set_processing_load(data["cpu_load"])
        if self.widget and hasattr(self.widget, "on_telemetry"):
            self.widget.on_telemetry(data)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        painter.setBrush(Theme.COLORS["node_bg"])
        painter.setPen(QPen(Theme.COLORS["node_border"], 1))
        painter.drawRoundedRect(self.boundingRect(), 5, 5)
        grad = QLinearGradient(0, 0, 0, Theme.DIMENSIONS["header_height"])
        if self.error_msg:
            grad.setColorAt(0, Theme.COLORS["header_error_start"])
            grad.setColorAt(1, Theme.COLORS["header_error_end"])
        else:
            grad.setColorAt(0, Theme.COLORS["header_normal_start"])
            grad.setColorAt(1, Theme.COLORS["header_normal_end"])
        painter.setBrush(grad)
        painter.drawRoundedRect(0, 0, self.width, Theme.DIMENSIONS["header_height"], 5, 5)

        if self._show_load and self._processing_load > 0:
            pct = min(self._processing_load, 100.0)
            bar_width = self.width * (pct / 100.0)
            if pct > 85:
                color = Theme.COLORS["load_high"]
            elif pct > 50:
                color = Theme.COLORS["load_medium"]
            else:
                color = Theme.COLORS["load_low"]
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRect(0, 0, bar_width, Theme.DIMENSIONS["header_height"])

        painter.setPen(Theme.COLORS["text_normal"])
        painter.drawText(QRectF(0, 0, self.width, Theme.DIMENSIONS["header_height"]), Qt.AlignCenter, self.node_name)

        # --- DRAW CLOCK ICON ---
        if self.can_be_master:
            icon_color = Theme.COLORS["clock_master"] if self.is_master else Theme.COLORS["clock_slave"]
            painter.setPen(QPen(icon_color, 1.5))
            painter.setBrush(Qt.NoBrush)

            # Position: Top right, padding
            cx = self.width - 15
            cy = Theme.DIMENSIONS["header_height"] / 2
            r = 6

            # Clock face
            painter.drawEllipse(QPointF(cx, cy), r, r)
            # Hands (3 o'clock and 12 o'clockish)
            painter.drawLine(QPointF(cx, cy), QPointF(cx, cy - 4))
            painter.drawLine(QPointF(cx, cy), QPointF(cx + 3, cy))

        if self.error_msg:
            painter.setPen(QPen(Theme.COLORS["error_border"], 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.boundingRect(), 5, 5)

        if self.isSelected():
            painter.setPen(QPen(Theme.COLORS["selection_outline"], 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.boundingRect(), 5, 5)

    def contextMenuEvent(self, event):
        menu = QMenu()
        if self.can_be_master:
            menu.addAction("Set Master Clock", lambda: self.controller.set_master_clock(self.nid))

        def delete_action():
            if self.isSelected():
                self.scene().delete_selection()
            else:
                self.controller.delete_node(self.nid)

        menu.addAction("Delete", delete_action)
        menu.exec(event.screenPos())

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.positionChanged.emit()
        elif change == QGraphicsItem.ItemSelectedHasChanged:
            self.setZValue(Theme.Z_LAYERS["node_selected"] if value else Theme.Z_LAYERS["node_normal"])
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        # Check if clicking the clock icon area (Top Right)
        if self.can_be_master:
            local_pos = event.pos()
            # Hit box: Top 30px (header), Rightmost 30px
            if local_pos.y() <= Theme.DIMENSIONS["header_height"] and local_pos.x() >= (self.width - 30):
                self.controller.set_master_clock(self.nid)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # Send move command only once when mouse is released
        # We calculate the delta based on THIS node's movement, then apply
        # that delta to find the previous position of all selected nodes.

        super().mouseReleaseEvent(event)

        # 1. Check if we actually moved
        if self.pos() == self._last_committed_pos:
            return

        # 2. Gather moves for all selected nodes
        moves = {}
        scene = self.scene()
        if scene:
            for item in scene.selectedItems():
                if isinstance(item, NodeItem):
                    # Current position (already updated by QGraphicsItem internals)
                    new_pos = (item.pos().x(), item.pos().y())

                    # Previous position
                    # We assume item._last_committed_pos holds the state before the drag started
                    prev_pos = (item._last_committed_pos.x(), item._last_committed_pos.y())

                    # Only record if it actually changed
                    if new_pos != prev_pos:
                        moves[item.nid] = (new_pos, prev_pos)
                        # Update the commit point so next drag starts fresh
                        item._last_committed_pos = item.pos()

        # 3. Send single batch command
        if moves:
            self.controller.move_nodes(moves)

    def update_single_param(self, param_name, value):
        """
        Update a single parameter efficiently without redrawing the whole graph.

        Args:
            param_name: Name of the parameter to update
            value: New value for the parameter
        """
        # First, update the internal data cache
        if param_name in self.params:
            self.params[param_name]["value"] = value

        # Check if param_name exists in param_controls (generic widgets)
        if param_name in self.param_controls:
            control = self.param_controls[param_name]["widget"]
            if hasattr(control, "update_from_backend"):
                control.update_from_backend(value)

        # Check if self.widget exists (custom UI)
        if self.widget and hasattr(self.widget, "update_from_params"):
            # Create a small dict with just the updated parameter
            update_dict = {param_name: value}
            self.widget.update_from_params(update_dict)


class GraphScene(QGraphicsScene):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.node_items = {}
        self.wire_items = {}
        self._last_reload_version = 0
        self.drag_start = None
        self.temp_wire = None
        self.drag_target = None
        self._show_load = False
        self.controller.graphUpdated.connect(self.reconcile)
        self.controller.telemetryUpdated.connect(self.on_telemetry_updated)
        self.controller.parameterUpdated.connect(self.on_parameter_update)

    def delete_selection(self):
        """
        Delete selected items using the batch controller method.
        """
        nodes_to_delete = []
        # Gather Nodes
        for item in self.selectedItems():
            if isinstance(item, NodeItem):
                nodes_to_delete.append(item.nid)

        # Gather Connections that are explicitly selected
        connections_to_delete = []
        for item in self.selectedItems():
            if isinstance(item, ConnectionItem) and item.logic_key:
                sid, sp, did, dp = item.logic_key
                connections_to_delete.append((sid, sp, did, dp))

        if not nodes_to_delete and not connections_to_delete:
            return

        self.controller.delete_selection(nodes_to_delete, connections_to_delete)

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

                # 1. Add item to scene (essential before building UI)
                self.addItem(item)

                # 2. Build UI now that scene is valid
                item.build_ui()

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

    def on_telemetry_updated(self, data):
        node_data = data.get("node_data", {})
        for nid, telemetry in node_data.items():
            if nid in self.node_items:
                node_item = self.node_items[nid]
                node_item.propagate_telemetry(telemetry)

    def on_parameter_update(self, data):
        """
        Handle granular parameter updates from the engine.

        Args:
            data: Dictionary containing node_id, param, and value
        """
        node_id = data.get("node_id")
        param = data.get("param")
        value = data.get("value")

        # Check if node_id exists in node_items
        if node_id in self.node_items:
            node_item = self.node_items[node_id]
            # Call update_single_param to handle the efficient update
            node_item.update_single_param(param, value)

    def toggle_load_view(self, show):
        self._show_load = show
        for item in self.node_items.values():
            item.set_show_load(show)

    def get_selected_structure(self):
        """
        Returns a dict containing selected nodes and connections.
        Only includes connections where both source and destination nodes are selected.
        Thread-safe: operates solely on UI snapshot data, no access to audio thread.
        """
        selected_nodes = []
        selected_connections = []

        # Get selected nodes - extract all data directly from NodeItem instances
        for item in self.selectedItems():
            if isinstance(item, NodeItem):
                # Extract params data from the cached params (updated via update_from_snapshot)
                params_data = {}
                for param_name, param_data in item.params.items():
                    params_data[param_name] = {
                        "value": param_data["value"],
                        "type": param_data["type"],
                        "meta": param_data["meta"],
                    }

                selected_nodes.append(
                    {
                        "id": item.nid,
                        "name": item.node_name,
                        "type": item.node_type,
                        "pos": (item.pos().x(), item.pos().y()),
                        "params": params_data,
                        "inputs": list(item.input_items.keys()),
                        "outputs": list(item.output_items.keys()),
                        "can_be_master": item.can_be_master,
                        "is_master": item.is_master,
                    }
                )

        # Get selected connections where both nodes are selected
        selected_node_ids = {node["id"] for node in selected_nodes}
        for item in self.selectedItems():
            if isinstance(item, ConnectionItem) and item.logic_key:
                src_id, src_port, dst_id, dst_port = item.logic_key
                if src_id in selected_node_ids and dst_id in selected_node_ids:
                    selected_connections.append(
                        {"src_id": src_id, "src_port": src_port, "dst_id": dst_id, "dst_port": dst_port}
                    )

        return {"nodes": selected_nodes, "connections": selected_connections}

    def copy_selection(self):
        """Copy selected nodes and connections to clipboard."""
        structure = self.get_selected_structure()
        if not structure["nodes"]:
            return

        # Serialize to JSON and copy to clipboard
        json_data = json.dumps(structure, indent=2)
        clipboard = QCoreApplication.instance().clipboard()
        clipboard.setText(json_data)

    def paste_selection(self, target_pos=None):
        """Paste nodes and connections from clipboard using batch command."""
        clipboard = QCoreApplication.instance().clipboard()
        clipboard_text = clipboard.text()

        if not clipboard_text:
            return

        try:
            structure = json.loads(clipboard_text)
        except json.JSONDecodeError:
            return

        if not structure.get("nodes"):
            return

        # Calculate offset
        if target_pos is not None and hasattr(target_pos, "x") and hasattr(target_pos, "y"):
            min_x = min_y = float("inf")
            max_x = max_y = float("-inf")

            for node in structure["nodes"]:
                x, y = node["pos"]
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

            clipboard_center_x = (min_x + max_x) / 2
            clipboard_center_y = (min_y + max_y) / 2

            offset_x = target_pos.x() - clipboard_center_x
            offset_y = target_pos.y() - clipboard_center_y
        else:
            offset_x = 50
            offset_y = 50

        # Generate new UUIDs for pasted nodes
        id_map = {}
        for node in structure["nodes"]:
            id_map[node["id"]] = str(uuid.uuid4())

        # Prepare Batch Data
        new_nodes_data = []
        for node in structure["nodes"]:
            new_node = node.copy()
            new_node["id"] = id_map[node["id"]]
            new_node["pos"] = (node["pos"][0] + offset_x, node["pos"][1] + offset_y)
            new_nodes_data.append(new_node)

        new_conns_data = []
        for conn in structure["connections"]:
            if conn["src_id"] in id_map and conn["dst_id"] in id_map:
                new_conns_data.append({
                    "src_id": id_map[conn["src_id"]],
                    "src_port": conn["src_port"],
                    "dst_id": id_map[conn["dst_id"]],
                    "dst_port": conn["dst_port"],
                })

        # Send Single Batch Command via Controller
        self.controller.paste_structure(new_nodes_data, new_conns_data)

    def selectAll(self):
        """Select all nodes in the scene."""
        for item in self.items():
            if isinstance(item, NodeItem):
                item.setSelected(True)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.scenePos(), QTransform())
        if item:
            super().contextMenuEvent(event)
            return
        menu = QMenu()
        add_menu = menu.addMenu("Add Node")
        click_pos = event.scenePos()
        structure = {}
        for class_name, cls in plugin_system.NODE_REGISTRY.items():
            cat = getattr(cls, "category", "Uncategorized")
            lbl = getattr(cls, "label", class_name)
            structure.setdefault(cat, []).append((lbl, class_name))
        for category in sorted(structure.keys()):
            sub_menu = add_menu.addMenu(category)
            nodes = structure[category]
            nodes.sort(key=lambda x: x[0])  # Sort by label
            for lbl, class_name in nodes:
                action = sub_menu.addAction(lbl)
                action.triggered.connect(
                    lambda c=False, n=class_name, p=(click_pos.x(), click_pos.y()): self.controller.add_node(n, p)
                )
        menu.exec(event.screenPos())
        event.accept()


class GraphView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._panning_mode = False

        svg_bytes = create_colored_logo("white")
        self._logo_renderer = QSvgRenderer(svg_bytes)

        self._generate_grid_texture()

    def _generate_grid_texture(self):
        pixmap = QPixmap(150, 150)
        pixmap.fill(Theme.COLORS["grid_bg"])
        painter = QPainter(pixmap)
        painter.setPen(QPen(Theme.COLORS["grid_major"], 1.5))
        painter.drawLine(0, 0, 150, 0)
        painter.drawLine(0, 0, 0, 150)
        painter.setPen(QPen(Theme.COLORS["grid_minor"], 1.0))
        for i in range(15, 150, 15):
            if i != 0:
                painter.drawLine(i, 0, i, 150)
                painter.drawLine(0, i, 150, i)
        painter.end()
        self.setBackgroundBrush(QBrush(pixmap))

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
        self._panning_mode = False
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier
        ):
            self._panning_mode = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake_event = QMouseEvent(
                event.type(), event.position(), event.globalPosition(), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier
            )
            super().mousePressEvent(fake_event)
            return

        pos = event.position().toPoint()
        item = self.itemAt(pos)
        if isinstance(item, SocketItem):
            self.scene().drag_start = item
            self.setDragMode(QGraphicsView.NoDrag)
            self.scene().temp_wire = ConnectionItem(item, self.mapToScene(pos))
            self.scene().temp_wire.temp_mode = True
            self.scene().temp_wire.setZValue(Theme.Z_LAYERS["temp_wire"])
            self.scene().addItem(self.scene().temp_wire)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.scene().temp_wire:
            pos = self.mapToScene(event.position().toPoint())
            # Look through the temp_wire to find the socket underneath
            items = self.items(event.position().toPoint())
            socket = None
            for item in items:
                if item is self.scene().temp_wire:
                    continue
                if isinstance(item, SocketItem):
                    socket = item
                    break
            self.scene().drag_target = socket
            start = self.scene().drag_start
            if socket and isinstance(socket, SocketItem):
                valid = (start.is_input != socket.is_input) and (start.node_id != socket.node_id)
                if valid:
                    self.scene().temp_wire.end_item = socket.scenePos()
                    self.scene().temp_wire.temp_color = Theme.COLORS["wire_temp_green"]
                else:
                    self.scene().temp_wire.end_item = pos
                    self.scene().temp_wire.temp_color = Theme.COLORS["wire_temp_red"]
            else:
                self.scene().temp_wire.end_item = pos
                self.scene().temp_wire.temp_color = Theme.COLORS["wire_temp_white"]
            self.scene().temp_wire.update_path()
            self.scene().temp_wire.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.scene().temp_wire:
            end = self.scene().drag_target
            start = self.scene().drag_start
            if end and isinstance(end, SocketItem) and start:
                if start.is_input != end.is_input and start.node_id != end.node_id:
                    src = start if not start.is_input else end
                    dst = end if end.is_input else start
                    self.scene().controller.connect_nodes(src.node_id, src.name, dst.node_id, dst.name)
            self.scene().removeItem(self.scene().temp_wire)
            self.scene().temp_wire = None
            self.scene().drag_start = None
            self.scene().drag_target = None
            self.setDragMode(QGraphicsView.RubberBandDrag)
        if self._panning_mode:
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self._panning_mode = False
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.scene().delete_selection()
        elif event.key() == Qt.Key_C and event.modifiers() & Qt.ControlModifier:
            self.scene().copy_selection()
        elif event.key() == Qt.Key_V and event.modifiers() & Qt.ControlModifier:
            # Calculate viewport center and map to scene coordinates
            viewport_center = self.viewport().rect().center()
            scene_pos = self.mapToScene(viewport_center)
            self.scene().paste_selection(target_pos=scene_pos)
        elif event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            self.scene().selectAll()
        super().keyPressEvent(event)

    def drawBackground(self, painter: QPainter, rect: QRectF):
        """
        Draws the background. The `rect` is the exposed
        area in scene coordinates, provided by the QGraphicsView framework.
        """
        super().drawBackground(painter, rect)

        if self._logo_renderer and self._logo_renderer.isValid():
            scale = 0.2
            w = self._logo_renderer.defaultSize().width() * scale
            h = self._logo_renderer.defaultSize().height() * scale
            logo_rect = QRectF(-w / 2, -h / 2, w, h)
            if rect.intersects(logo_rect):
                painter.save()
                painter.setOpacity(0.04)
                self._logo_renderer.render(painter, logo_rect)
                painter.restore()
