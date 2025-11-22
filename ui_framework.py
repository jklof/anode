from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
    QSlider,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, Slot
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QLinearGradient

# Constants
NODE_WIDTH = 160
HEADER_HEIGHT = 30
SOCKET_RADIUS = 6


class SocketItem(QGraphicsItem):
    def __init__(self, parent, name, is_input, slot_ref):
        super().__init__(parent)
        self.name = name
        self.is_input = is_input
        self.slot_ref = slot_ref  # Reference to core.InputSlot or OutputSlot
        self.setAcceptHoverEvents(True)
        self._color = QColor("#ff9900") if is_input else QColor("#00ccff")

    def boundingRect(self):
        return QRectF(-SOCKET_RADIUS, -SOCKET_RADIUS, SOCKET_RADIUS * 2, SOCKET_RADIUS * 2)

    def paint(self, painter, option, widget):
        painter.setBrush(self._color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.boundingRect())

        # Draw Label
        painter.setPen(QColor("white"))
        text_rect = QRectF(10 if self.is_input else -100, -10, 90, 20)
        align = Qt.AlignLeft if self.is_input else Qt.AlignRight
        painter.drawText(text_rect, align | Qt.AlignVCenter, self.name)


class ConnectionItem(QGraphicsPathItem):
    def __init__(self, start_pos, end_pos):
        super().__init__()
        self.setZValue(-1)  # Draw behind nodes
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.update_path()

    def update_path(self):
        path = QPainterPath()
        path.moveTo(self.start_pos)

        dx = self.end_pos.x() - self.start_pos.x()
        ctrl1 = QPointF(self.start_pos.x() + dx * 0.5, self.start_pos.y())
        ctrl2 = QPointF(self.end_pos.x() - dx * 0.5, self.end_pos.y())

        path.cubicTo(ctrl1, ctrl2, self.end_pos)
        self.setPath(path)

    def paint(self, painter, option, widget):
        pen = QPen(QColor("white"), 2)
        painter.setPen(pen)
        painter.drawPath(self.path())


class NodeItem(QGraphicsObject):
    positionChanged = Signal()

    def __init__(self, node_logic):
        super().__init__()
        self.node = node_logic
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        self.width = NODE_WIDTH
        self.height = HEADER_HEIGHT + 10

        # 1. Auto-Generate Sockets
        self.input_items = {}
        self.output_items = {}

        y = HEADER_HEIGHT + 10

        # Inputs
        for name, slot in self.node.inputs.items():
            item = SocketItem(self, name, True, slot)
            item.setPos(0, y)
            self.input_items[name] = item
            y += 20

        # Outputs
        y_out = HEADER_HEIGHT + 10
        for name, slot in self.node.outputs.items():
            item = SocketItem(self, name, False, slot)
            item.setPos(self.width, y_out)
            self.output_items[name] = item
            y_out += 20

        self.height = max(y, y_out) + 10

        # 2. Auto-Generate Controls (Widgets)
        if self.node.params:
            self.proxy = QGraphicsProxyWidget(self)
            self.widget = QWidget()
            self.layout = QVBoxLayout()
            self.widget.setLayout(self.layout)
            self.widget.setStyleSheet("background-color: transparent; color: white;")

            for param_name, param in self.node.params.items():
                self._add_slider(param_name, param)

            self.proxy.setWidget(self.widget)
            self.proxy.setPos(10, self.height)
            self.proxy.resize(self.width - 20, len(self.node.params) * 40)
            self.height += self.proxy.size().height() + 10

    def _add_slider(self, name, param):
        lbl = QLabel(f"{name}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)

        # Normalize value to 0-1000 based on min/max
        norm_val = (param.value - param.min) / (param.max - param.min)
        slider.setValue(int(norm_val * 1000))

        # Closure to capture param reference
        def on_change(val):
            float_val = param.min + (val / 1000.0) * (param.max - param.min)
            # THREAD SAFETY: We are on UI thread, calling set().
            # Engine calls sync() on audio thread. This is safe.
            param.set(float_val)
            lbl.setText(f"{name}: {float_val:.2f}")

        slider.valueChanged.connect(on_change)

        self.layout.addWidget(lbl)
        self.layout.addWidget(slider)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        # Body
        painter.setBrush(QColor(40, 40, 40))
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.drawRoundedRect(self.boundingRect(), 5, 5)

        # Header
        header_grad = QLinearGradient(0, 0, 0, HEADER_HEIGHT)
        header_grad.setColorAt(0, QColor(60, 60, 60))
        header_grad.setColorAt(1, QColor(50, 50, 50))
        painter.setBrush(header_grad)
        painter.drawRoundedRect(0, 0, self.width, HEADER_HEIGHT, 5, 5)

        # Title
        painter.setPen(QColor("white"))
        painter.drawText(QRectF(0, 0, self.width, HEADER_HEIGHT), Qt.AlignCenter, self.node.name)

        # Selection Border
        if self.isSelected():
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("orange"), 2))
            painter.drawRoundedRect(self.boundingRect(), 5, 5)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.positionChanged.emit()
        return super().itemChange(change, value)
