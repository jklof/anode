from PySide6.QtGui import QColor, QFont
from PySide6.QtCore import Qt


class Theme:
    # Colors
    COLORS = {
        # Node colors
        'node_bg': QColor(40, 40, 40),
        'node_border': QColor(20, 20, 20),
        'header_normal_start': QColor(60, 60, 60),
        'header_normal_end': QColor(50, 50, 50),
        'header_error_start': QColor(100, 60, 60),
        'header_error_end': QColor(80, 50, 50),

        # Socket colors
        'socket_input': QColor("#ff9900"),
        'socket_output': QColor("#00ccff"),

        # Wire colors
        'wire_normal': QColor("white"),
        'wire_selected': QColor("yellow"),
        'wire_hovered': QColor("#00ccff"),
        'wire_temp_white': QColor("white"),
        'wire_temp_red': QColor("red"),
        'wire_temp_green': QColor("green"),

        # Selection and error
        'selection_outline': QColor("#00ccff"),
        'error_border': QColor(255, 0, 0),

        # Clock icon
        'clock_master': QColor("#00FF00"),
        'clock_slave': QColor("#666666"),

        # Processing load colors
        'load_low': QColor(0, 255, 0, 100),
        'load_medium': QColor(255, 255, 0, 100),
        'load_high': QColor(255, 0, 0, 120),

        # Grid colors
        'grid_bg': QColor(30, 30, 30),
        'grid_major': QColor(50, 50, 50),
        'grid_minor': QColor(40, 40, 40),

        # Text colors
        'text_normal': QColor("white"),
        'text_widget': QColor("#e0e0e0"),

        # Logo
        'logo': QColor("white"),
    }

    # Dimensions
    DIMENSIONS = {
        'node_width': 160,
        'header_height': 30,
        'socket_radius': 6,
    }

    # Fonts
    FONTS = {
        'node_title': QFont("Arial", 10, QFont.Bold),
        'socket_label': QFont("Arial", 8),
    }

    # Z-layers
    Z_LAYERS = {
        'wire': -1.0,
        'node_normal': 0.0,
        'node_selected': 10.0,
        'temp_wire': 100.0,
        'socket': 10.0,
    }

    # Styles
    STYLES = {
        'generic_node_container': "#genericNodeContainer { background-color: transparent; } QLabel, QCheckBox, QLineEdit, QSpinBox { color: #e0e0e0; }",
    }
