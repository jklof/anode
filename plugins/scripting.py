import ast
import re
import traceback
import sys
import torch
import numpy as np
import math

from base import Node, InputSlot, OutputSlot, BLOCK_SIZE, CHANNELS, DTYPE
import plugin_system

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QPushButton, QLabel, QTextEdit
    from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QTextCursor, QFont
    from PySide6.QtCore import Qt, QTimer, QSignalBlocker

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

DEFAULT_CODE = """# Define inputs and outputs. Click "Apply" to update ports.
inputs = ['audio_in', 'gain']
outputs = ['audio_out']

# The 'state' dictionary persists between processing ticks.
if 'counter' not in state:
    state['counter'] = 0

# Retrieve input tensors
sig = audio_in
g = gain

# Perform processing and assign to output
audio_out = sig * g

# Log/debug via state if needed
state['counter'] += 1
"""


# ==============================================================================
# 1. AST Parser Helper
# ==============================================================================
def parse_ports(code_str: str) -> tuple[list[str], list[str]]:
    inputs = []
    outputs = []
    try:
        tree = ast.parse(code_str)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "inputs":
                            val = ast.literal_eval(node.value)
                            if isinstance(val, list):
                                inputs = [str(x) for x in val]
                            elif isinstance(val, dict):
                                inputs = [str(x) for x in val.keys()]
                        elif target.id == "outputs":
                            val = ast.literal_eval(node.value)
                            if isinstance(val, list):
                                outputs = [str(x) for x in val]
                            elif isinstance(val, dict):
                                outputs = [str(x) for x in val.keys()]
    except Exception:
        pass
    return inputs, outputs


# ==============================================================================
# 2. Python Syntax Highlighter
# ==============================================================================
if GUI_AVAILABLE:

    class PythonSyntaxHighlighter(QSyntaxHighlighter):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.highlighting_rules = []

            keyword_format = QTextCharFormat()
            keyword_format.setForeground(QColor("#ff79c6"))  # Pink
            keywords = [
                r"\bif\b",
                r"\belif\b",
                r"\belse\b",
                r"\bfor\b",
                r"\bwhile\b",
                r"\bin\b",
                r"\bis\b",
                r"\bnot\b",
                r"\band\b",
                r"\bor\b",
                r"\bdef\b",
                r"\bclass\b",
                r"\breturn\b",
                r"\bpass\b",
                r"\bcontinue\b",
                r"\bbreak\b",
                r"\btry\b",
                r"\bexcept\b",
                r"\bfinally\b",
                r"\braise\b",
                r"\bimport\b",
                r"\bfrom\b",
                r"\bas\b",
                r"\bwith\b",
            ]
            for word in keywords:
                self.highlighting_rules.append((re.compile(word), keyword_format))

            builtin_format = QTextCharFormat()
            builtin_format.setForeground(QColor("#8be9fd"))  # Cyan
            builtins = [r"\btorch\b", r"\bnp\b", r"\bmath\b", r"\bstate\b", r"\binputs\b", r"\boutputs\b"]
            for word in builtins:
                self.highlighting_rules.append((re.compile(word), builtin_format))

            number_format = QTextCharFormat()
            number_format.setForeground(QColor("#bd93f9"))  # Purple
            self.highlighting_rules.append((re.compile(r"\b[0-9]+\.?[0-9]*\b"), number_format))

            string_format = QTextCharFormat()
            string_format.setForeground(QColor("#f1fa8c"))  # Yellow
            self.highlighting_rules.append((re.compile(r"'.*?'"), string_format))
            self.highlighting_rules.append((re.compile(r'".*?"'), string_format))

            comment_format = QTextCharFormat()
            comment_format.setForeground(QColor("#6272a4"))  # Gray/Blue
            self.highlighting_rules.append((re.compile(r"#[^\n]*"), comment_format))

        def highlightBlock(self, text):
            for pattern, format in self.highlighting_rules:
                for match in pattern.finditer(text):
                    start, end = match.span()
                    self.setFormat(start, end - start, format)


# ==============================================================================
# 3. Node Logic Class
# ==============================================================================
class ScriptNode(Node):
    category = "Utilities"
    label = "Script Node"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_string_param("code", DEFAULT_CODE)
        self.compiled_code = None
        self.state_dict = {}
        self.error_line = -1
        self.graph = None

        self._recompile()

    def _recompile(self):
        code_str = self.params["code"].value
        inputs, outputs = parse_ports(code_str)

        # 1. Update Input Slots
        for name in list(self.inputs.keys()):
            if name not in inputs:
                slot = self.inputs.pop(name)
                slot.disconnect()
        for name in inputs:
            if name not in self.inputs:
                self.add_input(name)

        # 2. Update Output Slots
        for name in list(self.outputs.keys()):
            if name not in outputs:
                slot = self.outputs.pop(name)
                # Disconnect downstream slots referencing this output
                if hasattr(self, "graph") and self.graph:
                    for other in self.graph.nodes:
                        for inp in other.inputs.values():
                            for out_slot in list(inp.connected_outputs):
                                if out_slot == slot:
                                    inp.disconnect(out_slot)
        for name in outputs:
            if name not in self.outputs:
                self.add_output(name)

        self.request_graph_rebuild()

        # 3. Compile Python Bytecode
        try:
            self.compiled_code = compile(code_str, "<script>", "exec")
            self.error_msg = None
            self.error_line = -1
        except Exception as e:
            self.compiled_code = None
            self.error_msg = str(e)
            # Extracted line number for syntax errors
            if hasattr(e, "lineno"):
                self.error_line = e.lineno

    def on_ui_param_change(self, param_name: str):
        if param_name == "code":
            self.params["code"].sync()
            self._recompile()

    def load_state(self, data: dict):
        super().load_state(data)
        self._recompile()

    def get_telemetry(self) -> dict:
        return {"error_msg": self.error_msg, "error_line": self.error_line}

    def process(self):
        if not self.compiled_code:
            for out in self.outputs.values():
                out.buffer.zero_()
            return

        # Prepare context scope
        execution_scope = {"state": self.state_dict, "torch": torch, "np": np, "math": math}

        # Inject input values
        for name, inp in self.inputs.items():
            execution_scope[name] = inp.get_tensor()

        try:
            exec(self.compiled_code, execution_scope, execution_scope)
            self.error_msg = None
            self.error_line = -1

            # Retrieve outputs and copy to slot buffers safely
            for name, out in self.outputs.items():
                if name in execution_scope:
                    val = execution_scope[name]
                    if isinstance(val, torch.Tensor):
                        ch_to_copy = min(out.buffer.shape[0], val.shape[0])
                        frames_to_copy = min(out.buffer.shape[1], val.shape[1])
                        out.buffer[:ch_to_copy, :frames_to_copy].copy_(val[:ch_to_copy, :frames_to_copy])
                        if ch_to_copy < out.buffer.shape[0]:
                            out.buffer[ch_to_copy:].zero_()
                    elif isinstance(val, (float, int, bool)):
                        out.buffer.fill_(float(val))
                    else:
                        out.buffer.zero_()
                else:
                    out.buffer.zero_()

        except Exception as e:
            # Trace target script frame for accurate error line matching
            tb = sys.exc_info()[2]
            line_no = -1
            while tb:
                if tb.tb_frame.f_code.co_filename == "<script>":
                    line_no = tb.tb_lineno
                    break
                tb = tb.tb_next

            self.error_msg = str(e)
            self.error_line = line_no
            for out in self.outputs.values():
                out.buffer.zero_()


# ==============================================================================
# 4. Custom UI Widget
# ==============================================================================
if GUI_AVAILABLE:

    class ScriptNodeWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "ScriptNode"

        def __init__(self, node_proxy):
            super().__init__()
            self.proxy = node_proxy
            self.setMinimumSize(400, 300)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            # Editor
            self.editor = QPlainTextEdit()
            self.editor.setFont(QFont("Courier New", 10))
            self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
            self.highlighter = PythonSyntaxHighlighter(self.editor.document())

            # Populate Initial Code
            init_code = self.proxy.node_item.params["code"]["value"]
            self.editor.setPlainText(init_code)

            # Apply Button
            self.btn_apply = QPushButton("Apply Script")
            self.btn_apply.setFixedHeight(30)
            self.btn_apply.clicked.connect(self.on_apply)

            # Status Message
            self.lbl_status = QLabel("Status: OK")
            self.lbl_status.setStyleSheet("color: #00FF00; font-size: 10px; font-weight: bold;")

            layout.addWidget(self.editor)
            layout.addWidget(self.btn_apply)
            layout.addWidget(self.lbl_status)

        def on_apply(self):
            code_text = self.editor.toPlainText()
            self.proxy.set_parameter("code", code_text)

        def on_telemetry(self, data: dict):
            error_msg = data.get("error_msg")
            error_line = data.get("error_line", -1)

            if error_msg:
                self.lbl_status.setText(f"Error: {error_msg}")
                self.lbl_status.setStyleSheet("color: #ff5555; font-size: 10px; font-weight: bold;")
                if error_line > 0:
                    self.highlight_error_line(error_line)
            else:
                self.lbl_status.setText("Status: OK")
                self.lbl_status.setStyleSheet("color: #00FF00; font-size: 10px; font-weight: bold;")
                self.editor.setExtraSelections([])

        def highlight_error_line(self, line_num):
            selection = QTextEdit.ExtraSelection()
            selection.format.setBackground(QColor(100, 30, 30, 150))
            selection.format.setProperty(QTextCharFormat.FullWidthSelection, True)

            cursor = self.editor.textCursor()
            cursor.setPosition(0)
            cursor.movePosition(QTextCursor.NextBlock, QTextCursor.MoveAnchor, line_num - 1)
            selection.cursor = cursor
            self.editor.setExtraSelections([selection])

        def update_from_params(self, params):
            if "code" in params:
                code_val = params["code"]
                if self.editor.toPlainText() != code_val:
                    with QSignalBlocker(self.editor):
                        self.editor.setPlainText(code_val)
