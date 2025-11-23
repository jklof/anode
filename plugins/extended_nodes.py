import torch
from base import Node, BLOCK_SIZE, DTYPE


class Note(Node):
    def __init__(self, name="Note"):
        super().__init__(name)
        self.add_string_param("text", "Hello World")

    def process(self):
        pass


class Noise(Node):
    def __init__(self, name="Noise"):
        super().__init__(name)
        self.add_bool_param("enabled", True)
        self.add_float_param("amp", 0.1)
        self.out = self.add_output("out")

    def process(self):
        if self.params["enabled"].value:
            torch.rand(self.out.buffer.shape, out=self.out.buffer)
            self.out.buffer.mul_(2.0).sub_(1.0)
            self.out.buffer.mul_(self.params["amp"].value)
        else:
            self.out.buffer.zero_()


class Selector(Node):
    def __init__(self, name="Sel"):
        super().__init__(name)
        self.add_menu_param("source", ["Input A", "Input B"])
        self.in_a = self.add_input("A")
        self.in_b = self.add_input("B")
        self.out = self.add_output("out")

    def process(self):
        idx = int(self.params["source"].value)
        if idx == 0:
            self.out.buffer.copy_(self.in_a.get_tensor())
        else:
            self.out.buffer.copy_(self.in_b.get_tensor())
