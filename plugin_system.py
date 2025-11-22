import os
import sys
import importlib.util
import inspect
from typing import Dict, Type, Any, Optional
from core import Node

NODE_REGISTRY: Dict[str, Type[Node]] = {}
UI_REGISTRY: Dict[str, Type[Any]] = {}


def load_plugins(folder="plugins"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    sys.path.append(folder)

    for f in os.listdir(folder):
        if f.endswith(".py"):
            name = f[:-3]
            path = os.path.join(folder, f)
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)

                for mem_name, obj in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(obj, Node) and obj is not Node:
                        NODE_REGISTRY[obj.__name__] = obj
                        print(f"Registered Logic: {obj.__name__}")

                    if hasattr(obj, "IS_NODE_UI") and obj.IS_NODE_UI:
                        target = getattr(obj, "NODE_CLASS_NAME", None)
                        if target:
                            UI_REGISTRY[target] = obj
                            print(f"Registered UI: {obj.__name__}")

            except Exception as e:
                print(f"Failed to load plugin {name}: {e}")


def get_node_class(name):
    return NODE_REGISTRY.get(name)


def get_ui_class(node_class_name: str) -> Optional[Type[Any]]:
    return UI_REGISTRY.get(node_class_name)
