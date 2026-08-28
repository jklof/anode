import os
import sys
import importlib
import importlib.util
import inspect
import logging
from typing import Dict, Type, Any, Optional
from base import Node, CHANNELS

logger = logging.getLogger(__name__)

NODE_REGISTRY: Dict[str, Type[Node]] = {}
UI_REGISTRY: Dict[str, Type[Any]] = {}

# Track loaded modules to allow reloading
_loaded_modules = {}

# Top-level modules of the app itself. A plugin file with one of these names
# would shadow the real module on sys.path.
_PROJECT_MODULES = frozenset(
    {"base", "core", "commands", "controller", "main", "plugin_system", "theme", "ui_icons", "ui_system", "ffi_base"}
)


def load_plugins(folder="plugins"):
    """
    Scans the plugins folder. If modules are already loaded, it reloads them
    to pick up code changes. Registers Node and UI classes.
    """
    # Clear registries so we don't have stale references
    NODE_REGISTRY.clear()
    UI_REGISTRY.clear()
    # Refresh documentation cache so hot-reloads pick up new help metadata
    _DOC_CACHE.clear()

    if not os.path.exists(folder):
        os.makedirs(folder)

    abs_folder = os.path.abspath(folder)
    added_to_path = False
    if abs_folder not in sys.path:
        sys.path.insert(0, abs_folder)
        added_to_path = True

    try:
        logging.info(f"--- Loading Plugins from '{folder}' ---")

        for f in os.listdir(folder):
            if f.endswith(".py"):
                name = f[:-3]

                # Refuse plugins whose names would shadow an already-imported
                # module (stdlib, third-party) or one of the app's own modules.
                if name in _PROJECT_MODULES:
                    logging.warning(f"Skipping plugin '{name}': name collides with an app module.")
                    continue
                if name in sys.modules and name not in _loaded_modules:
                    logging.warning(f"Skipping plugin '{name}': name shadows already-imported module.")
                    continue

                try:
                    # Hot Reload Logic
                    if name in _loaded_modules:
                        # If previously loaded, force a reload of the module object
                        mod = importlib.reload(_loaded_modules[name])
                        logging.info(f"Reloaded: {name}")
                    else:
                        # First time load
                        mod = importlib.import_module(name)
                        _loaded_modules[name] = mod
                        logging.info(f"Loaded: {name}")

                    # Inspect the module for Nodes and UIs
                    for mem_name, obj in inspect.getmembers(mod, inspect.isclass):
                        if obj.__module__ != name:
                            continue
                        # Register Logic Class
                        if issubclass(obj, Node) and obj is not Node:
                            NODE_REGISTRY[obj.__name__] = obj

                        # Register UI Class
                        if hasattr(obj, "IS_NODE_UI") and obj.IS_NODE_UI:
                            target = getattr(obj, "NODE_CLASS_NAME", None)
                            if target:
                                UI_REGISTRY[target] = obj

                except Exception as e:
                    logging.exception(f"Failed to load plugin {name}: {e}")

    finally:
        if added_to_path:
            try:
                sys.path.remove(abs_folder)
            except ValueError:
                pass


# Lazy documentation cache: populated on first UI request per node type and
# cleared on plugin (re)load. Never touched by the audio path.
_DOC_CACHE: Dict[str, dict] = {}


def get_node_documentation(node_type: str) -> dict:
    """
    Returns a structured documentation dictionary for a node type.
    Constructs a lightweight instance once on first request, then caches.
    """
    if node_type in _DOC_CACHE:
        return _DOC_CACHE[node_type]

    cls = NODE_REGISTRY.get(node_type)
    if cls is None:
        return {}

    is_native = False
    try:
        from ffi_base import FFINode
        is_native = issubclass(cls, FFINode)
    except ImportError:
        is_native = False

    inputs, outputs, params = {}, {}, {}
    try:
        instance = cls()
        inputs = {
            k: {
                "help": getattr(v, "help", ""),
                "param_name": getattr(v, "param_name", None),
                "slot_type": getattr(v, "slot_type", "audio"),
            }
            for k, v in instance.inputs.items()
        }
        outputs = {}
        for k, v in instance.outputs.items():
            slot_type = getattr(v, "slot_type", "audio")
            if slot_type == "midi":
                outputs[k] = {
                    "help": getattr(v, "help", ""),
                    "slot_type": "midi",
                    "channels": 0,
                }
            else:
                channels = getattr(
                    v, "channels", getattr(getattr(v, "buffer", None), "shape", (CHANNELS,))[0]
                )
                outputs[k] = {
                    "help": getattr(v, "help", ""),
                    "slot_type": "audio",
                    "channels": channels,
                }
        params = {
            k: {
                "type": v.type,
                "value": v.value,
                "meta": v.meta,
                "help": v.meta.get("help", ""),
                "unit": v.meta.get("unit", ""),
            }
            for k, v in instance.params.items()
        }
    except Exception as e:
        logger.warning(f"Failed to inspect documentation for {node_type}: {e}")

    doc_text = (getattr(cls, "description", "") or "").strip() or (cls.__doc__ or "").strip()

    doc_data = {
        "type": node_type,
        "label": getattr(cls, "label", node_type),
        "category": getattr(cls, "category", "Uncategorized"),
        "is_native": is_native,
        "description": doc_text,
        "inputs": inputs,
        "outputs": outputs,
        "params": params,
    }

    _DOC_CACHE[node_type] = doc_data
    return doc_data


def get_node_class(name):
    return NODE_REGISTRY.get(name)


def get_ui_class(node_class_name: str) -> Optional[Type[Any]]:
    return UI_REGISTRY.get(node_class_name)
