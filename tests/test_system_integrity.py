import pytest
import inspect
from base import Node
import plugin_system


def test_plugin_metadata_integrity():
    """
    Ensure all registered nodes have valid categories and labels.
    This prevents UI menu fragmentation.
    """
    plugin_system.load_plugins("plugins")

    known_categories = {"Sources", "Utilities", "Effects", "I/O", "Visual", "Uncategorized", "MIDI"}

    for name, cls in plugin_system.NODE_REGISTRY.items():
        # 1. Check for Label
        label = getattr(cls, "label", "")
        assert label, f"Node {name} is missing a 'label' attribute"

        # 2. Check for Category
        category = getattr(cls, "category", "")
        assert category in known_categories, f"Node {name} has unknown category '{category}'. Valid: {known_categories}"


def test_node_documentation_integrity():
    """Verify that every registered node has descriptions, documented ports, and parameter help."""
    plugin_system.load_plugins("plugins")

    for node_type, cls in plugin_system.NODE_REGISTRY.items():
        doc = plugin_system.get_node_documentation(node_type)

        # 1. Class Description / Docstring
        assert doc["description"], f"Node '{node_type}' is missing a description or docstring"
        assert len(doc["description"]) >= 15, (
            f"Node '{node_type}' description is too short ({len(doc['description'])} chars)"
        )

        # 2. Port Documentation
        for port_name, p_info in doc["inputs"].items():
            assert p_info["help"], f"Node '{node_type}' input '{port_name}' is missing help text"
        for port_name, p_info in doc["outputs"].items():
            assert p_info["help"], f"Node '{node_type}' output '{port_name}' is missing help text"

        # 3. Parameter Documentation
        for param_name, p_info in doc["params"].items():
            assert p_info["help"], f"Node '{node_type}' param '{param_name}' is missing help text"
            assert isinstance(p_info["unit"], str), (
                f"Node '{node_type}' param '{param_name}' unit must be a string"
            )


def test_node_naming_logic():
    """
    Verify the base Node class correctly uses the 'label' attribute
    as the default instance name.
    """

    class LabeledNode(Node):
        label = "Friendly Name"

    class UnlabeledNode(Node):
        pass  # label defaults to ""

    n1 = LabeledNode()
    assert n1.name == "Friendly Name", "Did not use class label"

    n2 = UnlabeledNode()
    assert n2.name == "UnlabeledNode", "Did not fallback to class name"

    n3 = LabeledNode(name="Custom Override")
    assert n3.name == "Custom Override", "Init argument did not override label"
