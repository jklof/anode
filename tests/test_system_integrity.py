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

    known_categories = {"Sources", "Utilities", "Effects", "I/O", "Visual", "Uncategorized"}

    for name, cls in plugin_system.NODE_REGISTRY.items():
        # 1. Check for Label
        label = getattr(cls, "label", "")
        assert label, f"Node {name} is missing a 'label' attribute"

        # 2. Check for Category
        category = getattr(cls, "category", "")
        assert category in known_categories, f"Node {name} has unknown category '{category}'. Valid: {known_categories}"


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
