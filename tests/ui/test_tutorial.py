from nicegui.client import Client
from nicegui.page import page

from virtual_staining.ui.theme import PAGE_CSS
from virtual_staining.ui.tutorial import build_tutorial


def test_tutorial_navigation_and_numbered_steps_have_stable_layout() -> None:
    with Client(page("/tutorial-layout-test")) as client:
        build_tutorial()

    footer = next(
        element
        for element in client.elements.values()
        if "vs-tutorial-footer" in element._classes
    )
    step_numbers = [
        element
        for element in client.elements.values()
        if "vs-step-number" in element._classes
    ]

    assert len(footer.default_slot.children) == 3
    assert len(footer.default_slot.children[1].default_slot.children) == 7
    assert [element.text for element in step_numbers] == ["1", "2", "3", "4"]
    assert "place-items: center" in PAGE_CSS


def test_tutorial_evaluation_slide_explains_both_entry_paths_and_outputs() -> None:
    with Client(page("/tutorial-evaluation-guide-test")) as client:
        build_tutorial()

    labels = [getattr(element, "text", None) for element in client.elements.values()]

    assert "Existing run" in labels
    assert "Run from config" in labels
    assert "Aggregate metrics" in labels
    assert "Plots" in labels
    assert "Representative cases" in labels
    assert labels.count("Model") == 1
