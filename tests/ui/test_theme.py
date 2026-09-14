from pathlib import Path

from virtual_staining.ui.theme import (
    METRIC_QUALITY_STYLES,
    PAGE_CSS,
    metric_quality_style,
)


def test_metric_quality_palette_has_distinct_ordered_levels() -> None:
    expected = {
        "very_good": "Very good",
        "good": "Good",
        "fair": "Fair",
        "poor": "Poor",
        "very_poor": "Very poor",
        "unknown": "Not graded",
    }

    assert {level: label for level, (label, _css) in METRIC_QUALITY_STYLES.items()} == expected
    assert len({css for _label, css in METRIC_QUALITY_STYLES.values()}) == len(expected)
    assert metric_quality_style("very_good") == ("Very good", "vs-metric--very-good")


def test_header_uses_compact_hamburger_navigation() -> None:
    assert "grid-template-columns: auto minmax(0, 1fr) auto" in PAGE_CSS
    assert ".vs-menu-item" in PAGE_CSS
    assert ".vs-menu-trigger { width: 3.5rem; height: 3.5rem; }" in PAGE_CSS
    assert ".vs-menu-trigger .q-icon { font-size: 2rem; }" in PAGE_CSS


def test_ui_typography_has_a_readable_minimum_size() -> None:
    ui_directory = Path(__file__).parents[2] / "virtual_staining" / "ui"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(ui_directory.glob("*.py"))
    )

    assert "html { font-size: 18px; }" in PAGE_CSS
    assert "text-[11px]" not in source
    assert "text-xs" not in source
