import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

from nicegui import ui
from nicegui.client import Client
from nicegui.page import page
from PIL import Image

import virtual_staining.ui.experiments_page as experiments_page
from virtual_staining.applications.api import (
    ComparisonResult,
    RepresentativeSample,
    RunDescriptor,
)
from virtual_staining.ui.experiments_page import (
    _build_run_comparison,
    _normalise_comparison_mode,
    _render_overlap_comparison,
    _render_plots,
    _representative_card,
    _select_evaluation_plot_paths,
)


def test_comparison_mode_survives_paired_unpaired_paired_sequence() -> None:
    assert _normalise_comparison_mode("paired") == "paired"
    assert _normalise_comparison_mode("unpaired") == "unpaired"
    assert _normalise_comparison_mode("Paired (same samples)") == "paired"


def test_compare_ui_reuses_paired_result_after_switching_modes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakeService:
        supported_metrics = ("ssim",)

        def compare_runs(self, request):
            calls.append(request.mode)
            return ComparisonResult(
                mode=request.mode,
                metric=request.metric,
                higher_is_better=True,
                output_directory=tmp_path / request.mode,
                label_a="A",
                label_b="B",
                summary={"favors": "B"},
                plot_paths=(),
            )

    async def immediate_io_bound(function, *args):
        return function(*args)

    monkeypatch.setattr(experiments_page.run, "io_bound", immediate_io_bound)
    monkeypatch.setattr(experiments_page.ui, "notify", lambda *args, **kwargs: None)
    runs = tuple(
        RunDescriptor(name, name.upper(), tmp_path / name, True, sample_count=2)
        for name in ("a", "b")
    )

    with Client(page("/comparison-mode-sequence-test")) as client:
        _build_run_comparison(FakeService(), runs)  # type: ignore[arg-type]
        selects = [
            element for element in client.elements.values() if element.tag == "nicegui-select"
        ]
        compare_button = next(
            element
            for element in client.elements.values()
            if element.tag == "q-btn" and getattr(element, "text", None) == "Compare runs"
        )
        wrapper = next(iter(compare_button._event_listeners.values())).handler
        assert wrapper is not None and wrapper.__closure__ is not None
        execute = next(
            cell.cell_contents
            for cell in wrapper.__closure__
            if inspect.iscoroutinefunction(cell.cell_contents)
        )
        selects[0].set_value(str(runs[0].path))
        selects[1].set_value(str(runs[1].path))

        async def run_sequence() -> None:
            for mode in ("paired", "unpaired", "paired"):
                selects[3].set_value(mode)
                await execute()

        asyncio.run(run_sequence())

    assert calls == ["paired", "unpaired"]
    assert all(not select._props.get("disable", False) for select in selects)


def test_evaluation_dashboard_keeps_four_complementary_plots() -> None:
    paths = tuple(
        Path(name)
        for name in (
            "mae_histogram.png",
            "metrics_boxplot.png",
            "mse_histogram.png",
            "pcc_gray_histogram.png",
            "pcc_rgb_mean_histogram.png",
            "psnr_histogram.png",
            "rmse_histogram.png",
            "ssim_histogram.png",
        )
    )

    assert tuple(path.name for path in _select_evaluation_plot_paths(paths)) == (
        "ssim_histogram.png",
        "psnr_histogram.png",
        "mae_histogram.png",
        "pcc_rgb_mean_histogram.png",
    )


def test_evaluation_dashboard_renders_only_selected_plots(tmp_path: Path) -> None:
    paths: list[Path] = []
    for name in (
        "mae_histogram.png",
        "metrics_boxplot.png",
        "mse_histogram.png",
        "pcc_gray_histogram.png",
        "pcc_rgb_mean_histogram.png",
        "psnr_histogram.png",
        "rmse_histogram.png",
        "ssim_histogram.png",
    ):
        path = tmp_path / name
        Image.new("RGB", (8, 8), "white").save(path)
        paths.append(path)

    with Client(page("/evaluation-plots-test")) as client:
        _render_plots(tuple(paths), ui.column())

    rendered_images = [
        element for element in client.elements.values() if element.tag == "nicegui-image"
    ]
    rendered_sources = {str(element._props["src"]) for element in rendered_images}
    assert len(rendered_images) == 4
    assert not any("boxplot" in source or "pcc_gray" in source for source in rendered_sources)


def test_tiff_representative_is_rendered_as_generated_only_preview(tmp_path: Path) -> None:
    generated_path = tmp_path / "sample_target_generated.tif"
    Image.new("RGB", (8, 8), "purple").save(generated_path)
    sample = RepresentativeSample(
        kind="best",
        sample_id="sample",
        metric="ssim",
        value=0.9,
        source_path=None,
        generated_path=generated_path,
        target_path=None,
    )

    with Client(page("/representative-tiff-test")) as client:
        _representative_card(sample)

    texts = {getattr(element, "text", None) for element in client.elements.values()}
    image = next(element for element in client.elements.values() if element.tag == "nicegui-image")
    assert "Generated preview only" in texts
    assert "Restore the source and target dataset to build the full panel." in texts
    assert ".png" in str(image._props["src"])


def test_overlap_slider_blends_generated_image_over_target() -> None:
    with Client(page("/overlap-test")) as client:
        container = ui.column()
        _render_overlap_comparison(
            Image.new("RGB", (32, 32), "red"),
            Image.new("RGB", (32, 32), "blue"),
            container,
        )

    sliders = {
        element._props.get("aria-label"): element
        for element in client.elements.values()
        if element.tag == "q-slider"
    }
    slider = sliders["Generated image opacity"]
    generated_layer = next(
        element
        for element in client.elements.values()
        if "vs-overlap-generated" in element._classes
    )

    assert set(sliders) == {
        "Generated image opacity",
    }
    assert slider.value == 0
    assert generated_layer._style["opacity"] == "0"

    slider.set_value(75)

    assert generated_layer._style["opacity"] == "0.75"
    assert any(
        getattr(element, "text", None) == "75% generated" for element in client.elements.values()
    )


def test_comparison_play_button_animates_its_own_slider() -> None:
    with Client(page("/comparison-animation-test")) as client:
        container = ui.column()
        _render_overlap_comparison(
            Image.new("RGB", (32, 32), "red"),
            Image.new("RGB", (32, 32), "blue"),
            container,
        )

    button = next(
        element
        for element in client.elements.values()
        if element.tag == "q-btn" and element._props.get("aria-label") == "Play opacity animation"
    )
    slider = next(
        element
        for element in client.elements.values()
        if element.tag == "q-slider"
        and element._props.get("aria-label") == "Generated image opacity"
    )
    timer = next(element for element in client.elements.values() if element.tag == "nicegui-timer")
    speed_input = next(
        element
        for element in client.elements.values()
        if element.tag == "q-input"
        and element._props.get("aria-label") == "Opacity animation speed"
    )
    repeat_button = next(
        element
        for element in client.elements.values()
        if element.tag == "q-btn" and element._props.get("aria-label") == "Loop opacity animation"
    )
    speed_input.set_value(2)
    slider.set_value(30)

    click_listener = next(iter(button._event_listeners.values()))
    assert click_listener.handler is not None
    click_listener.handler(None)

    assert button.icon == "pause"
    assert timer.active is True
    assert slider.value == 0

    assert timer.callback is not None
    timer.callback()

    assert slider.value == 4

    repeat_listener = next(iter(repeat_button._event_listeners.values()))
    assert repeat_listener.handler is not None
    repeat_listener.handler(None)
    slider.set_value(100)
    timer.callback()

    assert repeat_button._props["aria-pressed"] == "true"
    assert slider.value == 0
    timer.callback()
    assert slider.value == 0


def test_horizontal_slider_reveals_generated_image_over_target() -> None:
    with Client(page("/horizontal-reveal-test")) as client:
        container = ui.column()
        _render_overlap_comparison(
            Image.new("RGB", (32, 32), "red"),
            Image.new("RGB", (32, 32), "blue"),
            container,
        )

    generated_layer = next(
        element for element in client.elements.values() if "vs-reveal-generated" in element._classes
    )
    divider = next(
        element for element in client.elements.values() if "vs-reveal-divider" in element._classes
    )

    assert not any(
        element.tag == "q-slider"
        and element._props.get("aria-label") == "Generated image horizontal reveal"
        for element in client.elements.values()
    )
    assert not any(
        element.tag == "q-btn"
        and element._props.get("aria-label")
        in {
            "Play reveal animation",
            "Loop reveal animation",
        }
        for element in client.elements.values()
    )
    assert not any(
        element.tag == "q-input" and element._props.get("aria-label") == "Reveal animation speed"
        for element in client.elements.values()
    )
    assert generated_layer._style["clip-path"] == "inset(0 50% 0 0)"
    assert divider._style["left"] == "50%"
    assert any(
        getattr(element, "text", None) == "50% revealed" for element in client.elements.values()
    )

    pointer_listener = next(iter(divider._event_listeners.values()))
    assert pointer_listener.handler is not None
    assert pointer_listener.js_handler is not None
    pointer_listener.handler(SimpleNamespace(args=20))

    assert generated_layer._style["clip-path"] == "inset(0 80% 0 0)"
    assert divider._style["left"] == "20%"


def test_comparison_panels_are_equal_height() -> None:
    with Client(page("/comparison-alignment-test")) as client:
        container = ui.column()
        _render_overlap_comparison(
            Image.new("RGB", (32, 32), "red"),
            Image.new("RGB", (32, 32), "blue"),
            container,
        )

    grid = next(
        element for element in client.elements.values() if "vs-comparison-grid" in element._classes
    )
    panels = [
        element for element in client.elements.values() if "vs-comparison-panel" in element._classes
    ]

    assert "items-stretch" in grid._classes
    assert len(panels) == 2
    assert all("h-full" in panel._classes for panel in panels)


def test_overlap_slider_requires_matching_image_dimensions() -> None:
    with Client(page("/overlap-size-test")) as client:
        container = ui.column()
        _render_overlap_comparison(
            Image.new("RGB", (32, 32)),
            Image.new("RGB", (64, 32)),
            container,
        )

    assert not any(element.tag == "q-slider" for element in client.elements.values())
    assert any(
        getattr(element, "text", None) == "Comparison unavailable"
        for element in client.elements.values()
    )
