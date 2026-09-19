import asyncio
import inspect
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from nicegui import ui
from nicegui.client import Client
from nicegui.page import page
from PIL import Image

import virtual_staining.ui.experiment_comparison as comparison_ui
from virtual_staining.applications.api import (
    ComparisonResult,
    RepresentativeSample,
    RunDescriptor,
)
from virtual_staining.ui.experiment_comparison import (
    build_run_comparison,
    normalise_comparison_mode,
    render_comparison,
)
from virtual_staining.ui.experiment_components import (
    AsyncRequestGuard,
    comparison_case_row,
    representative_card,
)
from virtual_staining.ui.experiment_evaluation import (
    render_plots,
    select_evaluation_plot_paths,
)
from virtual_staining.ui.image_comparison import (
    horizontal_reveal_png,
    instant_ab_gif,
    opacity_transition_gif,
    render_image_comparison,
)


def test_comparison_mode_survives_paired_unpaired_paired_sequence() -> None:
    assert normalise_comparison_mode("paired") == "paired"
    assert normalise_comparison_mode("unpaired") == "unpaired"
    assert normalise_comparison_mode("Paired (same samples)") == "paired"


def test_async_request_guard_invalidates_work_from_a_previous_page_state() -> None:
    guard = AsyncRequestGuard()
    first = guard.start()

    guard.invalidate()
    second = guard.start()

    assert guard.is_current(first) is False
    assert guard.is_current(second) is True


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

    monkeypatch.setattr(comparison_ui.run, "io_bound", immediate_io_bound)
    monkeypatch.setattr(comparison_ui.ui, "notify", lambda *args, **kwargs: None)
    runs = tuple(
        RunDescriptor(name, name.upper(), tmp_path / name, True, sample_count=2)
        for name in ("a", "b")
    )

    with Client(page("/comparison-mode-sequence-test")) as client:
        build_run_comparison(FakeService(), runs)  # type: ignore[arg-type]
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

    assert tuple(path.name for path in select_evaluation_plot_paths(paths)) == (
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
        render_plots(tuple(paths), ui.column())

    rendered_images = [
        element for element in client.elements.values() if element.tag == "nicegui-image"
    ]
    rendered_sources = {str(element._props["src"]) for element in rendered_images}
    assert len(rendered_images) == 4
    assert not any("boxplot" in source or "pcc_gray" in source for source in rendered_sources)


def test_evaluation_dashboard_uses_interactive_histograms_when_rows_are_loaded(
    tmp_path: Path,
) -> None:
    paths = []
    rows = []
    for name in (
        "ssim_histogram.png",
        "psnr_histogram.png",
        "mae_histogram.png",
        "pcc_rgb_mean_histogram.png",
    ):
        path = tmp_path / name
        Image.new("RGB", (8, 8), "white").save(path)
        paths.append(path)
    for index in range(8):
        rows.append(
            {
                "ssim": str(index / 10),
                "psnr": str(20 + index),
                "mae": str(index / 20),
                "pcc_rgb_mean": str(-0.4 + index / 10),
            }
        )

    with Client(page("/interactive-evaluation-plots-test")) as client:
        render_plots(tuple(paths), ui.column(), tuple(rows))

    charts = [element for element in client.elements.values() if element.tag == "nicegui-echart"]
    assert len(charts) == 4
    assert not any(element.tag == "nicegui-image" for element in client.elements.values())
    assert all(chart.options["animationDuration"] == 700 for chart in charts)
    assert all("saveAsImage" in chart.options["toolbox"]["feature"] for chart in charts)
    assert all("dataZoom" not in chart.options["toolbox"]["feature"] for chart in charts)
    assert all(chart.options["dataZoom"][0]["type"] == "inside" for chart in charts)
    assert all(chart.options["dataZoom"][0]["zoomOnMouseWheel"] == "shift" for chart in charts)
    assert all(chart.options["dataZoom"][0]["moveOnMouseWheel"] is False for chart in charts)
    assert all(chart.options["dataZoom"][0]["moveOnMouseMove"] is True for chart in charts)
    assert all(chart.options["xAxis"].get("name") is None for chart in charts)
    assert charts[0].options["xAxis"]["min"] == 0.0
    assert charts[0].options["xAxis"]["max"] == 1.0
    assert all(
        [series["type"] for series in chart.options["series"]] == ["line", "bar"]
        for chart in charts
    )
    assert all(chart.options["tooltip"]["trigger"] == "item" for chart in charts)
    assert all(chart.options["tooltip"]["renderMode"] == "html" for chart in charts)
    assert all(chart.options["tooltip"]["className"] == "vs-evaluation-tooltip" for chart in charts)
    assert all(chart.options["tooltip"]["showDelay"] == 0 for chart in charts)
    assert all(chart.options["tooltip"]["transitionDuration"] == 0 for chart in charts)
    assert all("opacity:1" in chart.options["tooltip"]["extraCssText"] for chart in charts)
    assert all(chart.options["toolbox"]["top"] == 0 for chart in charts)
    assert all(chart.options["grid"]["top"] == 58 for chart in charts)
    assert all(chart.options["series"][0]["lineStyle"]["opacity"] < 0.5 for chart in charts)
    assert all(chart.options["series"][1]["itemStyle"]["opacity"] == 1 for chart in charts)
    for chart in charts:
        curve = chart.options["series"][0]["data"]
        bars = chart.options["series"][1]["data"]
        bars_by_center = {item["value"][0]: item["value"][1] for item in bars}
        assert any(point[1] is None for point in curve)
        assert all(item["value"][1] > 0 for item in bars)
        assert all(
            point[1] is None or point[1] == bars_by_center.get(point[0], 0.0) for point in curve
        )
    texts = {getattr(element, "text", None) for element in client.elements.values()}
    assert (
        "Use Shift + Scroll to zoom, then hold and drag to move left or right; "
        "normal scrolling moves the page."
    ) in texts


def test_comparison_groups_cases_by_rank_and_balances_plot_cards(tmp_path: Path) -> None:
    image_path = tmp_path / "plot.png"
    Image.new("RGB", (80, 40), "white").save(image_path)

    def samples(label: str) -> tuple[RepresentativeSample, ...]:
        return tuple(
            RepresentativeSample(
                kind=kind,
                sample_id=f"{label}-{kind}",
                metric="ssim",
                value=value,
                source_path=image_path,
                generated_path=image_path,
                target_path=image_path,
            )
            for kind, value in (("worst", 0.1), ("best", 0.9), ("median", 0.5))
        )

    result = ComparisonResult(
        mode="paired",
        metric="ssim",
        higher_is_better=True,
        output_directory=tmp_path,
        label_a="Run A",
        label_b="Run B",
        summary={"favors": "Run A"},
        plot_paths=(image_path,) * 4,
        representatives_a=samples("a"),
        representatives_b=samples("b"),
    )

    with Client(page("/comparison-layout-test")) as client:
        render_comparison(
            result,
            ui.label(),
            ui.label(),
            ui.row(),
            ui.column(),
            ui.column(),
        )

    plot_cards = [
        element
        for element in client.elements.values()
        if "vs-statistical-plot-card" in element._classes
    ]
    ranked_labels = [
        element.text
        for element in sorted(client.elements.values(), key=lambda item: item.id)
        if getattr(element, "text", None)
        in {"Best cases", "Median cases", "Worst cases", "Run A", "Run B"}
    ]
    assert len(plot_cards) == 4
    assert all("h-full" in card._classes for card in plot_cards)
    rank_start = ranked_labels.index("Best cases")
    assert ranked_labels[rank_start:] == [
        "Best cases",
        "Run A",
        "Run B",
        "Median cases",
        "Run A",
        "Run B",
        "Worst cases",
        "Run A",
        "Run B",
    ]


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
        representative_card(sample)

    texts = {getattr(element, "text", None) for element in client.elements.values()}
    image = next(element for element in client.elements.values() if element.tag == "nicegui-image")
    assert "Generated preview only" in texts
    assert "Restore the source and target dataset to build the full panel." in texts
    assert ".png" in str(image._props["src"])


def test_opacity_overlap_has_no_manual_slider() -> None:
    with Client(page("/overlap-test")) as client:
        container = ui.column()
        render_image_comparison(
            Image.new("RGB", (32, 32), "red"),
            Image.new("RGB", (32, 32), "blue"),
            container,
        )

    generated_layer = next(
        element
        for element in client.elements.values()
        if "vs-overlap-generated" in element._classes
    )

    assert not any(element.tag == "q-slider" for element in client.elements.values())
    assert generated_layer._style["opacity"] == "0"
    assert any(
        element.tag == "q-btn" and element._props.get("aria-label") == "Save opacity transition GIF"
        for element in client.elements.values()
    )
    assert any(
        element.tag == "q-btn" and element._props.get("aria-label") == "Save instant comparison GIF"
        for element in client.elements.values()
    )


def test_comparison_play_button_animates_opacity_at_new_baseline_speed() -> None:
    with Client(page("/comparison-animation-test")) as client:
        container = ui.column()
        render_image_comparison(
            Image.new("RGB", (32, 32), "red"),
            Image.new("RGB", (32, 32), "blue"),
            container,
        )

    button = next(
        element
        for element in client.elements.values()
        if element.tag == "q-btn" and element._props.get("aria-label") == "Play opacity animation"
    )
    timer = next(
        element
        for element in client.elements.values()
        if "vs-opacity-animation-timer" in element._classes
    )
    generated_layer = next(
        element
        for element in client.elements.values()
        if "vs-overlap-generated" in element._classes
    )
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
    assert speed_input.value == 1.0

    click_listener = next(iter(button._event_listeners.values()))
    assert click_listener.handler is not None
    click_listener.handler(None)

    assert button.icon == "pause"
    assert timer.active is True
    assert generated_layer._style["opacity"] == "0.00"

    assert timer.callback is not None
    timer.callback()

    assert generated_layer._style["opacity"] == "0.04"

    repeat_listener = next(iter(repeat_button._event_listeners.values()))
    assert repeat_listener.handler is not None
    repeat_listener.handler(None)
    for _ in range(24):
        timer.callback()
    timer.callback()

    assert repeat_button._props["aria-pressed"] == "true"
    assert generated_layer._style["opacity"] == "0.00"
    timer.callback()
    assert generated_layer._style["opacity"] == "0.00"


def test_horizontal_slider_reveals_generated_image_over_target() -> None:
    with Client(page("/horizontal-reveal-test")) as client:
        container = ui.column()
        render_image_comparison(
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


def test_comparison_exports_gif_and_current_reveal_pixels() -> None:
    generated = Image.new("RGB", (10, 4), "red")
    target = Image.new("RGB", (10, 4), "blue")

    gif = opacity_transition_gif(generated, target)
    with Image.open(BytesIO(gif)) as animation:
        assert animation.format == "GIF"
        assert animation.n_frames == 26
        animation.seek(animation.n_frames - 1)
        assert animation.convert("RGB").getpixel((0, 0)) == (255, 0, 0)

    reveal = horizontal_reveal_png(generated, target, 40)
    with Image.open(BytesIO(reveal)) as image:
        assert image.getpixel((2, 0)) == (255, 0, 0)
        assert image.getpixel((4, 0)) == (255, 255, 255)
        assert image.getpixel((7, 0)) == (0, 0, 255)

    instant = instant_ab_gif(generated, target)
    with Image.open(BytesIO(instant)) as animation:
        assert animation.format == "GIF"
        assert animation.n_frames == 2
        assert animation.info["duration"] == 650
        animation.seek(1)
        assert animation.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_instant_comparison_starts_and_stops_without_transition() -> None:
    with Client(page("/instant-comparison-test")) as client:
        render_image_comparison(
            Image.new("RGB", (32, 32), "red"),
            Image.new("RGB", (32, 32), "blue"),
            ui.column(),
        )

    button = next(
        element
        for element in client.elements.values()
        if element.tag == "q-btn" and element._props.get("aria-label") == "Play instant comparison"
    )
    timer = next(
        element
        for element in client.elements.values()
        if "vs-hard-cut-animation-timer" in element._classes
    )
    layer = next(
        element
        for element in client.elements.values()
        if "vs-hard-cut-generated" in element._classes
    )
    listener = next(iter(button._event_listeners.values()))
    assert listener.handler is not None and timer.callback is not None

    listener.handler(None)
    timer.callback()
    assert timer.active is True
    assert button.icon == "pause"
    assert layer._style["opacity"] == "1"
    assert "transition" not in layer._style

    listener.handler(None)
    assert timer.active is False
    assert button.icon == "play_arrow"


def test_comparison_case_is_a_four_image_row_with_difference_map(tmp_path: Path) -> None:
    paths = {
        "source": tmp_path / "source.png",
        "generated": tmp_path / "generated.png",
        "target": tmp_path / "target.png",
    }
    Image.new("RGB", (8, 8), "white").save(paths["source"])
    Image.new("RGB", (8, 8), "red").save(paths["generated"])
    Image.new("RGB", (8, 8), "blue").save(paths["target"])
    sample = RepresentativeSample(
        kind="best",
        sample_id="sample",
        metric="ssim",
        value=0.9,
        source_path=paths["source"],
        generated_path=paths["generated"],
        target_path=paths["target"],
    )

    with Client(page("/comparison-case-row-test")) as client:
        comparison_case_row(sample)

    grid = next(
        element for element in client.elements.values() if "vs-case-grid" in element._classes
    )
    case_row = next(
        element for element in client.elements.values() if "vs-case-row" in element._classes
    )
    images = [element for element in client.elements.values() if element.tag == "nicegui-image"]
    texts = {getattr(element, "text", None) for element in client.elements.values()}
    assert "lg:grid-cols-4" in grid._classes
    assert {"w-full", "self-center"}.issubset(case_row._classes)
    assert len(images) == 4
    assert "Absolute difference" in texts


def test_comparison_panels_are_equal_height() -> None:
    with Client(page("/comparison-alignment-test")) as client:
        container = ui.column()
        render_image_comparison(
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
    assert "lg:grid-cols-3" in grid._classes
    assert len(panels) == 3
    assert all("h-full" in panel._classes for panel in panels)


def test_overlap_slider_requires_matching_image_dimensions() -> None:
    with Client(page("/overlap-size-test")) as client:
        container = ui.column()
        render_image_comparison(
            Image.new("RGB", (32, 32)),
            Image.new("RGB", (64, 32)),
            container,
        )

    assert not any(element.tag == "q-slider" for element in client.elements.values())
    assert any(
        getattr(element, "text", None) == "Comparison unavailable"
        for element in client.elements.values()
    )
