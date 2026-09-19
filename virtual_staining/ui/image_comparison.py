from __future__ import annotations

import logging
from collections.abc import Callable
from io import BytesIO
from time import monotonic

from nicegui import events, run, ui
from PIL import Image, ImageDraw

from virtual_staining.ui.theme import empty_state, section_heading

logger = logging.getLogger(__name__)


def render_image_comparison(
    generated: Image.Image,
    target: Image.Image,
    container,
) -> None:
    """Render the three pixel-aligned generated/target comparison tools."""
    with container:
        section_heading(
            "compare",
            "Generated / target comparison",
            "Inspect alignment with a blend, a draggable reveal, or an instant A/B cut.",
        )
        if generated.size != target.size:
            empty_state(
                "aspect_ratio",
                "Comparison unavailable",
                "The generated and target images must have matching dimensions.",
            )
            return

        with ui.row().classes(
            "vs-comparison-grid w-full grid grid-cols-1 lg:grid-cols-3 gap-4 items-stretch"
        ):
            with ui.column().classes(
                "vs-comparison-panel vs-subtle rounded-xl p-3 gap-3 min-w-0 h-full"
            ):
                _render_opacity_comparison(generated, target)
            with ui.column().classes(
                "vs-comparison-panel vs-subtle rounded-xl p-3 gap-3 min-w-0 h-full"
            ):
                _render_horizontal_reveal(generated, target)
            with ui.column().classes(
                "vs-comparison-panel vs-subtle rounded-xl p-3 gap-3 min-w-0 h-full"
            ):
                _render_hard_cut_comparison(generated, target)


def _render_opacity_comparison(generated: Image.Image, target: Image.Image) -> None:
    opacity_value = 0.0
    with ui.column().classes("vs-comparison-header w-full gap-2"):
        with ui.column().classes("gap-0 min-w-0"):
            ui.label("Opacity overlap").classes("text-base font-semibold text-slate-800")
            ui.label("Blend the generated image over the target.").classes("text-sm text-slate-500")
        with ui.row().classes("w-full items-center justify-end gap-1 flex-wrap"):
            play_button, speed_input, is_looping = _playback_controls("opacity")
            save_button = ui.button(icon="download").props(
                'round flat dense color=blue-grey-7 aria-label="Save opacity transition GIF"'
            )
            save_button.tooltip("Save transition GIF")
    with ui.element("div").classes(
        "vs-image-frame vs-overlap-frame w-full aspect-square rounded-lg"
    ):
        ui.image(target).props("fit=contain").classes("vs-image")
        generated_layer = (
            ui.image(generated)
            .props("fit=contain")
            .classes("vs-image vs-overlap-generated")
            .style("opacity: 0")
        )

    with ui.row().classes("w-full items-center justify-between gap-3"):
        ui.label("Target").classes("text-sm font-medium text-slate-500")
        opacity_label = ui.label("0% generated").classes("text-sm font-semibold text-slate-700")
        ui.label("Generated").classes("text-sm font-medium text-slate-500")

    def set_opacity(value: float) -> None:
        nonlocal opacity_value
        value = min(100.0, max(0.0, float(value)))
        opacity_value = value
        generated_layer.style(f"opacity: {value / 100:.2f}")
        opacity_label.set_text(f"{value:.0f}% generated")

    _attach_progress_animation(
        lambda: opacity_value,
        set_opacity,
        play_button,
        speed_input,
        is_looping,
        "opacity",
    )

    async def save_transition() -> None:
        save_button.disable()
        save_button.props("loading")
        try:
            content = await run.io_bound(
                opacity_transition_gif,
                generated,
                target,
                _animation_speed(speed_input.value),
            )
            assert content is not None
            ui.download(content, filename="opacity_transition.gif", media_type="image/gif")
        except (OSError, ValueError):
            logger.exception("Could not build opacity transition GIF")
            ui.notify("The transition GIF could not be created.", type="negative")
        finally:
            save_button.props(remove="loading")
            save_button.enable()

    save_button.on_click(save_transition)


def _render_horizontal_reveal(generated: Image.Image, target: Image.Image) -> None:
    reveal_value = 50.0
    with ui.column().classes("vs-comparison-header w-full gap-2"):
        with ui.column().classes("gap-0 min-w-0"):
            ui.label("Horizontal reveal").classes("text-base font-semibold text-slate-800")
            ui.label("Drag the divider to set the generated coverage.").classes(
                "text-sm text-slate-500"
            )
        with ui.row().classes("w-full items-center justify-end"):
            save_button = ui.button(icon="download").props(
                'round flat dense color=blue-grey-7 aria-label="Save horizontal reveal image"'
            )
            save_button.tooltip("Save reveal image")
    with ui.element("div").classes(
        "vs-image-frame vs-overlap-frame w-full aspect-square rounded-lg"
    ) as reveal_frame:
        ui.image(target).props("fit=contain").classes("vs-image")
        generated_layer = (
            ui.image(generated)
            .props("fit=contain")
            .classes("vs-image vs-reveal-generated")
            .style("clip-path: inset(0 50% 0 0)")
        )
        divider = ui.element("div").classes("vs-reveal-divider").style("left: 50%")

    reveal_label = ui.label("50% revealed").classes(
        "w-full text-center text-sm font-semibold text-slate-700"
    )

    def set_reveal(value: float) -> None:
        nonlocal reveal_value
        value = min(100.0, max(0.0, float(value)))
        reveal_value = value
        generated_layer.style(f"clip-path: inset(0 {100 - value:.0f}% 0 0)")
        divider.style(f"left: {value:.0f}%")
        reveal_label.set_text(f"{value:.0f}% revealed")

    def update_reveal_from_drag(event: events.GenericEventArguments) -> None:
        set_reveal(float(event.args))

    # The browser updates the image immediately while throttled events keep Python state
    # synchronized for exporting the current reveal position.
    divider.on(
        "pointerdown",
        update_reveal_from_drag,
        throttle=0.03,
        trailing_events=True,
        js_handler=f"""(event) => {{
            event.preventDefault();
            const handle = event.currentTarget;
            const frame = document.getElementById('{reveal_frame.html_id}');
            const layer = document.getElementById('{generated_layer.html_id}');
            if (!frame || !layer) return;
            handle.setPointerCapture(event.pointerId);
            const update = (clientX) => {{
                const bounds = frame.getBoundingClientRect();
                const value = Math.max(0, Math.min(100,
                    ((clientX - bounds.left) / bounds.width) * 100));
                layer.style.clipPath = 'inset(0 ' + (100 - value) + '% 0 0)';
                handle.style.left = value + '%';
                emit(value);
            }};
            const move = (moveEvent) => update(moveEvent.clientX);
            const finish = (finishEvent) => {{
                update(finishEvent.clientX);
                handle.removeEventListener('pointermove', move);
                handle.removeEventListener('pointerup', finish);
                handle.removeEventListener('pointercancel', finish);
                if (handle.hasPointerCapture(event.pointerId))
                    handle.releasePointerCapture(event.pointerId);
            }};
            handle.addEventListener('pointermove', move);
            handle.addEventListener('pointerup', finish, {{once: true}});
            handle.addEventListener('pointercancel', finish, {{once: true}});
            update(event.clientX);
        }}""",
    )

    def save_reveal() -> None:
        try:
            content = horizontal_reveal_png(generated, target, reveal_value)
            filename = f"horizontal_reveal_{round(reveal_value):03d}_percent.png"
            ui.download(content, filename=filename, media_type="image/png")
        except (OSError, ValueError):
            logger.exception("Could not build horizontal reveal image")
            ui.notify("The reveal image could not be created.", type="negative")

    save_button.on_click(save_reveal)


def _render_hard_cut_comparison(generated: Image.Image, target: Image.Image) -> None:
    showing_generated = False
    with ui.column().classes("vs-comparison-header w-full gap-2"):
        with ui.column().classes("gap-0 min-w-0"):
            ui.label("Instant A/B").classes("text-base font-semibold text-slate-800")
            ui.label("Alternate target and generated with a hard cut.").classes(
                "text-sm text-slate-500"
            )
        with ui.row().classes("w-full items-center justify-end gap-1"):
            play_button = ui.button(icon="play_arrow").props(
                'round flat dense color=primary aria-label="Play instant comparison"'
            )
            play_button.tooltip("Start instant comparison")
            save_button = ui.button(icon="download").props(
                'round flat dense color=blue-grey-7 aria-label="Save instant comparison GIF"'
            )
            save_button.tooltip("Save instant A/B GIF")

    with ui.element("div").classes(
        "vs-image-frame vs-overlap-frame w-full aspect-square rounded-lg"
    ):
        ui.image(target).props("fit=contain").classes("vs-image")
        generated_layer = (
            ui.image(generated)
            .props("fit=contain")
            .classes("vs-image vs-hard-cut-generated")
            .style("opacity: 0")
        )
    state_label = ui.label("Target").classes(
        "w-full text-center text-sm font-semibold text-slate-700"
    )

    def swap() -> None:
        nonlocal showing_generated
        showing_generated = not showing_generated
        generated_layer.style(f"opacity: {1 if showing_generated else 0}")
        state_label.set_text("Generated" if showing_generated else "Target")

    timer = ui.timer(0.65, swap, active=False, immediate=False)
    timer.classes("vs-hard-cut-animation-timer")

    def toggle() -> None:
        if timer.active:
            timer.deactivate()
            play_button.set_icon("play_arrow")
            play_button.props('aria-label="Play instant comparison"')
            play_button.tooltip("Start instant comparison")
        else:
            timer.activate()
            play_button.set_icon("pause")
            play_button.props('aria-label="Pause instant comparison"')
            play_button.tooltip("Stop instant comparison")

    play_button.on_click(toggle)

    async def save_transition() -> None:
        save_button.disable()
        save_button.props("loading")
        try:
            content = await run.io_bound(instant_ab_gif, generated, target)
            assert content is not None
            ui.download(content, filename="instant_ab.gif", media_type="image/gif")
        except (OSError, ValueError):
            logger.exception("Could not build instant A/B GIF")
            ui.notify("The instant A/B GIF could not be created.", type="negative")
        finally:
            save_button.props(remove="loading")
            save_button.enable()

    save_button.on_click(save_transition)


def _animation_speed(value: object) -> float:
    if not isinstance(value, str | int | float):
        return 1.0
    try:
        return min(4.0, max(0.25, float(value or 1.0)))
    except (TypeError, ValueError):
        return 1.0


def opacity_transition_gif(
    generated: Image.Image,
    target: Image.Image,
    speed: float = 1.0,
) -> bytes:
    """Build the same target-to-generated opacity transition shown by the player."""
    if generated.size != target.size:
        raise ValueError("Generated and target images must have matching dimensions.")
    generated_rgb = generated.convert("RGB")
    target_rgb = target.convert("RGB")
    frames = [Image.blend(target_rgb, generated_rgb, step / 25) for step in range(26)]
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=max(10, round(40 / _animation_speed(speed))),
        loop=0,
        disposal=2,
    )
    return output.getvalue()


def instant_ab_gif(
    generated: Image.Image,
    target: Image.Image,
    frame_duration_ms: int = 650,
) -> bytes:
    """Build the hard-cut target/generated animation shown by Instant A/B."""
    if generated.size != target.size:
        raise ValueError("Generated and target images must have matching dimensions.")
    frames = (target.convert("RGB"), generated.convert("RGB"))
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=[frames[1]],
        duration=max(40, int(frame_duration_ms)),
        loop=0,
        disposal=2,
    )
    return output.getvalue()


def horizontal_reveal_png(
    generated: Image.Image,
    target: Image.Image,
    percentage: float,
) -> bytes:
    """Build a static target image covered by generated pixels up to ``percentage``."""
    if generated.size != target.size:
        raise ValueError("Generated and target images must have matching dimensions.")
    percentage = min(100.0, max(0.0, float(percentage)))
    result = target.convert("RGB").copy()
    generated_rgb = generated.convert("RGB")
    boundary = round(result.width * percentage / 100)
    if boundary:
        result.paste(generated_rgb.crop((0, 0, boundary, result.height)), (0, 0))
    if 0 < boundary < result.width:
        # Keep the separator visible over both pale stains and dark tissue structures.
        draw = ImageDraw.Draw(result)
        draw.line((boundary, 0, boundary, result.height - 1), fill=(15, 23, 42), width=4)
        draw.line((boundary, 0, boundary, result.height - 1), fill=(255, 255, 255), width=2)
    output = BytesIO()
    result.save(output, format="PNG")
    return output.getvalue()


def _playback_controls(name: str):
    looping = False
    with ui.row().classes("items-center justify-end gap-1 flex-wrap"):
        speed_input = (
            ui.number(
                "Speed",
                value=1.0,
                min=0.25,
                max=4.0,
                precision=2,
                step=0.25,
                suffix="×",
            )
            .props(f'dense outlined hide-bottom-space aria-label="{name.title()} animation speed"')
            .classes("vs-playback-speed w-24")
        )
        repeat_button = ui.button(icon="repeat").props(
            f'round flat dense color=blue-grey-7 aria-label="Loop {name} animation" '
            'aria-pressed="false"'
        )
        repeat_button.tooltip("Toggle continuous replay")
        play_button = ui.button(icon="play_arrow").props(
            f'round flat dense color=primary aria-label="Play {name} animation"'
        )
        play_button.tooltip(f"Play {name} animation")

    def toggle_loop() -> None:
        nonlocal looping
        looping = not looping
        if looping:
            repeat_button.props(remove="flat")
            repeat_button.props('unelevated color=primary aria-pressed="true"')
        else:
            repeat_button.props(remove="unelevated")
            repeat_button.props('flat color=blue-grey-7 aria-pressed="false"')

    repeat_button.on_click(toggle_loop)
    return play_button, speed_input, lambda: looping


def _attach_progress_animation(
    get_value: Callable[[], float],
    set_value: Callable[[float], None],
    play_button,
    speed_input,
    is_looping: Callable[[], bool],
    name: str,
) -> None:
    """Animate one comparison from zero with configurable speed and replay."""
    can_resume = False
    zero_hold_until = 0.0

    def stop(*, resumable: bool = False) -> None:
        nonlocal can_resume
        can_resume = resumable
        timer.deactivate()
        play_button.set_icon("play_arrow")
        play_button.props(f'aria-label="Play {name} animation"')

    def advance() -> None:
        nonlocal zero_hold_until
        if monotonic() < zero_hold_until:
            return
        value = get_value()
        if value >= 100:
            if is_looping():
                set_value(0)
                # Hold the reset briefly so it is visible before the next replay.
                zero_hold_until = monotonic() + 0.6
                return
            stop(resumable=False)
            return
        speed = _animation_speed(speed_input.value)
        set_value(min(100.0, value + 4.0 * speed))

    timer = ui.timer(0.04, advance, active=False, immediate=False)
    timer.classes(f"vs-{name}-animation-timer")

    def toggle() -> None:
        nonlocal can_resume
        if timer.active:
            stop(resumable=True)
            return
        if not can_resume:
            set_value(0)
        can_resume = False
        play_button.set_icon("pause")
        play_button.props(f'aria-label="Pause {name} animation"')
        timer.activate()

    play_button.on_click(toggle)
