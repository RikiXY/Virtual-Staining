from __future__ import annotations

from nicegui import ui

from virtual_staining.applications.api import MetricQuality

# The palette is intentionally kept here so visual grading can be adjusted in one place.
# Numeric, direction-aware cutoffs live in metrics.METRIC_SPECS and reach the UI via the API.
METRIC_QUALITY_STYLES: dict[MetricQuality, tuple[str, str]] = {
    "very_good": ("Very good", "vs-metric--very-good"),
    "good": ("Good", "vs-metric--good"),
    "fair": ("Fair", "vs-metric--fair"),
    "poor": ("Poor", "vs-metric--poor"),
    "very_poor": ("Very poor", "vs-metric--very-poor"),
    "unknown": ("Not graded", "vs-metric--unknown"),
}

PAGE_CSS = """
:root {
  --vs-ink: #14213d;
  --vs-muted: #64748b;
  --vs-primary: #0f766e;
  --vs-primary-soft: #ccfbf1;
  --vs-accent: #0369a1;
  --vs-surface: #ffffff;
  --vs-bg: #f5f7fa;
  --vs-line: #dbe3ec;
}
html { font-size: 18px; }
body { background: var(--vs-bg); color: var(--vs-ink); font-size: 1rem; }
.q-btn, .q-tab__label { font-size: 1rem; }
.q-field__native, .q-field__input, .q-field__label { font-size: 1rem; }
.q-table { font-size: 1rem; }
.q-table th { font-size: .9375rem; }
.q-badge { font-size: .9375rem; }
.vs-header { background: rgba(255,255,255,.96); border-bottom: 1px solid var(--vs-line); }
.vs-menu-drawer { background: var(--vs-surface); }
.vs-menu-item {
  width: calc(100% - 1.5rem); min-height: 3rem; justify-content: flex-start;
  border-radius: .65rem; padding-inline: 1rem;
}
.vs-menu-item .q-btn__content { justify-content: flex-start; gap: .75rem; }
.vs-menu-trigger { width: 3.5rem; height: 3.5rem; }
.vs-menu-trigger .q-icon { font-size: 2rem; }
.vs-card {
  background: var(--vs-surface); border: 1px solid var(--vs-line);
  box-shadow: 0 1px 3px rgb(15 23 42 / .04);
}
.vs-subtle { background: #f8fafc; border: 1px solid #e2e8f0; }
.vs-image-frame {
  position: relative; background: #f8fafc; border: 1px solid #dbe3ec;
  overflow: hidden; min-height: 15rem;
}
.vs-image { position: absolute !important; inset: 0; width: 100%; height: 100% !important; }
.vs-image .q-img__image { object-fit: contain !important; }
.vs-overlap-frame { max-width: 48rem; margin-inline: auto; }
.vs-comparison-header { min-height: 3.5rem; }
.vs-comparison-panel { height: 100%; }
.vs-overlap-generated { pointer-events: none; transition: opacity 80ms linear; }
.vs-reveal-generated { pointer-events: none; transition: clip-path 60ms linear; }
.vs-reveal-divider {
  position: absolute; inset-block: 0; width: 2rem; z-index: 2; cursor: ew-resize;
  touch-action: none; transform: translateX(-50%); transition: left 60ms linear;
}
.vs-reveal-divider::before {
  content: ""; position: absolute; inset-block: 0; left: 50%; width: 2px;
  background: white; box-shadow: 0 0 0 1px rgb(15 23 42 / .3);
  transform: translateX(-1px);
}
.vs-reveal-divider::after {
  content: "↔"; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  display: grid; place-items: center; width: 2rem; height: 2rem; border-radius: 9999px;
  color: #0f766e; background: white; border: 1px solid #cbd5e1;
  box-shadow: 0 2px 6px rgb(15 23 42 / .2); font-size: .875rem; font-weight: 700;
}
.vs-metric { border-left: 4px solid #94a3b8; background: #f8fafc; }
.vs-metric--very-good { border-left-color: #16a34a; background: #f0fdf4; }
.vs-metric--good { border-left-color: #84a914; background: #f7fee7; }
.vs-metric--fair { border-left-color: #eab308; background: #fefce8; }
.vs-metric--poor { border-left-color: #f97316; background: #fff7ed; }
.vs-metric--very-poor { border-left-color: #dc2626; background: #fef2f2; }
.vs-metric--unknown { border-left-color: #94a3b8; background: #f8fafc; }
.vs-header-bar {
  display: grid; grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
}
.vs-section-tabs { align-self: center; width: auto; max-width: 100%; }
.vs-section-tabs .q-tabs__content { justify-content: center !important; }
.vs-section-tabs .q-tab { flex: 0 0 auto !important; text-transform: none; }
.q-uploader { box-shadow: none !important; border: 1px dashed #94a3b8; border-radius: .75rem; }
.q-dialog__inner > .vs-tutorial { max-width: 980px !important; }
.vs-tutorial-footer {
  display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  align-items: center;
}
.vs-step-number {
  display: grid !important; place-items: center; width: 1.5rem; height: 1.5rem;
  min-width: 1.5rem; padding: 0 !important; line-height: 1 !important;
}
@media (max-width: 640px) {
  .vs-image-frame { min-height: 12rem; }
  .vs-header-bar { gap: .75rem; }
  .vs-brand-copy .text-2xl { font-size: 1.25rem; }
  .vs-current-page { font-size: .9375rem; }
}
"""


def configure_theme() -> None:
    ui.page_title("Virtual Staining")
    ui.colors(primary="#0f766e", secondary="#0369a1", accent="#7c3aed")
    ui.add_css(PAGE_CSS)


def metric_quality_style(quality: MetricQuality) -> tuple[str, str]:
    """Return the short label and CSS class for one metric quality level."""
    return METRIC_QUALITY_STYLES[quality]


def section_heading(icon: str, title: str, description: str) -> None:
    with ui.row().classes("w-full items-start gap-3"):
        ui.icon(icon, size="sm").classes("text-teal-700 mt-1")
        with ui.column().classes("gap-0 min-w-0"):
            ui.label(title).classes("text-xl font-semibold text-slate-800")
            ui.label(description).classes("text-base text-slate-500")


def empty_state(icon: str, title: str, detail: str) -> None:
    with ui.column().classes("w-full items-center justify-center text-center gap-2 py-10 px-4"):
        ui.icon(icon, size="lg").classes("text-slate-300")
        ui.label(title).classes("font-medium text-slate-600")
        ui.label(detail).classes("text-base text-slate-400 max-w-lg")
