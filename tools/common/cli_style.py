from __future__ import annotations

import os
import sys


ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "orange": "\033[38;5;208m",
}


def use_color() -> bool:
    """Restituisce True se ha senso usare colori ANSI in console."""
    return os.environ.get("NO_COLOR") is None and sys.stdout.isatty()


def style(text: str, *names: str) -> str:
    """Applica uno stile ANSI al testo, se la colorazione è abilitata."""
    if not use_color():
        return text

    prefix = "".join(ANSI[name] for name in names if name in ANSI)
    return prefix + text + ANSI["reset"]


def print_section(title: str) -> None:
    """Stampa un'intestazione di sezione leggibile in CLI."""
    print()
    print(style(f"=== {title} ===", "bold", "cyan"))


def print_info(label: str, value: str) -> None:
    """Stampa una singola riga etichettata."""
    print(f"{style(label + ':', 'bold', 'blue')} {value}")


def color_metric(name: str, value: float) -> str:
    """Colora una metrica con soglie pensate per la sola lettura CLI."""
    if name == "ssim":
        if value >= 0.85:
            color = "green"
        elif value >= 0.75:
            color = "yellow"
        elif value >= 0.65:
            color = "orange"
        else:
            color = "red"
        return style(f"{value:.6f}", color)

    if name == "psnr":
        if value >= 25:
            color = "green"
        elif value >= 20:
            color = "yellow"
        elif value >= 15:
            color = "orange"
        else:
            color = "red"
        return style(f"{value:.4f}", color)

    if name == "mae":
        if value <= 0.06:
            color = "green"
        elif value <= 0.10:
            color = "yellow"
        elif value <= 0.16:
            color = "orange"
        else:
            color = "red"
        return style(f"{value:.6f}", color)

    if name == "rmse":
        if value <= 0.08:
            color = "green"
        elif value <= 0.12:
            color = "yellow"
        elif value <= 0.20:
            color = "orange"
        else:
            color = "red"
        return style(f"{value:.6f}", color)

    if name == "mse":
        if value <= 0.0036:
            color = "green"
        elif value <= 0.0100:
            color = "yellow"
        elif value <= 0.0256:
            color = "orange"
        else:
            color = "red"
        return style(f"{value:.6f}", color)

    if name in {"pcc_gray", "pcc_rgb_mean", "pcc_r", "pcc_g", "pcc_b"}:
        if value >= 0.95:
            color = "green"
        elif value >= 0.90:
            color = "yellow"
        elif value >= 0.80:
            color = "orange"
        else:
            color = "red"
        return style(f"{value:.6f}", color)

    return f"{value:.6f}"


