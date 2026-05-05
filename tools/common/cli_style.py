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


def metric_color_name(name: str, value: float) -> str:
    """Restituisce il nome del colore ANSI associato al valore della metrica."""
    if name == "ssim":
        if value >= 0.85:
            return "green"
        if value >= 0.75:
            return "yellow"
        if value >= 0.65:
            return "orange"
        return "red"

    if name == "psnr":
        if value >= 25:
            return "green"
        if value >= 20:
            return "yellow"
        if value >= 15:
            return "orange"
        return "red"

    if name == "mae":
        if value <= 0.06:
            return "green"
        if value <= 0.10:
            return "yellow"
        if value <= 0.16:
            return "orange"
        return "red"

    if name == "rmse":
        if value <= 0.08:
            return "green"
        if value <= 0.12:
            return "yellow"
        if value <= 0.20:
            return "orange"
        return "red"

    if name == "mse":
        if value <= 0.0036:
            return "green"
        if value <= 0.0100:
            return "yellow"
        if value <= 0.0256:
            return "orange"
        return "red"

    if name in {"pcc_gray", "pcc_rgb_mean", "pcc_r", "pcc_g", "pcc_b"}:
        if value >= 0.95:
            return "green"
        if value >= 0.90:
            return "yellow"
        if value >= 0.80:
            return "orange"
        return "red"

    return "cyan"


def color_metric(name: str, value: float) -> str:
    """Restituisce il valore della metrica formattato e colorato."""
    color = metric_color_name(name, value)

    if name == "psnr":
        formatted_value = f"{value:.4f}"
    else:
        formatted_value = f"{value:.6f}"

    return style(formatted_value, color)


def color_distance(value: float, good: float, warn: float) -> str:
    """Colora una distanza: più è piccola, meglio è."""
    if value <= good:
        return style(f"{value:.6f}", "green")
    if value <= warn:
        return style(f"{value:.6f}", "yellow")
    if value <= warn * 1.5:
        return style(f"{value:.6f}", "orange")
    return style(f"{value:.6f}", "red")


def color_pvalue(value: float) -> str:
    """Colora un p-value come forza dell'evidenza di differenza."""
    if value < 0.001:
        return style(f"{value:.6g}", "green")
    if value < 0.01:
        return style(f"{value:.6g}", "yellow")
    if value < 0.05:
        return style(f"{value:.6g}", "orange")
    return style(f"{value:.6g}", "red")


def color_share(value: float) -> str:
    """Colora una quota tra 0 e 1."""
    if value >= 0.70:
        return style(f"{value:.6f}", "green")
    if value >= 0.40:
        return style(f"{value:.6f}", "yellow")
    if value >= 0.20:
        return style(f"{value:.6f}", "orange")
    return style(f"{value:.6f}", "red")


def color_signed_delta(value: float) -> str:
    """Colora un delta signed: positivo meglio per B, negativo meglio per A."""
    if value > 0:
        return style(f"{value:.6f}", "green")
    if value < 0:
        return style(f"{value:.6f}", "red")
    return style(f"{value:.6f}", "yellow")

