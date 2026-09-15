"""Découverte des périphériques V4L2 et lecture de leurs capacités."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Iterable


COMMON_RESOLUTIONS = (
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1600, 1200),
    (1280, 720),
    (1024, 768),
    (800, 600),
    (640, 480),
    (320, 240),
)

COMMON_FPS = (120.0, 90.0, 60.0, 50.0, 30.0, 25.0, 24.0, 20.0, 15.0, 10.0, 5.0)

CONTROL_LABELS = {
    "brightness": "Luminosité",
    "contrast": "Contraste",
    "saturation": "Saturation",
    "hue": "Teinte",
    "gamma": "Gamma",
    "gain": "Gain",
    "sharpness": "Netteté",
    "backlight_compensation": "Compensation rétroéclairage",
    "power_line_frequency": "Fréquence secteur",
    "white_balance_automatic": "Balance des blancs auto",
    "white_balance_temperature": "Température des blancs",
    "exposure": "Exposition",
    "exposure_time_absolute": "Temps d'exposition",
    "exposure_auto": "Exposition auto",
    "exposure_dynamic_framerate": "FPS dynamique",
    "focus_automatic_continuous": "Mise au point auto",
    "focus_absolute": "Mise au point",
    "zoom_absolute": "Zoom",
    "pan_absolute": "Panoramique",
    "tilt_absolute": "Inclinaison",
    "auto_exposure": "Exposition auto",
}


@dataclass(frozen=True)
class FrameSize:
    width: int
    height: int
    fps: tuple[float, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class PixelFormat:
    fourcc: str
    description: str
    sizes: tuple[FrameSize, ...] = ()

    @property
    def label(self) -> str:
        extra = f" — {self.description}" if self.description else ""
        return f"{self.fourcc}{extra}"


@dataclass
class CameraControl:
    name: str
    ctrl_type: str
    minimum: float = 0
    maximum: float = 0
    step: float = 1
    default: float = 0
    value: float = 0
    menu: tuple[tuple[int, str], ...] = ()

    @property
    def label(self) -> str:
        return CONTROL_LABELS.get(self.name, self.name.replace("_", " ").capitalize())


@dataclass
class VideoDevice:
    path: str
    name: str
    bus_info: str = ""
    card: str = ""
    driver: str = ""
    formats: tuple[PixelFormat, ...] = ()
    controls: tuple[CameraControl, ...] = ()
    capture: bool = True
    virtual: bool = False

    @property
    def display_name(self) -> str:
        return self.name or self.card or self.path

    @property
    def subtitle(self) -> str:
        parts = [self.path]
        if self.bus_info:
            parts.append(self.bus_info)
        return " · ".join(parts)


def list_video_devices() -> list[VideoDevice]:
    """Liste les devices capables de capturer de la vidéo."""
    devices: list[VideoDevice] = []
    seen: set[str] = set()

    for path, name, bus in _enumerate_nodes():
        if path in seen:
            continue
        seen.add(path)
        if not _is_capture_device(path):
            continue
        info = _device_info(path)
        devices.append(
            VideoDevice(
                path=path,
                name=name or info.get("card") or os.path.basename(path),
                bus_info=bus or info.get("bus_info", ""),
                card=info.get("card", ""),
                driver=info.get("driver", ""),
                capture=True,
            )
        )

    devices.sort(key=lambda d: _device_sort_key(d.path))
    return devices


def probe_device(device: VideoDevice) -> VideoDevice:
    """Remplit formats et contrôles V4L2 pour un device."""
    if device.virtual:
        return device

    formats = _parse_formats(_run_v4l2(device.path, "--list-formats-ext"))
    if not formats:
        formats = _parse_formats(_run_v4l2(device.path, "--list-formats"))
    controls = _parse_controls(_run_v4l2(device.path, "--list-ctrls-menus"))
    if not controls:
        controls = _parse_controls(_run_v4l2(device.path, "--list-ctrls"))

    device.formats = tuple(formats)
    device.controls = tuple(controls)
    return device


def set_control(device_path: str, name: str, value: float | int | str) -> None:
    _run_v4l2(device_path, f"--set-ctrl={name}={value}", check=False)


def configure_video(
    device_path: str,
    fourcc: str,
    width: int,
    height: int,
    fps: float,
) -> None:
    """Impose le format au driver avant qu'OpenCV n'ouvre le device."""
    _run_v4l2(
        device_path,
        f"--set-fmt-video=width={width},height={height},pixelformat={fourcc}",
    )
    if fps > 0:
        _run_v4l2(device_path, f"--set-parm={fps:g}")


def fourcc_to_str(value: int) -> str:
    if value <= 0:
        return ""
    chars = [chr((value >> (8 * i)) & 0xFF) for i in range(4)]
    if not all(32 <= ord(c) < 127 for c in chars):
        return ""
    return "".join(chars)


def preferred_format(formats: Iterable[PixelFormat]) -> PixelFormat | None:
    formats = list(formats)
    if not formats:
        return None

    def score(fmt: PixelFormat) -> tuple:
        fourcc = fmt.fourcc.upper()
        # MJPG passe bien en USB ; H264 via OpenCV est souvent saccadé.
        codec = 0
        if fourcc in {"MJPG", "JPEG", "MJPEG"}:
            codec = 4
        elif fourcc in {"NV12", "NV21"}:
            codec = 3
        elif fourcc in {"YUYV", "YUY2", "UYVY"}:
            codec = 2
        elif fourcc in {"H264", "H265", "HEVC", "AVC1"}:
            codec = 1
        best = max(fmt.sizes, key=lambda s: s.width * s.height, default=None)
        pixels = (best.width * best.height) if best else 0
        return (codec, pixels)

    return max(formats, key=score)


def preferred_size(fmt: PixelFormat | None) -> FrameSize | None:
    if not fmt or not fmt.sizes:
        return None
    return max(fmt.sizes, key=lambda s: (s.width * s.height, _best_fps(s.fps)))


def preferred_fps(size: FrameSize | None) -> float:
    if not size or not size.fps:
        return 30.0
    if 30.0 in size.fps:
        return 30.0
    return _best_fps(size.fps)


def _best_fps(values: Iterable[float]) -> float:
    values = list(values)
    return max(values) if values else 30.0


def _enumerate_nodes() -> list[tuple[str, str, str]]:
    nodes: list[tuple[str, str, str]] = []
    listed = _parse_list_devices(_run_v4l2_global("--list-devices"))
    if listed:
        nodes.extend(listed)

    for path in sorted(glob.glob("/dev/video*"), key=_device_sort_key):
        if any(existing == path for existing, _, _ in nodes):
            continue
        name = _sysfs_name(path) or os.path.basename(path)
        nodes.append((path, name, ""))
    return nodes


def _parse_list_devices(text: str) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    name = ""
    bus = ""
    header = re.compile(r"^(?P<name>.+?)(?:\s+\((?P<bus>[^)]+)\))?:\s*$")

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            name = ""
            bus = ""
            continue
        if not line.startswith("\t") and not line.startswith(" "):
            match = header.match(line)
            if match:
                name = match.group("name").strip()
                bus = (match.group("bus") or "").strip()
            else:
                name = line.rstrip(":").strip()
                bus = ""
            continue
        path = line.strip()
        if path.startswith("/dev/video"):
            entries.append((path, name, bus))
    return entries


def _is_capture_device(path: str) -> bool:
    info = _run_v4l2(path, "--info")
    if not info:
        return os.path.exists(path)

    device_caps = _section_after(info, "Device Caps")
    haystack = device_caps or info
    if "Meta Capture" in haystack and "Video Capture" not in haystack:
        return False
    if "Video Capture" in haystack:
        return True

    formats = _run_v4l2(path, "--list-formats")
    return "Video Capture" in formats or "pixelformat" in formats.lower()


def _device_info(path: str) -> dict[str, str]:
    text = _run_v4l2(path, "--info")
    info: dict[str, str] = {}
    for key, pattern in (
        ("driver", r"Driver name\s*:\s*(.+)"),
        ("card", r"Card type\s*:\s*(.+)"),
        ("bus_info", r"Bus info\s*:\s*(.+)"),
    ):
        match = re.search(pattern, text)
        if match:
            info[key] = match.group(1).strip()
    return info


def _sysfs_name(path: str) -> str:
    index = path.replace("/dev/video", "")
    name_file = f"/sys/class/video4linux/video{index}/name"
    try:
        with open(name_file, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _parse_formats(text: str) -> list[PixelFormat]:
    formats: list[PixelFormat] = []
    current: dict | None = None
    current_size: dict | None = None

    fmt_re = re.compile(r"\[(\d+)\]:\s+'([^']+)'\s+\(([^)]*)\)")
    discrete_re = re.compile(r"Size:\s+Discrete\s+(\d+)x(\d+)")
    stepwise_re = re.compile(
        r"Size:\s+Stepwise\s+(\d+)x(\d+)\s+-\s+(\d+)x(\d+)"
        r"(?:\s+with step\s+(\d+)/(\d+))?"
    )
    fps_re = re.compile(r"Interval:.*\(([\d.]+)\s*fps\)", re.IGNORECASE)
    fps_range_re = re.compile(
        r"Interval:\s+Stepwise.*\(([\d.]+)\s*-\s*([\d.]+)\s*fps\)",
        re.IGNORECASE,
    )

    def flush_size() -> None:
        nonlocal current_size
        if current is not None and current_size is not None:
            current["sizes"].append(
                FrameSize(
                    width=current_size["width"],
                    height=current_size["height"],
                    fps=tuple(sorted(set(current_size["fps"]), reverse=True)),
                )
            )
        current_size = None

    def flush_format() -> None:
        nonlocal current
        flush_size()
        if current is not None:
            formats.append(
                PixelFormat(
                    fourcc=current["fourcc"],
                    description=current["description"],
                    sizes=tuple(current["sizes"]),
                )
            )
        current = None

    for line in text.splitlines():
        stripped = line.strip()
        match = fmt_re.search(stripped)
        if match:
            flush_format()
            current = {
                "fourcc": match.group(2),
                "description": match.group(3),
                "sizes": [],
            }
            continue

        if current is None:
            continue

        discrete = discrete_re.search(stripped)
        if discrete:
            flush_size()
            current_size = {
                "width": int(discrete.group(1)),
                "height": int(discrete.group(2)),
                "fps": [],
            }
            continue

        stepwise = stepwise_re.search(stripped)
        if stepwise:
            flush_size()
            min_w, min_h = int(stepwise.group(1)), int(stepwise.group(2))
            max_w, max_h = int(stepwise.group(3)), int(stepwise.group(4))
            for width, height in COMMON_RESOLUTIONS:
                if min_w <= width <= max_w and min_h <= height <= max_h:
                    current["sizes"].append(FrameSize(width=width, height=height, fps=COMMON_FPS))
            if not any(s.width == max_w and s.height == max_h for s in current["sizes"]):
                current["sizes"].append(FrameSize(width=max_w, height=max_h, fps=COMMON_FPS))
            current_size = None
            continue

        fps_range = fps_range_re.search(stripped)
        if fps_range and current_size is not None:
            low, high = float(fps_range.group(1)), float(fps_range.group(2))
            current_size["fps"].extend(fps for fps in COMMON_FPS if low <= fps <= high)
            continue

        fps = fps_re.search(stripped)
        if fps and current_size is not None:
            current_size["fps"].append(round(float(fps.group(1)), 3))

    flush_format()
    return formats


def _parse_controls(text: str) -> list[CameraControl]:
    controls: list[CameraControl] = []
    current: CameraControl | None = None
    ctrl_re = re.compile(
        r"^\s*([A-Za-z0-9_]+)\s+0x[0-9a-fA-F]+\s+\((int|bool|menu|int64|bitmask|button|ctrl_class|string|u8|u16|u32)\)"
        r"\s*:\s*(.*)$"
    )
    menu_re = re.compile(r"^\s*(\d+):\s+(.+)$")

    def kv(payload: str) -> dict[str, str]:
        return dict(re.findall(r"(min|max|step|default|value)=([^\s]+)", payload))

    for line in text.splitlines():
        match = ctrl_re.match(line)
        if match:
            name, ctrl_type, payload = match.group(1), match.group(2), match.group(3)
            if ctrl_type in {"button", "ctrl_class", "string", "bitmask"}:
                current = None
                continue
            fields = kv(payload)
            current = CameraControl(
                name=name,
                ctrl_type=ctrl_type,
                minimum=float(fields.get("min", 0)),
                maximum=float(fields.get("max", 1)),
                step=float(fields.get("step", 1) or 1),
                default=float(fields.get("default", 0)),
                value=float(fields.get("value", fields.get("default", 0))),
            )
            controls.append(current)
            continue

        menu = menu_re.match(line)
        if menu and current is not None and current.ctrl_type == "menu":
            current.menu = current.menu + ((int(menu.group(1)), menu.group(2).strip()),)

    return controls


def _section_after(text: str, title: str) -> str:
    match = re.search(rf"{re.escape(title)}\s*:\s*.*(?:\n[ \t]+.+)*", text)
    return match.group(0) if match else ""


def _device_sort_key(path: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path)
    return (int(match.group(1)) if match else 0, path)


def _v4l2_bin() -> str | None:
    return shutil.which("v4l2-ctl")


def _run_v4l2(device: str, *args: str, check: bool = False) -> str:
    binary = _v4l2_bin()
    if not binary:
        return ""
    try:
        result = subprocess.run(
            [binary, "--device", device, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=check,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout or result.stderr or ""


def _run_v4l2_global(*args: str) -> str:
    binary = _v4l2_bin()
    if not binary:
        return ""
    try:
        result = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout or ""
