# Digital Video Recorder

Video recorder for webcams and USB capture cards (V4L2) on Linux.

 - Lists /dev/video* devices
 - Live preview on click
 - Reads camera capabilities (format, resolution, FPS) and settings
 - H.264 recording (PyAV / libx264) to an MP4 file

## Prerequisites

 - [uv](https://docs.astral.sh/uv/)
 - Python 3.11+ (automatically installed by uv if needed)
 - `v4l-utils` (`v4l2-ctl`) for device enumeration and controls
 - A webcam / HDMI dongle, or the built-in test pattern

```bash
sudo apt install v4l-utils
```

## Install

```bash
uv sync
```

## Run

```bash
uv run dvr
```

On first launch, click a device on the left: the live preview starts and the right panel displays the detected formats / resolutions / FPS, along with the V4L2 sliders (brightness, exposure, etc.). Record saves a `dvr_YYYYMMDD_HHMMSS.mp4` file to `~/Videos/dvr` (or your chosen directory).
