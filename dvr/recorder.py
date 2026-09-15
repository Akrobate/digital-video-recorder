"""Enregistrement du flux : ffmpeg (H.264) avec repli OpenCV."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class FrameRecorder:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._writer: cv2.VideoWriter | None = None
        self._path = ""
        self._size: tuple[int, int] | None = None
        self._frames = 0

    @property
    def path(self) -> str:
        return self._path

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def active(self) -> bool:
        return self._proc is not None or self._writer is not None

    def start(self, directory: str, width: int, height: int, fps: float) -> str:
        self.stop()
        Path(directory).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"dvr_{stamp}.mp4")
        fps = fps if fps > 1 else 30.0
        width -= width % 2
        height -= height % 2
        self._size = (width, height)
        self._frames = 0
        self._path = path

        if shutil.which("ffmpeg"):
            self._proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    f"{fps:.3f}",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            return path

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("Impossible de créer le fichier d'enregistrement")
        self._writer = writer
        return path

    def write(self, frame: np.ndarray) -> None:
        if self._size is None:
            return
        width, height = self._size
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                self._abort()
                raise RuntimeError("ffmpeg a interrompu l'enregistrement") from exc
        elif self._writer is not None:
            self._writer.write(frame)
        self._frames += 1

    def stop(self) -> str:
        path = self._path
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            if proc.stdin:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._size = None
        return path

    def _abort(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass
            self._proc = None
        self._size = None


def default_output_dir() -> str:
    videos = Path.home() / "Videos"
    if videos.is_dir():
        return str(videos / "dvr")
    return str(Path.cwd() / "recordings")
