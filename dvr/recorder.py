"""Enregistrement H.264 via PyAV (FFmpeg in-process)."""

from __future__ import annotations

import os
import threading
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from av.error import FFmpegError


class FrameRecorder:
    def __init__(self) -> None:
        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._path = ""
        self._size: tuple[int, int] | None = None
        self._frames = 0
        self._lock = threading.Lock()

    @property
    def path(self) -> str:
        return self._path

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def active(self) -> bool:
        return self._container is not None

    def start(self, directory: str, width: int, height: int, fps: float) -> str:
        self.stop()
        Path(directory).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"dvr_{stamp}.mp4")
        fps = fps if fps > 1 else 30.0
        width -= width % 2
        height -= height % 2
        rate = Fraction(fps).limit_denominator(1001)

        try:
            container = av.open(path, mode="w")
        except FFmpegError as exc:
            raise RuntimeError("Impossible de créer le fichier d'enregistrement") from exc

        try:
            try:
                stream = container.add_stream("libx264", rate=rate)
            except FFmpegError:
                stream = container.add_stream("h264", rate=rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "veryfast", "crf": "18"}
        except FFmpegError as exc:
            container.close()
            raise RuntimeError("Impossible d'initialiser l'encodeur H.264") from exc

        with self._lock:
            self._container = container
            self._stream = stream
            self._size = (width, height)
            self._frames = 0
            self._path = path
        return path

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._container is None or self._stream is None or self._size is None:
                return
            width, height = self._size
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frame = np.ascontiguousarray(frame)
            try:
                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                video_frame.pts = self._frames
                for packet in self._stream.encode(video_frame):
                    self._container.mux(packet)
            except FFmpegError as exc:
                self._close_unlocked(flush=False)
                raise RuntimeError("L'enregistrement a été interrompu") from exc
            self._frames += 1

    def stop(self) -> str:
        with self._lock:
            path = self._path
            self._close_unlocked(flush=True)
            return path

    def _close_unlocked(self, *, flush: bool) -> None:
        container = self._container
        stream = self._stream
        self._container = None
        self._stream = None
        self._size = None
        if container is None:
            return
        try:
            if flush and stream is not None:
                for packet in stream.encode(None):
                    container.mux(packet)
        except FFmpegError:
            pass
        try:
            container.close()
        except FFmpegError:
            pass


def default_output_dir() -> str:
    videos = Path.home() / "Videos"
    if videos.is_dir():
        return str(videos / "dvr")
    return str(Path.cwd() / "recordings")
