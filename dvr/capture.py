"""Capture V4L2 via OpenCV, dans un thread Qt."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from dvr.devices import (
    FrameSize,
    PixelFormat,
    VideoDevice,
    VideoStandard,
    configure_video,
    device_index,
    fourcc_to_str,
)
from dvr.recorder import FrameRecorder


class CaptureWorker(QThread):
    opened = pyqtSignal(int, int, float, str)
    failed = pyqtSignal(str)
    record_failed = pyqtSignal(str)

    def __init__(
        self,
        device: VideoDevice,
        pixel_format: PixelFormat | None = None,
        size: FrameSize | None = None,
        fps: float = 30.0,
        standard: VideoStandard | None = None,
        input_index: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._device = device
        self._pixel_format = pixel_format
        self._size = size
        self._fps = fps
        self._standard = standard
        self._input_index = input_index
        self._running = False
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._recorder: FrameRecorder | None = None
        self._recorder_lock = threading.Lock()

    def take_latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._seq, self._latest

    def attach_recorder(self, recorder: FrameRecorder) -> None:
        with self._recorder_lock:
            self._recorder = recorder

    def detach_recorder(self) -> None:
        with self._recorder_lock:
            self._recorder = None

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        cap = None
        try:
            cap = self._open()
            if cap is None or not cap.isOpened():
                self.failed.emit(f"Impossible d'ouvrir {self._device.path}")
                return

            width, height, fps, fourcc = self._actual_format(cap)
            self.opened.emit(width, height, fps, fourcc)

            failures = 0
            limit = 90 if self._device.analog else 30
            while self._running:
                grabbed = cap.grab()
                if not grabbed:
                    failures += 1
                    if failures > limit:
                        self.failed.emit("Lecture de trames interrompue")
                        break
                    time.sleep(0.02 if self._device.analog else 0.005)
                    continue

                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    failures += 1
                    continue
                failures = 0

                copied = frame.copy()

                with self._recorder_lock:
                    recorder = self._recorder
                if recorder is not None:
                    try:
                        recorder.write(copied)
                    except RuntimeError as exc:
                        self.detach_recorder()
                        self.record_failed.emit(str(exc))

                with self._lock:
                    self._latest = copied
                    self._seq += 1
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()

    def _open(self) -> cv2.VideoCapture | None:
        if self._device.virtual:
            return self._open_virtual()

        fourcc = self._pixel_format.fourcc if self._pixel_format else ""
        width = self._size.width if self._size else 0
        height = self._size.height if self._size else 0
        standard = self._standard.name if self._standard else None
        if fourcc and width and height:
            configure_video(
                self._device.path,
                fourcc,
                width,
                height,
                self._fps,
                standard=standard,
                input_index=self._input_index,
            )

        cap = self._open_capture()
        if cap is None or not cap.isOpened():
            return cap

        self._apply_format(cap)
        warmup = 8 if self._device.analog else 4
        if self._device.analog:
            time.sleep(0.15)
        for _ in range(warmup):
            cap.grab()
        return cap

    def _open_capture(self) -> cv2.VideoCapture | None:
        sources: list[str | int] = []
        index = device_index(self._device.path)
        if self._device.analog and index is not None:
            sources.append(index)
        sources.append(self._device.path)
        if not self._device.analog and index is not None:
            sources.append(index)

        cap = None
        for source in sources:
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return cap
            cap.release()
            cap = None
        return cap

    def _apply_format(self, cap: cv2.VideoCapture) -> None:
        if self._size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size.height)
        if self._pixel_format:
            fourcc = cv2.VideoWriter_fourcc(*self._fourcc_chars(self._pixel_format.fourcc))
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if self._fps and not self._device.analog:
            cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def _actual_format(self, cap: cv2.VideoCapture) -> tuple[int, int, float, str]:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or (
            self._size.width if self._size else 640
        )
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or (
            self._size.height if self._size else 480
        )
        fps = cap.get(cv2.CAP_PROP_FPS) or self._fps or 30.0
        if fps <= 1:
            fps = (self._standard.fps if self._standard else 0) or self._fps or 30.0
        fourcc = fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
        if not fourcc and self._pixel_format:
            fourcc = self._pixel_format.fourcc
        return width, height, float(fps), fourcc

    def _open_virtual(self) -> cv2.VideoCapture:
        width = self._size.width if self._size else 1280
        height = self._size.height if self._size else 720
        return TestPatternCapture(width, height, self._fps)

    @staticmethod
    def _fourcc_chars(fourcc: str) -> str:
        return (fourcc + "    ")[:4]


class TestPatternCapture:
    """Source virtuelle pour tester preview / enregistrement sans caméra."""

    def __init__(self, width: int, height: int, fps: float) -> None:
        self._width = width
        self._height = height
        self._fps = fps or 30.0
        self._opened = True
        self._index = 0
        self._interval = 1.0 / self._fps
        self._last = 0.0
        self._buf: np.ndarray | None = None

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*"BGR3"))
        return 0.0

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def grab(self) -> bool:
        now = time.monotonic()
        wait = self._interval - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self._index += 1
        self._buf = self._frame()
        return True

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        if self._buf is None:
            return False, None
        return True, self._buf

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.grab():
            return False, None
        return self.retrieve()

    def release(self) -> None:
        self._opened = False

    def _frame(self) -> np.ndarray:
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        bands = 8
        band_w = max(1, self._width // bands)
        colors = (
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
            (255, 0, 255),
            (255, 255, 255),
            (40, 40, 40),
        )
        for i, color in enumerate(colors):
            frame[:, i * band_w : (i + 1) * band_w] = color

        seconds = self._index / self._fps
        label = f"TEST  {seconds:07.2f}s  {self._width}x{self._height}@{self._fps:.0f}"
        cv2.putText(
            frame,
            label,
            (24, self._height - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (24, self._height - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame
