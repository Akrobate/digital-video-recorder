"""Fenêtre principale : devices, preview, réglages, enregistrement."""

from __future__ import annotations

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCloseEvent, QColor, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from dvr.capture import CaptureWorker
from dvr.devices import (
    CameraControl,
    FrameSize,
    PixelFormat,
    VideoDevice,
    VideoInput,
    VideoStandard,
    list_video_devices,
    preferred_format,
    preferred_fps,
    preferred_size,
    preferred_standard,
    probe_device,
    set_control,
)
from dvr.recorder import FrameRecorder, default_output_dir


TEST_DEVICE = VideoDevice(
    path="test://pattern",
    name="Mire de test (virtuel)",
    bus_info="virtuel",
    card="Test pattern",
    virtual=True,
    formats=(
        PixelFormat(
            fourcc="BGR3",
            description="Mire couleur",
            sizes=(
                FrameSize(1920, 1080, (30.0, 25.0, 15.0)),
                FrameSize(1280, 720, (30.0, 25.0, 15.0)),
                FrameSize(640, 480, (30.0,)),
            ),
        ),
    ),
)


STYLESHEET = """
QWidget {
    background: #14171c;
    color: #e8edf4;
    font-family: "Ubuntu", "Noto Sans", sans-serif;
    font-size: 13px;
}
QLabel#title {
    font-size: 18px;
    font-weight: 600;
    color: #f4f7fb;
}
QLabel#section {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #8b97a8;
}
QLabel#hint {
    color: #8b97a8;
}
QListWidget {
    background: #1c2128;
    border: 1px solid #2b333d;
    border-radius: 8px;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    padding: 10px 8px;
    border-radius: 6px;
}
QListWidget::item:selected {
    background: #2b6cb0;
    color: #ffffff;
}
QListWidget::item:hover:!selected {
    background: #242b34;
}
QComboBox, QLineEdit {
    background: #1c2128;
    border: 1px solid #2b333d;
    border-radius: 6px;
    padding: 6px 8px;
    min-height: 18px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #1c2128;
    selection-background-color: #2b6cb0;
    border: 1px solid #2b333d;
}
QPushButton {
    background: #2a313b;
    border: 1px solid #3a4452;
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #343c48; }
QPushButton:pressed { background: #222830; }
QPushButton:disabled { color: #6b7480; background: #1c2128; }
QPushButton#record {
    background: #8b1e2d;
    border-color: #b02a3c;
    color: #fff;
}
QPushButton#record:hover { background: #a42436; }
QPushButton#record[recording="true"] {
    background: #c53030;
    border-color: #fc8181;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #2b333d;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background: #63b3ed;
    border-radius: 7px;
}
QFrame#panel {
    background: #1a1e24;
    border: 1px solid #2b333d;
    border-radius: 10px;
}
QLabel#preview {
    background: #0b0d10;
    border: 1px solid #2b333d;
    border-radius: 10px;
    color: #8b97a8;
}
QLabel#recdot {
    color: #fc8181;
    font-weight: 700;
}
QScrollArea { border: none; background: transparent; }
"""


class PreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("Sélectionnez un périphérique pour l'aperçu")
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._bgr = None

    def set_bgr(self, frame) -> None:
        self._bgr = frame
        self._paint()

    def clear_frame(self, message: str = "Sélectionnez un périphérique pour l'aperçu") -> None:
        self._bgr = None
        self.setPixmap(QPixmap())
        self.setText(message)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._paint()

    def _paint(self) -> None:
        if self._bgr is None:
            return
        fitted = _fit_bgr(self._bgr, self.width(), self.height())
        image = _bgr_to_qimage(fitted)
        self.setText("")
        self.setPixmap(QPixmap.fromImage(image))


def _fit_bgr(frame, max_w: int, max_h: int):
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0 or max_w <= 2 or max_h <= 2:
        return frame
    scale = min(max_w / width, max_h / height)
    if scale >= 0.98:
        return frame
    new_w = max(2, int(width * scale) & ~1)
    new_h = max(2, int(height * scale) & ~1)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _bgr_to_qimage(frame) -> QImage:
    if frame.ndim == 2:
        height, width = frame.shape
        return QImage(frame.data, width, height, frame.strides[0], QImage.Format_Grayscale8).copy()
    height, width = frame.shape[:2]
    fmt = QImage.Format_BGR888 if hasattr(QImage, "Format_BGR888") else QImage.Format_RGB888
    if fmt == QImage.Format_RGB888:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(frame.data, width, height, frame.strides[0], fmt).copy()


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Digital Video Recorder")
        self.resize(1280, 760)
        self.setStyleSheet(STYLESHEET)

        self._devices: list[VideoDevice] = []
        self._current: VideoDevice | None = None
        self._worker: CaptureWorker | None = None
        self._recorder = FrameRecorder()
        self._frame_size = (0, 0)
        self._stream_fps = 30.0
        self._stream_fourcc = ""
        self._preview_seq = -1
        self._rec_seconds = 0
        self._output_dir = default_output_dir()
        self._updating_combos = False

        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(1000)
        self._rec_timer.timeout.connect(self._tick_recording)

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(16)
        self._preview_timer.setTimerType(Qt.PreciseTimer)
        self._preview_timer.timeout.connect(self._poll_preview)

        self._build_ui()
        self.refresh_devices()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Digital Video Recorder")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        self.status_label = QLabel("Prêt")
        self.status_label.setObjectName("hint")
        header.addWidget(self.status_label)
        root.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_device_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.addWidget(self._build_settings_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 720, 320])
        root.addWidget(splitter, 1)

    def _build_device_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)

        label = QLabel("PÉRIPHÉRIQUES")
        label.setObjectName("section")
        layout.addWidget(label)

        self.device_list = QListWidget()
        self.device_list.itemClicked.connect(self._on_device_clicked)
        layout.addWidget(self.device_list, 1)

        refresh = QPushButton("Actualiser")
        refresh.clicked.connect(self.refresh_devices)
        layout.addWidget(refresh)
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.preview = PreviewLabel()
        layout.addWidget(self.preview, 1)

        controls = QHBoxLayout()
        self.rec_dot = QLabel("")
        self.rec_dot.setObjectName("recdot")
        self.timer_label = QLabel("00:00:00")
        self.stream_info = QLabel("Aucun flux")
        self.stream_info.setObjectName("hint")

        self.record_btn = QPushButton("Enregistrer")
        self.record_btn.setObjectName("record")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.toggle_record)

        self.folder_btn = QPushButton("Dossier…")
        self.folder_btn.clicked.connect(self.choose_folder)

        controls.addWidget(self.rec_dot)
        controls.addWidget(self.timer_label)
        controls.addSpacing(12)
        controls.addWidget(self.stream_info, 1)
        controls.addWidget(self.folder_btn)
        controls.addWidget(self.record_btn)
        layout.addLayout(controls)

        self.folder_label = QLabel(self._output_dir)
        self.folder_label.setObjectName("hint")
        layout.addWidget(self.folder_label)
        return panel

    def _build_settings_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)

        cap_label = QLabel("CAPACITÉS")
        cap_label.setObjectName("section")
        layout.addWidget(cap_label)

        self.format_combo = QComboBox()
        self.size_combo = QComboBox()
        self.fps_combo = QComboBox()
        self.standard_combo = QComboBox()
        self.input_combo = QComboBox()
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self.size_combo.currentIndexChanged.connect(self._on_size_changed)
        self.fps_combo.currentIndexChanged.connect(self._on_fps_changed)
        self.standard_combo.currentIndexChanged.connect(self._on_standard_changed)
        self.input_combo.currentIndexChanged.connect(self._on_input_changed)

        self.standard_label = QLabel("Standard TV")
        self.input_label = QLabel("Entrée analogique")
        layout.addWidget(self.standard_label)
        layout.addWidget(self.standard_combo)
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_combo)
        layout.addWidget(QLabel("Format pixel"))
        layout.addWidget(self.format_combo)
        layout.addWidget(QLabel("Résolution"))
        layout.addWidget(self.size_combo)
        layout.addWidget(QLabel("Images / seconde"))
        layout.addWidget(self.fps_combo)
        self._set_analog_widgets_visible(False)

        apply_btn = QPushButton("Appliquer le format")
        apply_btn.clicked.connect(self.apply_format)
        layout.addWidget(apply_btn)

        ctrl_label = QLabel("RÉGLAGES CAMÉRA")
        ctrl_label.setObjectName("section")
        layout.addWidget(ctrl_label)

        self.controls_host = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_host)
        self.controls_layout.setContentsMargins(0, 0, 8, 0)
        self.controls_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls_host)
        layout.addWidget(scroll, 1)
        return panel

    def refresh_devices(self) -> None:
        self.stop_capture()
        self.device_list.clear()
        self._devices = list_video_devices()
        self._devices.append(TEST_DEVICE)

        for device in self._devices:
            item = QListWidgetItem(f"{device.display_name}\n{device.subtitle}")
            item.setData(Qt.UserRole, device.path)
            self.device_list.addItem(item)

        if len(self._devices) == 1:
            self.set_status("Aucun device V4L2 — mire de test disponible")
        else:
            self.set_status(f"{len(self._devices) - 1} périphérique(s) détecté(s)")

    def _on_device_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        device = next((d for d in self._devices if d.path == path), None)
        if device is None:
            return
        if self._recorder.active:
            self._stop_recording(silent=True)
        self._current = probe_device(device)
        self._populate_capabilities(self._current)
        self.start_preview()

    def _populate_capabilities(self, device: VideoDevice) -> None:
        self._updating_combos = True
        self.format_combo.clear()
        self.size_combo.clear()
        self.fps_combo.clear()
        self.standard_combo.clear()
        self.input_combo.clear()

        analog = device.analog
        self._set_analog_widgets_visible(analog)
        for standard in device.standards:
            self.standard_combo.addItem(standard.label, standard)
        chosen_std = preferred_standard(device.standards, device.current_standard)
        if chosen_std:
            index = next(
                (
                    i
                    for i in range(self.standard_combo.count())
                    if self.standard_combo.itemData(i).name == chosen_std.name
                ),
                0,
            )
            self.standard_combo.setCurrentIndex(index)

        for video_input in device.inputs:
            self.input_combo.addItem(video_input.label, video_input)
        if device.current_input is not None:
            index = next(
                (
                    i
                    for i in range(self.input_combo.count())
                    if self.input_combo.itemData(i).index == device.current_input
                ),
                0,
            )
            self.input_combo.setCurrentIndex(index)
        self.input_label.setVisible(analog and bool(device.inputs))
        self.input_combo.setVisible(analog and bool(device.inputs))

        for fmt in device.formats:
            self.format_combo.addItem(fmt.label, fmt)
        chosen = preferred_format(device.formats, analog=analog)
        if not device.formats:
            fallback = PixelFormat(
                fourcc="YUYV" if analog else "MJPG",
                description="analogique" if analog else "détecté par OpenCV",
                sizes=(
                    FrameSize(720, 480, (29.97, 30.0, 25.0)),
                    FrameSize(720, 576, (25.0,)),
                    FrameSize(640, 480, (30.0, 15.0)),
                )
                if analog
                else (
                    FrameSize(1920, 1080, (30.0, 25.0, 15.0)),
                    FrameSize(1280, 720, (30.0, 25.0, 15.0)),
                    FrameSize(640, 480, (30.0, 15.0)),
                ),
            )
            self.format_combo.addItem(fallback.label, fallback)
            chosen = fallback
        if chosen:
            index = next(
                (
                    i
                    for i in range(self.format_combo.count())
                    if (self.format_combo.itemData(i) or chosen).fourcc == chosen.fourcc
                ),
                0,
            )
            self.format_combo.setCurrentIndex(index)
        self._fill_sizes()
        self._updating_combos = False
        self._rebuild_controls(device)

    def _fill_sizes(self) -> None:
        fmt = self._current_format()
        self.size_combo.clear()
        if not fmt:
            return
        for size in fmt.sizes:
            self.size_combo.addItem(size.label, size)
        best = preferred_size(fmt, self._current_standard())
        if best:
            index = next(
                (
                    i
                    for i in range(self.size_combo.count())
                    if self.size_combo.itemData(i).label == best.label
                ),
                0,
            )
            self.size_combo.setCurrentIndex(index)
        self._fill_fps()

    def _fill_fps(self) -> None:
        size = self._current_size()
        self.fps_combo.clear()
        values = size.fps if size and size.fps else (30.0, 25.0, 15.0)
        for fps in values:
            self.fps_combo.addItem(f"{fps:g} fps", fps)
        wanted = preferred_fps(size, self._current_standard())
        index = next(
            (
                i
                for i in range(self.fps_combo.count())
                if abs(self.fps_combo.itemData(i) - wanted) < 0.05
            ),
            0,
        )
        self.fps_combo.setCurrentIndex(index)

    def _set_analog_widgets_visible(self, visible: bool) -> None:
        self.standard_label.setVisible(visible)
        self.standard_combo.setVisible(visible)
        self.input_label.setVisible(visible)
        self.input_combo.setVisible(visible)

    def _rebuild_controls(self, device: VideoDevice) -> None:
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not device.controls:
            empty = QLabel("Aucun réglage exposé par ce périphérique.")
            empty.setObjectName("hint")
            empty.setWordWrap(True)
            self.controls_layout.addWidget(empty)
            self.controls_layout.addStretch()
            return

        for control in device.controls:
            self.controls_layout.addWidget(self._control_widget(device, control))
        self.controls_layout.addStretch()

    def _control_widget(self, device: VideoDevice, control: CameraControl) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if control.ctrl_type == "bool":
            checkbox = QCheckBox(control.label)
            checkbox.setChecked(bool(int(control.value)))
            checkbox.toggled.connect(
                lambda checked, name=control.name: self._apply_control(device, name, int(checked))
            )
            layout.addWidget(checkbox)
            return box

        if control.ctrl_type == "menu" and control.menu:
            layout.addWidget(QLabel(control.label))
            combo = QComboBox()
            for value, label in control.menu:
                combo.addItem(label, value)
            current = next(
                (i for i in range(combo.count()) if combo.itemData(i) == int(control.value)),
                0,
            )
            combo.setCurrentIndex(current)
            combo.currentIndexChanged.connect(
                lambda _i, c=combo, name=control.name: self._apply_control(device, name, c.currentData())
            )
            layout.addWidget(combo)
            return box

        header = QHBoxLayout()
        header.addWidget(QLabel(control.label))
        value_label = QLabel(self._format_ctrl_value(control.value))
        value_label.setObjectName("hint")
        header.addStretch()
        header.addWidget(value_label)
        layout.addLayout(header)

        slider = QSlider(Qt.Horizontal)
        minimum = int(control.minimum)
        maximum = int(control.maximum)
        step = max(int(control.step), 1)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum if maximum > minimum else minimum + 1)
        slider.setSingleStep(step)
        slider.setPageStep(step * 5)
        slider.setValue(int(control.value))
        slider.valueChanged.connect(
            lambda value, name=control.name, lbl=value_label: self._on_slider(device, name, value, lbl)
        )
        layout.addWidget(slider)
        return box

    def _on_slider(self, device: VideoDevice, name: str, value: int, label: QLabel) -> None:
        label.setText(self._format_ctrl_value(value))
        self._apply_control(device, name, value)

    def _apply_control(self, device: VideoDevice, name: str, value) -> None:
        if device.virtual:
            return
        try:
            set_control(device.path, name, value)
            self.set_status(f"{name} = {value}")
        except Exception as exc:  # noqa: BLE001
            self.set_status(str(exc))

    @staticmethod
    def _format_ctrl_value(value: float) -> str:
        if abs(value - int(value)) < 0.001:
            return str(int(value))
        return f"{value:.2f}"

    def _on_format_changed(self) -> None:
        if self._updating_combos:
            return
        self._updating_combos = True
        self._fill_sizes()
        self._updating_combos = False

    def _on_size_changed(self) -> None:
        if self._updating_combos:
            return
        self._updating_combos = True
        self._fill_fps()
        self._updating_combos = False

    def _on_fps_changed(self) -> None:
        return

    def _on_standard_changed(self) -> None:
        if self._updating_combos:
            return
        self._updating_combos = True
        self._fill_sizes()
        self._updating_combos = False
        self.start_preview()

    def _on_input_changed(self) -> None:
        if self._updating_combos:
            return
        self.start_preview()

    def apply_format(self) -> None:
        if self._current is None:
            return
        if self._recorder.active:
            self._stop_recording(silent=True)
        self.start_preview()

    def start_preview(self) -> None:
        if self._current is None:
            return
        self.stop_capture()
        self.preview.clear_frame("Ouverture du périphérique…")
        self.record_btn.setEnabled(False)

        worker = CaptureWorker(
            self._current,
            pixel_format=self._current_format(),
            size=self._current_size(),
            fps=self._current_fps(),
            standard=self._current_standard(),
            input_index=self._current_input_index(),
        )
        worker.opened.connect(self._on_opened)
        worker.failed.connect(self._on_capture_failed)
        worker.record_failed.connect(self._on_record_failed)
        worker.start()
        self._worker = worker
        self._preview_seq = -1
        self._preview_timer.start()
        self.set_status(f"Aperçu : {self._current.display_name}")

    def stop_capture(self) -> None:
        self._preview_timer.stop()
        if self._recorder.active:
            self._stop_recording(silent=True)
        worker = self._worker
        self._worker = None
        self._preview_seq = -1
        if worker is not None:
            worker.opened.disconnect(self._on_opened)
            worker.failed.disconnect(self._on_capture_failed)
            worker.record_failed.disconnect(self._on_record_failed)
            worker.detach_recorder()
            worker.stop()
            worker.wait(1500)
            if worker.isRunning():
                worker.terminate()
                worker.wait(500)
        self.preview.clear_frame()
        self.record_btn.setEnabled(False)
        self.stream_info.setText("Aucun flux")

    def _on_opened(self, width: int, height: int, fps: float, fourcc: str) -> None:
        self._frame_size = (width, height)
        self._stream_fps = fps
        self._stream_fourcc = fourcc
        label = f"{width}×{height}  ·  {fps:g} fps"
        if fourcc:
            label += f"  ·  {fourcc}"
        standard = self._current_standard()
        if standard:
            label += f"  ·  {standard.name}"
        self.stream_info.setText(label)
        self.record_btn.setEnabled(True)
        if (
            fourcc.upper() in {"YUYV", "YUY2", "UYVY"}
            and width * height >= 1280 * 720
            and not (self._current and self._current.analog)
        ):
            self.set_status(
                "Format non compressé à haute résolution — préférez MJPG pour un aperçu fluide"
            )

    def _on_capture_failed(self, message: str) -> None:
        self._preview_timer.stop()
        self.set_status(message)
        self.preview.clear_frame(message)
        self.record_btn.setEnabled(False)
        QMessageBox.warning(self, "Capture", message)

    def _on_record_failed(self, message: str) -> None:
        self._stop_recording(silent=True)
        QMessageBox.warning(self, "Enregistrement", message)

    def _poll_preview(self) -> None:
        worker = self._worker
        if worker is None:
            return
        seq, frame = worker.take_latest()
        if frame is None or seq == self._preview_seq:
            return
        self._preview_seq = seq
        self.preview.set_bgr(frame)

    def toggle_record(self) -> None:
        if self._recorder.active:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        width, height = self._frame_size
        if self._worker is not None:
            _seq, frame = self._worker.take_latest()
            if frame is not None:
                height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            QMessageBox.information(self, "Enregistrement", "Aucun flux à enregistrer.")
            return
        try:
            path = self._recorder.start(self._output_dir, width, height, self._stream_fps)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Enregistrement", str(exc))
            return
        if self._worker is not None:
            self._worker.attach_recorder(self._recorder)
        self._rec_seconds = 0
        self._rec_timer.start()
        self.record_btn.setText("Stop")
        self.record_btn.setProperty("recording", "true")
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)
        self.rec_dot.setText("● REC")
        self.timer_label.setText("00:00:00")
        self.set_status(f"Enregistrement vers {path}")

    def _stop_recording(self, silent: bool = False) -> None:
        if self._worker is not None:
            self._worker.detach_recorder()
        path = self._recorder.stop()
        self._rec_timer.stop()
        self.record_btn.setText("Enregistrer")
        self.record_btn.setProperty("recording", "false")
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)
        self.rec_dot.setText("")
        if path and not silent:
            self.set_status(f"Fichier enregistré : {path}")

    def _tick_recording(self) -> None:
        self._rec_seconds += 1
        hours, rem = divmod(self._rec_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        self.timer_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def choose_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Dossier d'enregistrement",
            self._output_dir,
        )
        if directory:
            self._output_dir = directory
            self.folder_label.setText(directory)

    def _current_format(self) -> PixelFormat | None:
        return self.format_combo.currentData()

    def _current_size(self) -> FrameSize | None:
        return self.size_combo.currentData()

    def _current_fps(self) -> float:
        value = self.fps_combo.currentData()
        standard = self._current_standard()
        if value:
            return float(value)
        if standard:
            return standard.fps
        return 30.0

    def _current_standard(self) -> VideoStandard | None:
        return self.standard_combo.currentData()

    def _current_input_index(self) -> int | None:
        video_input = self.input_combo.currentData()
        if isinstance(video_input, VideoInput):
            return video_input.index
        return None

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.stop_capture()
        event.accept()


def apply_dark_palette(app) -> None:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#14171c"))
    palette.setColor(QPalette.WindowText, QColor("#e8edf4"))
    palette.setColor(QPalette.Base, QColor("#1c2128"))
    palette.setColor(QPalette.Text, QColor("#e8edf4"))
    palette.setColor(QPalette.Button, QColor("#2a313b"))
    palette.setColor(QPalette.ButtonText, QColor("#e8edf4"))
    palette.setColor(QPalette.Highlight, QColor("#2b6cb0"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
