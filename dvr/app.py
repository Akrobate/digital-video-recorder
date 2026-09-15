"""Point d'entrée de l'application."""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from dvr.window import MainWindow, apply_dark_palette


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    app = QApplication(argv)
    app.setApplicationName("Digital Video Recorder")
    apply_dark_palette(app)
    window = MainWindow()
    window.show()
    return app.exec_()
