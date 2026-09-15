# Digital Video Recorder

Enregistreur vidéo pour webcams et cartes d'acquisition USB (V4L2) sous Linux.

- Liste les périphériques `/dev/video*`
- Aperçu en direct au clic
- Lecture des capacités (format, résolution, FPS) et des réglages caméra
- Enregistrement H.264 (ffmpeg) vers un fichier MP4

## Prérequis

- [uv](https://docs.astral.sh/uv/)
- Python 3.10+ (installé automatiquement par uv si besoin)
- `v4l-utils` (`v4l2-ctl`) pour l'énumération et les contrôles
- `ffmpeg` recommandé pour l'enregistrement H.264
- Une webcam / dongle HDMI, ou la mire de test intégrée

```bash
sudo apt install v4l-utils ffmpeg
```

## Installation

```bash
uv sync
```

## Lancement

```bash
uv run dvr
```

Au premier lancement, cliquez un périphérique à gauche : l'aperçu démarre et le panneau de droite propose les formats / résolutions / FPS scannés, plus les sliders V4L2 (luminosité, exposition, etc.). **Enregistrer** écrit un fichier `dvr_AAAAMMJJ_HHMMSS.mp4` dans `~/Videos/dvr` (ou le dossier choisi).
