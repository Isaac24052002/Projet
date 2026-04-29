# Neural Gesture Sculptor

Demo CV/Computer Vision en temps reel: creation et manipulation d'un objet 3D (reseau de neurones anime) par gestes devant webcam.

## Fonctionnalites (v1)

- `CIRCLE` (index en cercle): cree un objet 3D neuronal.
- `PINCH` (pouce-index): saisit et deplace l'objet.
- `TWO_HANDS`: met a l'echelle l'objet avec l'ecart entre mains.
- `ROTATE` (main ouverte): pivote l'objet (yaw/pitch).
- Overlay temps reel OpenCV avec HUD (geste courant, FPS, etat objet).

## Stack

- Python 3.10+
- OpenCV
- MediaPipe Hands
- NumPy

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Qualite code (optionnel mais recommande):

```bash
python3 -m pip install -r requirements-dev.txt
pytest
ruff check .
mypy .
```

Si votre version de `mediapipe` n'expose pas `mp.solutions`, telechargez le modele Tasks:

```bash
mkdir -p models
wget -O models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Lancement

```bash
python3 main.py
```

Mode ultra-fluide (si machine limitee):

```bash
NGS_WIDTH=960 NGS_HEIGHT=540 NGS_DET_WIDTH=512 python3 main.py
```

Mode custom (resolution, FPS cible, debug landmarks):

```bash
NGS_WIDTH=1280 NGS_HEIGHT=720 NGS_FPS=30 NGS_DET_WIDTH=640 NGS_DEBUG_LANDMARKS=0 python3 main.py
```

Reglages avances (camera + lissage + auto-tuning):

```bash
NGS_CAMERA_INDEX=0 \
NGS_PINCH_ALPHA=0.40 NGS_SCALE_ALPHA=0.55 NGS_ROTATE_ALPHA=0.65 \
NGS_LOW_FPS=24 NGS_HIGH_FPS=32 NGS_AUTO_TUNE_S=1.0 \
python3 main.py
```

Raccourcis:

- `q` ou `Esc`: quitter
- `c`: creer l'objet manuellement (test rapide)
- `x`: supprimer l'objet
- `d`: afficher/masquer les points de debug main (plus fluide si masque)
- `r`: reset position/scale/rotation de l'objet

## Structure

- `main.py`: boucle capture -> detection -> geste -> rendu.
- `gesture_engine.py`: detection `CIRCLE`, `PINCH`, `TWO_HANDS`, `ROTATE`.
- `neural_object.py`: graphe 3D `[4,6,6,3]`, projection perspective, animation pulse.
- `renderer.py`: overlay alpha et HUD.

## Reglages utiles

- Resolution cible: `1280x720` a `30 fps`.
- Seuil pinch: `40 px` sur `3 frames`.
- Cooldown cercle: `1 seconde`.
- Dead-zone rotation: `+/-5 deg`.
- Auto-tuning detection: ajuste la largeur de detection pour maintenir la fluidite.
- Index camera: `NGS_CAMERA_INDEX` (utile si plusieurs webcams).
- Lissage gestes: `NGS_PINCH_ALPHA`, `NGS_SCALE_ALPHA`, `NGS_ROTATE_ALPHA`.
- Seuils auto-tuning: `NGS_LOW_FPS`, `NGS_HIGH_FPS`, `NGS_AUTO_TUNE_S`.
- HUD enrichi: gesture, FPS, backend MediaPipe, largeur detection, etat debug.

## Demo portfolio

Ajoutez un GIF de demonstration (>= 10s) dans le depot, puis referencez-le ici:

```md
![Demo](assets/demo.gif)
```
