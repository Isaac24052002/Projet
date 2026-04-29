# Classification Animale en Temps Reel (Webcam)

Ce projet exploite un modele Keras deja entraine (`animal-10.keras`) pour predire un animal a partir du flux webcam en direct.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement

```bash
python3 predict.py
```

Avec une autre camera:

```bash
python3 predict.py --camera 1
```

Option detection de presence (MediaPipe):

```bash
python3 predict.py --use-mediapipe
```

Mode plein ecran:

```bash
python3 predict.py --fullscreen
```

Pour lancer en fenetre normale:

```bash
python3 predict.py --no-fullscreen
```

Mode surete renforcee (anti faux positifs):

```bash
python3 predict.py --min-confidence 55 --stable-frames 3
```

Mode plus fluide (si la webcam rame):

```bash
python3 predict.py --infer-every 4 --max-infer-fps 6 --crop-mode center-crop --retry-zoom 1 --no-fullscreen
```

Mode reconnaissance difficile (si certaines images sont peu reconnues):

```bash
python3 predict.py --preprocess auto --crop-mode center-crop --retry-zoom 0.85 --soft-margin 8 --min-confidence 45 --stable-frames 1 --smooth-alpha 0.2
```

## Fichiers

- `predict.py`: script principal.
- `animal-10.keras`: modele fourni.
- `labels.txt`: labels classes (1 par ligne, ordre exact de l'entrainement).
- `requirements.txt`: dependances Python.

## Optimisations integrees

- Chargement robuste du modele `.keras` avec fallback de reconstruction.
- Prechauffage du modele pour supprimer la latence du premier passage.
- Lissage temporel des probabilites (`--smooth-alpha`).
- Cadence d'inference reglable (`--infer-every`, `--max-infer-fps`).
- Capture webcam asynchrone (thread + buffer 1 frame) pour limiter les saccades.
- Verification MediaPipe moins frequente pour reduire la charge (`--mediapipe-every`).
- Cadrage d'inference configurable (`--crop-mode`) pour eviter la deformation des sujets.
- Pretraitement auto-adaptatif en cas de faible confiance (`--preprocess auto`) avec anti-rafale (`--auto-retry-cooldown`).
- Retry zoom ponctuel sur confiance faible (`--retry-zoom`) pour mieux reconnaitre les sujets petits/eloignes.
- Verrou de fiabilite base uniquement sur la confiance:
  - confiance minimale (`--min-confidence`)
  - marge souple stable (`--soft-margin`)
  - confirmation sur plusieurs frames (`--stable-frames`)
- Flux webcam plus reactif (`CAP_PROP_BUFFERSIZE=1` quand supporte).
- Tentative de capture MJPG + FPS cible pour reduire la latence webcam.
- Plein ecran desactive par defaut pour garder la fluidite (activable avec `--fullscreen`).
- Affichage classique (2 lignes) avec fond sombre semi-transparent et texte a fort contraste.
- FPS lisse affiche en direct.

## Reglages rapides

Si tu vois souvent `Indetermine`:

```bash
python3 predict.py --min-confidence 45 --soft-margin 8 --stable-frames 1 --smooth-alpha 0.25
```

Si la webcam n'est pas fluide:

```bash
python3 predict.py --infer-every 4 --max-infer-fps 6 --mediapipe-every 4 --crop-mode center-crop --retry-zoom 1 --no-fullscreen
```

## Commandes clavier

- `Q`: quitter.
- `F`: basculer plein ecran.
