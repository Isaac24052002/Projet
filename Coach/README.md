# Coach Sportif Temps Reel - Pose Analysis

Application Python locale qui analyse la posture via webcam en temps reel avec **MediaPipe Pose + OpenCV**.

Le projet detecte automatiquement l'orientation (face, profil gauche, profil droit), analyse trois exercices, affiche un feedback visuel immediat et exporte un rapport de seance.

## Exercices couverts

- `1` Squat
- `2` Push-up
- `3` Bicep curl

## Fonctionnalites principales

- Detection de pose en temps reel (33 landmarks)
- Detection auto orientation face/profil selon la visibilite des landmarks
- Analyse articulaire par angles (genou, coude, alignement)
- Optimisation anti faux positifs: etat neutre au repos, anti-jitter, amplitude minimale de repetition
- Squat optimise: validation de profondeur + alignement du dos adapte a la profondeur
- Squat optimise: suivi genou/cheville normalise (morphologie/camera) pour limiter les faux rejets
- Feedback visuel instantane:
  - Squelette vert si geste correct
  - Squelette rouge si geste incorrect
  - Squelette ambre en etat neutre (repos/instructions)
  - Message precis (ex: `Descends plus bas`, `Dos trop penche`, `Epaule qui bouge`)
- Compteur de repetitions via machine a etats (`haut` / `bas`)
- Panneau statistiques premium en temps reel (FPS lisse, reps invalides, jauges qualite/amplitude)
- Export de fin de seance:
  - rapport CSV
  - historique JSON
  - video demo `.mp4`

## Stack technique

- Python 3.10+
- opencv-python 4.8+
- mediapipe 0.10.14
- numpy 1.24+

## Structure du projet

- `main.py` boucle principale webcam, clavier, rendu, export
- `pose_detector.py` wrapper MediaPipe, landmarks, orientation
- `angle_calculator.py` calcul des angles et distances
- `exercise_analyzer.py` regles exercices + machine a etats
- `feedback_renderer.py` overlays OpenCV (stats, message, menu)
- `session_logger.py` generation CSV/JSON et nommage video
- `config.py` constantes, seuils, couleurs
- `requirements.txt` dependances

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement

```bash
python main.py
```

Option camera specifique:

```bash
python main.py --camera 1
```

Demarrer sur un exercice cible:

```bash
python main.py --exercice pushup
```

Choisir un profil de coaching:

```bash
python main.py --profil equilibre
```

Par defaut, l'application demarre en profil `strict`.

Profils disponibles:

- `strict` : validation exigeante (forme prioritaire)
- `equilibre` : compromis precision/fluidite (recommande)
- `tolerant` : validation plus souple pour demo fluide

En mode `strict`, une faute detectee pendant la repetition invalide immediatement le cycle.

Desactiver l'enregistrement video:

```bash
python main.py --sans-enregistrement
```

## Commandes clavier

- `1` -> Squat
- `2` -> Push-up
- `3` -> Bicep curl
- `q` -> Quitter la session

## Conseils de demo (important)

- Pour un comptage fiable des repetitions, place-toi de profil (gauche ou droit) pour les 3 exercices.
- Au repos, le systeme reste en etat neutre et n'encourage pas de fausse repetition.
- Pour le squat, la profondeur et l'alignement du dos sont evalues ensemble avec une tolerance dynamique.
- Le profil actif est affiche dans le panneau stats et dans le menu de droite.

## Fichiers generes

Tous les exports sont dans le dossier `sorties/`:

- `demo_seance_YYYYMMDD_HHMMSS.mp4`
- `rapport_seance_YYYYMMDD_HHMMSS.csv`
- `historique_seance_YYYYMMDD_HHMMSS.json`

## Remarques

- L'application est 100% locale (pas d'internet necessaire)
- Aucun GPU obligatoire
- Le rendu est entierement dans la fenetre OpenCV (pas de GUI externe)
