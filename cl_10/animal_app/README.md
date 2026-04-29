# Animal-10 Web App

Interface web simple pour classifier une image d'animal avec un modele Keras.

## Installation

1. Placer `animal-10.keras` a la racine du projet (un niveau au-dessus de `animal_app`).
2. Creer un environnement et installer les dependances :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r animal_app/requirements.txt
```

## Configuration

- Mettre les classes dans `animal_app/config.py` dans le bon ordre (`CLASS_NAMES`).
- Ajuster `IMAGE_SIZE` selon la taille d'entree du modele.
- Ajuster `CONFIDENCE_THRESHOLD` si besoin.

## Lancement

```bash
uvicorn animal_app.app:app --reload
```

Ouvrir ensuite `http://127.0.0.1:8000`.
