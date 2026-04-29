"""Configuration centrale du projet Coach Sportif Temps Reel."""

from copy import deepcopy
from pathlib import Path

# Couleurs en BGR (OpenCV)
COULEUR_VERT = (83, 200, 0)
COULEUR_ROUGE = (0, 0, 213)
COULEUR_BLEU_CLAIR = (255, 210, 120)
COULEUR_AMBRE = (0, 170, 255)
COULEUR_BLANC = (255, 255, 255)
COULEUR_NOIR = (0, 0, 0)
MODE_COULEUR_BINAIRE = True

# Reglages video
LARGEUR_VIDEO = 1280
HAUTEUR_VIDEO = 720
FPS_CIBLE = 30
NOM_FENETRE = "Coach Sportif Temps Reel"
FENETRE_FPS = 120

# Reglages visibilite pour detecter l'orientation
SEUIL_FACE = 0.6
SEUIL_PROFIL_HAUT = 0.7
SEUIL_PROFIL_BAS = 0.3
FENETRE_ORIENTATION = 8
SEUIL_VISIBILITE_POINT = 0.25

# Reglages anti faux positifs
SEUIL_VARIATION_ANGLE = 4.0
SEUIL_FRAMES_INACTIVITE = 20
MIN_FRAMES_PAR_REP = 6
FRAMES_RETENTION_FEEDBACK = 18
INVALIDER_REP_SUR_PREMIERE_ERREUR = False
BLOQUER_ORIENTATION_EXERCICE = False

# Reglages MediaPipe
CONFIANCE_DETECTION = 0.5
CONFIANCE_SUIVI = 0.5

# Reglages des exercices
EXERCICES = {
    "squat": {
        "nom": "Squat",
        "touche": "1",
        "orientations_valides": ("profil_gauche", "profil_droit"),
        "message_orientation_exercice": "Squat: mets-toi de profil pour une analyse fiable.",
        "seuil_haut": 155,
        "seuil_bas": 130,
        "angle_bas_valide": 90,
        "amplitude_min_cycle": 35,
        "angle_alignement_min_haut": 142,
        "angle_alignement_min_bas": 126,
        "deplacement_max": 0.65,
    },
    "pushup": {
        "nom": "Push-up",
        "touche": "2",
        "orientations_valides": ("profil_gauche", "profil_droit"),
        "message_orientation_exercice": "Push-up: mets-toi de profil pour une analyse fiable.",
        "seuil_haut": 155,
        "seuil_bas": 95,
        "angle_bas_valide": 90,
        "amplitude_min_cycle": 30,
        "angle_alignement_min": 160,
        "deplacement_max": 0.0,
    },
    "curl": {
        "nom": "Bicep Curl",
        "touche": "3",
        "orientations_valides": ("profil_gauche", "profil_droit"),
        "message_orientation_exercice": "Curl: mets-toi de profil pour une analyse fiable.",
        "seuil_haut": 150,
        "seuil_bas": 45,
        "angle_bas_valide": 35,
        "angle_haut_valide": 150,
        "amplitude_min_cycle": 55,
        "deplacement_max": 0.05,
    },
}

# Sorties
DOSSIER_SORTIES = Path("sorties")
DOSSIER_SORTIES.mkdir(parents=True, exist_ok=True)

# Textes d'aide
MESSAGE_ATTENTE = "Positionne-toi devant la camera."
MESSAGE_ORIENTATION = "Place-toi de face ou de profil pour une detection fiable."
MESSAGE_OK = "Super ! Continue."
MESSAGE_PRET = "En position. Descends pour lancer la repetition."
MESSAGE_INACTIVITE = "Pret. Lance une repetition complete."

# Profils demo
PROFIL_DEMO_PAR_DEFAUT = "strict"

_BASE_REGLAGES_ANTI_FAUX_POSITIFS = {
    "SEUIL_VARIATION_ANGLE": SEUIL_VARIATION_ANGLE,
    "SEUIL_FRAMES_INACTIVITE": SEUIL_FRAMES_INACTIVITE,
    "MIN_FRAMES_PAR_REP": MIN_FRAMES_PAR_REP,
    "FRAMES_RETENTION_FEEDBACK": FRAMES_RETENTION_FEEDBACK,
    "INVALIDER_REP_SUR_PREMIERE_ERREUR": INVALIDER_REP_SUR_PREMIERE_ERREUR,
    "BLOQUER_ORIENTATION_EXERCICE": BLOQUER_ORIENTATION_EXERCICE,
}
_BASE_EXERCICES = deepcopy(EXERCICES)

PROFILS_DEMO = {
    "strict": {
        "reglages": {
            "SEUIL_VARIATION_ANGLE": 3.0,
            "SEUIL_FRAMES_INACTIVITE": 18,
            "MIN_FRAMES_PAR_REP": 8,
            "FRAMES_RETENTION_FEEDBACK": 22,
            "INVALIDER_REP_SUR_PREMIERE_ERREUR": True,
            "BLOQUER_ORIENTATION_EXERCICE": False,
        },
        "exercices": {
            "squat": {
                "seuil_bas": 125,
                "angle_bas_valide": 88,
                "amplitude_min_cycle": 42,
                "angle_alignement_min_haut": 148,
                "angle_alignement_min_bas": 132,
                "deplacement_max": 0.58,
            },
            "pushup": {
                "angle_bas_valide": 86,
                "amplitude_min_cycle": 36,
                "angle_alignement_min": 164,
            },
            "curl": {
                "angle_bas_valide": 32,
                "angle_haut_valide": 154,
                "amplitude_min_cycle": 62,
                "deplacement_max": 0.045,
            },
        },
    },
    "equilibre": {
        "reglages": {
            "SEUIL_VARIATION_ANGLE": 4.0,
            "SEUIL_FRAMES_INACTIVITE": 20,
            "MIN_FRAMES_PAR_REP": 6,
            "FRAMES_RETENTION_FEEDBACK": 18,
            "INVALIDER_REP_SUR_PREMIERE_ERREUR": False,
            "BLOQUER_ORIENTATION_EXERCICE": False,
        },
        "exercices": {
            "squat": {
                "seuil_bas": 130,
                "angle_bas_valide": 90,
                "amplitude_min_cycle": 35,
                "angle_alignement_min_haut": 142,
                "angle_alignement_min_bas": 126,
                "deplacement_max": 0.65,
            },
            "pushup": {
                "angle_bas_valide": 90,
                "amplitude_min_cycle": 30,
                "angle_alignement_min": 160,
            },
            "curl": {
                "angle_bas_valide": 35,
                "angle_haut_valide": 150,
                "amplitude_min_cycle": 55,
                "deplacement_max": 0.05,
            },
        },
    },
    "tolerant": {
        "reglages": {
            "SEUIL_VARIATION_ANGLE": 5.5,
            "SEUIL_FRAMES_INACTIVITE": 24,
            "MIN_FRAMES_PAR_REP": 5,
            "FRAMES_RETENTION_FEEDBACK": 14,
            "INVALIDER_REP_SUR_PREMIERE_ERREUR": False,
            "BLOQUER_ORIENTATION_EXERCICE": False,
        },
        "exercices": {
            "squat": {
                "seuil_bas": 135,
                "angle_bas_valide": 95,
                "amplitude_min_cycle": 30,
                "angle_alignement_min_haut": 136,
                "angle_alignement_min_bas": 120,
                "deplacement_max": 0.74,
            },
            "pushup": {
                "angle_bas_valide": 95,
                "amplitude_min_cycle": 26,
                "angle_alignement_min": 154,
            },
            "curl": {
                "angle_bas_valide": 40,
                "angle_haut_valide": 146,
                "amplitude_min_cycle": 48,
                "deplacement_max": 0.06,
            },
        },
    },
}


def appliquer_profil_demo(nom_profil: str) -> str:
    """Applique un profil demo de facon idempotente et retourne son nom effectif."""
    profil = PROFILS_DEMO.get(nom_profil, PROFILS_DEMO[PROFIL_DEMO_PAR_DEFAUT])
    nom_effectif = nom_profil if nom_profil in PROFILS_DEMO else PROFIL_DEMO_PAR_DEFAUT

    # Reset complet sur la base pour eviter l'accumulation de modifications.
    globals().update(_BASE_REGLAGES_ANTI_FAUX_POSITIFS)
    for nom_exercice, regles in _BASE_EXERCICES.items():
        EXERCICES[nom_exercice] = deepcopy(regles)

    for cle, valeur in profil["reglages"].items():
        globals()[cle] = valeur

    for nom_exercice, regles in profil["exercices"].items():
        EXERCICES[nom_exercice].update(regles)

    return nom_effectif
