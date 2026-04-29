"""Fonctions d'affichage OpenCV pour le feedback visuel."""

from __future__ import annotations

import cv2
import numpy as np

import config


def _normaliser_rectangle(
    image,
    coin_haut_gauche: tuple[int, int],
    coin_bas_droit: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Contraint le rectangle aux bornes de l'image."""
    hauteur, largeur = image.shape[:2]

    x1, y1 = coin_haut_gauche
    x2, y2 = coin_bas_droit

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    x1 = max(0, min(x1, largeur))
    x2 = max(0, min(x2, largeur))
    y1 = max(0, min(y1, hauteur))
    y2 = max(0, min(y2, hauteur))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _dessiner_rectangle_transparent(
    image,
    coin_haut_gauche: tuple[int, int],
    coin_bas_droit: tuple[int, int],
    couleur: tuple[int, int, int],
    opacite: float,
) -> None:
    """Dessine un rectangle semi-transparent sur l'image."""
    rectangle = _normaliser_rectangle(image, coin_haut_gauche, coin_bas_droit)
    if rectangle is None:
        return

    x1, y1, x2, y2 = rectangle
    zone = image[y1:y2, x1:x2]
    if zone.size == 0:
        return

    calque = np.full_like(zone, couleur, dtype=zone.dtype)
    cv2.addWeighted(calque, opacite, zone, 1.0 - opacite, 0, zone)


def _dessiner_barre_progression(
    image,
    position: tuple[int, int],
    taille: tuple[int, int],
    valeur: float,
    valeur_max: float,
    couleur_remplissage: tuple[int, int, int],
) -> None:
    """Affiche une barre de progression simple et lisible."""
    x, y = position
    largeur, hauteur = taille
    largeur = max(largeur, 10)
    hauteur = max(hauteur, 8)

    valeur_bornee = 0.0
    if valeur_max > 0:
        valeur_bornee = max(0.0, min(1.0, valeur / valeur_max))

    cv2.rectangle(image, (x, y), (x + largeur, y + hauteur), (80, 80, 80), -1)
    largeur_remplie = int((largeur - 2) * valeur_bornee)
    if largeur_remplie > 0:
        cv2.rectangle(
            image,
            (x + 1, y + 1),
            (x + 1 + largeur_remplie, y + hauteur - 1),
            couleur_remplissage,
            -1,
        )
    cv2.rectangle(image, (x, y), (x + largeur, y + hauteur), config.COULEUR_BLANC, 1)


def dessiner_panneau_stats(image, donnees_stats: dict[str, str | float | int | None]) -> None:
    """Affiche les statistiques de seance dans un panneau fixe en haut a gauche."""
    _dessiner_rectangle_transparent(image, (12, 12), (460, 285), config.COULEUR_NOIR, 0.45)

    angle = donnees_stats.get("angle_principal")
    angle_texte = "-" if angle is None else f"{angle:.1f}"
    taux_reussite = float(donnees_stats.get("taux_reussite", 0.0) or 0.0)
    repetitions_invalides = int(donnees_stats.get("repetitions_invalides", 0) or 0)

    lignes = [
        f"Exercice: {donnees_stats.get('nom_exercice', '-')}",
        f"Profil: {str(donnees_stats.get('profil', '-')).capitalize()}",
        f"Orientation: {donnees_stats.get('orientation', '-')}",
        f"Reps correctes: {donnees_stats.get('repetitions_correctes', 0)}",
        f"Reps totales: {donnees_stats.get('repetitions_totales', 0)}",
        f"Reps invalides: {repetitions_invalides}",
        f"Taux correct: {taux_reussite:.1f}%",
        f"Etat mouvement: {donnees_stats.get('etat_mouvement', '-')}",
        f"Angle principal: {angle_texte}",
        f"FPS: {donnees_stats.get('fps', 0.0):.1f}",
        f"Duree: {donnees_stats.get('duree_sec', 0.0):.1f}s",
    ]

    y = 40
    for ligne in lignes:
        cv2.putText(
            image,
            ligne,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            config.COULEUR_BLANC,
            1,
            cv2.LINE_AA,
        )
        y += 22

    _dessiner_barre_progression(image, (24, 246), (250, 14), taux_reussite, 100.0, config.COULEUR_VERT)
    cv2.putText(
        image,
        "Qualite",
        (282, 258),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        config.COULEUR_BLANC,
        1,
        cv2.LINE_AA,
    )

    if angle is not None:
        _dessiner_barre_progression(image, (24, 266), (250, 12), float(angle), 180.0, config.COULEUR_BLEU_CLAIR)
        cv2.putText(
            image,
            "Amplitude angle",
            (282, 277),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            config.COULEUR_BLANC,
            1,
            cv2.LINE_AA,
        )


def dessiner_message_principal(
    image,
    message: str,
    est_correct: bool,
    niveau_feedback: str | None = None,
) -> None:
    """Affiche le message principal en bas de l'image."""
    hauteur, largeur = image.shape[:2]

    if config.MODE_COULEUR_BINAIRE:
        couleur_fond = config.COULEUR_VERT if est_correct else config.COULEUR_ROUGE
    else:
        if niveau_feedback == "neutre":
            couleur_fond = config.COULEUR_AMBRE
        else:
            couleur_fond = config.COULEUR_VERT if est_correct else config.COULEUR_ROUGE

    _dessiner_rectangle_transparent(
        image,
        (20, hauteur - 82),
        (largeur - 20, hauteur - 20),
        couleur_fond,
        0.35,
    )

    taille_police = 0.85
    epaisseur = 2
    texte = message
    while taille_police > 0.55:
        (largeur_texte, _), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, taille_police, epaisseur)
        if largeur_texte <= largeur - 80:
            break
        taille_police -= 0.05
        epaisseur = max(1, epaisseur - 1)

    (largeur_texte, _), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, taille_police, epaisseur)
    x_texte = max(32, (largeur - largeur_texte) // 2)

    cv2.putText(
        image,
        texte,
        (x_texte, hauteur - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        taille_police,
        config.COULEUR_BLANC,
        epaisseur,
        cv2.LINE_AA,
    )


def dessiner_menu_exercices(image, exercice_actuel: str, profil_actif: str = config.PROFIL_DEMO_PAR_DEFAUT) -> None:
    """Affiche les touches de controle de l'application."""
    hauteur, largeur = image.shape[:2]

    lignes = [
        "1 Squat | 2 Push-up | 3 Curl",
        "Q Quitter la session",
        f"Profil: {profil_actif.capitalize()}",
        f"Actif: {config.EXERCICES[exercice_actuel]['nom']}",
    ]

    largeur_max_texte = max(cv2.getTextSize(ligne, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0] for ligne in lignes)
    x1 = max(12, largeur - (largeur_max_texte + 42))
    x2 = largeur - 12

    _dessiner_rectangle_transparent(
        image,
        (x1, 12),
        (x2, 190),
        config.COULEUR_NOIR,
        0.45,
    )

    y = 45
    for ligne in lignes:
        cv2.putText(
            image,
            ligne,
            (x1 + 14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            config.COULEUR_BLEU_CLAIR,
            1,
            cv2.LINE_AA,
        )
        y += 35
