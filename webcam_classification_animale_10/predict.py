#!/usr/bin/env python3
"""Classification animale en temps reel depuis webcam."""

from __future__ import annotations

import argparse
from collections import deque
import queue
import json
import os
import sys
import threading
import zipfile
from pathlib import Path
from time import perf_counter, sleep

import cv2
import numpy as np

CANDIDATS_MODELE_PAR_DEFAUT = (
    "animal-10.keras",
    "animal_classifier.h5",
    "animal_classifier.keras",
    "animal_classifier",
)


class FluxWebcamAsynchrone:
    def __init__(self, capture) -> None:
        self._capture = capture
        self._buffer: deque[np.ndarray] = deque(maxlen=1)
        self._lock = threading.Lock()
        self._running = False
        self._reader_error = False
        self._thread: threading.Thread | None = None

    def demarrer(self) -> "FluxWebcamAsynchrone":
        self._running = True
        self._thread = threading.Thread(target=self._boucle_lecture, daemon=True)
        self._thread.start()
        return self

    def _boucle_lecture(self) -> None:
        while self._running:
            lecture_ok, frame = self._capture.read()
            if not lecture_ok:
                self._reader_error = True
                self._running = False
                break
            with self._lock:
                self._buffer.append(frame)

    def lire_dernier(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if not self._buffer:
                return False, None
            frame = self._buffer.pop()
            self._buffer.clear()
            return True, frame

    @property
    def en_erreur(self) -> bool:
        return self._reader_error

    def arreter(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.3)


class MoteurInferenceAsynchrone:
    def __init__(
        self,
        modele,
        etiquettes: list[str],
        taille_image: tuple[int, int],
        arguments: argparse.Namespace,
        detecteur_holistique,
        couleur_valide: tuple[int, int, int],
        couleur_attention: tuple[int, int, int],
        couleur_alerte: tuple[int, int, int],
        mode_pretraitement_actif: str,
    ) -> None:
        self._modele = modele
        self._etiquettes = etiquettes
        self._taille_image = taille_image
        self._arguments = arguments
        self._detecteur_holistique = detecteur_holistique
        self._couleur_valide = couleur_valide
        self._couleur_attention = couleur_attention
        self._couleur_alerte = couleur_alerte

        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._mode_pretraitement_actif = mode_pretraitement_actif
        self._probabilites_lissees: np.ndarray | None = None
        self._indice_candidat: int | None = None
        self._compteur_candidat = 0

        self._inference_autorisee = True
        self._frames_depuis_presence_check = arguments.mediapipe_every
        self._dernier_temps_retry_preprocess = -1e9
        self._dernier_temps_retry_zoom = -1e9

        self._ligne_1 = "Animal: Initialisation..."
        self._ligne_2 = "Preparation de l'inference"
        self._couleur_texte = couleur_attention

    def demarrer(self) -> "MoteurInferenceAsynchrone":
        self._running = True
        self._thread = threading.Thread(target=self._boucle, daemon=True)
        self._thread.start()
        return self

    def arreter(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def soumettre_trame(self, trame_bgr: np.ndarray) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(trame_bgr)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(trame_bgr)
            except queue.Full:
                pass

    def lire_etat(self) -> tuple[str, str, tuple[int, int, int], str]:
        with self._lock:
            return (
                self._ligne_1,
                self._ligne_2,
                self._couleur_texte,
                self._mode_pretraitement_actif,
            )

    def _mettre_a_jour_overlay(
        self,
        ligne_1: str,
        ligne_2: str,
        couleur: tuple[int, int, int],
    ) -> None:
        with self._lock:
            self._ligne_1 = ligne_1
            self._ligne_2 = ligne_2
            self._couleur_texte = couleur

    def _boucle(self) -> None:
        while self._running:
            try:
                trame = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if not self._running or trame is None:
                continue

            temps_actuel = perf_counter()

            if self._detecteur_holistique is not None:
                if self._frames_depuis_presence_check >= self._arguments.mediapipe_every:
                    self._inference_autorisee = sujet_present(trame, self._detecteur_holistique)
                    self._frames_depuis_presence_check = 1
                else:
                    self._frames_depuis_presence_check += 1
            else:
                self._inference_autorisee = True

            if not self._inference_autorisee:
                self._probabilites_lissees = None
                self._indice_candidat = None
                self._compteur_candidat = 0
                self._mettre_a_jour_overlay(
                    "Etat: Aucun sujet detecte",
                    "Inference en pause (MediaPipe)",
                    self._couleur_attention,
                )
                continue

            image_modele = preparer_image_modele(
                trame,
                self._taille_image,
                self._arguments.crop_mode,
            )
            probabilites_brutes = inferer_depuis_image(
                self._modele,
                image_modele,
                self._mode_pretraitement_actif,
            )

            if self._arguments.preprocess == "auto" and self._arguments.auto_retry_margin > 0.0:
                _, confiance_brute = calculer_confiance_max(probabilites_brutes)
                seuil_retry = min(
                    99.0,
                    self._arguments.min_confidence + self._arguments.auto_retry_margin,
                )
                peut_tenter_retry = (
                    (temps_actuel - self._dernier_temps_retry_preprocess)
                    >= self._arguments.auto_retry_cooldown
                )
                if confiance_brute < seuil_retry and peut_tenter_retry:
                    self._dernier_temps_retry_preprocess = temps_actuel
                    mode_alternatif = (
                        "minus-one-one"
                        if self._mode_pretraitement_actif == "zero-one"
                        else "zero-one"
                    )
                    probabilites_alt = inferer_depuis_image(
                        self._modele,
                        image_modele,
                        mode_alternatif,
                    )
                    _, confiance_alt = calculer_confiance_max(probabilites_alt)
                    if confiance_alt >= (confiance_brute + self._arguments.auto_retry_gain):
                        with self._lock:
                            self._mode_pretraitement_actif = mode_alternatif
                        probabilites_brutes = probabilites_alt
                        print(
                            "[INFO] Auto-preprocess bascule vers "
                            f"{mode_alternatif}."
                        )

            if self._arguments.retry_zoom < 0.999:
                _, confiance_brute = calculer_confiance_max(probabilites_brutes)
                seuil_zoom = min(
                    99.0,
                    self._arguments.min_confidence + self._arguments.auto_retry_margin,
                )
                peut_tenter_zoom = (
                    (temps_actuel - self._dernier_temps_retry_zoom)
                    >= self._arguments.retry_zoom_cooldown
                )
                if confiance_brute < seuil_zoom and peut_tenter_zoom:
                    self._dernier_temps_retry_zoom = temps_actuel
                    trame_zoom = extraire_zoom_central(trame, self._arguments.retry_zoom)
                    image_zoom = preparer_image_modele(
                        trame_zoom,
                        self._taille_image,
                        self._arguments.crop_mode,
                    )
                    probabilites_zoom = inferer_depuis_image(
                        self._modele,
                        image_zoom,
                        self._mode_pretraitement_actif,
                    )
                    _, confiance_zoom = calculer_confiance_max(probabilites_zoom)
                    if confiance_zoom >= (confiance_brute + self._arguments.retry_zoom_gain):
                        probabilites_brutes = probabilites_zoom

            self._probabilites_lissees = lisser_probabilites(
                self._probabilites_lissees,
                probabilites_brutes,
                self._arguments.smooth_alpha,
            )

            indice_classe, confiance = calculer_confiance_max(self._probabilites_lissees)
            confiance = max(0.0, min(100.0, confiance))

            seuil_souple = max(0.0, self._arguments.min_confidence - self._arguments.soft_margin)
            if confiance >= seuil_souple:
                if indice_classe == self._indice_candidat:
                    self._compteur_candidat += 1
                else:
                    self._indice_candidat = indice_classe
                    self._compteur_candidat = 1
            else:
                self._indice_candidat = None
                self._compteur_candidat = 0

            prediction_fiable = (
                confiance >= self._arguments.min_confidence
                or (
                    confiance >= seuil_souple
                    and self._compteur_candidat >= (self._arguments.stable_frames + 2)
                )
            )

            if not prediction_fiable:
                nom_animal = "Indetermine"
                ligne_etat = (
                    f"Top1: {self._etiquettes[indice_classe]} ({confiance:.1f}%) "
                    f"| min {self._arguments.min_confidence:.0f}%"
                )
                couleur_texte = self._couleur_alerte
            elif self._compteur_candidat < self._arguments.stable_frames:
                nom_animal = "Validation..."
                ligne_etat = (
                    f"Candidat: {self._etiquettes[indice_classe]} "
                    f"({self._compteur_candidat}/{self._arguments.stable_frames})"
                )
                couleur_texte = self._couleur_attention
            elif confiance >= self._arguments.high_confidence:
                nom_animal = self._etiquettes[indice_classe]
                ligne_etat = "Prediction confiante"
                couleur_texte = self._couleur_valide
            elif confiance < self._arguments.min_confidence:
                nom_animal = self._etiquettes[indice_classe]
                ligne_etat = f"Confiance basse mais stable: {confiance:.1f}%"
                couleur_texte = self._couleur_attention
            else:
                nom_animal = self._etiquettes[indice_classe]
                ligne_etat = f"Confiance: {confiance:.1f}%"
                couleur_texte = self._couleur_attention

            if nom_animal == "Validation...":
                ligne_1 = "Animal: Validation..."
            else:
                ligne_1 = f"Animal: {nom_animal}"

            self._mettre_a_jour_overlay(ligne_1, ligne_etat, couleur_texte)


def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classification animale en temps reel via webcam."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index de la webcam (defaut: 0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Chemin du modele (.keras/.h5/SavedModel). Si vide, detection auto.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="labels.txt",
        help="Chemin du fichier labels (1 classe/ligne).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=45.0,
        help="Seuil minimum de confiance (defaut: 45).",
    )
    parser.add_argument(
        "--high-confidence",
        type=float,
        default=70.0,
        help="Seuil de confiance elevee pour la couleur verte (defaut: 70).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Largeur cible webcam (defaut: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Hauteur cible webcam (defaut: 480).",
    )
    parser.add_argument(
        "--use-mediapipe",
        action="store_true",
        help="Active une detection de presence avant inference.",
    )
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plein ecran desactive par defaut (utiliser --fullscreen pour l'activer).",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.30,
        help="Lissage temporel des probabilites [0-0.99] (defaut: 0.30).",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=1,
        help="Nombre de frames consecutives fiables avant validation (defaut: 1).",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=4,
        help="Lance une inference toutes les N frames (defaut: 4).",
    )
    parser.add_argument(
        "--max-infer-fps",
        type=float,
        default=6.0,
        help="Limite le debit d'inference en FPS (0 = illimite, defaut: 6).",
    )
    parser.add_argument(
        "--mediapipe-every",
        type=int,
        default=4,
        help="Frequence de verification MediaPipe (1 = chaque frame, defaut: 4).",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("center-crop", "stretch", "letterbox"),
        default="center-crop",
        help=(
            "Mode de cadrage avant resize: center-crop (defaut), "
            "stretch ou letterbox."
        ),
    )
    parser.add_argument(
        "--preprocess",
        choices=("auto", "zero-one", "minus-one-one"),
        default="auto",
        help="Pretraitement image: auto, zero-one (x/255) ou minus-one-one ((x/127.5)-1).",
    )
    parser.add_argument(
        "--auto-retry-margin",
        type=float,
        default=6.0,
        help="Marge de confiance declenchant le test de pretraitement alternatif en mode auto.",
    )
    parser.add_argument(
        "--auto-retry-gain",
        type=float,
        default=2.0,
        help="Gain minimum (%%) pour basculer vers l'autre pretraitement en mode auto.",
    )
    parser.add_argument(
        "--auto-retry-cooldown",
        type=float,
        default=0.40,
        help="Delai minimal (secondes) entre deux tests de pretraitement alternatif (defaut: 0.40).",
    )
    parser.add_argument(
        "--retry-zoom",
        type=float,
        default=0.85,
        help=(
            "Zoom central applique ponctuellement si confiance faible (0<z<=1, "
            "defaut: 0.85, 1 desactive)."
        ),
    )
    parser.add_argument(
        "--retry-zoom-gain",
        type=float,
        default=1.0,
        help="Gain minimum (%%) requis pour accepter la prediction issue du retry zoom.",
    )
    parser.add_argument(
        "--retry-zoom-cooldown",
        type=float,
        default=0.35,
        help="Delai minimal (secondes) entre deux retries zoom (defaut: 0.35).",
    )
    parser.add_argument(
        "--soft-margin",
        type=float,
        default=8.0,
        help=(
            "Marge (%%) sous --min-confidence acceptee si la prediction est stable "
            "sur plusieurs frames (defaut: 8)."
        ),
    )
    return parser.parse_args()


def valider_arguments(arguments: argparse.Namespace) -> None:
    if not (0.0 <= arguments.smooth_alpha < 1.0):
        raise ValueError("--smooth-alpha doit etre compris entre 0.0 et 0.99.")
    if arguments.stable_frames < 1:
        raise ValueError("--stable-frames doit etre >= 1.")
    if arguments.infer_every < 1:
        raise ValueError("--infer-every doit etre >= 1.")
    if arguments.max_infer_fps < 0.0:
        raise ValueError("--max-infer-fps doit etre >= 0.")
    if arguments.mediapipe_every < 1:
        raise ValueError("--mediapipe-every doit etre >= 1.")
    if arguments.auto_retry_margin < 0.0:
        raise ValueError("--auto-retry-margin doit etre >= 0.")
    if arguments.auto_retry_gain < 0.0:
        raise ValueError("--auto-retry-gain doit etre >= 0.")
    if arguments.auto_retry_cooldown < 0.0:
        raise ValueError("--auto-retry-cooldown doit etre >= 0.")
    if not (0.0 < arguments.retry_zoom <= 1.0):
        raise ValueError("--retry-zoom doit etre dans ]0, 1].")
    if arguments.retry_zoom_gain < 0.0:
        raise ValueError("--retry-zoom-gain doit etre >= 0.")
    if arguments.retry_zoom_cooldown < 0.0:
        raise ValueError("--retry-zoom-cooldown doit etre >= 0.")
    if arguments.soft_margin < 0.0:
        raise ValueError("--soft-margin doit etre >= 0.")


def trouver_chemin_modele(modele_utilisateur: str, dossier_racine: Path) -> Path:
    if modele_utilisateur:
        chemin_modele = Path(modele_utilisateur).expanduser()
        if not chemin_modele.is_absolute():
            chemin_modele = dossier_racine / chemin_modele
        if chemin_modele.exists():
            return chemin_modele
        raise FileNotFoundError(
            f"Modele introuvable: {chemin_modele}. Verifie l'option --model."
        )

    for candidat in CANDIDATS_MODELE_PAR_DEFAUT:
        chemin_candidat = dossier_racine / candidat
        if chemin_candidat.exists():
            return chemin_candidat

    fichiers_keras = sorted(dossier_racine.glob("*.keras"))
    if fichiers_keras:
        return fichiers_keras[0]

    fichiers_h5 = sorted(dossier_racine.glob("*.h5"))
    if fichiers_h5:
        return fichiers_h5[0]

    raise FileNotFoundError(
        "Aucun modele trouve. Ajoute un fichier .keras/.h5 ou utilise --model."
    )


def charger_etiquettes(chemin_etiquettes: Path, nombre_classes: int) -> list[str]:
    if chemin_etiquettes.exists():
        etiquettes = [
            ligne.strip()
            for ligne in chemin_etiquettes.read_text(encoding="utf-8").splitlines()
            if ligne.strip()
        ]
    else:
        etiquettes = []

    if not etiquettes:
        print(
            f"[WARN] labels introuvables ({chemin_etiquettes})."
            " Utilisation de labels generiques class_0..class_n.",
            file=sys.stderr,
        )
        return [f"class_{indice}" for indice in range(nombre_classes)]

    if len(etiquettes) < nombre_classes:
        print(
            f"[WARN] labels insuffisants ({len(etiquettes)}/{nombre_classes}). "
            "Complements generiques ajoutes.",
            file=sys.stderr,
        )
        etiquettes.extend(
            f"class_{indice}" for indice in range(len(etiquettes), nombre_classes)
        )
        return etiquettes

    if len(etiquettes) > nombre_classes:
        print(
            f"[WARN] labels en trop ({len(etiquettes)}>{nombre_classes}). "
            "Les premiers labels seront utilises.",
            file=sys.stderr,
        )
        return etiquettes[:nombre_classes]

    return etiquettes


def cadrer_trame_inference(
    trame_bgr: np.ndarray,
    mode_cadrage: str,
) -> np.ndarray:
    if mode_cadrage == "stretch":
        return trame_bgr

    hauteur, largeur = trame_bgr.shape[:2]
    if mode_cadrage == "center-crop":
        cote = min(hauteur, largeur)
        y0 = (hauteur - cote) // 2
        x0 = (largeur - cote) // 2
        return trame_bgr[y0 : y0 + cote, x0 : x0 + cote]

    cote = max(hauteur, largeur)
    pad_y0 = (cote - hauteur) // 2
    pad_y1 = cote - hauteur - pad_y0
    pad_x0 = (cote - largeur) // 2
    pad_x1 = cote - largeur - pad_x0
    return cv2.copyMakeBorder(
        trame_bgr,
        pad_y0,
        pad_y1,
        pad_x0,
        pad_x1,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def extraire_zoom_central(trame_bgr: np.ndarray, ratio_zoom: float) -> np.ndarray:
    if ratio_zoom >= 0.999:
        return trame_bgr

    hauteur, largeur = trame_bgr.shape[:2]
    nouvelle_hauteur = max(2, int(round(hauteur * ratio_zoom)))
    nouvelle_largeur = max(2, int(round(largeur * ratio_zoom)))

    y0 = max(0, (hauteur - nouvelle_hauteur) // 2)
    x0 = max(0, (largeur - nouvelle_largeur) // 2)
    y1 = min(hauteur, y0 + nouvelle_hauteur)
    x1 = min(largeur, x0 + nouvelle_largeur)
    return trame_bgr[y0:y1, x0:x1]


def preparer_image_modele(
    trame_bgr: np.ndarray,
    taille_image: tuple[int, int],
    mode_cadrage: str,
) -> np.ndarray:
    trame_cadree = cadrer_trame_inference(trame_bgr, mode_cadrage)
    trame_rgb = cv2.cvtColor(trame_cadree, cv2.COLOR_BGR2RGB)
    image_redimensionnee = cv2.resize(trame_rgb, taille_image, interpolation=cv2.INTER_AREA)
    return image_redimensionnee.astype(np.float32)


def normaliser_image_modele(
    image_rgb_float: np.ndarray,
    mode_pretraitement: str,
) -> np.ndarray:
    if mode_pretraitement == "minus-one-one":
        image_normalisee = (image_rgb_float / 127.5) - 1.0
    else:
        image_normalisee = image_rgb_float / 255.0

    return np.expand_dims(image_normalisee, axis=0)


def pretraiter_trame(
    trame_bgr: np.ndarray,
    taille_image: tuple[int, int],
    mode_pretraitement: str,
    mode_cadrage: str = "center-crop",
) -> np.ndarray:
    image_float = preparer_image_modele(trame_bgr, taille_image, mode_cadrage)
    return normaliser_image_modele(image_float, mode_pretraitement)


def inferer_depuis_image(
    modele,
    image_rgb_float: np.ndarray,
    mode_pretraitement: str,
) -> np.ndarray:
    lot_image = normaliser_image_modele(image_rgb_float, mode_pretraitement)
    return inferer_probabilites(modele, lot_image)



def inferer_depuis_trame(
    modele,
    trame_bgr: np.ndarray,
    taille_image: tuple[int, int],
    mode_pretraitement: str,
    mode_cadrage: str = "center-crop",
) -> np.ndarray:
    image_float = preparer_image_modele(trame_bgr, taille_image, mode_cadrage)
    return inferer_depuis_image(modele, image_float, mode_pretraitement)


def mode_pretraitement_auto(modele) -> str:
    nom_modele = str(getattr(modele, "name", "")).lower()
    noms_couches = " ".join(
        str(getattr(couche, "name", "")).lower()
        for couche in getattr(modele, "layers", [])
    )

    if "mobilenet" in nom_modele or "mobilenet" in noms_couches:
        return "minus-one-one"
    return "zero-one"


def ecrire_texte_lisible(
    trame: np.ndarray,
    texte: str,
    position: tuple[int, int],
    couleur: tuple[int, int, int],
    echelle: float,
    epaisseur: int,
) -> None:
    cv2.putText(
        trame,
        texte,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        echelle,
        (0, 0, 0),
        epaisseur + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        trame,
        texte,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        echelle,
        couleur,
        epaisseur,
        cv2.LINE_AA,
    )


def afficher_overlay_classique(
    trame: np.ndarray,
    ligne_1: str,
    ligne_2: str,
    couleur_ligne_2: tuple[int, int, int],
) -> None:
    hauteur, largeur = trame.shape[:2]
    x = 18
    y = 18
    padding_interne = 16
    max_largeur = max(280, largeur - 36)

    taille_l1, _ = cv2.getTextSize(ligne_1, cv2.FONT_HERSHEY_SIMPLEX, 0.86, 2)
    taille_l2, _ = cv2.getTextSize(ligne_2, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    largeur_boite = min(max_largeur, max(taille_l1[0], taille_l2[0]) + padding_interne * 2)
    hauteur_boite = 92

    x1 = min(largeur, x + largeur_boite)
    y1 = min(hauteur, y + hauteur_boite)
    if x1 <= x or y1 <= y:
        return

    roi = trame[y:y1, x:x1]
    overlay = roi.copy()
    overlay[:] = (10, 10, 10)
    cv2.addWeighted(overlay, 0.62, roi, 0.38, 0.0, roi)
    cv2.rectangle(trame, (x, y), (x1, y1), (245, 245, 245), 1)

    ecrire_texte_lisible(
        trame=trame,
        texte=ligne_1,
        position=(x + padding_interne, y + 35),
        couleur=(255, 255, 255),
        echelle=0.86,
        epaisseur=2,
    )
    ecrire_texte_lisible(
        trame=trame,
        texte=ligne_2,
        position=(x + padding_interne, y + 72),
        couleur=couleur_ligne_2,
        echelle=0.72,
        epaisseur=2,
    )


def afficher_pied(trame: np.ndarray, texte: str) -> None:
    hauteur, largeur = trame.shape[:2]
    hauteur_bande = 42
    y0 = hauteur - hauteur_bande
    y0 = max(0, y0)
    roi = trame[y0:hauteur, 0:largeur]
    overlay = roi.copy()
    overlay[:] = (12, 12, 12)
    cv2.addWeighted(overlay, 0.52, roi, 0.48, 0.0, roi)

    y = trame.shape[0] - 15
    cv2.putText(
        trame,
        texte,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )


def regler_plein_ecran(nom_fenetre: str, activer: bool) -> bool:
    try:
        mode = cv2.WINDOW_FULLSCREEN if activer else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(nom_fenetre, cv2.WND_PROP_FULLSCREEN, mode)
        if hasattr(cv2, "WND_PROP_ASPECT_RATIO") and hasattr(cv2, "WINDOW_FREERATIO"):
            cv2.setWindowProperty(
                nom_fenetre,
                cv2.WND_PROP_ASPECT_RATIO,
                cv2.WINDOW_FREERATIO,
            )
        return activer
    except cv2.error:
        return False


def construire_detecteur_presence(activer: bool):
    if not activer:
        return None

    try:
        import mediapipe as mp
    except ModuleNotFoundError:
        print(
            "[WARN] MediaPipe n'est pas installe. --use-mediapipe desactive.",
            file=sys.stderr,
        )
        return None

    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def sujet_present(trame_bgr: np.ndarray, detecteur_holistique) -> bool:
    if detecteur_holistique is None:
        return True
    trame_rgb = cv2.cvtColor(trame_bgr, cv2.COLOR_BGR2RGB)
    resultats = detecteur_holistique.process(trame_rgb)
    return any(
        [
            resultats.pose_landmarks is not None,
            resultats.face_landmarks is not None,
            resultats.left_hand_landmarks is not None,
            resultats.right_hand_landmarks is not None,
        ]
    )


def reconstruire_modele_depuis_archive_keras(tf, chemin_modele: Path):
    try:
        with zipfile.ZipFile(chemin_modele) as archive:
            configuration_modele = json.loads(archive.read("config.json"))
    except Exception as exc:
        raise ValueError(f"Lecture archive .keras impossible: {exc}") from exc

    if configuration_modele.get("class_name") != "Sequential":
        raise ValueError("Fallback uniquement supporte pour modele Sequential.")

    sequence = configuration_modele.get("config", {})
    configurations_couches = sequence.get("layers", [])
    if len(configurations_couches) < 2:
        raise ValueError("Configuration de couches invalide dans le modele.")

    configuration_entree = configurations_couches[0].get("config", {})
    batch_shape = configuration_entree.get("batch_shape")
    if not batch_shape or len(batch_shape) < 4:
        raise ValueError("Impossible de lire la forme d'entree depuis config.json.")

    forme_entree = tuple(int(dimension) for dimension in batch_shape[1:4])
    tenseur_entree = tf.keras.Input(
        shape=forme_entree,
        name=configuration_entree.get("name", "input"),
    )
    tenseur_sortie = tenseur_entree

    for configuration_couche in configurations_couches[1:]:
        couche = tf.keras.utils.deserialize_keras_object(configuration_couche)
        tenseur_sortie = couche(tenseur_sortie)
        if isinstance(tenseur_sortie, (list, tuple)):
            if len(tenseur_sortie) == 1:
                tenseur_sortie = tenseur_sortie[0]
            else:
                raise ValueError(
                    f"Layer {couche.name} renvoie {len(tenseur_sortie)} sorties; "
                    "reconstruction automatique impossible."
                )

    nom_modele = sequence.get("name", "modele_reconstruit")
    modele_reconstruit = tf.keras.Model(
        inputs=tenseur_entree,
        outputs=tenseur_sortie,
        name=f"{nom_modele}_recovered",
    )
    modele_reconstruit.load_weights(chemin_modele)
    return modele_reconstruit


def charger_modele_robuste(tf, chemin_modele: Path):
    try:
        return tf.keras.models.load_model(chemin_modele, compile=False)
    except Exception as exc:
        if chemin_modele.suffix.lower() != ".keras":
            raise

        print(
            "[WARN] Chargement direct du modele impossible "
            f"({type(exc).__name__}). Tentative de reconstruction .keras...",
            file=sys.stderr,
        )
        modele_reconstruit = reconstruire_modele_depuis_archive_keras(tf, chemin_modele)
        print("[INFO] Modele reconstruit avec succes depuis l'archive .keras.")
        return modele_reconstruit


def inferer_probabilites(modele, lot_images: np.ndarray) -> np.ndarray:
    sorties = modele(lot_images, training=False)
    probabilites = np.asarray(sorties, dtype=np.float32).squeeze()

    if probabilites.ndim != 1:
        probabilites = probabilites.reshape(-1)

    probabilites = np.maximum(probabilites, 0.0)
    somme = float(np.sum(probabilites))

    if somme > 0.0:
        return probabilites / somme

    if probabilites.size == 0:
        return np.array([1.0], dtype=np.float32)

    return np.full((probabilites.size,), 1.0 / probabilites.size, dtype=np.float32)


def lisser_probabilites(
    probabilites_precedentes: np.ndarray | None,
    probabilites_courantes: np.ndarray,
    alpha_lissage: float,
) -> np.ndarray:
    if probabilites_precedentes is None:
        return probabilites_courantes

    probabilites_lissees = (
        alpha_lissage * probabilites_precedentes
        + (1.0 - alpha_lissage) * probabilites_courantes
    )
    somme = float(np.sum(probabilites_lissees))
    if somme > 0.0:
        probabilites_lissees = probabilites_lissees / somme
    return probabilites_lissees


def calculer_confiance_max(probabilites: np.ndarray) -> tuple[int, float]:
    ordre = np.argsort(probabilites)[::-1]
    indice_1 = int(ordre[0])
    confiance_1 = float(probabilites[indice_1]) * 100.0
    return indice_1, confiance_1


def ouvrir_camera(index_camera: int):
    if sys.platform.startswith("win"):
        capture = cv2.VideoCapture(index_camera, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture

    return cv2.VideoCapture(index_camera)


def main() -> int:
    arguments = parser_arguments()

    try:
        valider_arguments(arguments)
    except ValueError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 1

    dossier_racine = Path(__file__).resolve().parent

    # Reduction des logs TensorFlow pour une sortie terminal propre.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print(
            "[ERREUR] TensorFlow n'est pas installe.\n"
            "Installe les dependances avec:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    try:
        chemin_modele = trouver_chemin_modele(arguments.model, dossier_racine)
    except FileNotFoundError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Chargement du modele: {chemin_modele}")
    try:
        modele = charger_modele_robuste(tf, chemin_modele)
    except Exception as exc:
        print(f"[ERREUR] Echec du chargement du modele: {exc}", file=sys.stderr)
        return 1

    forme_entree = getattr(modele, "input_shape", None)
    forme_sortie = getattr(modele, "output_shape", None)

    if not forme_entree or len(forme_entree) < 4:
        print("[ERREUR] Forme d'entree invalide pour ce modele.", file=sys.stderr)
        return 1

    hauteur_entree = int(forme_entree[1])
    largeur_entree = int(forme_entree[2])
    taille_image = (largeur_entree, hauteur_entree)

    if forme_sortie is None:
        print("[ERREUR] Forme de sortie invalide pour ce modele.", file=sys.stderr)
        return 1

    nombre_classes = int(forme_sortie[-1])
    chemin_etiquettes = Path(arguments.labels).expanduser()
    if not chemin_etiquettes.is_absolute():
        chemin_etiquettes = dossier_racine / chemin_etiquettes
    etiquettes = charger_etiquettes(chemin_etiquettes, nombre_classes)

    print(
        f"[INFO] Entree modele: {largeur_entree}x{hauteur_entree} | Classes: {nombre_classes} | "
        f"Camera index: {arguments.camera}"
    )
    print(
        "[INFO] Mode surete: "
        f"min_conf={arguments.min_confidence:.1f}% | "
        f"smooth_alpha={arguments.smooth_alpha:.2f} | "
        f"stable_frames={arguments.stable_frames} | "
        f"soft_margin={arguments.soft_margin:.1f}%"
    )
    print(
        "[INFO] Performance: "
        f"infer_every={arguments.infer_every} | "
        f"max_infer_fps={arguments.max_infer_fps:.1f} | "
        f"preprocess={arguments.preprocess} | "
        f"crop={arguments.crop_mode} | "
        f"retry_zoom={arguments.retry_zoom:.2f}"
    )

    if arguments.preprocess == "auto":
        mode_pretraitement_actif = mode_pretraitement_auto(modele)
        print(f"[INFO] Preprocess auto initial: {mode_pretraitement_actif}")
    else:
        mode_pretraitement_actif = arguments.preprocess

    # Prechauffage: supprime la latence du premier passage d'inference.
    lot_preechauffage = np.zeros((1, hauteur_entree, largeur_entree, 3), dtype=np.float32)
    _ = inferer_probabilites(modele, lot_preechauffage)

    detecteur_holistique = construire_detecteur_presence(arguments.use_mediapipe)

    capture = ouvrir_camera(arguments.camera)
    if not capture.isOpened():
        print(
            f"[ERREUR] Impossible d'ouvrir la webcam index {arguments.camera}.",
            file=sys.stderr,
        )
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, arguments.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, arguments.height)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FPS, 30)

    largeur_effective = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    hauteur_effective = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps_effectif = float(capture.get(cv2.CAP_PROP_FPS))
    if fps_effectif > 0.0:
        print(
            "[INFO] Camera effective: "
            f"{largeur_effective}x{hauteur_effective} @ {fps_effectif:.1f} FPS"
        )
    else:
        print(
            "[INFO] Camera effective: "
            f"{largeur_effective}x{hauteur_effective}"
        )

    flux_capture: FluxWebcamAsynchrone | None = FluxWebcamAsynchrone(capture).demarrer()

    couleur_valide = (40, 255, 40)
    couleur_attention = (0, 220, 255)
    couleur_alerte = (80, 80, 255)
    nom_fenetre = "Classification Animale - Webcam"

    cv2.namedWindow(nom_fenetre, cv2.WINDOW_NORMAL)
    if not arguments.fullscreen:
        cv2.resizeWindow(nom_fenetre, arguments.width, arguments.height)
    plein_ecran_actif = regler_plein_ecran(nom_fenetre, arguments.fullscreen)
    if arguments.fullscreen and not plein_ecran_actif:
        print("[WARN] Plein ecran indisponible avec ce backend OpenCV.", file=sys.stderr)

    moteur_inference: MoteurInferenceAsynchrone | None = MoteurInferenceAsynchrone(
        modele=modele,
        etiquettes=etiquettes,
        taille_image=taille_image,
        arguments=arguments,
        detecteur_holistique=detecteur_holistique,
        couleur_valide=couleur_valide,
        couleur_attention=couleur_attention,
        couleur_alerte=couleur_alerte,
        mode_pretraitement_actif=mode_pretraitement_actif,
    ).demarrer()

    frames_depuis_inference = arguments.infer_every
    dernier_temps_inference = 0.0
    intervalle_min_inference = (
        (1.0 / arguments.max_infer_fps) if arguments.max_infer_fps > 0.0 else 0.0
    )

    fps_lisse = 0.0
    temps_precedent = perf_counter()

    try:
        while True:
            frame_lue, trame = flux_capture.lire_dernier() if flux_capture else (False, None)
            if not frame_lue:
                if flux_capture and flux_capture.en_erreur:
                    print("[ERREUR] Lecture webcam impossible.", file=sys.stderr)
                    break
                sleep(0.002)
                continue

            temps_actuel = perf_counter()
            delta_t = temps_actuel - temps_precedent
            temps_precedent = temps_actuel
            fps_instantane = (1.0 / delta_t) if delta_t > 0.0 else 0.0
            if fps_lisse <= 0.0:
                fps_lisse = fps_instantane
            else:
                fps_lisse = 0.85 * fps_lisse + 0.15 * fps_instantane

            frames_depuis_inference += 1
            temps_depuis_inference = temps_actuel - dernier_temps_inference
            soumettre_maintenant = (
                frames_depuis_inference >= arguments.infer_every
                and temps_depuis_inference >= intervalle_min_inference
            )
            if soumettre_maintenant and moteur_inference is not None:
                moteur_inference.soumettre_trame(trame.copy())
                frames_depuis_inference = 0
                dernier_temps_inference = temps_actuel

            if moteur_inference is not None:
                ligne_1, ligne_2, couleur_texte, mode_pretraitement_actif = (
                    moteur_inference.lire_etat()
                )
            else:
                ligne_1 = "Animal: Initialisation..."
                ligne_2 = "Preparation de l'inference"
                couleur_texte = couleur_attention

            afficher_overlay_classique(
                trame=trame,
                ligne_1=ligne_1,
                ligne_2=ligne_2,
                couleur_ligne_2=couleur_texte,
            )

            afficher_pied(
                trame,
                "Q: Quitter | "
                f"F: Plein ecran {'ON' if plein_ecran_actif else 'OFF'} | "
                f"FPS: {fps_lisse:.1f} | PRE: {mode_pretraitement_actif}",
            )

            cv2.imshow(nom_fenetre, trame)
            touche = cv2.waitKey(1) & 0xFF
            if touche in (ord("q"), ord("Q")):
                break
            if touche in (ord("f"), ord("F")):
                plein_ecran_actif = regler_plein_ecran(nom_fenetre, not plein_ecran_actif)

            if cv2.getWindowProperty(nom_fenetre, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if moteur_inference is not None:
            moteur_inference.arreter()
        if flux_capture is not None:
            flux_capture.arreter()
        capture.release()
        cv2.destroyAllWindows()
        if detecteur_holistique is not None:
            detecteur_holistique.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
