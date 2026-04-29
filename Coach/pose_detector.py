"""Module de detection de pose base sur MediaPipe."""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2

import config


NOMS_POINTS_UTILS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

POINTS_VISIBILITE_COTE = {
    "left": (
        "left_shoulder",
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_elbow",
        "left_wrist",
    ),
    "right": (
        "right_shoulder",
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_elbow",
        "right_wrist",
    ),
}


def _charger_modules_mediapipe() -> tuple[Any, Any]:
    """Charge MediaPipe de facon compatible selon la version installee."""
    try:
        import mediapipe as module_mediapipe
    except ModuleNotFoundError as erreur:
        raise RuntimeError(
            "MediaPipe est introuvable. Installe les dependances avec: "
            "`pip install -r requirements.txt`."
        ) from erreur

    if hasattr(module_mediapipe, "solutions") and hasattr(module_mediapipe.solutions, "pose"):
        return module_mediapipe.solutions.pose, module_mediapipe.solutions.drawing_utils

    try:
        from mediapipe.python.solutions import drawing_utils as module_dessin
        from mediapipe.python.solutions import pose as module_pose

        return module_pose, module_dessin
    except Exception as erreur:
        version = getattr(module_mediapipe, "__version__", "inconnue")
        raise RuntimeError(
            "La version de MediaPipe installee ne fournit pas `solutions.pose` "
            f"(version detectee: {version}). "
            "Installe une version compatible avec: "
            "`pip uninstall -y mediapipe && pip install mediapipe==0.10.14`."
        ) from erreur


class PoseDetector:
    """Encapsule MediaPipe Pose pour simplifier son usage dans la boucle video."""

    def __init__(self) -> None:
        self._mp_pose, self._mp_dessin = _charger_modules_mediapipe()
        self._indices_utiles = {
            nom: self._mp_pose.PoseLandmark[nom.upper()].value for nom in NOMS_POINTS_UTILS
        }
        self._styles_dessin: dict[tuple[int, int, int], tuple[Any, Any]] = {}
        self._historique_orientation: deque[str] = deque(maxlen=config.FENETRE_ORIENTATION)
        self._modele = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=config.CONFIANCE_DETECTION,
            min_tracking_confidence=config.CONFIANCE_SUIVI,
        )

    def detecter_pose(self, image_bgr: Any) -> Any:
        """Retourne le resultat MediaPipe pour une frame BGR."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        resultat = self._modele.process(image_rgb)
        image_rgb.flags.writeable = True
        return resultat

    def extraire_landmarks(self, resultat_pose: Any) -> dict[str, dict[str, float]]:
        """Extrait les landmarks dans un dictionnaire simple et lisible."""
        if not resultat_pose or not resultat_pose.pose_landmarks:
            return {}

        points_pose: dict[str, dict[str, float]] = {}
        landmarks = resultat_pose.pose_landmarks.landmark
        for nom, indice in self._indices_utiles.items():
            landmark = landmarks[indice]
            points_pose[nom] = {
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "visibilite": float(landmark.visibility),
            }
        return points_pose

    @staticmethod
    def _visibilite_moyenne(points_pose: dict[str, dict[str, float]], cote: str) -> float:
        noms = POINTS_VISIBILITE_COTE[cote]
        valeurs = [points_pose[n]["visibilite"] for n in noms if n in points_pose]
        if not valeurs:
            return 0.0
        return float(sum(valeurs) / len(valeurs))

    def _orientation_majoritaire(self) -> str:
        if not self._historique_orientation:
            return "incertaine"

        compteur: dict[str, int] = {}
        for valeur in self._historique_orientation:
            compteur[valeur] = compteur.get(valeur, 0) + 1
        return max(compteur, key=compteur.get)

    def detecter_orientation(self, points_pose: dict[str, dict[str, float]]) -> str:
        """Detecte automatiquement si la personne est de face, profil gauche ou profil droit."""
        if not points_pose:
            self._historique_orientation.clear()
            return "inconnue"

        visibilite_gauche = self._visibilite_moyenne(points_pose, "left")
        visibilite_droite = self._visibilite_moyenne(points_pose, "right")

        orientation_inst = "incertaine"
        if visibilite_gauche > config.SEUIL_FACE and visibilite_droite > config.SEUIL_FACE:
            orientation_inst = "face"
        elif visibilite_gauche > config.SEUIL_PROFIL_HAUT and visibilite_droite < config.SEUIL_PROFIL_BAS:
            orientation_inst = "profil_gauche"
        elif visibilite_droite > config.SEUIL_PROFIL_HAUT and visibilite_gauche < config.SEUIL_PROFIL_BAS:
            orientation_inst = "profil_droit"

        if orientation_inst in {"face", "profil_gauche", "profil_droit"}:
            self._historique_orientation.append(orientation_inst)
            return self._orientation_majoritaire()

        if self._historique_orientation:
            return self._orientation_majoritaire()
        return "incertaine"

    @staticmethod
    def _moyenne_points(point_gauche: dict[str, float], point_droit: dict[str, float]) -> tuple[float, float] | None:
        if not point_gauche or not point_droit:
            return None
        return (
            (point_gauche["x"] + point_droit["x"]) / 2.0,
            (point_gauche["y"] + point_droit["y"]) / 2.0,
        )

    @staticmethod
    def _choisir_point_par_visibilite(
        point_gauche: dict[str, float] | None,
        point_droit: dict[str, float] | None,
    ) -> tuple[float, float] | None:
        point_gauche_valide = point_gauche and point_gauche["visibilite"] >= config.SEUIL_VISIBILITE_POINT
        point_droit_valide = point_droit and point_droit["visibilite"] >= config.SEUIL_VISIBILITE_POINT

        if point_gauche_valide and point_droit_valide:
            if point_gauche["visibilite"] >= point_droit["visibilite"]:
                return (point_gauche["x"], point_gauche["y"])
            return (point_droit["x"], point_droit["y"])
        if point_gauche_valide:
            return (point_gauche["x"], point_gauche["y"])
        if point_droit_valide:
            return (point_droit["x"], point_droit["y"])

        if point_gauche and point_droit:
            if point_gauche["visibilite"] >= point_droit["visibilite"]:
                return (point_gauche["x"], point_gauche["y"])
            return (point_droit["x"], point_droit["y"])
        if point_gauche:
            return (point_gauche["x"], point_gauche["y"])
        if point_droit:
            return (point_droit["x"], point_droit["y"])
        return None

    def _point_articulation(
        self,
        points_pose: dict[str, dict[str, float]],
        nom_simple: str,
        orientation: str,
    ) -> tuple[float, float] | None:
        point_gauche = points_pose.get(f"left_{nom_simple}")
        point_droit = points_pose.get(f"right_{nom_simple}")

        if orientation == "profil_gauche":
            if point_gauche and point_gauche["visibilite"] >= config.SEUIL_VISIBILITE_POINT:
                return (point_gauche["x"], point_gauche["y"])
            return self._choisir_point_par_visibilite(point_gauche, point_droit)

        if orientation == "profil_droit":
            if point_droit and point_droit["visibilite"] >= config.SEUIL_VISIBILITE_POINT:
                return (point_droit["x"], point_droit["y"])
            return self._choisir_point_par_visibilite(point_gauche, point_droit)

        if orientation == "face":
            if point_gauche and point_droit:
                return self._moyenne_points(point_gauche, point_droit)
            return self._choisir_point_par_visibilite(point_gauche, point_droit)

        return self._choisir_point_par_visibilite(point_gauche, point_droit)

    def _styles_pour_couleur(self, couleur: tuple[int, int, int]) -> tuple[Any, Any]:
        styles = self._styles_dessin.get(couleur)
        if styles is not None:
            return styles

        style_landmark = self._mp_dessin.DrawingSpec(color=couleur, thickness=2, circle_radius=2)
        style_connexion = self._mp_dessin.DrawingSpec(color=couleur, thickness=2)
        self._styles_dessin[couleur] = (style_landmark, style_connexion)
        return style_landmark, style_connexion

    def recuperer_points_utiles(
        self,
        points_pose: dict[str, dict[str, float]],
        orientation: str,
    ) -> dict[str, tuple[float, float] | None]:
        """Retourne les articulations utiles en fonction de l'orientation detectee."""
        if not points_pose:
            return {}

        return {
            "epaule": self._point_articulation(points_pose, "shoulder", orientation),
            "coude": self._point_articulation(points_pose, "elbow", orientation),
            "poignet": self._point_articulation(points_pose, "wrist", orientation),
            "hanche": self._point_articulation(points_pose, "hip", orientation),
            "genou": self._point_articulation(points_pose, "knee", orientation),
            "cheville": self._point_articulation(points_pose, "ankle", orientation),
        }

    def dessiner_squelette(self, image_bgr: Any, resultat_pose: Any, couleur: tuple[int, int, int]) -> None:
        """Dessine le squelette avec la couleur de feedback."""
        if not resultat_pose or not resultat_pose.pose_landmarks:
            return

        style_landmark, style_connexion = self._styles_pour_couleur(couleur)

        self._mp_dessin.draw_landmarks(
            image_bgr,
            resultat_pose.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=style_landmark,
            connection_drawing_spec=style_connexion,
        )

    def fermer(self) -> None:
        """Libere proprement les ressources MediaPipe."""
        self._modele.close()
