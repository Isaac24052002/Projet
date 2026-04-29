"""Fonctions utilitaires pour calculer les angles articulaires."""

import numpy as np


def calculer_angle(
    point_a: tuple[float, float] | None,
    point_b: tuple[float, float] | None,
    point_c: tuple[float, float] | None,
) -> float | None:
    """Retourne l'angle en degres au point B a partir des points A, B et C.

    Si un point est manquant, la fonction renvoie None.
    """
    if point_a is None or point_b is None or point_c is None:
        return None

    try:
        radians = np.arctan2(point_c[1] - point_b[1], point_c[0] - point_b[0]) - np.arctan2(
            point_a[1] - point_b[1], point_a[0] - point_b[0]
        )
        angle = abs(np.degrees(radians))
        if angle > 180:
            angle = 360 - angle
        return float(angle)
    except Exception:
        # On protege le pipeline temps reel en cas de donnee inattendue.
        return None


def distance_2d(point_a: tuple[float, float] | None, point_b: tuple[float, float] | None) -> float:
    """Calcule la distance euclidienne 2D entre deux points."""
    if point_a is None or point_b is None:
        return 0.0
    return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))
