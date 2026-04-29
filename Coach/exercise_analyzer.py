"""Analyse des exercices et comptage des repetitions avec machine a etats."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot, inf

import config
from angle_calculator import calculer_angle, distance_2d


@dataclass(slots=True)
class EtatExercice:
    """Etat interne pour suivre la progression d'un exercice."""

    repetitions_totales: int = 0
    repetitions_correctes: int = 0
    repetitions_invalides: int = 0
    etat_mouvement: str = "haut"
    cycle_actif: bool = False
    mouvement_valide: bool = True
    donnees_completes: bool = True
    frames_cycle: int = 0
    frames_statiques: int = 0
    frames_message_prioritaire: int = 0
    dernier_angle_principal: float | None = None
    angle_min_cycle: float = inf
    angle_max_cycle: float = 0.0
    alignement_min_cycle: float = inf
    deplacement_max_cycle: float = 0.0
    position_epaule_reference: tuple[float, float] | None = None
    dernier_message: str = config.MESSAGE_ATTENTE


class ExerciseAnalyzer:
    """Centralise les regles d'analyse des exercices."""

    def __init__(self) -> None:
        self.exercice_actuel = "squat"
        self.etats = {nom_exercice: EtatExercice() for nom_exercice in config.EXERCICES}

    def changer_exercice(self, nouvel_exercice: str) -> None:
        """Change l'exercice actif si la cle existe."""
        if nouvel_exercice in config.EXERCICES:
            self.exercice_actuel = nouvel_exercice
            etat = self.etats[nouvel_exercice]
            if etat.dernier_message == config.MESSAGE_ATTENTE:
                etat.dernier_message = f"Exercice actif: {config.EXERCICES[nouvel_exercice]['nom']}"

    @staticmethod
    def _taux_reussite(repetitions_totales: int, repetitions_correctes: int) -> float:
        if repetitions_totales <= 0:
            return 0.0
        return (repetitions_correctes / repetitions_totales) * 100.0

    @staticmethod
    def _angle_alignement_squat(
        point_epaule: tuple[float, float] | None,
        point_hanche: tuple[float, float] | None,
    ) -> float | None:
        """Retourne un score d'alignement du dos sur [0, 180] (plus haut = meilleur)."""
        if point_epaule is None or point_hanche is None:
            return None

        vecteur_x = point_epaule[0] - point_hanche[0]
        vecteur_y = point_epaule[1] - point_hanche[1]
        norme = hypot(vecteur_x, vecteur_y)
        if norme <= 1e-6:
            return None

        cosinus = max(-1.0, min(1.0, (-vecteur_y) / norme))
        inclinaison = degrees(acos(cosinus))
        return 180.0 - inclinaison

    @staticmethod
    def _deplacement_genou_normalise(
        point_genou: tuple[float, float] | None,
        point_cheville: tuple[float, float] | None,
    ) -> float:
        if point_genou is None or point_cheville is None:
            return 0.0

        longueur_tibia = distance_2d(point_genou, point_cheville)
        if longueur_tibia <= 1e-6:
            return 0.0
        return abs(point_genou[0] - point_cheville[0]) / longueur_tibia

    @staticmethod
    def _seuil_alignement_squat(regles: dict[str, object], angle_genou: float | None) -> float:
        """Calcule un seuil d'alignement dynamique selon la profondeur du squat."""
        seuil_haut = float(regles["angle_alignement_min_haut"])
        seuil_bas = float(regles["angle_alignement_min_bas"])
        if angle_genou is None:
            return seuil_haut

        angle_haut = float(regles["seuil_haut"])
        angle_bas = float(regles["angle_bas_valide"])
        denominateur = max(angle_haut - angle_bas, 1.0)
        progression = (angle_haut - angle_genou) / denominateur
        progression = max(0.0, min(1.0, progression))
        return seuil_haut - progression * (seuil_haut - seuil_bas)

    @staticmethod
    def _reset_cycle(etat: EtatExercice) -> None:
        etat.cycle_actif = False
        etat.mouvement_valide = True
        etat.donnees_completes = True
        etat.frames_cycle = 0
        etat.angle_min_cycle = inf
        etat.angle_max_cycle = 0.0
        etat.alignement_min_cycle = inf
        etat.deplacement_max_cycle = 0.0
        etat.position_epaule_reference = None

    @staticmethod
    def _demarrer_cycle(
        etat: EtatExercice,
        angle_principal: float | None,
        angle_alignement: float | None,
        deplacement: float,
        point_epaule: tuple[float, float] | None,
    ) -> None:
        etat.cycle_actif = True
        etat.mouvement_valide = True
        etat.donnees_completes = True
        etat.frames_cycle = 0
        etat.angle_min_cycle = angle_principal if angle_principal is not None else inf
        etat.angle_max_cycle = angle_principal if angle_principal is not None else 0.0
        etat.alignement_min_cycle = angle_alignement if angle_alignement is not None else inf
        etat.deplacement_max_cycle = deplacement
        etat.position_epaule_reference = point_epaule

        if angle_principal is None:
            etat.donnees_completes = False
        if angle_alignement is None:
            etat.donnees_completes = False

    @staticmethod
    def _construire_retour(
        etat: EtatExercice,
        est_correct: bool,
        message: str,
        angle_principal: float | None,
        niveau_feedback: str,
    ) -> dict[str, str | bool | int | float | None]:
        taux_reussite = ExerciseAnalyzer._taux_reussite(etat.repetitions_totales, etat.repetitions_correctes)
        return {
            "est_correct": est_correct,
            "niveau_feedback": niveau_feedback,
            "message": message,
            "angle_principal": angle_principal,
            "repetitions_totales": etat.repetitions_totales,
            "repetitions_correctes": etat.repetitions_correctes,
            "repetitions_invalides": etat.repetitions_invalides,
            "taux_reussite": taux_reussite,
        }

    @staticmethod
    def _mettre_a_jour_inactivite(etat: EtatExercice, angle_principal: float | None) -> None:
        if angle_principal is None:
            etat.frames_statiques = 0
            etat.dernier_angle_principal = None
            return

        if etat.dernier_angle_principal is None:
            etat.frames_statiques = 0
            etat.dernier_angle_principal = angle_principal
            return

        if abs(angle_principal - etat.dernier_angle_principal) < config.SEUIL_VARIATION_ANGLE:
            etat.frames_statiques += 1
        else:
            etat.frames_statiques = 0

        etat.dernier_angle_principal = angle_principal

    @staticmethod
    def _mettre_a_jour_cycle(
        etat: EtatExercice,
        angle_principal: float | None,
        angle_alignement: float | None,
        deplacement: float,
    ) -> None:
        if not etat.cycle_actif:
            return

        etat.frames_cycle += 1

        if angle_principal is None:
            etat.donnees_completes = False
        else:
            etat.angle_min_cycle = min(etat.angle_min_cycle, angle_principal)
            etat.angle_max_cycle = max(etat.angle_max_cycle, angle_principal)

        if angle_alignement is None:
            etat.donnees_completes = False
        else:
            etat.alignement_min_cycle = min(etat.alignement_min_cycle, angle_alignement)

        etat.deplacement_max_cycle = max(etat.deplacement_max_cycle, deplacement)

    def _valider_cycle(self, nom_exercice: str, etat: EtatExercice) -> tuple[bool, str]:
        """Valide la repetition selon les contraintes de l'exercice."""
        regles = config.EXERCICES[nom_exercice]

        if not etat.mouvement_valide:
            return False, etat.dernier_message

        if not etat.donnees_completes:
            return False, "Rep non comptee: points insuffisants."

        if etat.frames_cycle < config.MIN_FRAMES_PAR_REP:
            return False, "Mouvement trop rapide."

        amplitude_cycle = etat.angle_max_cycle - etat.angle_min_cycle
        if amplitude_cycle < regles["amplitude_min_cycle"]:
            return False, "Amplitude insuffisante."

        if nom_exercice == "squat":
            if etat.angle_min_cycle >= regles["angle_bas_valide"]:
                return False, "Descends plus bas."
            if etat.alignement_min_cycle < regles["angle_alignement_min_bas"]:
                return False, "Dos trop penche."
            if etat.deplacement_max_cycle > regles["deplacement_max"]:
                return False, "Genou trop en avant."
            return True, "Bonne profondeur !"

        if nom_exercice == "pushup":
            if etat.angle_min_cycle >= regles["angle_bas_valide"]:
                return False, "Descends plus bas."
            if etat.alignement_min_cycle < regles["angle_alignement_min"]:
                return False, "Corps pas aligne."
            return True, "Push-up valide !"

        # curl
        if etat.angle_min_cycle > regles["angle_bas_valide"]:
            return False, "Amplitude trop faible."
        if etat.angle_max_cycle < regles["angle_haut_valide"]:
            return False, "Tends davantage le bras."
        if etat.deplacement_max_cycle > regles["deplacement_max"]:
            return False, "Epaule qui bouge."
        return True, "Bicep curl valide !"

    def _analyser_squat(
        self,
        _etat: EtatExercice,
        points_corps: dict[str, tuple[float, float] | None],
    ) -> tuple[bool, str, float | None, float | None, float]:
        point_epaule = points_corps.get("epaule")
        point_hanche = points_corps.get("hanche")
        point_genou = points_corps.get("genou")
        point_cheville = points_corps.get("cheville")

        angle_genou = calculer_angle(point_hanche, point_genou, point_cheville)
        angle_dos = self._angle_alignement_squat(point_epaule, point_hanche)

        deplacement_genou = self._deplacement_genou_normalise(point_genou, point_cheville)

        if angle_genou is None:
            return False, "Genou non detecte.", None, angle_dos, deplacement_genou

        regles = config.EXERCICES["squat"]
        seuil_alignement = self._seuil_alignement_squat(regles, angle_genou)

        if angle_genou <= regles["seuil_bas"]:
            if angle_genou >= regles["angle_bas_valide"]:
                return False, "Descends plus bas.", angle_genou, angle_dos, deplacement_genou
            if angle_dos is not None and angle_dos < seuil_alignement:
                return False, "Dos trop penche.", angle_genou, angle_dos, deplacement_genou
            if deplacement_genou > regles["deplacement_max"]:
                return False, "Genou trop en avant.", angle_genou, angle_dos, deplacement_genou
            return True, "Bonne profondeur !", angle_genou, angle_dos, deplacement_genou

        if angle_dos is not None and angle_dos < regles["angle_alignement_min_bas"] - 5:
            return False, "Redresse le dos.", angle_genou, angle_dos, deplacement_genou

        return True, config.MESSAGE_OK, angle_genou, angle_dos, deplacement_genou

    def _analyser_pushup(
        self,
        _etat: EtatExercice,
        points_corps: dict[str, tuple[float, float] | None],
    ) -> tuple[bool, str, float | None, float | None, float]:
        point_epaule = points_corps.get("epaule")
        point_coude = points_corps.get("coude")
        point_poignet = points_corps.get("poignet")
        point_hanche = points_corps.get("hanche")
        point_cheville = points_corps.get("cheville")

        angle_coude = calculer_angle(point_epaule, point_coude, point_poignet)
        angle_corps = calculer_angle(point_epaule, point_hanche, point_cheville)

        if angle_coude is None:
            return False, "Coude non detecte.", None, angle_corps, 0.0

        regles = config.EXERCICES["pushup"]
        if angle_coude <= regles["seuil_bas"]:
            if angle_coude >= regles["angle_bas_valide"]:
                return False, "Descends plus bas.", angle_coude, angle_corps, 0.0
            if angle_corps is not None and angle_corps < regles["angle_alignement_min"]:
                return False, "Corps pas aligne.", angle_coude, angle_corps, 0.0
            return True, "Bonne descente !", angle_coude, angle_corps, 0.0

        if angle_corps is not None and angle_corps < regles["angle_alignement_min"]:
            return False, "Garde le corps aligne.", angle_coude, angle_corps, 0.0

        return True, config.MESSAGE_OK, angle_coude, angle_corps, 0.0

    def _analyser_curl(
        self,
        etat: EtatExercice,
        points_corps: dict[str, tuple[float, float] | None],
    ) -> tuple[bool, str, float | None, float | None, float]:
        point_epaule = points_corps.get("epaule")
        point_coude = points_corps.get("coude")
        point_poignet = points_corps.get("poignet")

        angle_coude = calculer_angle(point_epaule, point_coude, point_poignet)

        deplacement_epaule = 0.0
        if etat.position_epaule_reference and point_epaule:
            deplacement_epaule = distance_2d(point_epaule, etat.position_epaule_reference)

        if angle_coude is None:
            return False, "Bras non detecte.", None, None, deplacement_epaule

        regles = config.EXERCICES["curl"]
        if deplacement_epaule > regles["deplacement_max"]:
            return False, "Epaule qui bouge.", angle_coude, angle_coude, deplacement_epaule

        if angle_coude <= regles["seuil_bas"]:
            if angle_coude > regles["angle_bas_valide"]:
                return False, "Monte encore la main.", angle_coude, angle_coude, deplacement_epaule
            return True, "Bonne contraction !", angle_coude, angle_coude, deplacement_epaule

        return True, config.MESSAGE_OK, angle_coude, angle_coude, deplacement_epaule

    def analyser(
        self,
        points_corps: dict[str, tuple[float, float] | None],
        orientation: str,
    ) -> dict[str, str | bool | int | float | None]:
        """Analyse une frame et retourne le feedback associe."""
        etat = self.etats[self.exercice_actuel]
        regles = config.EXERCICES[self.exercice_actuel]

        if etat.frames_message_prioritaire > 0:
            etat.frames_message_prioritaire -= 1

        if orientation in {"inconnue", "incertaine"}:
            if etat.cycle_actif:
                etat.donnees_completes = False
            etat.frames_statiques = 0
            etat.dernier_angle_principal = None
            etat.dernier_message = config.MESSAGE_ORIENTATION
            return self._construire_retour(etat, False, etat.dernier_message, None, "neutre")

        if not points_corps:
            if etat.cycle_actif:
                etat.donnees_completes = False
            etat.frames_statiques = 0
            etat.dernier_angle_principal = None
            etat.dernier_message = config.MESSAGE_ATTENTE
            return self._construire_retour(etat, False, etat.dernier_message, None, "neutre")

        orientations_valides = regles.get("orientations_valides")
        if (
            config.BLOQUER_ORIENTATION_EXERCICE
            and orientations_valides
            and orientation not in orientations_valides
        ):
            if etat.cycle_actif:
                etat.donnees_completes = False
            etat.dernier_message = str(regles.get("message_orientation_exercice", config.MESSAGE_ORIENTATION))
            return self._construire_retour(etat, False, etat.dernier_message, None, "neutre")

        if self.exercice_actuel == "squat":
            est_correct, message, angle_principal, angle_alignement, deplacement = self._analyser_squat(
                etat, points_corps
            )
            point_epaule = points_corps.get("epaule")
        elif self.exercice_actuel == "pushup":
            est_correct, message, angle_principal, angle_alignement, deplacement = self._analyser_pushup(
                etat, points_corps
            )
            point_epaule = points_corps.get("epaule")
        else:
            est_correct, message, angle_principal, angle_alignement, deplacement = self._analyser_curl(
                etat, points_corps
            )
            point_epaule = points_corps.get("epaule")

        feedback_niveau = "ok" if est_correct else "erreur"

        if not est_correct:
            etat.dernier_message = message
            if config.INVALIDER_REP_SUR_PREMIERE_ERREUR and etat.cycle_actif:
                etat.mouvement_valide = False
            if angle_principal is None and etat.cycle_actif:
                etat.donnees_completes = False

        self._mettre_a_jour_inactivite(etat, angle_principal)

        if angle_principal is not None:
            if etat.etat_mouvement == "haut" and angle_principal <= regles["seuil_bas"]:
                etat.etat_mouvement = "bas"
                self._demarrer_cycle(etat, angle_principal, angle_alignement, deplacement, point_epaule)

            self._mettre_a_jour_cycle(etat, angle_principal, angle_alignement, deplacement)

            if etat.etat_mouvement == "bas" and angle_principal >= regles["seuil_haut"] and etat.cycle_actif:
                etat.etat_mouvement = "haut"
                etat.repetitions_totales += 1
                repetition_valide, message_repetition = self._valider_cycle(self.exercice_actuel, etat)
                if repetition_valide:
                    etat.repetitions_correctes += 1
                else:
                    etat.repetitions_invalides += 1
                    est_correct = False
                    feedback_niveau = "erreur"
                message = message_repetition
                etat.frames_message_prioritaire = config.FRAMES_RETENTION_FEEDBACK
                self._reset_cycle(etat)

        if (
            feedback_niveau == "ok"
            and etat.frames_message_prioritaire <= 0
            and not etat.cycle_actif
            and etat.etat_mouvement == "haut"
        ):
            feedback_niveau = "neutre"
            est_correct = False
            if etat.frames_statiques >= config.SEUIL_FRAMES_INACTIVITE:
                message = config.MESSAGE_INACTIVITE
            else:
                message = config.MESSAGE_PRET

        etat.dernier_message = message
        return self._construire_retour(etat, est_correct, message, angle_principal, feedback_niveau)

    def obtenir_stats_exercice_actuel(self) -> dict[str, str | int | float]:
        """Expose les statistiques de l'exercice en cours pour le rendu visuel."""
        etat = self.etats[self.exercice_actuel]
        return {
            "nom_exercice": config.EXERCICES[self.exercice_actuel]["nom"],
            "repetitions_totales": etat.repetitions_totales,
            "repetitions_correctes": etat.repetitions_correctes,
            "repetitions_invalides": etat.repetitions_invalides,
            "taux_reussite": self._taux_reussite(etat.repetitions_totales, etat.repetitions_correctes),
            "etat_mouvement": etat.etat_mouvement,
        }

    def obtenir_resume_seance(
        self,
        durees_par_exercice: dict[str, float],
        duree_totale: float,
        fps_moyen: float,
    ) -> list[dict[str, str | int | float]]:
        """Construit le resume de seance pour l'export CSV/JSON."""
        lignes = []
        for nom_exercice, etat in self.etats.items():
            duree_exercice = float(durees_par_exercice.get(nom_exercice, 0.0))
            if etat.repetitions_totales == 0 and duree_exercice <= 0:
                continue

            lignes.append(
                {
                    "exercice": config.EXERCICES[nom_exercice]["nom"],
                    "repetitions_totales": etat.repetitions_totales,
                    "repetitions_correctes": etat.repetitions_correctes,
                    "repetitions_invalides": etat.repetitions_invalides,
                    "pourcentage_correct": round(
                        self._taux_reussite(etat.repetitions_totales, etat.repetitions_correctes),
                        2,
                    ),
                    "duree_exercice_sec": round(duree_exercice, 2),
                    "duree_totale_sec": round(duree_totale, 2),
                    "fps_moyen": round(fps_moyen, 2),
                }
            )

        if not lignes:
            lignes.append(
                {
                    "exercice": config.EXERCICES[self.exercice_actuel]["nom"],
                    "repetitions_totales": 0,
                    "repetitions_correctes": 0,
                    "repetitions_invalides": 0,
                    "pourcentage_correct": 0.0,
                    "duree_exercice_sec": round(float(durees_par_exercice.get(self.exercice_actuel, 0.0)), 2),
                    "duree_totale_sec": round(duree_totale, 2),
                    "fps_moyen": round(fps_moyen, 2),
                }
            )

        return lignes
