"""Gestion des exports de seance (CSV, JSON, video)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import config


class SessionLogger:
    """Cree et sauvegarde les livrables de fin de seance."""

    CHAMPS_CSV = [
        "exercice",
        "repetitions_totales",
        "repetitions_correctes",
        "repetitions_invalides",
        "pourcentage_correct",
        "duree_exercice_sec",
        "duree_totale_sec",
        "fps_moyen",
    ]

    def __init__(self, dossier_sortie: Path | None = None) -> None:
        self.dossier_sortie = dossier_sortie or config.DOSSIER_SORTIES
        self.dossier_sortie.mkdir(parents=True, exist_ok=True)
        self.horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")

    def chemin_video(self) -> Path:
        """Retourne le chemin du fichier video de demonstration."""
        return self.dossier_sortie / f"demo_seance_{self.horodatage}.mp4"

    @classmethod
    def _normaliser_lignes(cls, lignes_resume: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
        lignes_normalisees: list[dict[str, str | int | float]] = []
        for ligne in lignes_resume:
            ligne_normalisee = {
                "exercice": str(ligne.get("exercice", "-")),
                "repetitions_totales": int(ligne.get("repetitions_totales", 0)),
                "repetitions_correctes": int(ligne.get("repetitions_correctes", 0)),
                "repetitions_invalides": int(ligne.get("repetitions_invalides", 0)),
                "pourcentage_correct": float(ligne.get("pourcentage_correct", 0.0)),
                "duree_exercice_sec": float(ligne.get("duree_exercice_sec", 0.0)),
                "duree_totale_sec": float(ligne.get("duree_totale_sec", 0.0)),
                "fps_moyen": float(ligne.get("fps_moyen", 0.0)),
            }
            lignes_normalisees.append(ligne_normalisee)
        return lignes_normalisees

    @staticmethod
    def _calculer_resume_global(
        lignes_resume: list[dict[str, str | int | float]],
    ) -> dict[str, float | int]:
        repetitions_totales = sum(int(ligne.get("repetitions_totales", 0)) for ligne in lignes_resume)
        repetitions_correctes = sum(int(ligne.get("repetitions_correctes", 0)) for ligne in lignes_resume)
        repetitions_invalides = sum(int(ligne.get("repetitions_invalides", 0)) for ligne in lignes_resume)
        duree_totale_sec = max((float(ligne.get("duree_totale_sec", 0.0)) for ligne in lignes_resume), default=0.0)
        fps_moyen = max((float(ligne.get("fps_moyen", 0.0)) for ligne in lignes_resume), default=0.0)
        taux_global = (repetitions_correctes / repetitions_totales * 100.0) if repetitions_totales > 0 else 0.0
        return {
            "repetitions_totales": repetitions_totales,
            "repetitions_correctes": repetitions_correctes,
            "repetitions_invalides": repetitions_invalides,
            "taux_reussite_global": round(taux_global, 2),
            "duree_totale_sec": round(duree_totale_sec, 2),
            "fps_moyen": round(fps_moyen, 2),
        }

    def enregistrer_rapport_csv(self, lignes_resume: list[dict[str, str | int | float]]) -> Path:
        """Enregistre le rapport de seance au format CSV."""
        chemin_csv = self.dossier_sortie / f"rapport_seance_{self.horodatage}.csv"
        lignes_normalisees = self._normaliser_lignes(lignes_resume)
        with chemin_csv.open("w", newline="", encoding="utf-8") as fichier_csv:
            writer = csv.DictWriter(fichier_csv, fieldnames=self.CHAMPS_CSV)
            writer.writeheader()
            writer.writerows(lignes_normalisees)
        return chemin_csv

    def enregistrer_historique_json(self, lignes_resume: list[dict[str, str | int | float]]) -> Path:
        """Enregistre le resume en JSON pour conserver un historique."""
        chemin_json = self.dossier_sortie / f"historique_seance_{self.horodatage}.json"
        lignes_normalisees = self._normaliser_lignes(lignes_resume)
        contenu = {
            "horodatage": self.horodatage,
            "resume_global": self._calculer_resume_global(lignes_normalisees),
            "seance": lignes_normalisees,
        }
        chemin_json.write_text(json.dumps(contenu, ensure_ascii=False, indent=2), encoding="utf-8")
        return chemin_json
