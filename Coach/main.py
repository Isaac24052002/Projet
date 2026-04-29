"""Point d'entree du projet Coach Sportif Temps Reel."""

from __future__ import annotations

import argparse
from collections import deque
import time

import cv2

import config
from exercise_analyzer import ExerciseAnalyzer
from feedback_renderer import dessiner_menu_exercices, dessiner_message_principal, dessiner_panneau_stats
from pose_detector import PoseDetector
from session_logger import SessionLogger


def parser_arguments() -> argparse.Namespace:
    """Definit les options de lancement."""
    parser = argparse.ArgumentParser(description="Coach sportif temps reel avec MediaPipe + OpenCV")
    parser.add_argument("--camera", type=int, default=0, help="Index de la camera (defaut: 0)")
    parser.add_argument("--largeur", type=int, default=config.LARGEUR_VIDEO, help="Largeur de capture video")
    parser.add_argument("--hauteur", type=int, default=config.HAUTEUR_VIDEO, help="Hauteur de capture video")
    parser.add_argument("--fps", type=int, default=config.FPS_CIBLE, help="FPS cible demandes a la webcam")
    parser.add_argument(
        "--exercice",
        choices=tuple(config.EXERCICES.keys()),
        default="squat",
        help="Exercice selectionne au demarrage",
    )
    parser.add_argument(
        "--profil",
        choices=tuple(config.PROFILS_DEMO.keys()),
        default=config.PROFIL_DEMO_PAR_DEFAUT,
        help="Profil de coaching: strict, equilibre ou tolerant",
    )
    parser.add_argument(
        "--sans-enregistrement",
        action="store_true",
        help="Desactive l'export video MP4 en fin de session",
    )
    return parser.parse_args()


def touche_vers_exercice(touche_clavier: int) -> str | None:
    """Convertit la touche clavier en nom d'exercice."""
    if touche_clavier == ord("1"):
        return "squat"
    if touche_clavier == ord("2"):
        return "pushup"
    if touche_clavier == ord("3"):
        return "curl"
    return None


def main() -> None:
    """Lance la boucle webcam, l'analyse de pose et les exports de fin de session."""
    arguments = parser_arguments()
    profil_actif = config.appliquer_profil_demo(arguments.profil)
    cv2.setUseOptimized(True)

    detecteur_pose = PoseDetector()
    analyseur_exercice = ExerciseAnalyzer()
    analyseur_exercice.changer_exercice(arguments.exercice)
    logger_seance = SessionLogger()

    capture_video = cv2.VideoCapture(arguments.camera)
    capture_video.set(cv2.CAP_PROP_FRAME_WIDTH, arguments.largeur)
    capture_video.set(cv2.CAP_PROP_FRAME_HEIGHT, arguments.hauteur)
    capture_video.set(cv2.CAP_PROP_FPS, arguments.fps)
    capture_video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture_video.isOpened():
        detecteur_pose.fermer()
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    largeur_video = int(capture_video.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.LARGEUR_VIDEO
    hauteur_video = int(capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.HAUTEUR_VIDEO
    fps_capture = float(capture_video.get(cv2.CAP_PROP_FPS))
    if fps_capture <= 0:
        fps_capture = float(arguments.fps)

    chemin_video = None
    enregistreur_video = None
    if not arguments.sans_enregistrement:
        chemin_video = logger_seance.chemin_video()
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        enregistreur_video = cv2.VideoWriter(str(chemin_video), codec, fps_capture, (largeur_video, hauteur_video))

        if not enregistreur_video.isOpened():
            enregistreur_video = None
            print("Avertissement: l'enregistrement video est desactive.")

    durees_par_exercice = {nom_exercice: 0.0 for nom_exercice in config.EXERCICES}

    instant_debut = time.perf_counter()
    instant_precedent = instant_debut
    instant_debut_exercice = instant_debut
    historique_fps: deque[float] = deque(maxlen=config.FENETRE_FPS)
    somme_fps = 0.0

    cv2.namedWindow(config.NOM_FENETRE, cv2.WINDOW_NORMAL)

    try:
        while True:
            lecture_ok, image = capture_video.read()
            if not lecture_ok:
                print("Lecture webcam interrompue.")
                break

            image = cv2.flip(image, 1)
            instant_actuel = time.perf_counter()

            delta_temps = max(instant_actuel - instant_precedent, 1e-6)
            fps_frame = 1.0 / delta_temps
            instant_precedent = instant_actuel

            if len(historique_fps) == historique_fps.maxlen:
                somme_fps -= historique_fps[0]
            historique_fps.append(fps_frame)
            somme_fps += fps_frame
            fps_moyen_courant = somme_fps / len(historique_fps)

            resultat_pose = detecteur_pose.detecter_pose(image)
            points_pose = detecteur_pose.extraire_landmarks(resultat_pose)
            orientation = detecteur_pose.detecter_orientation(points_pose)
            points_utiles = detecteur_pose.recuperer_points_utiles(points_pose, orientation)

            retour_analyse = analyseur_exercice.analyser(points_utiles, orientation)
            niveau_feedback = str(
                retour_analyse.get(
                    "niveau_feedback",
                    "ok" if bool(retour_analyse.get("est_correct", False)) else "erreur",
                )
            )

            if config.MODE_COULEUR_BINAIRE:
                couleur_squelette = config.COULEUR_VERT if bool(retour_analyse.get("est_correct", False)) else config.COULEUR_ROUGE
            else:
                if niveau_feedback == "ok":
                    couleur_squelette = config.COULEUR_VERT
                elif niveau_feedback == "neutre":
                    couleur_squelette = config.COULEUR_AMBRE
                else:
                    couleur_squelette = config.COULEUR_ROUGE

            detecteur_pose.dessiner_squelette(image, resultat_pose, couleur_squelette)

            stats_exercice = analyseur_exercice.obtenir_stats_exercice_actuel()
            donnees_panneau = {
                **stats_exercice,
                "orientation": orientation,
                "profil": profil_actif,
                "fps": fps_moyen_courant,
                "duree_sec": instant_actuel - instant_debut,
                "angle_principal": retour_analyse["angle_principal"],
            }

            dessiner_panneau_stats(image, donnees_panneau)
            dessiner_message_principal(
                image,
                str(retour_analyse["message"]),
                bool(retour_analyse["est_correct"]),
                niveau_feedback=niveau_feedback,
            )
            dessiner_menu_exercices(image, analyseur_exercice.exercice_actuel, profil_actif)

            if enregistreur_video is not None:
                enregistreur_video.write(image)

            cv2.imshow(config.NOM_FENETRE, image)
            touche = cv2.waitKey(1) & 0xFF

            if touche == ord("q"):
                break

            exercice_demande = touche_vers_exercice(touche)
            if exercice_demande and exercice_demande != analyseur_exercice.exercice_actuel:
                durees_par_exercice[analyseur_exercice.exercice_actuel] += instant_actuel - instant_debut_exercice
                analyseur_exercice.changer_exercice(exercice_demande)
                instant_debut_exercice = instant_actuel

    except KeyboardInterrupt:
        print("\nInterruption manuelle de la session.")

    finally:
        instant_fin = time.perf_counter()
        durees_par_exercice[analyseur_exercice.exercice_actuel] += instant_fin - instant_debut_exercice
        duree_totale = instant_fin - instant_debut

        fps_moyen_final = somme_fps / len(historique_fps) if historique_fps else 0.0

        resume_seance = analyseur_exercice.obtenir_resume_seance(
            durees_par_exercice=durees_par_exercice,
            duree_totale=duree_totale,
            fps_moyen=fps_moyen_final,
        )

        chemin_csv = logger_seance.enregistrer_rapport_csv(resume_seance)
        chemin_json = logger_seance.enregistrer_historique_json(resume_seance)

        capture_video.release()
        if enregistreur_video is not None:
            enregistreur_video.release()
        cv2.destroyAllWindows()
        detecteur_pose.fermer()

        print("Seance terminee.")
        print(f"Profil demo: {profil_actif}")
        print(f"Rapport CSV: {chemin_csv}")
        print(f"Historique JSON: {chemin_json}")
        if enregistreur_video is not None and chemin_video is not None:
            print(f"Video demo: {chemin_video}")


if __name__ == "__main__":
    main()
