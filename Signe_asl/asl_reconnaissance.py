import os
import time
import json
import queue
import threading
import subprocess
import shutil
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

TITRE_APP = "ASL - Reconnaissance temps reel"

CAMERA_INDEX = 0
CAMERA_LARGEUR = 640
CAMERA_HAUTEUR = 480
CAMERA_FPS = 30
CAMERA_BUFFER = 1
MIROIR_IMAGE = True
NORMALISER_MAIN_GAUCHE = True
TAILLE_ENTREE_PAR_DEFAUT = (224, 224)
INFER_EVERY_N_FRAMES = 1
DRAW_LANDMARKS = True
DRAW_RECT = True
WINDOW_RESIZABLE = False

SEUIL_CONFIANCE = 0.70
SEUIL_CONFIANCE_HAUT = 0.85
FENETRE_LISSAGE = 15
PADDING_MAIN = 35
MARGE_TOP2 = 0.10
MIN_TAILLE_ROI = 60
MIN_OCCURRENCES = max(3, int(FENETRE_LISSAGE * 0.35))
MODEL_COMPLEXITY = 0
PREPROCESS_DEFAULT = "auto"
MODE_VERITE = False
DELTA_SCORE = 0.06
CALIB_SAMPLES = 40
CALIB_MESSAGE_SEC = 2.0

DELAI_REPETITION = 1.8
VITESSE_VOIX = 140
LANGUE_VOIX = "fr"

COULEUR_VERT = (50, 220, 100)
COULEUR_ROUGE = (0, 0, 255)
COULEUR_JAUNE = (0, 220, 255)
COULEUR_BANDEAU = (35, 35, 35)
COULEUR_PANNEAU = (20, 20, 20)
COULEUR_TEXTE = (240, 240, 240)

POLICE = cv2.FONT_HERSHEY_SIMPLEX


def trouver_fichier_par_defaut(noms, env_var):
    valeur_env = os.getenv(env_var)
    if valeur_env:
        chemin = Path(valeur_env).expanduser()
        if chemin.is_file():
            return chemin

    base = Path(__file__).resolve().parent
    for nom in noms:
        chemin = base / nom
        if chemin.is_file():
            return chemin

    for chemin in base.glob("*.h5"):
        return chemin

    return None


def charger_classes(chemin_classes):
    with open(chemin_classes, "r", encoding="utf-8") as fichier:
        data = json.load(fichier)

    if isinstance(data, dict):
        classes = data.get("classes")
        if not classes:
            raise ValueError("Le JSON ne contient pas la cle 'classes'.")
        return classes

    if isinstance(data, list):
        return data

    raise ValueError("Format class_names.json invalide.")


def lire_version_keras_h5(chemin_modele):
    try:
        import h5py

        with h5py.File(chemin_modele, "r") as fichier:
            version = fichier.attrs.get("keras_version")
    except Exception:  # noqa: BLE001
        return None

    if isinstance(version, bytes):
        version = version.decode("utf-8")
    return version


def charger_modele_keras(keras_module, chemin_modele):
    try:
        return keras_module.models.load_model(chemin_modele, compile=False, safe_mode=False)
    except TypeError:
        return keras_module.models.load_model(chemin_modele, compile=False)


def charger_modele(chemin_modele):
    erreurs = []
    version_keras = lire_version_keras_h5(chemin_modele)

    if version_keras:
        print(f"Version Keras du modele: {version_keras}")

    if version_keras and version_keras.startswith("3"):
        try:
            import keras

            try:
                modele = charger_modele_keras(keras, chemin_modele)
                return modele, f"keras {keras.__version__}"
            except Exception as exc:  # noqa: BLE001
                erreurs.append(f"keras: {exc}")
        except ModuleNotFoundError:
            erreurs.append("keras non installe")

    if not version_keras or version_keras.startswith("2"):
        try:
            import tf_keras as keras

            try:
                modele = keras.models.load_model(chemin_modele, compile=False)
                return modele, "tf_keras"
            except Exception as exc:  # noqa: BLE001
                erreurs.append(f"tf_keras: {exc}")
        except ModuleNotFoundError:
            erreurs.append("tf_keras non installe")

    try:
        import tensorflow as tf
        try:
            modele = charger_modele_keras(tf.keras, chemin_modele)
            return modele, "tf.keras"
        except Exception as exc:  # noqa: BLE001
            erreurs.append(f"tf.keras: {exc}")
    except ModuleNotFoundError as exc:
        erreurs.append(f"tensorflow non installe: {exc}")

    message = "Impossible de charger le modele .h5. "
    if version_keras:
        message += f"(modele Keras {version_keras}). "
    message += " / ".join(erreurs)
    raise RuntimeError(message)


def obtenir_taille_entree(modele):
    taille = TAILLE_ENTREE_PAR_DEFAUT
    try:
        shape = modele.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) >= 3 and shape[1] and shape[2]:
            taille = (int(shape[2]), int(shape[1]))
    except Exception:  # noqa: BLE001
        pass
    return taille


def detecter_preprocess_mode(modele):
    try:
        for layer in modele.layers:
            nom = layer.__class__.__name__.lower()
            lname = getattr(layer, "name", "").lower()
            if "rescal" in nom or "rescal" in lname:
                return "raw"
    except Exception:  # noqa: BLE001
        pass
    return "mobilenet_v2"


def pretraiter_roi(roi, taille_entree, mode):
    image = cv2.resize(roi, taille_entree)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32")
    if mode == "raw":
        image = image
    elif mode == "mobilenet_v2":
        image = (image / 127.5) - 1.0
    else:
        image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def score_prediction(preds):
    if preds.size == 0:
        return 0.0
    conf = float(np.max(preds))
    if preds.size >= 2:
        top2 = np.partition(preds, -2)[-2:]
        marge = float(np.max(top2) - np.min(top2))
    else:
        marge = conf
    return conf + 0.5 * marge


def analyser_predictions(preds):
    if preds.ndim == 2:
        preds = preds[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    if preds.size >= 2:
        top2 = np.partition(preds, -2)[-2:]
        marge = float(np.max(top2) - np.min(top2))
    else:
        marge = conf
    return preds, idx, conf, marge


def pad_carre(image, taille_cible):
    h, w = image.shape[:2]
    taille = max(taille_cible, h, w)
    carre = np.zeros((taille, taille, 3), dtype=image.dtype)
    y_offset = (taille - h) // 2
    x_offset = (taille - w) // 2
    carre[y_offset : y_offset + h, x_offset : x_offset + w] = image
    return carre


def extraire_roi_carre(frame, x_min, y_min, x_max, y_max, largeur, hauteur):
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None, None

    taille = int(max(w, h) + 2 * PADDING_MAIN)
    if taille <= 0:
        return None, None

    centre_x = (x_min + x_max) / 2
    centre_y = (y_min + y_max) / 2
    x1 = int(centre_x - taille / 2)
    y1 = int(centre_y - taille / 2)
    x2 = x1 + taille
    y2 = y1 + taille

    x1c = max(x1, 0)
    y1c = max(y1, 0)
    x2c = min(x2, largeur)
    y2c = min(y2, hauteur)

    rect = (x1c, y1c, x2c, y2c)
    if taille < MIN_TAILLE_ROI:
        return None, rect

    if x2c <= x1c or y2c <= y1c:
        return None, rect

    roi = frame[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        return None, rect

    roi = pad_carre(roi, taille)
    return roi, rect


def extraire_main_label(results, index, miroir):
    if not results.multi_handedness or len(results.multi_handedness) <= index:
        return None

    label = results.multi_handedness[index].classification[0].label
    if miroir:
        if label == "Left":
            return "Right"
        if label == "Right":
            return "Left"
    return label


def lisser_predictions(historique):
    if not historique:
        return None, 0.0, 0

    valides = [(idx, conf) for idx, conf in historique if idx >= 0]
    if not valides:
        return None, 0.0, 0

    indices = [idx for idx, _ in valides]
    idx_majoritaire, count = Counter(indices).most_common(1)[0]
    confs = [conf for idx, conf in valides if idx == idx_majoritaire]
    conf_moy = float(np.mean(confs)) if confs else 0.0
    return idx_majoritaire, conf_moy, count


def formater_label_affiche(label):
    if label == "space":
        return "SPC"
    if label == "del":
        return "DEL"
    return label


def formater_label_voix(label):
    if label == "space":
        return "espace"
    if label == "del":
        return "effacer"
    return label


def dessiner_texte_centre(image, texte, centre_x, centre_y, echelle, epaisseur, couleur):
    taille, _ = cv2.getTextSize(texte, POLICE, echelle, epaisseur)
    x = int(centre_x - taille[0] / 2)
    y = int(centre_y + taille[1] / 2)
    cv2.putText(image, texte, (x, y), POLICE, echelle, couleur, epaisseur, cv2.LINE_AA)


def dessiner_barre_confiance(panneau, x, y, largeur, hauteur, confiance, seuil_confiance, seuil_confiance_haut):
    if confiance < seuil_confiance:
        return

    if confiance >= seuil_confiance_haut:
        couleur = COULEUR_VERT
    elif confiance >= seuil_confiance:
        couleur = COULEUR_JAUNE
    else:
        couleur = COULEUR_ROUGE

    cv2.rectangle(panneau, (x, y), (x + largeur, y + hauteur), (60, 60, 60), -1)
    largeur_remplie = int(largeur * confiance)
    cv2.rectangle(panneau, (x, y), (x + largeur_remplie, y + hauteur), couleur, -1)
    texte = f"{int(confiance * 100)}%"
    cv2.putText(panneau, texte, (x + largeur + 10, y + hauteur - 2), POLICE, 0.6, COULEUR_TEXTE, 2)


class MoteurVocal:
    def __init__(self, langue, vitesse, delai_repetition):
        self.langue = langue
        self.vitesse = vitesse
        self.delai_repetition = delai_repetition
        self.queue = queue.Queue(maxsize=4)
        self.dernier_texte = None
        self.dernier_temps = 0.0
        self.actif = shutil.which("espeak") is not None

        if self.actif:
            self.thread = threading.Thread(target=self._boucle, daemon=True)
            self.thread.start()

    def _boucle(self):
        while True:
            texte = self.queue.get()
            if texte is None:
                break
            subprocess.run(
                ["espeak", "-v", self.langue, "-s", str(self.vitesse), texte],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def dire(self, texte):
        if not self.actif:
            return False

        maintenant = time.time()
        if texte == self.dernier_texte and (maintenant - self.dernier_temps) < self.delai_repetition:
            return False

        self.dernier_texte = texte
        self.dernier_temps = maintenant
        try:
            self.queue.put_nowait(texte)
            return True
        except queue.Full:
            return False

    def reinitialiser(self):
        self.dernier_texte = None
        self.dernier_temps = 0.0

    def arreter(self):
        if not self.actif:
            return
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass


class CameraStream:
    def __init__(self, index, largeur, hauteur, fps, buffer_size):
        self.cap = cv2.VideoCapture(index)
        self.ok = self.cap.isOpened()
        if not self.ok:
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, largeur)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hauteur)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        except Exception:  # noqa: BLE001
            pass

        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        if self.ok and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.ok:
            self.cap.release()


def main():
    chemin_modele = trouver_fichier_par_defaut(
        ["best_asl_model.h5", "best_asl_model_3_.h5"],
        "ASL_MODEL_PATH",
    )
    chemin_classes = trouver_fichier_par_defaut(["class_names.json"], "ASL_CLASSES_PATH")

    if not chemin_modele or not chemin_modele.is_file():
        print("Modele .h5 introuvable. Ajoutez best_asl_model.h5 dans le dossier.")
        return
    if not chemin_classes or not chemin_classes.is_file():
        print("class_names.json introuvable. Ajoutez le fichier dans le dossier.")
        return

    print(f"Modele: {chemin_modele}")
    print(f"Classes: {chemin_classes}")

    classes = charger_classes(chemin_classes)
    modele, source = charger_modele(str(chemin_modele))
    taille_entree = obtenir_taille_entree(modele)
    print(f"Chargement modele OK ({source}). Taille entree: {taille_entree}")

    seuil_confiance = SEUIL_CONFIANCE
    seuil_confiance_haut = SEUIL_CONFIANCE_HAUT
    marge_top2 = MARGE_TOP2
    min_occurrences = MIN_OCCURRENCES
    delta_score = DELTA_SCORE

    mode_pretraitement = PREPROCESS_DEFAULT.lower()
    if mode_pretraitement == "auto":
        mode_pretraitement = detecter_preprocess_mode(modele)
        print(f"Pretraitement auto: {mode_pretraitement}")
    elif mode_pretraitement == "auto_frame":
        print("Pretraitement auto_frame actif")
    else:
        print(f"Pretraitement force: {mode_pretraitement}")

    mode_utilise = mode_pretraitement
    calib_active = False
    calib_conf = deque(maxlen=CALIB_SAMPLES)
    calib_marge = deque(maxlen=CALIB_SAMPLES)
    calib_message_time = 0.0

    try:
        sortie = modele.output_shape
        if isinstance(sortie, list):
            sortie = sortie[0]
        nb_classes_modele = int(sortie[-1])
    except Exception:  # noqa: BLE001
        nb_classes_modele = None

    if nb_classes_modele and nb_classes_modele != len(classes):
        print(
            "Attention: nombre de classes different entre le modele "
            f"({nb_classes_modele}) et class_names.json ({len(classes)})."
        )

    moteur_vocal = MoteurVocal(LANGUE_VOIX, VITESSE_VOIX, DELAI_REPETITION)
    if not moteur_vocal.actif:
        print("espeak non detecte, synthese vocale desactivee.")

    cv2.setUseOptimized(True)
    camera = CameraStream(CAMERA_INDEX, CAMERA_LARGEUR, CAMERA_HAUTEUR, CAMERA_FPS, CAMERA_BUFFER)
    if not camera.ok:
        print("Webcam inaccessible.")
        return

    if WINDOW_RESIZABLE:
        cv2.namedWindow(TITRE_APP, cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow(TITRE_APP, cv2.WINDOW_AUTOSIZE)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    historique = deque(maxlen=FENETRE_LISSAGE)
    fps = 0.0
    dernier_temps = time.time()
    frame_index = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.60,
    ) as hands:
        while True:
            ret, frame = camera.read()
            if not ret:
                continue

            frame_index += 1
            do_infer = (frame_index % INFER_EVERY_N_FRAMES) == 0

            if MIROIR_IMAGE:
                frame = cv2.flip(frame, 1)
            hauteur, largeur = frame.shape[:2]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            mode_utilise = mode_pretraitement

            main_detectee = False
            roi = None

            main_label = None
            if results.multi_hand_landmarks:
                main_detectee = True
                landmarks = results.multi_hand_landmarks[0]
                main_label = extraire_main_label(results, 0, MIROIR_IMAGE)
                xs = [lm.x for lm in landmarks.landmark]
                ys = [lm.y for lm in landmarks.landmark]
                x_min = int(min(xs) * largeur)
                x_max = int(max(xs) * largeur)
                y_min = int(min(ys) * hauteur)
                y_max = int(max(ys) * hauteur)

                roi, rect = extraire_roi_carre(frame, x_min, y_min, x_max, y_max, largeur, hauteur)
                if rect and DRAW_RECT:
                    x1, y1, x2, y2 = rect
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COULEUR_VERT, 2)

                if NORMALISER_MAIN_GAUCHE and main_label == "Left" and roi is not None:
                    roi = cv2.flip(roi, 1)

                if DRAW_LANDMARKS:
                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            if main_detectee and roi is not None and roi.size > 0 and do_infer:
                if mode_pretraitement == "auto_frame":
                    modes = []
                    for mode in ("raw", "rescale", "mobilenet_v2"):
                        entree = pretraiter_roi(roi, taille_entree, mode)
                        preds = modele.predict(entree, verbose=0)
                        preds, idx_m, conf_m, marge_m = analyser_predictions(preds)
                        score_m = score_prediction(preds)
                        modes.append((mode, idx_m, conf_m, marge_m, score_m))

                    modes = sorted(modes, key=lambda item: item[4], reverse=True)
                    best_mode, best_idx, best_conf, best_marge, best_score = modes[0]
                    second_score = modes[1][4] if len(modes) > 1 else 0.0

                    counts = Counter(idx for _, idx, _, _, _ in modes if idx >= 0)
                    idx_agree = -1
                    if counts:
                        idx_agree, count_agree = counts.most_common(1)[0]
                    else:
                        count_agree = 0

                    if count_agree >= 2:
                        candidats = [item for item in modes if item[1] == idx_agree]
                        candidat = max(candidats, key=lambda item: item[2])
                        idx, conf, marge = candidat[1], candidat[2], candidat[3]
                        mode_utilise = "mix"
                    elif MODE_VERITE:
                        score_diff = best_score - second_score
                        if score_diff >= delta_score and best_conf >= seuil_confiance_haut:
                            idx, conf, marge = best_idx, best_conf, best_marge
                            mode_utilise = best_mode
                        else:
                            idx, conf, marge = -1, 0.0, 0.0
                            mode_utilise = "indet"
                    else:
                        idx, conf, marge = best_idx, best_conf, best_marge
                        mode_utilise = best_mode

                    conf_calib = best_conf
                    marge_calib = best_marge
                else:
                    entree = pretraiter_roi(roi, taille_entree, mode_pretraitement)
                    preds = modele.predict(entree, verbose=0)
                    _, idx, conf, marge = analyser_predictions(preds)
                    conf_calib = conf
                    marge_calib = marge

                if calib_active and conf_calib > 0:
                    calib_conf.append(conf_calib)
                    calib_marge.append(marge_calib)
                    if len(calib_conf) >= CALIB_SAMPLES:
                        med_conf = float(np.median(calib_conf))
                        med_marge = float(np.median(calib_marge))
                        seuil_confiance = max(0.7, min(0.92, med_conf - 0.05))
                        seuil_confiance_haut = max(seuil_confiance + 0.05, min(0.97, med_conf + 0.03))
                        marge_top2 = max(0.08, min(0.25, med_marge * 0.85))
                        delta_score = max(0.05, min(0.12, med_marge * 0.5))
                        calib_active = False
                        calib_message_time = time.time()
                        print(
                            "Calibration OK -> "
                            f"conf={seuil_confiance:.2f}, conf_haut={seuil_confiance_haut:.2f}, "
                            f"marge={marge_top2:.2f}, delta={delta_score:.2f}"
                        )

                if idx >= 0 and (conf >= seuil_confiance_haut or marge >= marge_top2):
                    historique.append((idx, conf))
                else:
                    historique.append((-1, 0.0))
            else:
                if not main_detectee:
                    historique.clear()
                    moteur_vocal.reinitialiser()
                elif roi is None or roi.size == 0:
                    historique.append((-1, 0.0))

            idx_lisse, conf_lisse, count_lisse = lisser_predictions(historique)
            label_affiche = None
            if idx_lisse is not None:
                label = classes[idx_lisse]
                if label != "nothing":
                    if conf_lisse >= seuil_confiance and count_lisse >= min_occurrences:
                        label_affiche = label
                    elif conf_lisse >= seuil_confiance_haut and count_lisse >= max(3, min_occurrences // 2):
                        label_affiche = label

            largeur_panneau = 320
            panneau = np.zeros((hauteur, largeur_panneau, 3), dtype=np.uint8)
            panneau[:] = COULEUR_PANNEAU

            if label_affiche:
                texte_label = formater_label_affiche(label_affiche)
                cv2.putText(panneau, "Signe detecte", (20, 50), POLICE, 0.8, COULEUR_TEXTE, 2)
                dessiner_barre_confiance(
                    panneau,
                    20,
                    70,
                    180,
                    16,
                    conf_lisse,
                    seuil_confiance,
                    seuil_confiance_haut,
                )
                dessiner_texte_centre(
                    panneau,
                    texte_label,
                    largeur_panneau // 2,
                    hauteur // 2,
                    4.5,
                    10,
                    COULEUR_VERT,
                )

                label_voix = formater_label_voix(label_affiche)
                moteur_vocal.dire(label_voix)
            else:
                if main_detectee:
                    if calib_active:
                        cv2.putText(
                            panneau,
                            f"Calibration {len(calib_conf)}/{CALIB_SAMPLES}",
                            (20, 50),
                            POLICE,
                            0.7,
                            COULEUR_TEXTE,
                            2,
                        )
                    else:
                        cv2.putText(panneau, "Signe non confirme", (20, 50), POLICE, 0.7, COULEUR_TEXTE, 2)
                else:
                    cv2.putText(panneau, "Main absente...", (20, 50), POLICE, 0.7, COULEUR_TEXTE, 2)

            frame_finale = np.hstack([frame, panneau])

            largeur_totale = largeur + largeur_panneau
            cv2.rectangle(frame_finale, (0, 0), (largeur_totale, 40), COULEUR_BANDEAU, -1)
            cv2.putText(frame_finale, TITRE_APP, (10, 26), POLICE, 0.7, COULEUR_TEXTE, 2)
            if mode_pretraitement == "auto_frame":
                if mode_utilise in ("raw", "rescale", "mobilenet_v2", "mix", "indet"):
                    preproc_label = f"auto/{mode_utilise}"
                else:
                    preproc_label = "auto_frame"
            else:
                preproc_label = mode_pretraitement
            cv2.putText(
                frame_finale,
                f"PREPROC {preproc_label}",
                (220, 26),
                POLICE,
                0.6,
                COULEUR_TEXTE,
                2,
            )

            if calib_message_time and (time.time() - calib_message_time) < CALIB_MESSAGE_SEC:
                cv2.putText(
                    frame_finale,
                    "CALIB OK",
                    (420, 26),
                    POLICE,
                    0.6,
                    COULEUR_TEXTE,
                    2,
                )

            couleur_statut = COULEUR_VERT if main_detectee else COULEUR_ROUGE
            cv2.circle(frame_finale, (largeur_totale - 30, 20), 8, couleur_statut, -1)

            temps_actuel = time.time()
            delta = temps_actuel - dernier_temps
            if delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / delta)
            dernier_temps = temps_actuel
            cv2.putText(
                frame_finale,
                f"FPS {fps:.1f}",
                (largeur_totale - 140, 26),
                POLICE,
                0.6,
                COULEUR_TEXTE,
                2,
            )

            if main_label:
                cv2.putText(
                    frame_finale,
                    f"MAIN {main_label}",
                    (largeur_totale - 260, 26),
                    POLICE,
                    0.6,
                    COULEUR_TEXTE,
                    2,
                )

            cv2.putText(
                frame_finale,
                "Q / ESC : Quitter",
                (10, hauteur - 10),
                POLICE,
                0.6,
                COULEUR_TEXTE,
                2,
            )

            cv2.imshow(TITRE_APP, frame_finale)
            touche = cv2.waitKey(1) & 0xFF
            if touche in (ord("c"), ord("C")):
                calib_active = True
                calib_conf.clear()
                calib_marge.clear()
                calib_message_time = 0.0
                print("Calibration: gardez un signe stable pendant 2-3 secondes")
            if touche in (ord("q"), 27):
                break

    moteur_vocal.arreter()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
