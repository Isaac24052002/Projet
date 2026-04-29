import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from .config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    PREPROCESSING,
    BASE_WEIGHTS,
)

# TensorFlow import is intentionally inside a try to give a clear error message
try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - runtime dependency check
    tf = None
    TF_IMPORT_ERROR = exc
else:
    TF_IMPORT_ERROR = None

try:
    import h5py
except Exception:  # pragma: no cover - optional for fallback
    h5py = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "animal-10.keras"

app = FastAPI(title="Animal-10 Classifier")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

model = None


def _extract_weights() -> Path:
    weights_path = Path(tempfile.gettempdir()) / "model.weights.h5"
    if weights_path.exists() and weights_path.stat().st_mtime >= MODEL_PATH.stat().st_mtime:
        return weights_path

    with zipfile.ZipFile(MODEL_PATH, "r") as zf:
        zf.extract("model.weights.h5", path=weights_path.parent)
    return weights_path


def _infer_num_classes(weights_path: Path) -> int:
    if h5py is None:
        raise RuntimeError("h5py is required to infer num classes from weights.")
    with h5py.File(weights_path, "r") as f:
        # Keras 3 weights structure: /layers/dense_2/vars/0 is kernel (in_features, out_features)
        kernel = f["layers"]["dense_2"]["vars"]["0"]
        return int(kernel.shape[1])


def _load_top_weights_from_h5(model: "tf.keras.Model", weights_path: Path) -> None:
    with h5py.File(weights_path, "r") as f:
        layers_group = f["layers"]
        for layer_name in ("dense", "dense_1", "dense_2"):
            if layer_name not in layers_group:
                continue
            vars_group = layers_group[layer_name]["vars"]
            weights = []
            for key in sorted(vars_group.keys(), key=lambda x: int(x)):
                weights.append(vars_group[key][()])
            if weights:
                model.get_layer(layer_name).set_weights(weights)


def _rebuild_model(num_classes: int) -> "tf.keras.Model":
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
        weights=BASE_WEIGHTS,
        name="functional",
    )
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(base.output)
    x = tf.keras.layers.Dense(1024, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_2")(x)
    return tf.keras.Model(inputs=base.input, outputs=outputs, name="animal10_rebuilt")


def load_model_once() -> None:
    global model
    if model is not None:
        return
    if TF_IMPORT_ERROR is not None:
        raise RuntimeError(
            "TensorFlow is not installed. Install dependencies from requirements.txt"
        ) from TF_IMPORT_ERROR
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Place animal-10.keras at project root."
        )
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except ValueError as exc:
        # Fallback: rebuild known architecture and load weights from .keras zip
        if "expects 1 input(s), but it received 2 input tensors" not in str(exc):
            raise
        weights_path = _extract_weights()
        num_classes = _infer_num_classes(weights_path)
        model = _rebuild_model(num_classes)
        try:
            model.load_weights(weights_path)
        except Exception as load_exc:
            # The saved weights may only include the top classifier.
            # Manually load dense layer weights and continue with base imagenet weights.
            if h5py is None:
                raise RuntimeError(
                    "Weights loading failed and h5py is missing for manual load.\n"
                    f"Details: {load_exc}"
                ) from load_exc
            _load_top_weights_from_h5(model, weights_path)


def preprocess_image(contents: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    if PREPROCESSING == "mobilenet_v2":
        array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    else:
        array = array / 255.0
    array = np.expand_dims(array, axis=0)
    return array


def predict(image_batch: np.ndarray) -> dict:
    load_model_once()

    preds = model.predict(image_batch, verbose=0)
    preds = np.asarray(preds)

    # Handle common model output shapes
    if preds.ndim == 2 and preds.shape[0] == 1:
        scores = preds[0]
    else:
        scores = preds.reshape(-1)

    # If logits, convert to probabilities
    if scores.min() < 0 or scores.max() > 1:
        exp = np.exp(scores - np.max(scores))
        scores = exp / exp.sum()

    top_index = int(np.argmax(scores))
    top_score = float(scores[top_index])

    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "label": None,
            "message": "Je n'arrive pas a identifier cet animal.",
        }

    label = CLASS_NAMES[top_index] if top_index < len(CLASS_NAMES) else f"class_{top_index}"
    return {
        "label": label,
        "message": f"L'image que vous venez d'uploader est celle d'un {label}.",
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image_batch = preprocess_image(contents)
        result = predict(image_batch)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
