# Configuration for the animal classifier app

# Update with your class names in the exact order used during training
CLASS_NAMES = [
'Chien',
'Cheval',
'Éléphant',
'Papillon',
'Poule',
'Chat',
'Vache',
'Mouton',
'Araignée',
'Écureuil']
# Update to your model input size (width, height)
IMAGE_SIZE = (224, 224)

# Preprocessing: "mobilenet_v2" or "rescale_0_1"
PREPROCESSING = "mobilenet_v2"

# Base model weights: "imagenet" or None
BASE_WEIGHTS = "imagenet"

# If the top score is below this threshold, the app will say it cannot identify
CONFIDENCE_THRESHOLD = 0.60
