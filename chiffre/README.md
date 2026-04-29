# chiffre

## Aperçu
Reconnaissance de chiffres avec CNN.

## Fichier principal
- `chiffre.ipynb`

## Modèles utilisés
- CNN Keras (`Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `Sequential`)

## Résultats (sorties sauvegardées)
- `accuracy` test: environ `0.99` (support `10000`)
- `val_accuracy` observée: autour de `0.99` selon les epochs

## Notes
- Les pertes affichées semblent élevées sur certaines epochs; vérifier l'échelle/normalisation de la loss.
