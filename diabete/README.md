# diabete

## Aperçu
Classification du risque diabète avec modèles ML classiques et deep learning.

## Fichier principal
- `diabete.ipynb`

## Modèles utilisés
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC`
- `XGBClassifier`
- `LGBMClassifier`
- Réseau de neurones (`Sequential`, `Dense`, `Dropout`)

## Résultats (sorties sauvegardées)
- Logs d'entraînement NN: `val_accuracy` observée jusqu'à environ `0.69`
- Les sorties visibles ne montrent pas clairement la métrique finale comparative de tous les modèles.

## Notes
- Ajouter un tableau final par modèle (`accuracy`, `recall`, `F1`, `ROC-AUC`) rendra la comparaison plus propre.
