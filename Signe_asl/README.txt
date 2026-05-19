ASL - Reconnaissance temps reel (Python 3.12.3)

1) Installer dependances systeme
   sudo apt-get update
   sudo apt-get install python3.12 python3.12-venv espeak

2) Creer et activer l'environnement virtuel
   python3.12 -m venv venv_asl
   source venv_asl/bin/activate

   Note: utilisez toujours l'environnement virtuel pour lancer le script.

3) Installer dependances Python
   pip install -r requirements.txt

4) Lancer
   python asl_reconnaissance.py

Calibration (stabilite)
- Dans l'appli, appuyez sur C et gardez un signe stable 2-3 secondes.
- Les seuils sont ajustes automatiquement.
