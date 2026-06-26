import ssl
import urllib.request

# 1. On désactive globalement la vérification des certificats SSL pour urllib
ssl._create_default_https_context = ssl._create_unverified_context

# 2. On importe le point d'entrée de la commande de bricknet
from bricknet.__main__ import _cmd_fetch_meshes


# 3. On simule les arguments attendus par la fonction avec les bons attributs
class DummyArgs:
    dest = None  # Permet d'éviter l'AttributeError et d'utiliser le dossier par défaut


args = DummyArgs()

print("Lancement du téléchargement des meshes (SSL bypassé)...")
_cmd_fetch_meshes(args)
print("Téléchargement terminé avec succès !")
