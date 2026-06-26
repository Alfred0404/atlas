import json
import bricknet.data

# Patch Windows
bricknet.data._part_names = lambda: json.loads(
    (bricknet.data._DATA / "part_names.json").read_text(encoding="utf-8")
)

import bricknet
from bricknet.core import StudSub
from bricknet.graph import _part_info

print("--- 1. Énumérateurs de ports (StudSub) ---")
print([e.name for e in StudSub])

catalog = bricknet.load_catalog()
part_id = catalog.stem_to_id.get("43093")

if part_id is not None:
    info = _part_info(part_id)
    print("\n--- 2. Ports physiques disponibles sur la pièce 43093 ---")
    # Affiche les 10 premières clés de connexion (ps, pp, conn_index)
    for key in list(info.flat_of.keys())[:10]:
        print(key)
else:
    print("Erreur : Pièce 43093 introuvable dans le catalogue.")
