import json
import bricknet.data

# Patch Windows
bricknet.data._part_names = lambda: json.loads(
    (bricknet.data._DATA / "part_names.json").read_text(encoding="utf-8")
)

import bricknet

# Le test qui fonctionnait
content = """1 16 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
1 1 0 24 0 1 0 0 0 1 0 0 0 1 3001.dat"""

graph = bricknet.parse_ldr(content)
tree = bricknet.sample_tree(graph, 0)

print("--- Structure brute de l'objet Tree ---")
print(type(tree))
print("Attributs/Clés :", dir(tree) if not isinstance(tree, dict) else tree.keys())

# Si c'est un dictionnaire ou s'il a des propriétés, on l'affiche en JSON
try:
    print(json.dumps(tree, indent=2))
except Exception:
    # Si c'est un objet custom, on inspecte ses slots internes
    print(vars(tree))
