import json
import bricknet.data

# Patch Windows
bricknet.data._part_names = lambda: json.loads(
    (bricknet.data._DATA / "part_names.json").read_text(encoding="utf-8")
)

import bricknet


content_2 = """1 16 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
1 1 0 24 0 1 0 0 0 1 0 0 0 1 3001.dat"""

for i, content in enumerate([content_2], 1):
    try:
        graph = bricknet.parse_ldr(content)
        print(f"\n--- Test {i} ---")
        print(f"Edges détectées par BrickNet : {graph.edges}")

        tree = bricknet.sample_tree(graph, 0)
        print("Séquence sérialisée :")
        print(bricknet.serialize_tree(tree))
    except Exception as e:
        print(f"Erreur Test {i} : {e}")
