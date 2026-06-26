import json
import bricknet.data

# Patch Windows obligatoire
bricknet.data._part_names = lambda: json.loads(
    (bricknet.data._DATA / "part_names.json").read_text(encoding="utf-8")
)

import bricknet
from bricknet.core import Tree, Part, StudEdge, StudSub, AxleEdge, AxleSub
from bricknet.graph import tree_to_graph

TEXT_TO_STEM = {
    "brick 2x4": "3001",
    "plate 2x2": "3021",
    "brick 1x1": "3005",
    "plate 1x1": "3024",
    "technic axle pin with friction": "43093",  # Ajout de la nouvelle brique générée
}


def parse_bricknet_text_to_tree(text):
    catalog = bricknet.load_catalog()
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    node_to_idx = {}
    parts_list = []
    edges_list = []

    # 1. Extraction des pièces
    part_lines = [l for l in lines if "|" in l]
    for idx, line in enumerate(part_lines):
        node_letter, rest = line.split(" ", 1)
        part_name, color_name = rest.split("|")

        stem = TEXT_TO_STEM.get(part_name.strip(), "3001")
        part_id = catalog.stem_to_id.get(stem)

        if part_id is None:
            raise ValueError(f"Pièce non reconnue dans le catalogue BrickNet: {stem}")

        # Mapping couleur (72 = Light Bluish Grey standard pour le Technic)
        color_id = 72 if "grey" in color_name.lower() else 16

        node_to_idx[node_letter] = idx
        parts_list.append(Part(part_id=part_id, color=color_id))

    # 2. Extraction dynamique des connexions
    edge_lines = [l for l in lines if "|" not in l]
    for line in edge_lines:
        tokens = line.split()
        if len(tokens) >= 2:
            parent_node = tokens[0]

            # Détection du domaine géométrique
            is_technic = "axle" in tokens or "pin" in tokens

            if is_technic:
                # Routage Technic (AxleSub / AxleEdge)
                p_sub_str = "pin" if "pin" in tokens[:3] else "axle"

                # Dans le domaine Technic, le réceptacle (socket) prend le même type que l'insert
                if "socket" in tokens or "pin" in tokens[3:]:
                    c_sub_str = "pin"
                else:
                    c_sub_str = "axle"

                edges_list.append(
                    AxleEdge(
                        parent=node_to_idx[parent_node],
                        child=len(parts_list) - 1,
                        parent_sub=getattr(AxleSub, p_sub_str, AxleSub.axle),
                        child_sub=getattr(
                            AxleSub, c_sub_str, AxleSub.axle
                        ),  # Le fallback fatal 'hole' est supprimé
                        parent_conn=0,
                        child_conn=0,
                        yaw=0,
                    )
                )
            else:
                # Routage System classique (StudSub / StudEdge)
                edges_list.append(
                    StudEdge(
                        parent=node_to_idx[parent_node],
                        child=len(parts_list) - 1,
                        parent_sub=getattr(
                            StudSub,
                            "stud" if "stud" in tokens else "hole",
                            StudSub.stud,
                        ),
                        child_sub=getattr(
                            StudSub,
                            "hole" if "hole" in tokens else "stud",
                            StudSub.hole,
                        ),
                        parent_conn=0,
                        child_conn=0,
                        yaw=0,
                    )
                )

    return Tree(parts=tuple(parts_list), edges=tuple(edges_list))


# --- Zone de Test ---
sequence_sample = """
a brick 2x4 | blue
b brick 2x4 | red
a stud b hole a
c brick 2x4 | yellow
b stud c hole a
"""

print("1. Parsing de l'arbre syntaxique...")
custom_tree = parse_bricknet_text_to_tree(sequence_sample)
print(f"Arbre reconstruit : {len(custom_tree.parts)} pièces.")

print("2. Génération de la Topologie (tree_to_graph)...")
topological_graph = tree_to_graph(custom_tree)

print("3. Résolution Cinématique et Export (graph_to_ldr)...")
# On utilise leur solveur géométrique natif
try:
    ldraw_text = bricknet.graph_to_ldr(topological_graph)
    with open("atlas_reconstructed.ldr", "w", encoding="utf-8") as f:
        f.write(ldraw_text)
    print("🎉 Exportation LDraw réussie !")
except AttributeError:
    # Si la version de la librairie exige un appel explicite à decode_graph d'abord
    realized_graph = bricknet.decode_graph(topological_graph)
    ldraw_text = bricknet.graph_to_ldr(realized_graph)
    with open("atlas_reconstructed.ldr", "w", encoding="utf-8") as f:
        f.write(ldraw_text)
    print("🎉 Exportation LDraw réussie (via decode_graph) !")
