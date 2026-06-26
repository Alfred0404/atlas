import json

path_to_titan = r"C:\Users\derfl\Downloads\paths_pt.jsonl\paths_pt.jsonl"

with open(path_to_titan, "r", encoding="utf-8") as f:
    data = json.loads(f.readline())

    print("--- Métadonnées de la ligne ---")
    print(f"Source : {data.get('source')}")
    print(f"Nombre total de nœuds (briques) : {data.get('nodes')}")
    print(f"Index de l'échantillon (Sample Index) : {data.get('sample_index')}")

    print("\n--- Contenu de la clé 'path' (Aperçu brut) ---")
    path_data = data.get("path")
    print(f"Type de data['path'] : {type(path_data)}")

    if isinstance(path_data, list):
        print(f"Longueur du chemin : {len(path_data)} étapes")
        print("Aperçu des 3 premières étapes :")
        for i, step in enumerate(path_data[:3]):
            print(f"Étape {i} : {step}")
    else:
        print(path_data)
