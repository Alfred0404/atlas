import json

path_to_titan = r"C:\Users\derfl\Downloads\paths_pt.jsonl\paths_pt.jsonl"
output_jsonl = r"atlas_dev_dataset.jsonl"
lines_to_extract = 5000

print(f"Extraction et formatage de {lines_to_extract} lignes...")

with open(path_to_titan, "r", encoding="utf-8") as infile:
    with open(output_jsonl, "w", encoding="utf-8") as outfile:
        for i in range(lines_to_extract):
            line = infile.readline()
            if not line:
                break

            raw_data = json.loads(line)

            # On extrait la séquence textuelle
            target_text = raw_data.get("path", "").strip()

            # Formatage pour le Fine-Tuning Supervisé (SFT)
            # Puisqu'on est sur le dataset de pré-entraînement générique,
            # on simule un prompt d'instruction générique ou vide.
            formatted_example = {
                "instruction": "Generate a Lego assembly sequence.",
                "output": target_text,
            }

            outfile.write(json.dumps(formatted_example, ensure_ascii=False) + "\n")

print(f"Terminé ! Ton mini-dataset d'entraînement est prêt : {output_jsonl}")
