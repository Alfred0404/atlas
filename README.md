<!-- Ancre vers le haut -->

<br />
<div align="center">
  <a href="https://github.com/Alfred0404/atlas">
    <img src="public/logo_atlas.png" alt="Logo" width="300">
  </a>

  <h3 align="center">ATLAS</h3>

  <p align="center">
    Autoregressive Transformer Lego Assembly Synthesis
    <br />
    <a href="https://github.com/Alfred0404/atlas"><strong>Parcourir le repo »</strong></a>
  </p>
</div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

## À propos du projet

ATLAS implémente un pipeline ML complet : parsing de fichiers LDraw `.mpd`, tokenisation de briques en séquences, entraînement d'un transformer decoder-only, et génération de nouveaux modèles LEGO. Chaque brique est encodée en une séquence fixe de 6 tokens `[part, x, y, z, rotation, couleur]`, et le modèle apprend à prédire la prochaine brique de manière autorégressif.

<p align="center">
    <img src="public/generation_example.png" alt="Exemple de génération du modèle ATLAS actuel" style="border-radius:5px" width="500">
    <br />
    <em>Exemple de génération du modèle ATLAS actuel</em>
</p>

## Getting Started

Prérequis

- Python 3.10+
- pip

Cloner le dépôt :

```bash
git clone https://github.com/Alfred0404/atlas.git
cd atlas
```

Installer les dépendances :

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Pour l'entraînement GPU (CUDA 12.4) :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Architecture

Le pipeline transforme des fichiers LDraw bruts en séquences tokenisées, puis entraîne un transformer decoder-only pour la génération autorégressif.

### Structure du Projet

```
ATLAS/
├── atlas_config.json           # Vocabulaire et configuration spatiale
├── dataset/
│   ├── mpd_files/              # Fichiers .mpd / .ldr organisés par thème
│   │   ├── Town/               # Ex : Classic Town + sous-thèmes fusionnés
│   │   ├── Star Wars/
│   │   └── ...
│   ├── graph_sets/             # Graphes d'assemblage (.npz) par thème
│   │   └── Town/
│   └── technic_blacklist.txt   # Sets Technic/Bionicle exclus
├── tokenized_sets/             # Séquences tokenisées (.npy)
├── checkpoints/                # Checkpoints du modèle
│   └── graph/
│       └── Town/               # Checkpoints par thème
├── generated_sets/             # Sorties .mpd générées
├── src/
│   ├── config.py               # Constantes globales et chemins
│   ├── main.py                 # Point d'entrée traitement du dataset
│   ├── group_by_theme.py       # Classement des .mpd par thème via Rebrickable API
│   ├── visualize_graph.py      # Visualisation des graphes d'assemblage
│   ├── core/
│   │   ├── tokenizer.py        # Conversion brique → tokens
│   │   └── vocabulary.py       # Gestion du vocabulaire
│   ├── data/
│   │   ├── builder.py          # Pipeline de traitement end-to-end
│   │   ├── parser.py           # Parser MPD/LDraw et flattening
│   │   ├── adjacency.py        # Tri BFS par adjacence spatiale
│   │   ├── augmentation.py     # Augmentation (rotations, miroir, permutations)
│   │   └── sequence_dataset.py # Dataset PyTorch sur les .npy
│   ├── file_io/
│   │   ├── mpd_writer.py       # Export MPD
│   │   ├── scraper.py          # Téléchargement de datasets
│   │   └── omr_scraper.py      # Scraper OMR
│   ├── geometry/
│   │   ├── conn_parser.py      # Extraction studs/anti-studs depuis .dat
│   │   ├── col_parser.py       # Parser collision boxes (.col)
│   │   ├── port.py             # Dataclass Port (position, normale, type)
│   │   ├── lego_part.py        # LegoPart + PartDatabase (cache)
│   │   ├── snap.py             # Détection connexions stud/anti-stud
│   │   ├── spatial_hash.py     # Grille voxel pour collision O(1)
│   │   └── lego_core.py        # Moteur principal (graphe + validation)
│   ├── maths/
│   │   ├── rotations.py        # 24 rotations orthogonales discrètes
│   │   └── transforms.py       # Extraction positions/rotations
│   ├── model/
│   │   ├── config.py           # ModelConfig dataclass
│   │   ├── transformer.py      # ATLASTransformer (nn.Module)
│   │   ├── train.py            # Trainer (AdamW + cosine LR)
│   │   ├── generate.py         # Génération par sampling
│   │   ├── train_model.py      # Point d'entrée entraînement
│   │   └── generate_model.py   # Point d'entrée génération
│   └── utils/
│       └── logging.py          # Configuration du logging
└── tests/
    ├── test_adjacency.py
    ├── test_augmentation.py
    ├── test_integration.py
    ├── test_rotations.py
    ├── test_vocabulary.py
    └── test_geometry/
        ├── test_port.py
        ├── test_conn_parser.py
        ├── test_col_parser.py
        ├── test_snap.py
        ├── test_spatial_hash.py
        └── test_lego_core.py
```

### Pipeline de données

1. **Parsing** : `MPDParser` lit les fichiers `.mpd` et aplatit les sous-modèles hiérarchiques en placements de briques world-space
2. **Tri par adjacence** : `sort_bricks_by_adjacency` ordonne les briques via BFS closest-first (KDTree + distance de Chebyshev normalisée)
3. **Centrage** : recentrage sur l'origine avec snapping sur grille (10 LDU en X/Z, 8 LDU en Y)
4. **Vocabulaire** : `VocabularyManager` gère les mappings parts/couleurs/rotations, persistés dans `atlas_config.json`
5. **Tokenisation** : chaque brique $\rightarrow$ 6 tokens `[part_id, x_bin, y_bin, z_bin, rotation, color]`
6. **Augmentation** : rotations globales 90°/180°/270°, miroir X, permutations BFS $\rightarrow$ 8× multiplicateur

**Ranges de tokens** :

- **Spéciaux** : `PAD=0, SOS=1, EOS=2, UNK=3` $\Rightarrow$ indices 0 $\rightarrow$ 3
- **Rotations** : 24 orientations orthogonales $\Rightarrow$ indices 4 $\rightarrow$ 27
- **Positions** : 1000 bins par axe (précision 2 LDU, range ±1000) $\Rightarrow$ indices 28 $\rightarrow$ 3027
- **Couleurs** : à partir de 3028
- **Parts** : offset dynamique après les couleurs

### Modèle

**ATLASTransformer** — ~7M paramètres, decoder-only :

- **Embeddings** : token + position absolue + champ intra-brique (7 champs : SOS + 6 par brique)
- **Architecture** : 6 couches, 8 têtes d'attention, 256-dim, 1024-dim FFN, pre-norm
- **Masquage** : attention causale + masquage de logits par champ (contraint les sorties aux ranges de tokens valides)

**Entraînement** : AdamW, linear warmup $\rightarrow$ cosine annealing, gradient clipping à 1.0, masquage de logits activé pendant le training.

**Génération** : sampling temperature + top-k avec masquage de logits par champ. Export `.mpd` via `mpd_writer.py`.

### Moteur Géométrique (LegoCore)

Convertit les fichiers `.mpd` en graphes d'assemblage `G=(V, E)` où V = briques et E = connexions stud/anti-stud.

```
.mpd → MPDParser → list[RawBrickData] → LegoCore.from_raw_bricks() → G=(V, E)
```

- **ConnParser** parse récursivement les fichiers LDraw `.dat` pour extraire les positions des studs (male) et anti-studs (female), en déterminant la hauteur réelle de chaque pièce depuis la géométrie (pas depuis les références stud4.dat).
- **snap.py** matche les ports male/female entre paires de briques (KDTree, tolérance 2 LDU, alignement des normales).
- **LegoCore** maintient le graphe complet : placement, validation, suppression de briques, export.

Les fichiers `.dat` sont lus depuis la bibliothèque LDraw installée (`C:/Users/Public/Documents/LDraw/parts/`).

## Utilisation

### Classement par thème (première utilisation)

Les fichiers `.mpd` bruts doivent d'abord être classés par thème via l'API Rebrickable :

```bash
# Aperçu sans déplacer les fichiers
python src/group_by_theme.py --dry-run

# Classement effectif (--merge fusionne les sous-thèmes Town en un seul dossier)
python src/group_by_theme.py --merge

# Options
#   --workers N   Nombre de workers parallèles pour les appels API (défaut : 5)
#   --dry-run     Affiche les déplacements sans les exécuter
#   --merge       Applique les fusions définies dans MERGE_GROUPS
```

Les groupes de fusion sont configurables dans `src/group_by_theme.py` (`MERGE_GROUPS`). Par défaut, les sous-thèmes Town (Classic Town, Traffic, Police, Airport, Fire, Harbor, Gas Station, Town Jr.) sont fusionnés dans `Town/`.

### Entraînement par thème (Graph Transformer)

```bash
# 1. Construire le dataset de graphes pour un thème
python graph_build_dataset.py --theme "Town"
# → dataset/graph_sets/Town/*.npz
# → dataset/graph_vocab_Town.pt

# 2. Entraîner le modèle sur ce thème
python graph_train_model.py --theme "Town"
# → checkpoints/graph/Town/latest.pt, best.pt

# Options communes
#   --device cuda|cpu   Forcer le device
#   --epochs N          Nombre d'époques
#   --no-resume         Ignorer le checkpoint existant
#   --live-plot         Affichage temps réel de la loss
```

### Flux séquentiel (encoder-decoder)

```bash
python src/main.py                   # Tokenisation du dataset
python -m src.model.train_model      # Entraînement
python -m src.model.generate_model   # Génération
```

Les fichiers `.mpd` générés sont sauvegardés dans `generated_sets/` et peuvent être ouverts dans n'importe quel viewer LDraw (Studio, LDView, etc.).

## Entraînement

### Choix du thème

L'entraînement sur le dataset complet (~2 200 sets, tous thèmes confondus) fait plafonner la loss à ~15 nats : le modèle doit simultanément apprendre des distributions quasi-indépendantes (Star Wars, City, Creator…) qui partagent très peu de pièces ou de patterns structurels. Entraîner sur un thème unique cohérent est donc préférable.

Le thème **Town** (Classic Town + sous-thèmes fusionnés) est le plus représenté avec ~370 fichiers et offre la meilleure cohérence structurelle (bâtiments, véhicules, infrastructure urbaine).

### Pipeline Graph Transformer

Le pipeline Graph Transformer utilise 3 passes :
1. **Pass 1** : scan du vocabulaire (parts + couleurs) sur tous les fichiers du thème
2. **Pass 1.5** : préchauffage du cache `PartDatabase` pour toutes les pièces connues
3. **Pass 2** : construction parallèle des graphes `.npz` (N workers, `--jobs`)

### Résultats Actuels

#### ✅ Pipeline fonctionnel

Le pipeline complet fonctionne de bout en bout : classement par thème, construction des graphes, entraînement et génération de fichiers `.mpd` valides.

#### ⚠️ Qualité de génération à améliorer

Les modèles générés ne sont pas encore réalistes — la qualité de génération dépend fortement du thème choisi et du nombre d'epochs.

**Pistes d'amélioration :**

1. Augmenter le dataset via scraping (OMR)
2. Affiner les hyperparamètres par thème
3. Génération conditionnée par le thème (token de thème en entrée)

## Contribuer

Les contributions sont bienvenues : signalez des bugs via des issues et proposez des pull requests pour les améliorations.

## Licence

Distribué sous la licence du projet. Voir `LICENSE.txt` pour les détails.

<p align="center">
	<a href="https://github.com/Alfred0404/atlas/LICENSE"><img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=License&message=MIT&logoColor=d9e0ee&colorA=363a4f&colorB=b7bdf8"/></a>
</p>

## Ressources

| Ressource | Description |
| :-------- | :---------- |
| [LDraw Parts Library](https://library.ldraw.org/parts/list) | Bibliothèque officielle des pièces LDraw |
| [LDraw File Format Spec](https://www.ldraw.org/article/547.html) | Spécification du format de fichier LDraw |
| [Seymouria Sets](https://www.seymouria.pl/Download/official-lego-sets-ldr.php) | Sets LEGO officiels au format LDR |
| [BrickGPT](https://avalovelace1.github.io/BrickGPT/) | Projet similaire de génération LEGO par transformer |
| [PointGPT](https://github.com/CGuangyan-BIT/PointGPT) | Transformer autoregressif pour nuages de points |

<p align="center">
	<img src="https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/footers/gray0_ctp_on_line.svg?sanitize=true" />
</p>

<!-- LINKS & IMAGES -->
<!-- Contributors -->

[contributors-shield]: https://img.shields.io/github/contributors/Alfred0404/atlas.svg?style=for-the-badge
[contributors-url]: https://github.com/Alfred0404/atlas/graphs/contributors

<!-- Forks -->

[forks-shield]: https://img.shields.io/github/forks/Alfred0404/atlas.svg?style=for-the-badge
[forks-url]: https://github.com/Alfred0404/atlas/network/members

<!-- Stars -->

[stars-shield]: https://img.shields.io/github/stars/Alfred0404/atlas.svg?style=for-the-badge
[stars-url]: https://github.com/Alfred0404/atlas/stargazers

<!-- Issues -->

[issues-shield]: https://img.shields.io/github/issues/Alfred0404/atlas.svg?style=for-the-badge
[issues-url]: https://github.com/Alfred0404/atlas/issues

<!-- License -->

[license-shield]: https://img.shields.io/github/license/Alfred0404/atlas.svg?style=for-the-badge
[license-url]: https://github.com/Alfred0404/atlas/blob/master/LICENSE.txt

<!-- Linkedin -->

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/alfred-de-vulpian
