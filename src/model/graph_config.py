from dataclasses import dataclass


@dataclass
class GraphModelConfig:
    # ---- Vocabulary sizes (set from dataset before instantiating model) ----
    n_parts:  int = 501    # number of distinct part IDs (500 top parts + 1 for UNK at index 0)
    n_colors: int = 128    # number of distinct colors

    # ---- Architecture ------------------------------------------------------
    d_model:       int   = 512
    n_heads:       int   = 8
    n_layers:      int   = 8
    d_ff:          int   = 2048
    dropout:       float = 0.1

    # ---- Token/port limits -------------------------------------------------
    max_open_ports: int = 256   # must match graph_dataset.MAX_OPEN_PORTS
    max_port_id:    int = 32    # must match graph_builder.MAX_PORT_ID
    n_rot_steps:    int = 4     # discrete rotations: 0°/90°/180°/270°

    # ---- Input feature sizes (fixed by graph_builder/graph_dataset) --------
    node_cont_dim:  int = 9    # translation(3) + rotation_6d(6)
    edge_attr_dim:  int = 9    # relative rotation matrix (3×3 flattened)
    port_feat_dim:  int = 9    # world_pos(3)+world_nrm(3)+node_idx+port_id+type

    # ---- Training ----------------------------------------------------------
    batch_size:               int   = 32
    learning_rate:            float = 3e-4
    weight_decay:             float = 0.01
    max_epochs:               int   = 100
    warmup_steps:             int   = 500
    grad_clip_norm:           float = 1.0
    checkpoint_dir:           str   = "./checkpoints/graph"
    log_interval:             int   = 20
    checkpoint_interval:      int   = 500   # save latest.pt every N steps (0 = disable)
    early_stopping_patience:  int   = 10    # stop if val loss doesn't improve for N epochs (0 = disable)

    # ---- Generation --------------------------------------------------------
    max_gen_bricks: int   = 400
    temperature:    float = 0.8
    top_k:          int   = 50
