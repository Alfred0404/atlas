"""Visualize the assembly graph of a LEGO set.

Usage:
    python visualize_graph.py dataset/mpd_files/10002-1.mpd
    python visualize_graph.py dataset/mpd_files/10002-1.mpd --3d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.data.parser import MPDParser
from src.geometry.lego_core import LegoCore
from src.geometry.lego_part import PartDatabase


def build_graph(mpd_path: str) -> tuple[LegoCore, str]:
    """Parse an MPD and build its LegoCore assembly graph."""
    parser = MPDParser(mpd_path)
    if not parser._submodels:
        raise ValueError(f"No submodels found in {mpd_path}")

    model_name = list(parser._submodels.keys())[0]
    parser.flatten(model_name, np.eye(4))
    if not parser.raw_data:
        raise ValueError(f"No bricks parsed from {mpd_path}")

    part_db = PartDatabase()
    core = LegoCore.from_raw_bricks(parser.raw_data, part_db)
    return core, model_name


def to_networkx(core: LegoCore) -> nx.Graph:
    """Convert a LegoCore graph to a NetworkX graph."""
    G = nx.Graph()
    for node in core.nodes.values():
        pos = node.world_matrix[:3, 3]
        G.add_node(
            node.node_id,
            part_id=node.part_id,
            color=node.color,
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
        )
    for edge in core.edges:
        if not G.has_edge(edge.node_a, edge.node_b):
            G.add_edge(edge.node_a, edge.node_b)
    return G


def draw_2d(G: nx.Graph, title: str) -> None:
    """Draw the graph in 2D using spring layout, colored by connectivity."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    degrees = dict(G.degree())
    node_colors = [degrees.get(n, 0) for n in G.nodes()]
    vmin = min(node_colors) if node_colors else 0
    vmax = max(node_colors) if node_colors else 1

    # --- Left: spring layout ---
    ax = axes[0]
    pos = nx.spring_layout(G, seed=42, k=1.5 / max(len(G) ** 0.5, 1))
    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_size=40,
        with_labels=False,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        vmin=vmin,
        vmax=vmax,
        edge_color="#88888844",
        width=0.5,
    )
    ax.set_title(f"Spring layout  ({G.number_of_nodes()} bricks, {G.number_of_edges()} connections)")
    ax.axis("off")

    # --- Right: spatial top-down (X vs Z) ---
    ax = axes[1]
    spatial_pos = {n: (G.nodes[n]["x"], G.nodes[n]["z"]) for n in G.nodes()}
    nx.draw_networkx(
        G,
        spatial_pos,
        ax=ax,
        node_size=30,
        with_labels=False,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        vmin=vmin,
        vmax=vmax,
        edge_color="#88888844",
        width=0.5,
    )
    ax.set_title("Spatial top-down (X vs Z)")
    ax.set_xlabel("X (LDU)")
    ax.set_ylabel("Z (LDU)")
    ax.set_aspect("equal")

    # Colorbar legend for node degree
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Node degree (connections)")

    # Stats
    components = nx.number_connected_components(G)
    isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)
    fig.suptitle(f"{title}\n{components} components, {isolated} isolated nodes", fontsize=13)
    plt.tight_layout()
    plt.show()


def draw_3d(G: nx.Graph, title: str) -> None:
    """Draw the graph in 3D using actual brick positions."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    xs = [G.nodes[n]["x"] for n in G.nodes()]
    ys = [G.nodes[n]["z"] for n in G.nodes()]
    zs = [-G.nodes[n]["y"] for n in G.nodes()]  # flip Y for natural up

    degrees = dict(G.degree())
    colors = [degrees.get(n, 0) for n in G.nodes()]
    vmin = min(colors) if colors else 0
    vmax = max(colors) if colors else 1

    sc = ax.scatter(xs, ys, zs, c=colors, cmap=plt.cm.plasma, s=30, depthshade=True, vmin=vmin, vmax=vmax)

    for u, v in G.edges():
        ax.plot(
            [G.nodes[u]["x"], G.nodes[v]["x"]],
            [G.nodes[u]["z"], G.nodes[v]["z"]],
            [-G.nodes[u]["y"], -G.nodes[v]["y"]],
            color="#88888844",
            linewidth=0.5,
        )

    ax.set_xlabel("X (LDU)")
    ax.set_ylabel("Z (LDU)")
    ax.set_zlabel("Height")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Node degree (connections)")

    components = nx.number_connected_components(G)
    isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)
    ax.set_title(f"{title}\n{G.number_of_nodes()} bricks, {G.number_of_edges()} conn, {components} components, {isolated} isolated")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LEGO set assembly graph")
    parser.add_argument("mpd_file", help="Path to .mpd file")
    parser.add_argument("--3d", dest="three_d", action="store_true", help="3D spatial view")
    args = parser.parse_args()

    core, model_name = build_graph(args.mpd_file)
    G = to_networkx(core)

    nodes, edges = core.get_graph()
    print(f"Set: {model_name}")
    print(f"Bricks: {len(nodes)}")
    print(f"Connections: {len(edges)}")
    print(f"Components: {nx.number_connected_components(G)}")
    print(f"Isolated: {sum(1 for n in G.nodes() if G.degree(n) == 0)}")

    title = Path(args.mpd_file).stem
    if args.three_d:
        draw_3d(G, title)
    else:
        draw_2d(G, title)


if __name__ == "__main__":
    main()
