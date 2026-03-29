"""Flask backend for the ATLAS Graph Viewer.

Parses an uploaded .mpd file using the existing ATLAS geometry engine
and returns the assembly graph as JSON.

Usage:
    pip install flask
    python webapp/server.py        (from project root)
    # → http://localhost:5000
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

# Add project root to path so src.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory
import numpy as np

from src.data.parser import MPDParser
from src.geometry.lego_core import LegoCore
from src.geometry.lego_part import PartDatabase

app = Flask(__name__, static_folder=".")


def _arr(x: np.ndarray | list | None) -> list | None:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/parse", methods=["POST"])
def parse_mpd():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename or "model.mpd"
    if not filename.lower().endswith((".mpd", ".ldr")):
        return jsonify({"error": "File must be an .mpd or .ldr file"}), 400

    suffix = ".ldr" if filename.lower().endswith(".ldr") else ".mpd"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb") as f:
        file.save(f)
        tmp_path = f.name

    try:
        parser = MPDParser(tmp_path)

        if parser._submodels:
            model_name = list(parser._submodels.keys())[0]
            parser.flatten(model_name, np.eye(4))
        else:
            # Flat file with no 0 FILE headers — inject a synthetic submodel
            # from all type-1 lines in the file so flatten() can process them.
            brick_lines = [l for l in parser.lines if l.startswith("1 ")]
            if not brick_lines:
                return jsonify({"error": "No bricks found in file"}), 400
            synthetic = "__main__\n"
            parser._submodels[synthetic] = brick_lines
            parser.flatten(synthetic, np.eye(4))

        if not parser.raw_data:
            return jsonify({"error": "No bricks parsed from the MPD file"}), 400

        raw_bricks = parser.raw_data
        part_db = PartDatabase()
        core = LegoCore.from_raw_bricks(raw_bricks, part_db)
        nodes, edges = core.get_graph()

        # Degree per node
        degree: dict[int, int] = {n.node_id: 0 for n in nodes}
        for e in edges:
            degree[e.node_a] += 1
            degree[e.node_b] += 1

        # Cache port world positions for edge endpoint rendering
        port_world: dict[tuple[int, int], list] = {}

        nodes_out = []
        for node in nodes:
            pos = node.world_matrix[:3, 3]
            rot = node.world_matrix[:3, :3]

            lp = core.part_db.get_or_default(node.part_id)
            ports_out = []
            for port in lp.ports:
                w = port.transformed(node.world_matrix)
                wp = _arr(w.local_position)
                port_world[(node.node_id, port.port_id)] = wp
                ports_out.append({
                    "port_id":        port.port_id,
                    "type":           port.port_type,
                    "local_position": _arr(port.local_position),
                    "world_position": wp,
                    "normal":         _arr(w.normal),
                })

            nodes_out.append({
                "id":       node.node_id,
                "part_id":  node.part_id,
                "color":    node.color,
                "position": _arr(pos),
                "rotation": _arr(rot),
                "ports":    ports_out,
                "degree":   degree.get(node.node_id, 0),
            })

        edges_out = []
        for edge in edges:
            edges_out.append({
                "node_a":               edge.node_a,
                "node_b":               edge.node_b,
                "port_a_id":            edge.port_a_id,
                "port_b_id":            edge.port_b_id,
                "port_a_world_position": port_world.get((edge.node_a, edge.port_a_id)),
                "port_b_world_position": port_world.get((edge.node_b, edge.port_b_id)),
            })

        # Connected components (BFS)
        adj: dict[int, list[int]] = {n.node_id: [] for n in nodes}
        for e in edges:
            adj[e.node_a].append(e.node_b)
            adj[e.node_b].append(e.node_a)
        visited: set[int] = set()
        components = 0
        for node in nodes:
            if node.node_id not in visited:
                components += 1
                stack = [node.node_id]
                while stack:
                    nid = stack.pop()
                    if nid in visited:
                        continue
                    visited.add(nid)
                    stack.extend(adj[nid])

        return jsonify({
            "model_name": os.path.splitext(filename)[0],
            "stats": {
                "num_bricks":      len(nodes),
                "num_connections": len(edges),
                "num_components":  components,
                "isolated":        sum(1 for n in nodes if degree.get(n.node_id, 0) == 0),
            },
            "nodes": nodes_out,
            "edges": edges_out,
        })

    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    print("ATLAS Graph Viewer → http://localhost:5000")
    app.run(debug=True, port=5000)
