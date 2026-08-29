#!/usr/bin/env python3
"""Render a protein, its alpha-complex mesh, and spanning edges in PyMOL.

Run with the PyMOL executable, for example::

    pymol -cq pymol_alpha_complex_publication.py -- \
        --pdb path/to/case.pdb --ply path/to/case.ply \
        --edges path/to/case.edges.pdb --png case.png --session case.pse

The edge PDB is expected to contain CONECT records, as produced by the
full-atom alpha-complex failure analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pymol import cmd
from pymol.cgo import COLOR, CYLINDER, SPHERE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True, help="protein PDB")
    parser.add_argument("--ply", type=Path, required=True, help="alpha-complex PLY")
    parser.add_argument(
        "--edges", type=Path, required=True, help="edge sidecar PDB with CONECT records"
    )
    parser.add_argument("--png", type=Path, help="ray-traced publication PNG")
    parser.add_argument("--session", type=Path, help="PyMOL session to save")
    parser.add_argument(
        "--mesh-transparency",
        type=float,
        default=0.38,
        help="alpha-complex transparency in [0, 1] (default: 0.38)",
    )
    parser.add_argument(
        "--edge-radius",
        type=float,
        default=0.16,
        help="edge stick radius in Angstroms (default: 0.16)",
    )
    parser.add_argument(
        "--mesh-edge-radius",
        type=float,
        default=0.025,
        help="alpha-complex triangle-edge radius (default: 0.025)",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def load_edge_cgo(path: Path, radius: float) -> None:
    atoms: dict[int, tuple[float, float, float]] = {}
    bonds: set[tuple[int, int]] = set()
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atoms[int(line[6:11])] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        elif line.startswith("CONECT"):
            fields = line.split()
            source = int(fields[1])
            for target_text in fields[2:]:
                target = int(target_text)
                bonds.add(tuple(sorted((source, target))))
    if not bonds:
        raise ValueError(f"No CONECT bonds found in edge PDB: {path}")
    color = (0.05, 0.55, 0.85)
    edge_cgo: list[float] = []
    for source, target in sorted(bonds):
        if source not in atoms or target not in atoms:
            raise ValueError(f"Edge PDB references an unknown atom: {source}, {target}")
        edge_cgo.extend(
            [
                COLOR,
                *color,
                CYLINDER,
                *atoms[source],
                *atoms[target],
                radius,
                *color,
                *color,
            ]
        )
        endpoint_radius = radius * 1.12
        edge_cgo.extend(
            [
                SPHERE,
                *atoms[source],
                endpoint_radius,
                SPHERE,
                *atoms[target],
                endpoint_radius,
            ]
        )
    cmd.load_cgo(edge_cgo, "spanning_edges")


def load_mesh_wireframe(path: Path, radius: float) -> None:
    lines = path.read_text().splitlines()
    header_end = lines.index("end_header")
    vertex_count = next(
        int(line.split()[2])
        for line in lines[:header_end]
        if line.startswith("element vertex ")
    )
    face_count = next(
        int(line.split()[2])
        for line in lines[:header_end]
        if line.startswith("element face ")
    )
    data = lines[header_end + 1 :]
    vertices = [tuple(map(float, data[i].split()[:3])) for i in range(vertex_count)]
    faces = []
    for line in data[vertex_count : vertex_count + face_count]:
        fields = list(map(int, line.split()))
        faces.append(fields[1 : fields[0] + 1])
    edges = {
        tuple(sorted((face[i], face[(i + 1) % len(face)])))
        for face in faces
        for i in range(len(face))
    }
    color = (0.20, 0.06, 0.02)
    wire_cgo: list[float] = []
    for source, target in sorted(edges):
        wire_cgo.extend(
            [
                COLOR,
                *color,
                CYLINDER,
                *vertices[source],
                *vertices[target],
                radius,
                *color,
                *color,
            ]
        )
    cmd.load_cgo(wire_cgo, "alpha_complex_edges")


def configure_scene(args: argparse.Namespace) -> None:
    pdb_path = require_file(args.pdb, "protein PDB")
    ply_path = require_file(args.ply, "alpha-complex PLY")
    edge_path = require_file(args.edges, "edge PDB")
    if not 0.0 <= args.mesh_transparency <= 1.0:
        raise ValueError("--mesh-transparency must be between 0 and 1")
    if args.edge_radius <= 0.0:
        raise ValueError("--edge-radius must be positive")
    if args.mesh_edge_radius <= 0.0:
        raise ValueError("--mesh-edge-radius must be positive")

    cmd.reinitialize()
    cmd.load(str(pdb_path), "protein")
    cmd.load(str(ply_path), "alpha_complex")
    load_edge_cgo(edge_path, args.edge_radius)
    load_mesh_wireframe(ply_path, args.mesh_edge_radius)

    cmd.set_color("publication_protein", [0.35, 0.42, 0.50])
    cmd.set_color("publication_mesh", [0.90, 0.30, 0.12])
    cmd.set_color("publication_edges", [0.05, 0.55, 0.85])

    cmd.hide("everything", "all")
    cmd.show("cartoon", "protein and polymer")
    cmd.color("publication_protein", "protein")
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_sampling", 20)
    cmd.set("cartoon_oval_length", 1.2)
    cmd.set("cartoon_oval_width", 0.25)

    cmd.show("everything", "alpha_complex")
    cmd.color("publication_mesh", "alpha_complex")
    cmd.set("cgo_transparency", args.mesh_transparency, "alpha_complex")
    cmd.set("surface_quality", 1)
    cmd.set("two_sided_lighting", 1)
    cmd.show("everything", "spanning_edges")
    cmd.show("everything", "alpha_complex_edges")

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ambient", 0.35)
    cmd.set("direct", 0.65)
    cmd.set("specular", 0.15)
    cmd.set("shininess", 20)
    cmd.set("depth_cue", 0)
    cmd.orient("all")
    cmd.zoom("all", 2.0)

    if args.session:
        args.session.parent.mkdir(parents=True, exist_ok=True)
        cmd.save(str(args.session))
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        cmd.png(str(args.png), width=2400, height=2400, dpi=300, ray=1)


def main() -> None:
    args = parse_args()
    configure_scene(args)
    if args.png or args.session:
        cmd.quit()
    else:
        cmd.set("auto_show_spheres", 0)


if __name__ == "__main__":
    main()
