#!/usr/bin/env python3
"""Merge the completed MaSIF benchmark sweeps into the plot input table."""

import csv
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TABLE = SCRIPT_DIR / "masif_surface_benchmark_serial" / "surface_speed_vertices.csv"
NEW_SURFACES = (
    SCRIPT_DIR
    / "masif_surface_benchmark_all_serial_20260821"
    / "pinder_benchmark_raw.csv"
)
NEW_MSMS = (
    SCRIPT_DIR / "masif_msms_simplified_serial_20260821" / "pinder_benchmark_raw.csv"
)
OUTPUT = TABLE


def summarize(
    raw_path,
    source_method,
    param,
    label,
    grid_scale,
    *,
    output_method=None,
    time_column="t_gen",
    vertices_column="n_verts",
):
    data = pd.read_csv(raw_path)
    data = data[(data["method"] == source_method) & (data["param"] == param)]
    data = data[(~data["crash"]) & (data[time_column] > 0)]
    if data.empty:
        raise ValueError(
            f"No successful rows for {source_method} {param} in {raw_path}"
        )
    return {
        "method": output_method or source_method,
        "label": label,
        "grid_scale": grid_scale,
        "time_ms": data[time_column].mean() * 1000.0,
        "time_std": data[time_column].std(ddof=1) * 1000.0,
        "vertices": data[vertices_column].mean(),
        "vertices_std": data[vertices_column].std(ddof=1),
    }


def main():
    with TABLE.open(newline="") as handle:
        old_rows = list(csv.DictReader(handle))

    replace = {
        ("edtsurf", "2.0"),
        ("nanoshaper", "0.3"),
        ("nanoshaper", "2.0"),
        ("nanoshaper", "0.6"),
        ("msms", ""),
        ("msms_simplified", ""),
    }
    rows = [
        row for row in old_rows if (row["method"], row["grid_scale"]) not in replace
    ]
    rows.extend(
        [
            summarize(NEW_SURFACES, "edtsurf", 2.0, "EDTsurf", "2.0"),
            summarize(NEW_SURFACES, "nanoshaper", 0.3, "NanoShaper", "0.3"),
            summarize(NEW_SURFACES, "nanoshaper", 0.6, "NanoShaper", "0.6"),
            summarize(NEW_SURFACES, "nanoshaper", 2.0, "NanoShaper", "2.0"),
            summarize(
                NEW_MSMS,
                "msms",
                0.1,
                "MSMS",
                "",
                time_column="t_gen",
                vertices_column="n_verts_raw",
            ),
            summarize(
                NEW_MSMS,
                "msms",
                0.1,
                "MSMS simplified",
                "",
                output_method="msms_simplified",
                time_column="t_total",
                vertices_column="n_verts",
            ),
        ]
    )

    order = {
        ("alpha_complex", ""): 0,
        ("edtsurf", "0.3"): 1,
        ("edtsurf", "0.4"): 2,
        ("edtsurf", "0.5"): 3,
        ("edtsurf", "2.0"): 4,
        ("nanoshaper", "0.3"): 5,
        ("nanoshaper", "0.4"): 6,
        ("nanoshaper", "0.5"): 7,
        ("nanoshaper", "0.6"): 8,
        ("nanoshaper", "2.0"): 9,
        ("msms", ""): 10,
        ("msms_simplified", ""): 11,
    }
    rows.sort(key=lambda row: order[(row["method"], row["grid_scale"])])
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT} ({len(rows)} methods)")


if __name__ == "__main__":
    main()
