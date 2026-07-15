#!/bin/bash
# Download/resume MISATO MD data and the official sequence-clustered splits.
set -euo pipefail

DATA_DIR=${1:-/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/data/misato}
mkdir -p "$DATA_DIR/splits"

# -c is intentional: the file is 132.8 GB and interrupted transfers are common.
wget -c -O "$DATA_DIR/MD.hdf5" \
    https://zenodo.org/records/7711953/files/MD.hdf5

for split in train val test; do
    wget -O "$DATA_DIR/splits/${split}.txt" \
        "https://zenodo.org/records/7711953/files/${split}_MD.txt"
done

echo "Downloaded MD.hdf5 and official MISATO splits to $DATA_DIR"
echo "Next: sbatch alphasurf/tasks/misato_binding_site/preprocess.sh"
