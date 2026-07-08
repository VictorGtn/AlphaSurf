#!/bin/bash
#SBATCH --job-name=cath_download
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

set -euo pipefail

# S3F/ProtSSN use the same CATH v4.3.0 S40 dompdb from HuggingFace:
# https://huggingface.co/datasets/tyang816/cath
# We download and extract the raw PDBs here. AlphaSurf generates CGAL
# alpha-complex surfaces on-the-fly from these PDBs (ProteinLoader), so
# we do NOT run S3F's preload_dataset.py or process_surface.py — those
# produce dMaSIF point-cloud surfaces for S3F's specific surface branch.

CATH_DIR="${CATH_DIR:-/cluster/CBIO/data2/vgertner/alphasurf/data/cath}"
mkdir -p "$CATH_DIR"
cd "$CATH_DIR"

CURL_TLS_OPTS=(--retry 5)
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    CURL_TLS_OPTS+=(--cacert /etc/ssl/certs/ca-certificates.crt)
fi
if ! curl -fLsI "${CURL_TLS_OPTS[@]}" "https://huggingface.co" >/dev/null 2>&1; then
    echo "TLS verification failed; retrying with --insecure." >&2
    CURL_TLS_OPTS+=(-k)
fi

DOMPDB_TAR="$CATH_DIR/dompdb.tar"
DOMPDB_DIR="$CATH_DIR/dompdb"

if [ ! -f "$DOMPDB_TAR" ]; then
    echo "Downloading dompdb.tar from HuggingFace (tyang816/cath)..."
    curl -fL "${CURL_TLS_OPTS[@]}" -o "$DOMPDB_TAR" \
        "https://huggingface.co/datasets/tyang816/cath/resolve/main/dompdb.tar"
fi

if [ ! -d "$DOMPDB_DIR" ]; then
    echo "Extracting dompdb.tar..."
    tar -xf "$DOMPDB_TAR" -C "$CATH_DIR"
fi

echo "Done."
echo "  dompdb: $DOMPDB_DIR"
echo "  count:  $(ls "$DOMPDB_DIR" 2>/dev/null | wc -l) domains"
