#!/bin/bash
#SBATCH --job-name=proteingym_download
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=log/proteingym_download/%x_%j.log
#SBATCH --error=log/proteingym_download/%x_%j.err

set -euo pipefail

DATA_DIR="${PROTEINGYM_DIR:-/cluster/CBIO/data2/vgertner/alphasurf/data/proteingym}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading ProteinGym data to: $DATA_DIR"

PROTEINGYM_VERSION="${PROTEINGYM_VERSION:-v1.3}"
BASE_URL="https://marks.hms.harvard.edu/proteingym/ProteinGym_${PROTEINGYM_VERSION}"
DMS_URL="${BASE_URL}/DMS_ProteinGym_substitutions.zip"
AF2_URL="${BASE_URL}/ProteinGym_AF2_structures.zip"

# marks.hms.harvard.edu is signed by InCommon RSA OV SSL CA 3, which is missing
# from this cluster's CA bundle (curl 60 / wget "unable to get local issuer
# certificate"). Probe with --cacert, then fall back to -k if that fails.
CURL_TLS_OPTS=(--retry 5)
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    CURL_TLS_OPTS+=(--cacert /etc/ssl/certs/ca-certificates.crt)
fi
if ! curl -fLsI "${CURL_TLS_OPTS[@]}" "$DMS_URL" >/dev/null 2>&1; then
    echo "TLS verification failed; retrying with --insecure (cluster CA bundle is stale)." >&2
    CURL_TLS_OPTS+=(-k)
fi
CURL_TRANSFER_OPTS=(-C - -fL "${CURL_TLS_OPTS[@]}")

if [ ! -f "$DATA_DIR/DMS_ProteinGym_substitutions.zip" ]; then
    echo "Fetching DMS substitutions archive..."
    curl "${CURL_TRANSFER_OPTS[@]}" -o "$DATA_DIR/DMS_ProteinGym_substitutions.zip" "$DMS_URL"
fi

if [ ! -d "$DATA_DIR/substitutions" ]; then
    echo "Extracting DMS substitutions..."
    unzip -q -o "$DATA_DIR/DMS_ProteinGym_substitutions.zip" -d "$DATA_DIR/substitutions"
fi

if [ ! -f "$DATA_DIR/ProteinGym_AF2_structures.zip" ]; then
    echo "Fetching AF2 structures archive..."
    curl "${CURL_TRANSFER_OPTS[@]}" -o "$DATA_DIR/ProteinGym_AF2_structures.zip" "$AF2_URL"
fi

if [ ! -d "$DATA_DIR/af2_structures" ]; then
    echo "Extracting AF2 structures..."
    unzip -q -o "$DATA_DIR/ProteinGym_AF2_structures.zip" -d "$DATA_DIR/af2_structures"
fi

echo "Done."
echo "  DMS assays:   $DATA_DIR/substitutions/"
echo "  AF2 structures: $DATA_DIR/af2_structures/"
