#!/usr/bin/env bash
# Download mizu-trading 1s OHLCV ndjson files for ES and NQ.
# Saves to data/mizu/es/ and data/mizu/nq/.
# Supports resume: curl -C - skips already-complete files.
#
# Usage:
#   bash scripts/download_mizu_data.sh          # both ES and NQ
#   bash scripts/download_mizu_data.sh es       # ES only
#   bash scripts/download_mizu_data.sh nq       # NQ only

set -euo pipefail
cd "$(dirname "$0")/.."

BASE="https://raw.githubusercontent.com/mizu-trading/market-data/main"

ES_DIR="GLBX-20251209-3BSFEH3FL9"
NQ_DIR="GLBX-20251209-BGNUG45EGV"
ES_FILE="glbx-mdp3-20240902-20251208.ohlcv-1s"
NQ_FILE="glbx-mdp3-20240902-20251208.ohlcv-1s"
ES_PARTS=37   # part00..part36
NQ_PARTS=40   # part00..part39

download_parts() {
    local remote_dir="$1"
    local remote_file="$2"
    local local_dir="$3"
    local n_parts="$4"
    local label="$5"

    mkdir -p "$local_dir"
    echo "=== Downloading $label ($n_parts parts → $local_dir) ==="

    for i in $(seq 0 $((n_parts - 1))); do
        part=$(printf "part%02d" "$i")
        fname="${remote_file}.${part}.ndjson"
        url="${BASE}/${remote_dir}/${fname}"
        dest="${local_dir}/${fname}"

        if [ -f "$dest" ]; then
            # Check if file is complete by comparing Content-Length
            remote_size=$(curl -sI "$url" | awk '/content-length/{print $2}' | tr -d '\r')
            local_size=$(wc -c < "$dest" | tr -d ' ')
            if [ "$local_size" = "$remote_size" ]; then
                echo "  [skip] $fname  (already complete)"
                continue
            fi
        fi

        echo "  [fetch] $fname"
        curl -# -C - -o "$dest" "$url" || {
            echo "  [warn] failed, will retry next run: $fname"
        }
    done
    echo "  Done: $label"
}

TARGET="${1:-both}"

case "$TARGET" in
    es|ES)
        download_parts "$ES_DIR" "$ES_FILE" "data/mizu/es" "$ES_PARTS" "ES"
        ;;
    nq|NQ)
        download_parts "$NQ_DIR" "$NQ_FILE" "data/mizu/nq" "$NQ_PARTS" "NQ"
        ;;
    *)
        download_parts "$ES_DIR" "$ES_FILE" "data/mizu/es" "$ES_PARTS" "ES"
        download_parts "$NQ_DIR" "$NQ_FILE" "data/mizu/nq" "$NQ_PARTS" "NQ"
        ;;
esac

echo ""
echo "All done. Run: python src/convert_mizu_ndjson.py"
