#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="Create Clump JSON"
#SBATCH --error="./new_logs/job-%j-create_clump_json_script.err"
#SBATCH --output="./new_logs/job-%j-create_clump_json_script.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="hplp"

module purge
module load miniforge
source /home/bue6zr/.bashrc

set -exo pipefail

echo "Running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

conda deactivate
conda activate netsec2

export PYTHONPATH=$(pwd)

mkdir -p new_logs
mkdir -p data/tmp
mkdir -p data/final

FINAL_DOH="data/final/benign_doh.json"
FINAL_NDOH="data/final/ndoh.json"

# Start final JSON arrays
echo "[" > "$FINAL_DOH"
echo "[" > "$FINAL_NDOH"

DOH_FIRST=1
NDOH_FIRST=1

append_json_array () {
    local SRC_FILE=$1
    local DST_FILE=$2
    local FIRST_FLAG=$3

    if [ ! -f "$SRC_FILE" ]; then
        echo "$FIRST_FLAG"
        return
    fi

    # Remove outer [ ] from all.json
    CONTENT=$(python3 - "$SRC_FILE" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r") as f:
    data = json.load(f)
for obj in data:
    print(json.dumps(obj))
PY
)

    if [ -z "$CONTENT" ]; then
        echo "$FIRST_FLAG"
        return
    fi

    if [ "$FIRST_FLAG" -eq 0 ]; then
        echo "," >> "$DST_FILE"
    fi

    echo "$CONTENT" | paste -sd, - >> "$DST_FILE"

    echo 0
}

PCAP_ROOT="../dataset/PCAPs/DoHBenign-NonDoH"

find "$PCAP_ROOT" -type f -name "*.pcap" | sort | while read -r f; do
    echo "======================================"
    echo "Processing PCAP: $f"
    echo "======================================"

    # Clean temporary working directory before each PCAP
    rm -rf data/tmp
    mkdir -p data/tmp/doh
    mkdir -p data/tmp/ndoh

    # Run dohlyzer on this PCAP only
    python3 meter/dohlyzer.py -f "$f" -s data/tmp/

    # Aggregate temporary DoH flow JSON files
    if compgen -G "data/tmp/doh/*.json" > /dev/null; then
        python3 meter/clump_aggregator.py data/tmp/doh/ --json
        DOH_FIRST=$(append_json_array "data/tmp/doh/all.json" "$FINAL_DOH" "$DOH_FIRST")
    fi

    # Aggregate temporary Non-DoH flow JSON files
    if compgen -G "data/tmp/ndoh/*.json" > /dev/null; then
        python3 meter/clump_aggregator.py data/tmp/ndoh/ --json
        NDOH_FIRST=$(append_json_array "data/tmp/ndoh/all.json" "$FINAL_NDOH" "$NDOH_FIRST")
    fi

    # Delete temporary JSONs before next PCAP
    rm -rf data/tmp/doh/*.json
    rm -rf data/tmp/ndoh/*.json
done

# Close final JSON arrays
echo "]" >> "$FINAL_DOH"
echo "]" >> "$FINAL_NDOH"

echo "Final DoH JSON: $FINAL_DOH"
echo "Final NonDoH JSON: $FINAL_NDOH"
echo "Done"
exit