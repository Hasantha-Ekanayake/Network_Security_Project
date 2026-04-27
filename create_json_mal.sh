#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="Create Clump Malicious JSON"
#SBATCH --error="./new_logs/job-%j-create_clump_mal_json_script.err"
#SBATCH --output="./new_logs/job-%j-create_clump_mal_json_script.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="hplp"

set -e

module purge
module load miniforge
source /home/bue6zr/.bashrc

echo "Running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

conda deactivate
conda activate netsec2

export PYTHONPATH=$(pwd)

mkdir -p new_logs
mkdir -p data/tmp2
mkdir -p data/final

FINAL_MAL="data/final/malicious_doh.json"

# Start final JSON arrays
echo "[" > "$FINAL_MAL"

MAL_FIRST=1

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

PCAP_ROOT="../dataset/PCAPs/DoHMalicious"

find "$PCAP_ROOT" -type f -name "*.pcap" | sort | while read -r f; do
    echo "======================================"
    echo "Processing PCAP: $f"
    echo "======================================"

    # Clean temporary working directory before each PCAP
    rm -rf data/tmp2
    mkdir -p data/tmp2/doh
    mkdir -p data/tmp2/ndoh

    # Run dohlyzer on this PCAP only
    python3 meter/dohlyzer.py -f "$f" -s data/tmp2/

    # Aggregate temporary DoH flow JSON files
    if compgen -G "data/tmp2/doh/*.json" > /dev/null; then
        python3 meter/clump_aggregator.py data/tmp2/doh/ --json
        DOH_FIRST=$(append_json_array "data/tmp2/doh/all.json" "$FINAL_MAL" "$MAL_FIRST")
    fi

    # Delete temporary JSONs before next PCAP
    rm -rf data/tmp2/doh/*.json
    rm -rf data/tmp2/ndoh/*.json
done

# Close final JSON arrays
echo "]" >> "$FINAL_MAL"

echo "Final malicious DoH JSON: $FINAL_MAL"
echo "Done"
exit