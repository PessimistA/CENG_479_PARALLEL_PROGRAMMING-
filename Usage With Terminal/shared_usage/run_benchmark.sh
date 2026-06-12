#!/usr/bin/env bash

# Orchestrates the complete Non-Local Means benchmark pipeline
# Validates user inputs, compiles the CUDA source, processes target images, and aggregates JSON results

# Instruct the shell to terminate execution immediately if any command fails
set -e

# Validate and parse required command-line arguments for image directory and test iterations
IMAGES_DIR="${1:?Usage: $0 <images_dir> [repeats]}"
REPEATS="${2:-3}"

# Establish absolute paths to ensure reliable execution regardless of the current working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/nlm_benchmark"
RESULTS_ROOT="${IMAGES_DIR}/benchmark_results"
MASTER_JSON="${RESULTS_ROOT}/all_results.json"

echo "=============================================="
echo "  Compiling nlm_benchmark.cu ..."
echo "=============================================="

# Compile the CUDA benchmark source code using native architecture optimizations for maximum performance
nvcc -O3 -arch=native -o "${BINARY}" "${SCRIPT_DIR}/nlm_benchmark.cu"

echo "  Compiled → ${BINARY}"
echo ""

# Ensure the root directory for storing benchmark outputs is available before processing begins
mkdir -p "${RESULTS_ROOT}"

# Define the acceptable image extensions for the benchmark pipeline
EXTENSIONS=("png" "jpg" "jpeg" "bmp" "tga")

# Dynamically construct the search command parameters to locate supported image formats
FIND_ARGS=()
for ext in "${EXTENSIONS[@]}"; do
    FIND_ARGS+=(-iname "*.${ext}" -o)
done

# Remove the trailing logical OR operator to finalize the find expression syntax
unset 'FIND_ARGS[${#FIND_ARGS[@]}-1]'

# Retrieve matching image files from the target directory without traversing into subdirectories
mapfile -t IMAGES < <(find "${IMAGES_DIR}" -maxdepth 1 \( "${FIND_ARGS[@]}" \) | sort)

# Terminate the script safely if no valid images are located in the specified directory
if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "No images found in ${IMAGES_DIR}"
    exit 1
fi

echo "Found ${#IMAGES[@]} image(s) in ${IMAGES_DIR}"
echo "Repeats per test : ${REPEATS}"
echo "Results root     : ${RESULTS_ROOT}"
echo ""

# Iterate through all discovered images and execute the compiled benchmark binary for each
for IMG in "${IMAGES[@]}"; do
    BASENAME="$(basename "${IMG%.*}")"
    OUT_DIR="${RESULTS_ROOT}/${BASENAME}"
    
    mkdir -p "${OUT_DIR}"

    echo "=============================================="
    echo "  Processing: ${IMG}"
    echo "  Output dir: ${OUT_DIR}"
    echo "=============================================="

    "${BINARY}" "${IMG}" "${OUT_DIR}" "${REPEATS}"
    echo ""
done

echo "=============================================="
echo "  Merging results → ${MASTER_JSON}"
echo "=============================================="

# Export directory paths to make them accessible to the embedded Python environment
export RESULTS_ROOT MASTER_JSON

# Utilize an embedded Python script to aggregate individual JSON results into a single master file
python3 - <<'PYEOF'
import json, glob, sys, os

results_root = os.environ["RESULTS_ROOT"]
master_json  = os.environ["MASTER_JSON"]

all_records = []
pattern = os.path.join(results_root, "*", "results.json")

# Sequentially load and append data from every generated JSON fragment
for path in sorted(glob.glob(pattern)):
    with open(path) as f:
        try:
            data = json.load(f)
            all_records.extend(data)
        except json.JSONDecodeError as e:
            print(f"  Warning: could not parse {path}: {e}")

# Write the consolidated dataset back to the storage
with open(master_json, "w") as f:
    json.dump(all_records, f, indent=2)

print(f"  Merged {len(all_records)} record(s) from {len(list(glob.glob(pattern)))} file(s)")
PYEOF

echo ""
echo "=============================================="
echo "  SUMMARY TABLE"
echo "=============================================="

# Utilize an embedded Python script to parse the master JSON and format a readable console summary
python3 - <<'PYEOF'
import json, os

master_json = os.environ["MASTER_JSON"]

with open(master_json) as f:
    records = json.load(f)

if not records:
    print("No records found.")
    exit()

# Format the column headers to include structural search and patch dimension details
header = f"{'Image':<20} {'Res':>6} {'S/P':>7} {'CPU avg(s)':>12} {'GPU avg(s)':>12} {'Speedup':>9} {'MSE':>10} {'MaxAbsErr':>11}"
sep    = "-" * len(header)
print(sep)
print(header)
print(sep)

# Loop through the parsed records and print aligned data rows for terminal display
for r in records:
    sp_str = f"{r.get('search_radius', 7)}/{r.get('patch_radius', 2)}"
    print(
        f"{r['image']:<20} "
        f"{r['resolution']:>6} "
        f"{sp_str:>7} "
        f"{r['cpu_avg_s']:>12.4f} "
        f"{r['gpu_avg_s']:>12.4f} "
        f"{r['speedup']:>9.2f}x "
        f"{r['mse']:>10.4f} "
        f"{r['max_abs_error']:>11.1f}"
    )

print(sep)
print(f"\nFull results saved to: {master_json}")
PYEOF

echo ""
echo "Done. All outputs are in: ${RESULTS_ROOT}"