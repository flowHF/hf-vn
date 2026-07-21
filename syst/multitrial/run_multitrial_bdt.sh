#!/bin/bash

# Default flags
do_compile_fitter=false
do_cutset_generation=false
do_fits=false
do_plots=false
workers=1

# --- Command Line Argument Handling ---
if [ "$#" -lt 3 ]; then
    echo "Usage:"
    echo "  $0 <config_modifies_fit> <config_default> <output_dir> [--do_configs] [--do_fits] [--do_plots] [--workers <workers>]"
    exit 1
fi
# if [ "$#" -ne 3 ]; then
#     echo "Usage: $0 <config_modifies_fit> <config_default> <output_dir>"
#     exit 1
# fi

export config_modifies_fit="$1"
export config_default="$2"
export output_dir="$3"

shift 3

# ------------------------------------------------------------
# Parse optional flags
# ------------------------------------------------------------

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --do_configs)
            do_cutset_generation=true
            ;;
        --do_fits)
            do_fits=true
            ;;
        --do_plots)
            do_plots=true
            ;;
        --do_compile)
            do_compile_fitter=true
            ;;
        --workers)
            workers="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done


# Normalize working directories
SCRIPT_HOME=$(pwd)
OUTPUT_DIR=$(realpath "$output_dir")
BASE_ROOT=$(realpath "$OUTPUT_DIR/syst/multitrial/bdt")

export OUTPUT_DIR
export BASE_ROOT

### Examples
# export config_modifies_fit="/home/mdicosta/DFlowOO/SP_bdt_OO/Preliminary/syst_multitrial_bdt_scan.yml"
# export config_default="/home/mdicosta/DFlowOO/SP_bdt_OO/Preliminary/config_020.yml"
# export output_dir="/home/mdicosta/DFlowOO/SP_bdt_OO/Preliminary/cutvar_020_combined"

generate_cutset() {
    local yaml_file="$1"
    local pt_dir="$2"

    # Safety checks
    if [[ ! -f "$yaml_file" ]]; then
        echo "ERROR: YAML file not found: $yaml_file"
        return 1
    fi
    if [[ -z "$pt_dir" ]]; then
        echo "ERROR: pt_dir is empty"
        return 1
    fi

    # Extract trial number from filename
    local trial_num
    trial_num=$(basename "$yaml_file" | sed -E 's/config_trial_([0-9]+)\.yml/\1/')

    # Target directory for this trial from directory of config_trial
    # yaml_file_dir=$(dirname "$yaml_file")
    # local target_dir="$target_base_dir/$trial_num"
    local target_dir=$(dirname "$yaml_file")

    # Create target directory
    mkdir -p "$target_dir"

    # Log start
    echo "[$trial_num] Generating YAML cutsets for file $yaml_file into $target_dir"

    # Run Python script
    if python3 "$path_to_src/src/make_cutsets_cfgs.py" "$yaml_file" -o "$target_dir"; then
        echo "[$trial_num] Cutsets generated successfully"
    else
        echo "[$trial_num] Failed to generate cutsets"
        return 1
    fi
}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ROOT_MAX_THREADS=1

# Export function and necessary variables
export -f generate_cutset
export path_to_src="/home/mdicosta/alice/hf-vn/"

mkdir -p "$OUTPUT_DIR/syst/multitrial/bdt"

if [ "$do_compile_fitter" = true ]; then
    echo "Compiling InvMassFitter and VnVsMassFitter ..."
    rootcling -f $path_to_src/invmassfitter/vnfitter_dict.cxx -c /home/mdicosta/alice/hf-vn/invmassfitter/InvMassFitter.h /home/mdicosta/alice/hf-vn/invmassfitter/VnVsMassFitter.h /home/mdicosta/alice/hf-vn/invmassfitter/LinkDefVnFitter.h

    echo "Compiling fitter once for every trial ..."
    g++ -shared -fPIC `root-config --cflags --libs` \
        $path_to_src/invmassfitter/InvMassFitter.cxx $path_to_src/invmassfitter/VnVsMassFitter.cxx $path_to_src/invmassfitter/vnfitter_dict.cxx \
        -o $path_to_src/invmassfitter/libvnfitter.so
    echo "Compilation done!"
fi

if [ "$do_cutset_generation" = true ]; then
    echo "Generating YAML files for BDT multitrial ..."
    # Generate YAML file list with indices
    python3 $path_to_src/syst/multitrial/make_configs_multitrial.py $config_default -m $config_modifies_fit -bm -o $OUTPUT_DIR > "$OUTPUT_DIR/syst/multitrial/bdt/log_make_cutsets.txt" 2>&1
    echo "Yaml files generated!"
fi

if [ "$do_fits" = true ]; then

    export path_to_src
    export output_dir

    process_pt_bin() {

        local pt_dir
        pt_dir=$(realpath "$1")

        echo "Processing PT bin → ${pt_dir##*/}"

        # Sequential execution inside pt bin
        find "$pt_dir" -type f -name "config_trial_*.yml" | sort | while read -r file; do

            [[ -z "$file" ]] && continue

            log_file="${file%.yml}.log"
            mkdir -p "$(dirname "$log_file")"

            echo "  -> Processing $file"

            python3 "$path_to_src/run_analysis.py" "$file" --combined \
                > "$log_file" 2>&1

        done

        echo "Finished PT bin → ${pt_dir##*/}"
    }

    export -f process_pt_bin

    base_dir=$(realpath "$output_dir/syst/multitrial/bdt")

    # Print command that will be launched for debugging
    echo "Launching parallel execution with workers=$workers on PT bins in $base_dir"
    echo "Command: find \"$base_dir\" -maxdepth 1 -type d -name \"pt_*\" -print0 | xargs -0 -P \"$workers\" -I {} bash -c 'process_pt_bin \"\$@\"' _ {}"

    # Print output of find \"$base_dir\" -maxdepth 1 -type d -name \"pt_*\" -print0
    echo "PT bins found:"
    find "$base_dir" -maxdepth 1 -type d -name "pt_*" -print0 | xargs -0 -I {} echo "  {}"

    find "$base_dir" -maxdepth 1 -type d -name "pt_*" -print0 | \
    xargs -0 -P "$workers" -I {} bash -c 'process_pt_bin "$@"' _ {}

fi

# --- Produce final plots ---
if [ "$do_plots" = true ]; then
    echo "Producing final systematic plots ..."
    python3 $path_to_src/syst/multitrial/produce_bdt_multitrial_syst_plots.py "$config_default" "$OUTPUT_DIR" > "$OUTPUT_DIR/syst/multitrial/bdt/log_produce_scan_plots.txt" 2>&1
    echo "Final plots produced!"
fi
