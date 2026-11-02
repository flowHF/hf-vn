#!/bin/bash

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ROOT_MAX_THREADS=1

# -----------------------------
# Default values
# -----------------------------
n_parallel=1
path_to_src=""
config_default=""
config_modifies_fit=""
output_dir="."

do_cms_fits=false
do_compile_fitter=false
do_cutset_generation=false
do_projections=false
do_sim_fits=false
do_v2_vs_frac=false
produce_plots=false

max_chi2=10
min_signif=5
max_signif=1000
force_prompt_enhanced=false

# -----------------------------
# Parse command line arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)
            path_to_src="$2"
            shift 2
            ;;
        --config)
            config_default="$2"
            shift 2
            ;;
        --mod-config)
            config_modifies_fit="$2"
            shift 2
            ;;
        --outdir)
            output_dir="$2"
            shift 2
            ;;
        --nproc)
            n_parallel="$2"
            shift 2
            ;;
        --do-cutset)
            do_cutset_generation=true
            shift
            ;;
        --do-compile)
            do_compile_fitter=true
            shift
            ;;
        --do-projections)
            do_projections=true
            shift
            ;;
        --do-simfits)
            do_sim_fits=true
            shift
            ;;
        --do-cmsfits)
            do_cms_fits=true
            shift
            ;;
        --do-v2frac)
            do_v2_vs_frac=true
            shift
            ;;
        --produce-plots)
            produce_plots=true
            shift
            ;;
        --max-chi2)
            max_chi2="$2"
            shift 2
            ;;
        --min-signif)
            min_signif="$2"
            shift 2
            ;;
        --max-signif)
            max_signif="$2"
            shift 2
            ;;
        --force-prompt-enhanced)
            force_prompt_enhanced=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Recap of the configuration
echo "Configuration:"
echo "  Source code path: $path_to_src"
echo "  Default config: $config_default"
echo "  Modifies fit config: $config_modifies_fit"
echo "  Output directory: $output_dir"
echo "  Number of parallel processes: $n_parallel"
echo "  Do cutset generation: $do_cutset_generation"
echo "  Do compile fitter: $do_compile_fitter"
echo "  Do projections: $do_projections"
echo "  Do simultaneous fits: $do_sim_fits"
echo "  Do CMS fits: $do_cms_fits"
echo "  Do v2 vs frac extraction: $do_v2_vs_frac"
echo "  Produce plots: $produce_plots"
echo "  Max chi2 for plots: $max_chi2"
echo "  Min significance for plots: $min_signif"
echo "  Max significance for plots: $max_signif"
echo "  Force prompt enhanced: $force_prompt_enhanced"

export path_to_src

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

    # Target directory for this trial
    local target_dir="$pt_dir/trials/$trial_num"

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

get_v2_vs_fracs() {
    local yaml_file="$1"
    local reference_frac_dir="$2"
    trial_num=$(basename "$yaml_file" | sed -E 's/config_trial_([0-9]+)\.yml/\1/')
    target_dir="$dir/trials/$trial_num"
    echo "[$trial_num] Computing v2 vs frac for file $yaml_file by running $path_to_src/src/get_v2_vs_frac.py"
    python3 "$path_to_src/src/get_v2_vs_frac.py" "$yaml_file" "$target_dir/raw_yields/" "$reference_frac_dir" --multitrial
}

# Export function and necessary variables
export -f generate_cutset
export -f get_v2_vs_fracs
export dir  # make sure $dir is visible inside the function

mkdir -p "$output_dir/syst/multitrial/fit"

if [ "$do_cutset_generation" = true ]; then
    # Generate YAML file list with indices
    python3 $path_to_src/syst/multitrial/make_configs_multitrial.py $config_default -m $config_modifies_fit -fm -o $output_dir > "$output_dir/syst/multitrial/fit/log_make_cutsets.txt" 2>&1
    echo "Yaml files generated!"
fi

if [ "$do_compile_fitter" = true ]; then
    echo "Compiling InvMassFitter and VnVsMassFitter ..."
    rootcling -f $path_to_src/invmassfitter/vnfitter_dict.cxx -c /home/mdicosta/alice/hf-vn/invmassfitter/VnVsMassFitter.h /home/mdicosta/alice/hf-vn/invmassfitter/LinkDefVnFitter.h

    echo "Compiling fitter once for every trial ..."
    g++ -shared -fPIC `root-config --cflags --libs` \
        $path_to_src/invmassfitter/VnVsMassFitter.cxx $path_to_src/invmassfitter/vnfitter_dict.cxx \
        -o $path_to_src/invmassfitter/libvnfitter.so
    echo "Compilation done!"
fi

# pt_dirs=($(ls -d "$output_dir"/syst/multitrial/fit/pt_*))
pt_dirs=("$output_dir"/syst/multitrial/fit/pt_120_160)

# Find YAML files, sort numerically by trial number, one per line
for dir in "${pt_dirs[@]}"; do
    find "$dir" -maxdepth 3 -type f -name "config_trial_*.yml" \
    | sort -V \
    > "$dir/multitrial_configs.txt"
done

# Loop over pt_dirs for cutset generation, projections, fitting with simultaneous fits
for dir in "${pt_dirs[@]}"; do
    echo -e "\nProcessing directory $dir"

    # Record start time for directory
    dir_start=$(date +%s)

    # Find YAML files in the current directory
    yaml_files=($(find "$dir" -maxdepth 3 -type f -name "config_trial_*.yml"))

    # Count them
    echo "----> Number of yaml files found in $dir: $(wc -l < "$dir/multitrial_configs.txt")"

    # --- Perform projections ---
    if [ "$do_projections" = true ]; then
        log_file_projections="$dir/log_projections.txt"
        start=$(date +%s)
        python3 $path_to_src/src/proj_thn.py $config_default --multitrial_folder "$dir" --multitrial_workers $n_parallel > "$log_file_projections" 2>&1
        end=$(date +%s)
        echo -e "✅ Projections performed in $((end - start)) seconds. Starting fitting ..."
    fi

    # --- Perform fitting ---
    if [ "$do_sim_fits" = true ]; then
        log_file_fits="$dir/log_simfits.txt"
        start=$(date +%s)
        python3 $path_to_src/syst/multitrial/run_fits.py "${yaml_files[@]}" $output_dir --nproc $n_parallel > "$log_file_fits" 2>&1
        end=$(date +%s)
        echo -e "✅ Fitting done in $((end - start)) seconds."
    fi
done

# pt_dirs=("$output_dir"/syst/multitrial/fit/pt_*)
pt_dirs=("$output_dir"/syst/multitrial/fit/pt_120_160)

if [ "$do_cms_fits" = true ]; then
    log_file_fits="$dir/log_yieldfits.txt"
    start=$(date +%s)
    parallel -j "$n_parallel" \
        python3 "$path_to_src/src/get_vn_by_yield_extraction.py" "$config_default" \
        --multitrial \
        --multitrial_configs "{}/multitrial_configs.txt" \
        '>' "{}/log_yieldfits.txt" '2>&1' \
        ::: "${pt_dirs[@]}"

    end=$(date +%s)
    echo -e "----> CMS Fitting done in $((end - start)) seconds. Starting v2 vs frac extraction ..."
fi

# --- Obtain v2 vs frac ---
if [ "$do_v2_vs_frac" = true ]; then
    for dir in "${pt_dirs[@]}"; do
        # Record start time for directory
        dir_start=$(date +%s)
        log_file_v2_vs_frac="$dir/log_v2_vs_frac.txt"
        yaml_files=($(find "$dir" -maxdepth 3 -type f -name "config_trial_*.yml"))
        reference_frac_dir="$output_dir/frac/"
        start=$(date +%s)
        parallel -j $n_parallel get_v2_vs_fracs ::: "${yaml_files[@]}" ::: "$reference_frac_dir" > "$log_file_v2_vs_frac" 2>&1
        # Total time for this pt_dir
        dir_end=$(date +%s)
        echo "-----------------------------------------------"
        echo -e "✅ V2 vs frac for $dir extracted in $((dir_end - dir_start)) seconds.\n"
    done
fi

# --- Evaluate systematics and produce final plots ---
export max_chi2=10
export min_signif=5
export max_signif=1000
export force_prompt_enhanced=true
if [ "$produce_plots" = true ]; then
    echo "Producing final systematic plots ..."
    mkdir -p "$output_dir/syst/multitrial/fit/summary"
    cmd=(
        python3 "$path_to_src/syst/multitrial/produce_fit_multitrial_syst_plots.py"
        "$config_default"
        "$output_dir"
        --multitrial_type fit
        --max_chi2 "$max_chi2"
        --min_signif "$min_signif"
        --max_signif "$max_signif"
    )

    if [ "$force_prompt_enhanced" = "true" ]; then
        cmd+=(--force_prompt_enhanced)
    fi

    "${cmd[@]}" > "$output_dir/syst/multitrial/fit/summary/log_produce_plots.txt" 2>&1    
    echo "Final plots produced!"
fi
