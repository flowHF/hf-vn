#!/bin/bash

export OUTDIR="/home/mdicosta/DMesonEsE/RedQCalib/2025_pass1_v2"
export INDIR="/data/mdicosta/DzeroEsE/Train733368/CheckCalibs/"

mkdir -p "$OUTDIR"

# Copy this file to the output directory for reference
cp "$0" "$OUTDIR/$(basename "$0")"

# Number of simultaneous ROOT jobs
N_JOBS=1

INPUT_FILE="/home/mdicosta/alice/hf-vn/calibqvec/run_sor_eor_pbpb_2025.txt"

# wagonSuffixFT0A
# wagonSuffixFT0C
# wagonSuffixFT0M
# wagonSuffixFV0A
# wagonSuffixTPCPOS
# wagonSuffixTPCNEG
# wagonSuffixTPCALL

run_calibrations_check() {

    local run=$1

    echo "Q-Vector -> RUN=$run"

    root -b -l <<EOF
gSystem->Load("checkCalibrations_C.so");

checkCalibrations("$OUTDIR", "$INDIR", $run,
                  "_id60687", "RefB",
                  "_id60687", "RefA",
                  "_id60689", "RefB",
                  "_id60689", "",
                  "_id60688", "RefA",
                  "_id60688", "RefB",
                  "_id60687", "",
                  true);

checkCalibrations("$OUTDIR", "$INDIR", $run,
                  "_id60687", "RefB",
                  "_id60687", "RefA",
                  "_id60689", "RefB",
                  "_id60689", "",
                  "_id60688", "RefA",
                  "_id60688", "RefB",
                  "_id60687", "",
                  false);
.q
EOF
}
export -f run_calibrations_check

echo "================================================="
echo "Starting Q-Vector Calibrations checks"
echo "================================================="

root -b -l <<'EOF'
gSystem->SetBuildDir(".", kTRUE);
gSystem->CompileMacro("/home/mdicosta/alice/hf-vn/calibqvec/checkCalibrations.C","kO");
.q
EOF

tail -n +2 "$INPUT_FILE" | \
    parallel -j ${N_JOBS} --colsep '\s+' \
    run_calibrations_check {1}

echo
echo "================================================="
echo "Calibration checks completed successfully."
echo "================================================="
