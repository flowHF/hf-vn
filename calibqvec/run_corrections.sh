#!/bin/bash

# ==============================================================================
# OPTIONS
# ==============================================================================

GAIN_EQ=false
Q_VEC=false
QA_PLOTS=false
LOAD_CCDB_GAIN=false
LOAD_CCDB_SP=false
LOAD_CCDB_ESE=false
APASS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gain)
            GAIN_EQ=true
            shift
            ;;
        --qvec)
            Q_VEC=true
            shift
            ;;
        --qa)
            QA_PLOTS=true
            shift
            ;;
        --ccdb_gain)
            LOAD_CCDB_GAIN=true
            shift
            ;;
        --ccdb_sp)
            LOAD_CCDB_SP=true
            shift
            ;;
        --ccdb_ese)
            LOAD_CCDB_ESE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gain] [--qvec] [--qa] [--ccdb_gain] [--ccdb_sp] [--ccdb_ese]"
            exit 1
            ;;
    esac
done

echo "GAIN_EQ   = $GAIN_EQ"
echo "Q_VEC     = $Q_VEC"
echo "QA_PLOTS  = $QA_PLOTS"
echo "LOAD_CCDB_GAIN = $LOAD_CCDB_GAIN"
echo "LOAD_CCDB_SP   = $LOAD_CCDB_SP"
echo "LOAD_CCDB_ESE  = $LOAD_CCDB_ESE"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_FILE="/home/mdicosta/alice/hf-vn/calibqvec/run_sor_eor_oo_2025.txt"

export OUTDIR="/home/mdicosta/DFlowOO/Paper/MCClosure/QVecCalib_test"
export INDIR_GAIN="/home/mdicosta/DFlowOO/Paper/MCClosure/TrainsGainEqCalib/BTOT_FT0C_FT0A___FV0A_FT0A_FT0M/Train747387/"
export INDIR_QVECCORR="/dummy/path/"

# Create output directory
mkdir -p "$OUTDIR"

# Copy this file to the output directory for reference
cp "$0" "$OUTDIR/$(basename "$0")"

# Number of simultaneous ROOT jobs
N_JOBS=1

# ==============================================================================
# BLOCK 1: GAIN EQUALISATION
# ==============================================================================

run_gain_equalisation_corrections() {

    local run=$1
    local sor=$2
    local eor=$3

    echo "Gain Equalisation -> RUN=$run"

    root -b -l <<EOF
gSystem->Load("calibGain_C.so");
calibGain("$OUTDIR", "$INDIR_GAIN", $run, true);
.q
EOF
}
export -f run_gain_equalisation_corrections

if [ "$GAIN_EQ" = true ]; then

    echo "================================================="
    echo "Starting Gain Equalisation"
    echo "================================================="

    root -b -l <<'EOF'
gSystem->SetBuildDir(".", kTRUE);
gSystem->CompileMacro("/home/mdicosta/alice/hf-vn/calibqvec/calibGain.C","kO");
.q
EOF

    tail -n +2 "$INPUT_FILE" | \
        parallel -j ${N_JOBS} --colsep '\s+' \
        run_gain_equalisation_corrections {1} {2} {3}
fi

# ==============================================================================
# BLOCK 2: Q-VECTOR CORRECTIONS
# ==============================================================================

run_qvec_corrections() {

    local run=$1
    local sor=$2
    local eor=$3

    echo "Q-Vector -> RUN=$run"

    root -b -l <<EOF
gSystem->Load("calibQVecs_C.so");

calibQVecs("$OUTDIR", "$INDIR_QVECCORR", $run,
           "_id60252", "RefA",
           "_id60251", "",
           "_id60252", "RefB",
           "_id60252", "",
           "_id60251", "RefA",
           "_id60251", "RefB",
           "_id60250", "",
           true);

calibQVecs("$OUTDIR", "$INDIR_QVECCORR", $run,
           "_id60252", "RefA",
           "_id60251", "",
           "_id60252", "RefB",
           "_id60252", "",
           "_id60251", "RefA",
           "_id60251", "RefB",
           "_id60250", "",
           false);

.q
EOF
}
export -f run_qvec_corrections

if [ "$Q_VEC" = true ]; then

    echo "================================================="
    echo "Starting Q-Vector Corrections"
    echo "================================================="

    root -b -l <<'EOF'
gSystem->SetBuildDir(".", kTRUE);
gSystem->CompileMacro("/home/mdicosta/alice/hf-vn/calibqvec/calibQVecs.C","kO");
.q
EOF

    tail -n +2 "$INPUT_FILE" | \
        parallel -j ${N_JOBS} --colsep '\s+' \
        run_qvec_corrections {1} {2} {3}
fi

# ==============================================================================
# BLOCK 3: QA PLOTS
# ==============================================================================

if [ "$QA_PLOTS" = true ]; then

    echo "================================================="
    echo "Producing QA plots"
    echo "================================================="

    root -b -l <<'EOF'
gSystem->SetBuildDir(".", kTRUE);
gSystem->CompileMacro("/home/mdicosta/alice/hf-vn/calibqvec/produceQa.C","kO");
.q
EOF

    root -b -l <<EOF
gSystem->Load("produceQa_C.so");
produceQa("$OUTDIR", "$INPUT_FILE");
.q
EOF

fi

# ==============================================================================
# BLOCK 4: LOAD TO CCDB
# ==============================================================================

load_run_corrections_to_ccdb() {

    local run=$1
    local sor=$2
    local eor=$3
    local load_gain=$4
    local load_sp=$5
    local load_ese=$6
    local apass=$7

    # Check that apass != ""
    if [ -z "$apass" ]; then
        echo "Error: 'apass' is empty, cannot load to CCDB without specifying an apass."
        return 1
    fi

    echo "CCDB Upload -> RUN=$run"

    root -b -l <<EOF
gSystem->Load("loadCorrectionsToCCDB_C.so");

loadCorrectionsToCCDB(
    "$OUTDIR",
    $run,
    $sor,
    $eor,
    "Users/m/mdicosta/QVecCalib/LHC25j10_v1",
    $load_gain,
    $load_sp,
    $load_ese,
    "$apass");

.q
EOF
}
export -f load_run_corrections_to_ccdb

if [ "$LOAD_CCDB_GAIN" = true ] || [ "$LOAD_CCDB_SP" = true ] || [ "$LOAD_CCDB_ESE" = true ]; then

    echo "================================================="
    echo "Uploading corrections to CCDB"
    echo "================================================="

    root -b -l <<'EOF'
gSystem->SetBuildDir(".", kTRUE);
gSystem->CompileMacro("/home/mdicosta/alice/hf-vn/calibqvec/loadCorrectionsToCCDB.C","kO");
.q
EOF

    tail -n +2 "$INPUT_FILE" | \
        parallel -j ${N_JOBS} --colsep '\s+' \
        load_run_corrections_to_ccdb {1} {2} {3} ${LOAD_CCDB_GAIN} ${LOAD_CCDB_SP} ${LOAD_CCDB_ESE} ${APASS}
fi

echo
echo "================================================="
echo "All requested tasks completed successfully"
echo "================================================="
