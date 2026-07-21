#!/bin/bash

# ==============================================================================
# OPTIONS
# ==============================================================================

GAIN_EQ=false
Q_VEC=false
QA_PLOTS=false
LOAD_CCDB=false

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
        --ccdb)
            LOAD_CCDB=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gain] [--qvec] [--qa] [--ccdb]"
            exit 1
            ;;
    esac
done

echo "GAIN_EQ   = $GAIN_EQ"
echo "Q_VEC     = $Q_VEC"
echo "QA_PLOTS  = $QA_PLOTS"
echo "LOAD_CCDB = $LOAD_CCDB"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_FILE="run_sor_eor_pbpb_2025.txt"

export OUTDIR="/home/mdicosta/DMesonEsE/RedQCalib/2025_pass1"
export INDIR="/data/mdicosta/DplusFlow/EsE/Train687991"

# Number of simultaneous ROOT jobs
N_JOBS=8

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
calibGain("$OUTDIR", "$INDIR", $run, true);
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
gSystem->CompileMacro("calibGain.C","kO");
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

calibQVecs("$OUTDIR", "$INDIR", $run,
           "_id54810", "RefB",
           "_id54810", "RefA",
           "_id54812", "RefB",
           "_id54812", "",
           "_id54811", "RefA",
           "_id54811", "RefB",
           "_id54810", "",
           true);

calibQVecs("$OUTDIR", "$INDIR", $run,
           "_id54810", "RefB",
           "_id54810", "RefA",
           "_id54812", "RefB",
           "_id54812", "",
           "_id54811", "RefA",
           "_id54811", "RefB",
           "_id54810", "",
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
gSystem->CompileMacro("calibQVecs.C","kO");
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
gSystem->CompileMacro("produceQa.C","kO");
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

    echo "CCDB Upload -> RUN=$run"

    root -b -l <<EOF
gSystem->Load("loadCorrectionsToCCDB_C.so");

loadCorrectionsToCCDB(
    "$OUTDIR",
    $run,
    $sor,
    $eor,
    "Users/m/mdicosta/QVecCalib/2025_pass1_v1",
    true,
    true,
    false);

.q
EOF
}
export -f load_run_corrections_to_ccdb

if [ "$LOAD_CCDB" = true ]; then

    echo "================================================="
    echo "Uploading corrections to CCDB"
    echo "================================================="

    root -b -l <<'EOF'
gSystem->SetBuildDir(".", kTRUE);
gSystem->CompileMacro("loadCorrectionsToCCDB.C","kO");
.q
EOF

    tail -n +2 "$INPUT_FILE" | \
        parallel -j ${N_JOBS} --colsep '\s+' \
        load_run_corrections_to_ccdb {1} {2} {3}
fi

echo
echo "================================================="
echo "All requested tasks completed successfully"
echo "================================================="
