#!/bin/bash

load_reso_file_to_ccdb() {

    local reso_file_path=$1
    local ccdb_file_path=$2
    local timestamp_file=$3
    local apass=$4
    local year=$5
    LOAD=$6

    echo "CCDB Upload -> Reso File=$reso_file_path, Timestamp File=$timestamp_file ---> CCDB Path=$ccdb_file_path"

    root -b -l <<EOF
gSystem->Load("LoadResolutionToCCDB_C.so");

LoadResolutionToCCDB(
    "$reso_file_path",
    "$ccdb_file_path",
    "$timestamp_file",
    "$apass",
    $year,
    $LOAD
    );

.q
EOF
}
export -f load_reso_file_to_ccdb

echo "================================================="
echo "Uploading resolution to CCDB"
echo "================================================="

root -b -l <<'EOF'
gSystem->SetBuildDir(".", kTRUE);
gSystem->CompileMacro("LoadResolutionToCCDB.C","kO");
.q
EOF

# load_reso_file_to_ccdb '/data/shared/ScalarProdResolutions/2023/Train748160/reso_0100.root' \
#                        'Users/m/mdicosta/Test/UploadResolutions/' \
#                        '/home/mdicosta/alice/hf-vn/datasets_info/run_sor_eor_pbpb_2023.txt' \
#                        'apass4' \
#                        2023 \
#                        true

# load_reso_file_to_ccdb '/data/shared/ScalarProdResolutions/2024/Train748051/reso_good_runs_0100.root' \
#                        'Users/m/mdicosta/Test/UploadResolutions/' \
#                        '/home/mdicosta/alice/hf-vn/datasets_info/run_sor_eor_pbpb_2024.txt' \
#                        'apass3' \
#                        2024 \
#                        true

# load_reso_file_to_ccdb '/data/shared/ScalarProdResolutions/2025/Train748050/reso_0100.root' \
#                        'Users/m/mdicosta/Test/UploadResolutions/' \
#                        '/home/mdicosta/alice/hf-vn/datasets_info/run_sor_eor_pbpb_2025.txt' \
#                        'apass1' \
#                        2025 \
#                        true

load_reso_file_to_ccdb '/data/shared/ScalarProdResolutions/2026/Train787448/reso_0100.root' \
                       'Users/m/mdicosta/Test/UploadResolutions/' \
                       '/home/mdicosta/alice/hf-vn/datasets_info/run_sor_eor_pbpb_2026.txt' \
                       'apass1' \
                       2026 \
                       true
