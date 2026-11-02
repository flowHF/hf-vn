#!/bin/bash

# \brief Bash script to run the azimuthal correlation analysis
# \usage ./AnalysisExecution.sh

# Create the output directory
# gSystem->Exec("rm -f DhCorrelationFitter_cxx*")
# gSystem->Exec("rm -f FitCorrel_C*")
log_file="Output_AnalysisExecution/stdoutFitCorrel_010_negDeta.log"
# add the include path for yaml-cpp and load the library
# gSystem->AddIncludePath("-I/home/wuct/Software/miniforge3/envs/alice/include");
# gSystem->Load("/home/wuct/Software/miniforge3/envs/alice/lib/libyaml-cpp.so");
root -b -l <<'EOF' 2>&1 | tee "$log_file"
gSystem->SetBuildDir(".", kTRUE)
.L DhCorrelationFitter.cxx++
.L FitCorrel.C++
.x FitCorrel.C("config_CorrAnalysis_v2_010_negDeta.json")
.q
EOF

