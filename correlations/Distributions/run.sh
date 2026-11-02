OPTION="-b --configuration json://cfg.json"
MEMORY="--aod-memory-rate-limit 4000000000 --shm-segment-size 24000000000"
# CPU="--pipeline hf-correlator-charm-hadrons-reduced:1"
# export OMP_NUM_THREADS=8
LOGFILE="log.txt"

o2-analysis-hf-correlator-flow-charm-hadrons-reduced $OPTION $MEMORY --aod-writer-json OutputDir.json --aod-file @input_derived_data.txt 1 > $LOGFILE 2>&1

# report status
OUTPUT="/home/wuct/MetaData/DATA/OO/apass2/corr/results/fthlook_Biao/large/SE/k40100/"
mkdir -p $OUTPUT
rc=$?
if [ $rc -eq 0 ]; then
  echo "No problems!"
  mv AnalysisResults.root $OUTPUT/AnalysisResults_oriBDT_fulldeltaEta.root #/AnalysisResults_0d8DeltaEta.root
  echo "Results moved to $OUTPUT" #AnalysisResults_0d8DeltaEta.root
else
  echo "Error: Exit code $rc"
  echo "Check the log file $LOGFILE"
  exit $rc
fi

# touch ao2ds_to_merge.txt
# echo AO2D.root > ao2ds_to_merge.txt
# o2-aod-merger --input ao2ds_to_merge.txt --max-size 1 --output Tree.root
# rm ao2ds_to_merge.txt
# rm localhost*
