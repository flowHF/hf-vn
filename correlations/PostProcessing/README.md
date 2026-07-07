# Post-processing macros for angular correlation part

## Table of contents
1. [Code description](#project_overview)
2. [Execution](#execution)

---

## Code description <a name="project_overview"></a>

### Project description

This project provides a framework to perform fits on angular correlation distributions. The code is organized into two main components:
  * Angular correlation extraction:
      * ExtractOutputCorrel.C
      * DhCorrelationExtraction.cxx
      * DhCorrelationExtraction.h
  * Angular correlation distribution fit:
      * FitCorrel.C
      * DhCorrelationFitter.cxx
      * DhCorrelationFitter.h
Each component handles a specific stage of the post-processing workflow, from data extraction to fitting of the final correlation distributions.

## Execution <a name="execution"></a>
The configuration parameters are defined in dedicated config.json files, organized by centrality bin.

To run the full analysis, execute:

<pre> ```bash ./AnalysisExecution.sh ``` </pre>

To merge the output files corresponding to positive and negative Δη, use:

<pre> ```bash ./run_hadd.sh ``` </pre>

## New Execution Instructions
To run the full analysis, follow these steps:
- configure `cfg.json` and `input_derived_data.txt` first, then in `correlations/Distributions/run.sh`, set `$OPTION`, `$MEMORY`, and `$OUTPUT` as needed. Finally, execute:
    ```
    bash correlations/Distributions/run.sh
    ```
    for both SE and ME.

- configure `config_CorrAnalysis_v2_010_negDeta.yaml` and `config_CorrAnalysis_v2_010_negDeta.json` first, then execute:
    ```
    python3 correlations/PostProcessing/extract_output_correl.py correlations/PostProcessing/config_CorrAnalysis_v2_010_negDeta.yaml
    ```
    to obtain the correlation distributions.
- execute:
    ```
    bash src/ry_interface.py correlations/PostProcessing/config_CorrAnalysis_v2_010_negDeta.json
    ```
    to perform the mass fits and extract the raw yields.
- execute:
    ```
    bash correlations/PostProcessing/AnalysisExecution.sh
    ```
    to perform the correlation fits and extract the vn values.

**Or** execute the full workflow which locates in `correlations/PostProcessing/debug.ipynb` (note that the configuration files `config_CorrAnalysis_v2_010_negDeta.yaml` need to be set up properly before running the notebook):
    ```
    jupyter notebook correlations/PostProcessing/debug.ipynb
    ```

