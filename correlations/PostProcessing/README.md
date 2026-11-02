# Post-processing macros for angular correlation part

## Table of contents
1. [Code description](#project_overview)
2. [Execution](#execution)
3. [debug.ipynb — full workflow driver](#debug_nb)
4. [debug_sys.ipynb — systematic trial driver](#debug_sys_nb)
5. [syst/ — correlation systematic toolchain](#syst_dir)

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

## Execution <a name="execution"></a> (**deprecated**)
The configuration parameters are defined in dedicated config yaml files, organized by centrality bin.
(In-use configs: `config_CorrAnalysis_v2_010_negDeta.yaml`, `config_CorrAnalysis_v2_010_sys.yaml`,
`config_CorrAnalysis_v2_20_50.yaml`, `config_sys_trails.yaml`; the old per-centrality `*.json`
configs and `config_2_3.yaml` were dropped in the 2026-09 cleanup.)

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

- configure `config_CorrAnalysis_v2_010_negDeta.yaml` first, then execute:
    ```
    python3 correlations/PostProcessing/ExtractOutputCorrel.py correlations/PostProcessing/config_CorrAnalysis_v2_010_negDeta.yaml
    ```
    to obtain the correlation distributions.
- execute:
    ```
    python3 src/ry_interface.py correlations/PostProcessing/config_CorrAnalysis_v2_010_negDeta.yaml
    ```
    to perform the mass fits and extract the raw yields.
- execute:
    ```
    python3 correlations/PostProcessing/FitCorrel.py correlations/PostProcessing/config_CorrAnalysis_v2_010_negDeta.yaml
    ```
    to perform the correlation fits and extract the vn values.

**Or** execute the full workflow which locates in `correlations/PostProcessing/debug.ipynb` (note that the configuration files `config_CorrAnalysis_v2_010_negDeta.yaml` need to be set up properly before running the notebook):
    ```
    jupyter notebook correlations/PostProcessing/debug.ipynb
    ```

---

## debug.ipynb — full workflow driver <a name="debug_nb"></a>

Two cells. Drives the whole 2PC chain (extract → mass fit → correlation fit → final results →
prompt v2) over a list of Δη ranges, one generated config per range. Each stage is launched as a
subprocess and logged next to its config, so it is a thin orchestration layer over
`ExtractOutputCorrel.py`, `src/ry_interface.py`, `FitCorrel.py`, `construct_v2_mass.py`,
`src/get_vn_vs_mass.py`.

### Cell 0 — stage driver

* `run_command(cmd, log, label)` — runs one stage, writes stdout/stderr into `log`, prints a
  warning if the exit code is non-zero (failures do not stop the loop).
* `task_modify_config(config_path, inner_edge, outer_edge, task_LM=None)` — deep-copies the base
  config, sets `deltaEtaBins = [[-outer, -inner], [inner, outer]]`, sets `rebinDeltaPhi = 4` for
  `method: MassBinning`, and rewrites `suffix` as `{inner}d{outer}_{final_suffix}`. `final_suffix`
  is set by the banner-marked line in this cell (`AppDeltaPhi` for Δφ binning, `AppMass` for mass
  binning). With `task_LM=` the SE/ME/mass input files, `outdir` and `nDeltaPhiBins` are re-pointed
  to the `task_LM` block of the base config (used to build the LM template). The generated config
  is written to `{outdir}/CorrelExtract_{suffix}/{base_name}_{suffix}.yaml`.
* Stage functions (DeltaPhiBinning path in parentheses):

  | function | what it runs |
  |---|---|
  | `task_extract_correl` | `python3 ExtractOutputCorrel.py <cfg>` — correlation distributions |
  | `task_fit_mass` | `python3 src/ry_interface.py <cfg>` — mass fits, `PairYieldsVsPhi.root` (DeltaPhiBinning) |
  | `task_extract_ry_trigger` | `construct_v2_mass.py extract-ry-trigger <cfg>` (MassBinning only, before FitCorrel) |
  | `task_fit_correl` | `python3 FitCorrel.py <cfg>` — `CorrPhiD0_FinalPlots.root` |
  | `task_construct_v2_mass` | `construct_v2_mass.py build-mass-v2` (MassBinning only) |
  | `task_simfit` | `get_vn_vs_mass.py` per associated-pT bin (MassBinning only) |
  | `produce_final_results` | `CorrPhiD0_FinalPlots.root` → `final_results.root`, every histogram scaled by 1/0.07 |

* Main loop: `for inner_edge, outer_edge in product(list_inner_edges, list_outer_edges) + special_cases`
  (pairs with `outer <= inner + 0.1` are skipped). Which stages actually run is controlled by
  comment/uncomment inside the loop — as committed, only **Step 3 (correlation fit)** and
  **Produce final results** are active.

### Cell 1 — packaging

Derives a results suffix from the generated config
(`temp{raw|rawspline|smoothspline|Gaus|GausPeriodic}_{fixed|free}_{BL|noBL}`, optionally prefixed
with `hand_suffix`, e.g. `k2050_rebin_loose_k60100`), then moves every entry of each
`CorrelExtract_*` directory except `CorrelationsResults*` and `results_*` into
`CorrelExtract_*/results_{suffix}/`.

### Usage

1. Start Jupyter **from `correlations/PostProcessing`** — cell 0 derives `PROJECT_ROOT` from
   `os.getcwd()/../..`.
2. In cell 0 set:
   * `config` — the base config (`config_CorrAnalysis_v2_010_negDeta.yaml`, or
     `config_CorrAnalysis_v2_20_50.yaml` for the 20–50 bin),
   * `list_inner_edges` / `list_outer_edges` — the Δη inner/outer edges to scan
     (e.g. `["0.2", "0.3", "0.4"]` × `["1.3"]`),
   * `special_cases` — extra (inner, outer) pairs to append,
   * `hand_suffix` (cell 1) — free label put in front of the results directory name.
3. Run cell 0, then cell 1 to package the results.

---

## debug_sys.ipynb — systematic trial driver <a name="debug_sys_nb"></a>

Eight cells. Produces the systematic-trial variations (mass-fit × correlation-fit parameter
combinations, LM-template variations, HM/LM binning combinations) whose reduced spread becomes the
fit-procedure systematic. **Cells must be run in order**; every cell reuses the objects defined in
the previous ones (`config`, `trial_config`, `config_trials`, `main`, …).

* **cell 0 — `main(config, list_inner_edges, list_outer_edges, special_cases=[], doextract=True,
  do_fit_mass=True, do_fit_correl=True, do_produce_final=True, do_prompt_only=False)`**
  Same stage set as `debug.ipynb` cell 0, but every stage is switchable. `PROJECT_ROOT` is
  hardcoded to `/home/wuct/ALICE/reps/hf-vn-dev/dev`. Suffixes that already start with `trial_`
  are preserved instead of being regenerated from the Δη edges. At the end of each config it runs
  `produce_final_results` (1/0.07 scaling) and `compute_prompt_v2_unfold.py`
  (`configs/v2_prompt_v2_method_check.yml`), so each trial directory ends up with
  `final_results.root` and `v2_prompt*.root`.
* **cell 1 — `modify_config_sys(config, trial_config)`** expands `config_sys_trails.yaml`
  (per pT bin: `MassMin` × `MassMax` × `Rebin` × `BkgFunc` and `nDeltaPhiBins`) into one config per
  trial, `suffix = trial_<mM><MM><reb><bkg><ndphi>`, with `outdir = {sys_outdir}/sys_fit` and
  `task_LM.outdir = {sys_outdir}/LM_sys_fit`; the generated configs go to
  `{sys_outdir}/sys_fit/config/config_<trial_id>.yaml`. Returns `{trial_id: config_path}`. The same
  cell defines `reset_config(…)`, which regenerates all trial configs after re-pointing the results
  root.
* **cell 2 — central value + all trials.** Runs the central config first with
  `do_fit_correl=False, do_produce_final=True` (reuses the existing correlation fit and only
  re-produces `final_results.root` + prompt v2), then loops over the trial configs in parallel
  (`ThreadPoolExecutor`), staging the per-trial mass-fit outputs. Note the `sys.exit(0)` right after
  the central call: it is the "central only" escape hatch, comment it out to run the trials.
  Set `do_prompt_only=True` to skip extraction/fitting and only re-run the prompt-v2 step.
* **cell 3 — LM variation.** HM = the trial's own mass fit, LM = the **central** template; suffix
  `trial_LM_<id>`; re-runs only FitCorrel + final results + prompt v2.
* **cell 4 — unified HM/LM binning combinations.** `RUN_COMBOS = ["H16L16", "H16L32", "H32L16",
  "H32L32"]`, suffix `trial_<code>_H<hm>L<lm>`. Asymmetric combos scale the LM
  `hPairsYields_vs_DeltaPhi` by `lm_bins/hm_bins` (H32L16 → 0.5, H16L32 → 2.0, symmetric → plain
  copy); ry_trigger is untouched, so F stays unchanged.
* **cell 5** — empty (scratch).
* **cell 6 — central 32-bin LM reused by every 32-bin trial.** Builds `config_lm32_central.yaml`
  once (extract + mass fit, idempotent), then runs each 32-bin trial as suffix `trial_LM_<id>_32`.
* **cell 7 — final results + prompt v2 for the `_15`/`_32` variants.** Self-contained: re-points the
  results root (`.../fifth`) inside the variant configs, scales 1/0.07, then runs
  `produce_final_results` + `compute_prompt_v2_unfold.py` per variant.

### Usage

1. Check the central (non-sys) results exist — cells 0–2 reuse them and only vary the parts you
   switch on.
2. Configure:
   * `config` (cell 2) — `config_CorrAnalysis_v2_010_sys.yaml`; its `sys_outdir` defines where all
     trial outputs go (default
     `/home/wuct/MetaData/DATA/OO/apass2/corr/results/fifth/k020_gausPer/k60100_loose2to2d5_d20/sys`),
   * `trial_config` (cell 2) — `config_sys_trails.yaml`; the trial grid per pT bin.
3. Run cells 0 → 1 → 2, then whichever variant cells you need (3, 4, 6, 7). Cell 2's `sys.exit(0)`
   must be commented out for the trials to run; expect a long run (each trial = one mass fit +
   one correlation fit + prompt v2).
4. Reduce the trials with `syst/fit/produce_fit_syst.py` (next section) — it reads
   `{sys_outdir}/sys_fit/CorrelExtract_trial_*`.

---

## syst/ — correlation systematic toolchain <a name="syst_dir"></a>

Everything specific to the **correlation systematic uncertainty** of the prompt-v2 result lives
here (moved in during the 2026-09 re-organization; the old paths `src/compute_prompt_v2_unfold_sys.py`
and `src/compare_2pc_sp_sys.py` no longer exist — the latter was replaced by `plot/plot_2pc_sp.py`,
both are still available on the `dev_bak` branch).

```
syst/
├── fit/
│   ├── produce_fit_syst.py              # (1) fit-procedure systematic from the trial runs
│   ├── LMFitSys.py                      # (2) LM-template systematic (sampling + refit)
│   ├── LMFitSys_prompt.py               # (3) propagate (2) through the prompt-v2 unfolding
│   └── compute_prompt_v2_unfold_sys.py  # (4) FD-ratio scan systematic
└── plot/
    ├── plot_syst.py                     # (5) combine (1)(2)(4) + stat → total band
    ├── syst_config.yml                  # per-pT systematic values consumed by (5)
    └── plot_2pc_sp.py                   # (6) 2PC vs SP comparison + ratio
```

Dependencies: (1) and (2) are independent; (3) needs (2); (4) needs the central prompt-v2 inputs;
(5) needs (1) + (3) + (4) + the central `v2_prompt_ratio_0d2_1d3.root`; (6) needs (5).

### (1) Fit-procedure systematic — `fit/produce_fit_syst.py`

```bash
python3 correlations/PostProcessing/syst/fit/produce_fit_syst.py \
        correlations/PostProcessing/config_CorrAnalysis_v2_010_sys.yaml \
        [--max_chi2 5] [--max_mass_chi2 10] [--ref ref.root]
```

* Input: `{sys_outdir}/sys_fit/CorrelExtract_trial_*` written by `debug_sys.ipynb` (cells 1–2, plus
  the variant groups of cells 4/6), reading each trial's `v2_prompt*.root`, correlation-fit
  χ²/NDF and mass-fit χ²/NDF; the central `{outdir}/CorrelExtract_{suffix}/v2_prompt*.root` is the
  reference (override with `--ref`).
* Per pT bin: trials are cut on χ²/NDF (`--max_chi2`, `--max_mass_chi2`), and the systematic is
  `sqrt(mean² + rms²)` of `(v2_trial − v2_ref)` — the same formula as
  `syst/multitrial/produce_fit_multitrial_syst_plots.py` in the **repo-root** `syst/` (different
  from this one). Trials are grouped by
  the `_H32L32 / _H32L16 / _H16L32 / _H16L16` suffix; legacy names (`…2`, `_15`, `_32`, `HM_*`,
  `LM_*`) fall into no group.
* Output: `{sys_outdir}/results/fit/` — `TotalSystV2.root` (`hSystV2_vs_pT`,
  `hRelaSystV2_vs_pT`), `SystV2_vs_pT.{png,pdf}`, `RelaSystV2_vs_pT.png`, per-pT
  `pt_<i>_<j>/syst_pt_<i>_<j>.png`.

### (2) LM-template systematic — `fit/LMFitSys.py`

```bash
python3 correlations/PostProcessing/syst/fit/LMFitSys.py [--n-samples 100] [--pt-only N] [--load-from-exist]
```

* Compiles `DhCorrelationFitter.cxx` with ACLiC (path is resolved from this file's directory), then
  for each (PtCand, PtHad) bin: reads the data `CorrPhiD0…root` and the `hLMtemplate…root`, samples
  the LM parameters from their covariance with a multivariate Gaussian (`--n-samples`, default 100),
  rebuilds the `GausPeriodic` LM templates, refits with `DhCorrelationFitter` and collects the Δv2
  of every sample.
* `--load-from-exist` re-uses the already produced `LMSysResults/PtCand_i/trial_*/v2_result.txt`
  instead of refitting (use the same `--n-samples` as the original run); `--pt-only N` processes a
  single pT bin.
* Paths are hardcoded at the top of the file (`BASE = .../sys/central`,
  `EXTRACT = {BASE}/CorrelExtract_0d2_1d3`). Outputs into
  `{EXTRACT}/CorrelationFitResults/LMSysResults/`:
  `PtCand_i/trial_XXX/v2_result.txt`, `probV2_PtCand.root`, `hProbV2_PtCand*.png`,
  `v2_vs_pT_LMSys.{root,png}`, `v2Syst_vs_pT.png`, `LMSys/LMFits_PtCand*.png`.

### (3) LM systematic → prompt v2 — `fit/LMFitSys_prompt.py`

```bash
python3 correlations/PostProcessing/syst/fit/LMFitSys_prompt.py [--n-samples N]
```

* Reads the per-trial Δv2 from `LMSysResults/PtCand_i/*/v2_result.txt` together with the prompt
  fractions (`hPromptFracCorr` / `hFDFracCorr`, same objects as `compute_prompt_v2_unfold.py`,
  config `configs/v2_prompt_v2_method_check.yml`) and propagates every sample through
  `v2p = v2obs / (f_prompt + r·f_FD)` (r = 0.5, scale 1/0.07), per pT bin.
* Output: `LMSysResults/prompt/v2_vs_pT_LMSys.{root,png}` and `v2Syst_vs_pT.png` — the `hSyst`
  histogram is the `lm_syst` input of step (5) — plus `prompt/trial/final_results.root`; the
  `prompt/` directory is also copied into `{…}/sys/results`.

### (4) FD-ratio systematic — `fit/compute_prompt_v2_unfold_sys.py`

```bash
python3 correlations/PostProcessing/syst/fit/compute_prompt_v2_unfold_sys.py \
        configs/v2_prompt_v2_method_check.yml \
        [--r-step 0.05] [--r-start 0.0] [--r-end 1.0] [--central-r 0.5] \
        [--sys-method asymmetric|half-range|rms|geom-mean]
```

* Scans `r = v2_FD / v2_prompt` from `--r-start` to `--r-end` and converts the spread of the
  resulting v2_prompt curves into the systematic; `--sys-method` picks the treatment
  (`asymmetric` = max|dev| per side, `half-range` = (max−min)/2, `rms`, `geom-mean`); the central
  value and its statistical error are taken at `--central-r`.
* Output: the config's `output.file`
  (`…/k60100_loose2to2d5_d20/sys/results/v2_prompt_ratio_0d2_1d3.root`) with
  `hV2Prompt`, `gV2Prompt`, `gSysUncAsym`, `gTotUnc`, `ratio_scan/` (all individual r curves),
  plus the summary PNGs `…_syst_vs_pT.png` and `…_v2_stat_syst.png`.

### (5) Combine everything — `plot/plot_syst.py`

```bash
python3 correlations/PostProcessing/syst/plot/plot_syst.py \
        [--central-file <v2_prompt_ratio_0d2_1d3.root>] [--yaml syst_config.yml]
```

* `syst_config.yml` holds the **hand-maintained** numbers per pT bin for
  `D0 → k60100_loose2to2d5_d20`: `pt_bins`, `fit_syst` (from step 1), `lm_syst` (from step 3),
  `fd_syst_low` / `fd_syst_high` (from step 4), `fp_syst` (universal scalar). Update it after
  re-running (1)/(3)/(4).
* Draws, widest first: total box (thick outline), then fd / lm / fit bands (transparent, staggered
  widths), then the central points with statistical errors.
* Output: `syst/plot/v2_syst.{png,root}` — `gCentral`, `gFitSyst`, `gLMSyst`, `gFDSyst`, `gFPSyst`,
  `gTotal`.

### (6) 2PC vs SP comparison — `plot/plot_2pc_sp.py`

```bash
cd correlations/PostProcessing/syst/plot && python3 plot_2pc_sp.py
```

* No CLI options: reads `v2_syst.root` (2PC, this directory) and `v2_prompt_wsyst_d0_020.root`
  (SP reference, same directory), and writes `v2_compare_SP_vs_2PC.png` (with ratio panel) and
  `v2_compare_SP_vs_2PC_no_ratio.png` **into the current working directory**.
