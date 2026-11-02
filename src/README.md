# Output

- [compute_efficiencies.py](https://github.com/flowHF/hf-vn/blob/dev/src/compute_efficiencies.py)
  - output directory: `{outdir}/vn_extr_{suffix}_{method}/eff`
  - output Name: `eff_{iCut}.root`

- [cut_variation.py](https://github.com/flowHF/hf-vn/blob/dev/src/cut_variation.py)
  - output directory: `{outdir}/vn_extr_{suffix}_{method}/cutVar`
  - output Name: `cutVar.root`

- [data_driven_fraction.py](https://github.com/flowHF/hf-vn/blob/dev/src/data_driven_fraction.py)
  - output directory: `{outdir}/vn_extr_{suffix}_{method}/frac`
  - output Name: `frac_{iCut}.root`

- [get_v2_vs_frac.py](https://github.com/flowHF/hf-vn/blob/dev/src/get_v2_vs_frac.py)
  - output directory: `{outdir}/vn_extr_{suffix}_{method}/v2`
  - output Name: `v2VsFrac.root`

- [get_vn_vs_mass.py](https://github.com/flowHF/hf-vn/blob/dev/src/get_vn_vs_mass.py)
  - output directory: `{outdir}/vn_extr_{suffix}_{method}/ry`
  - output Name: `raw_yields_{iCut}.root`

- [make_cutsets_cfgs.py](https://github.com/flowHF/hf-vn/blob/dev/src/make_cutsets_cfgs.py)
  - output directory: `{outdir}/vn_extr_{suffix}_{method}/cutsets`
  - output Name: `cutsets_{iCut}.yml`

- [proj_thn.py](https://github.com/flowHF/hf-vn/blob/dev/src/proj_thn.py)
  - output directory: `{outdir}/vn_extr_{suffix}_{method}/proj_thn`
  - output Name: `proj_{iCut}.root`

- [pre_process.py](https://github.com/flowHF/hf-vn/blob/dev/src/pre_process.py)
  - output directory: `{outdirPrep}/preprocess`
  - output Name: `AnalysisResults_pt_{ptMin*10}_{ptMax*10}.root`