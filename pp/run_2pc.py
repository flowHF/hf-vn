#!/usr/bin/env python3

"""
Driver for the 2PC analysis pipeline.

Usage:
    python run_2pc.py config.yaml

Pipeline:
    preprocess -> correlations

Each step can be enabled/disabled from the configuration:

operations:
  dopreprocess: true
  docorrelations: true
  dodeltaphi: true
  dofourier: true
  dosimfit: true
  dojetshape: true

If preprocessing is disabled, the correlation step uses the
existing files in paths.output_dir.

Correlation outputs are written to:
    <paths.output_dir>/<correlation.subdir>/
"""

import os
import sys
import yaml

from src.preprocess import run_preprocess
from src.correlations import run_correlations
from src.deltaphi import run_deltaphi
from src.fourier import run_fourier
from src.simfit import run_simultaneous_fit
from src.jetshape import run_jetshape

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    if len(sys.argv) != 2:
        sys.exit(__doc__)

    config_file = sys.argv[1]

    if not os.path.isfile(config_file):
        sys.exit(
            f"Configuration file does not exist: {config_file}"
        )

    with open(config_file) as f:
        raw_yaml = f.read()

    cfg = yaml.safe_load(raw_yaml)

    if not cfg:
        sys.exit("Empty configuration")

    if "output_dir" not in cfg:
        sys.exit("Missing output_dir in configuration")

    operations = cfg.get("operations", {})
    do_preprocess = bool(
        operations.get("dopreprocess", False)
    )
    do_corrections = bool(
        operations.get("docorrelations", False)
    )
    do_deltaphi = bool(
        operations.get("dodeltaphi", False)
    )
    do_fourier = bool(
        operations.get("dofourier", False)
    )
    do_simfit = bool(
        operations.get("dosimfit", False)
    )
    do_jetshape = bool(
        operations.get("dojetshape", False)
    )


    print("=" * 70)
    print(" 2PC ANALYSIS")
    print("=" * 70)
    print(f"Config        : {config_file}")
    print(f"Preprocess    : {do_preprocess}")
    print(f"Correlations  : {do_corrections}")
    print(f"deltaphi vs mass    : {do_deltaphi}")
    print(f"Fourier       : {do_fourier}")
    print(f"Simultaneous Fit: {do_simfit}")
    print(f"Jet Shape     : {do_jetshape}")

    # ------------------------------------------------------------------
    # Step 1: preprocessing
    # ------------------------------------------------------------------

    if do_preprocess:
        run_preprocess(config_file)
    else:
        print()
        print(
            "[preprocess] Skipping -- "
            "using existing files in "
            f"{cfg['output_dir']}/preprocessed"
        )

    # ------------------------------------------------------------------
    # Step 2: correlations
    # ------------------------------------------------------------------

    if do_corrections:
        run_correlations(cfg)
    else:
        print()
        print("[correlations] Skipping")

    #------------------------------------------------------------------
    # Optional: Δφ vs mass maps
    #------------------------------------------------------------------

    if do_deltaphi:
        run_deltaphi(cfg)
    else:
        print()
        print("[deltaphi] Skipping")

    #------------------------------------------------------------------
    # Step 3: Fourier analysis
    #------------------------------------------------------------------

    if do_fourier:
        run_fourier(cfg)
    else:
        print()
        print("[fourier] Skipping")

    #------------------------------------------------------------------
    # Step 4: Simultaneous fit
    #------------------------------------------------------------------

    if do_simfit:
        run_simultaneous_fit(cfg)
    else:
        print()
        print("[simfit] Skipping")

    #------------------------------------------------------------------
    # Check 0: Jet shape analysis
    #------------------------------------------------------------------
    if do_jetshape:
        run_jetshape(cfg)

    print()
    print("=" * 70)
    print(" DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()