#!/usr/bin/env python3

import argparse
import os
from multiprocessing import Pool
from alive_progress import alive_bar
script_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.join(script_dir, '../..', 'src'))
from get_vn_vs_mass import get_vn_vs_mass
from ROOT import gSystem
os.sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../utils")
from utils import merge_cutsets_fits
from pathlib import Path

# Load your precompiled library once
script_dir = os.path.dirname(os.path.realpath(__file__))
gSystem.Load(f"{script_dir}/../../invmassfitter/libvnfitter.so")

def fit_proj(args):
    yaml_file, cutset_file, proj_file = args
    print(f"Fitting projection file: {proj_file} with config: {yaml_file} and cutset: {cutset_file}")
    get_vn_vs_mass(yaml_file, cutset_file, proj_file, batch=True, isMultitrial=True) # isMultitrial=False)

def merge_trial(yaml_file):
    trial_dir = os.path.dirname(yaml_file)
    raw_dir = Path(trial_dir) / "raw_yields"

    if not raw_dir.exists():
        print(f"Raw yields dir {raw_dir} does not exist. Skipping.")
        return

    print(f"Merging cutsets in {raw_dir}")
    merge_cutsets_fits(raw_dir, verbose=False)

def main():
    parser = argparse.ArgumentParser(description="Run vn vs mass for multiple YAMLs")
    parser.add_argument('yaml_files', nargs='+', help='List of YAML config files')
    parser.add_argument('reference_dir', help='Directory containing reference files')
    parser.add_argument('--nproc', type=int, default=8, help='Number of parallel processes')
    args = parser.parse_args()

    # Prepare all (yaml, proj_file) pairs
    all_tasks = []
    for yaml_file in args.yaml_files:
        trial_dir = os.path.dirname(yaml_file)
        proj_dir = os.path.join(trial_dir, "projs")
        # check if proj_dir exists
        if not os.path.exists(proj_dir):
            print(f"Projection directory {proj_dir} does not exist. Skipping.")
            continue
        proj_files = sorted([os.path.join(proj_dir, f)
                             for f in os.listdir(proj_dir) if f.startswith("proj_") and f.endswith(".root")])
        cutset_dir = os.path.join(args.reference_dir, "cutsets")
        cutset_files = sorted([os.path.join(cutset_dir, f) for f in os.listdir(cutset_dir) if f.startswith("cutset_") and f.endswith(".yml")])
        all_tasks.extend([(yaml_file, cutset_file, pf) for cutset_file, pf in zip(cutset_files, proj_files)])

    # Parallel execution
    with Pool(args.nproc) as pool:
        for i, _ in enumerate(pool.imap_unordered(fit_proj, all_tasks), 1):
            print(f"[{i}/{len(all_tasks)}] Completed")

    # Parallel merging (after all fits are done)
    with Pool(args.nproc) as pool:
        for i, _ in enumerate(pool.imap_unordered(merge_trial, args.yaml_files), 1):
            print(f"[{i}/{len(args.yaml_files)}] Merge completed")

if __name__ == "__main__":
    main()
