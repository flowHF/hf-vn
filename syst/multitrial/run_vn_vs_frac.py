#!/usr/bin/env python3

from pathlib import Path
import shutil
import argparse
from itertools import product
import os
import sys
import yaml
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../src")
from data_driven_fraction import main_data_driven_frac  # pylint: disable=import
from get_v2_vs_frac import main_v2_vs_frac  # pylint: disable=import

parser = argparse.ArgumentParser()
parser.add_argument("reference_cfg", metavar="text", help="path to the cut variation file")
parser.add_argument("base_dir", help="Directory containing trials_cutset_*")
parser.add_argument('--n_trials', "-n", type=int, default=1000, help='number of trials to consider for the multitrial systematic (default: 1000)')
parser.add_argument("--move_files", action='store_true', help="Enable debug output")
parser.add_argument("--do_df_frac", action='store_true', help="Enable debug output")
parser.add_argument("--do_vn_vs_frac", action='store_true', help="Enable debug output")
args = parser.parse_args()

with open(args.reference_cfg, 'r') as ymlCfgFile:
    ref_cfg = yaml.load(ymlCfgFile, yaml.FullLoader)

ref_folder = f"{ref_cfg['outdir']}/vn_extr_{ref_cfg['suffix']}_combined/"

base_dir = Path(args.base_dir)
trials_dir = base_dir / "trials"
trials_dir.mkdir(exist_ok=True)

# Discover cutsets
cutsets = sorted(base_dir.glob("trials_cutset_*"))

total_trials = 0
bkg_dirs_list = []
for i_cutset, cutset_path in enumerate(cutsets):
    print(f"\nProcessing cutset {i_cutset}: {cutset_path}")
    bkg_dirs = sorted(cutset_path.glob("bkg_*"))
    bkg_dirs_list.append(bkg_dirs)
    if i_cutset == 0:
        total_trials = len(bkg_dirs)
    else: # Cover the combinatorics
        total_trials = total_trials * len(bkg_dirs)

# Perform all the combinatorial combinations of trials from different cutsets
trials = list(product(*bkg_dirs_list))

# Select randomly n_trials if needed
if args.n_trials < len(trials):
    print(f"\nSelecting {args.n_trials} random trials out of {len(trials)} total")
    trials = random.sample(trials, args.n_trials)

def copy_trial_files(i_trial, trial, trials_dir):
    print(f"[COPY] Trial {i_trial} starting")
    dst_root = trials_dir / str(i_trial)
    dst_root.mkdir(parents=True, exist_ok=True)

    for i_cutset, src_dir in enumerate(trial):
        for src_file in src_dir.rglob("*_00.root"):
            if not src_file.is_file():
                continue

            rel_path = src_file.relative_to(src_dir)

            # replace suffix on the PATH STRING, not Path.replace()
            new_name = rel_path.name.replace("_00.root", f"_0{i_cutset}.root")
            dst_file = dst_root / rel_path.parent / new_name

            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # print(f"Copying {src_file} → {dst_file}")
            shutil.copy2(src_file, dst_file)

if args.move_files:
    print(f"\nCopying files using 20 workers\n")

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(copy_trial_files, i_trial, trial, trials_dir)
            for i_trial, trial in enumerate(trials)
        ]

        for f in as_completed(futures):
            f.result()  # raise errors immediately

def run_df_frac_trial(i_trial, trials_dir, ref_folder):
    trial_dir = trials_dir / str(i_trial)
    eff_path = trial_dir / "effs"

    print(f"[DF-FRAC] Trial {i_trial} starting")
    main_data_driven_frac(
        cutVarFile=f"{ref_folder}/cutVar/cutVar.root",
        effPath=eff_path,
        batch=True,
        outputDir=trial_dir
    )
    print(f"[DF-FRAC] Trial {i_trial} done")

if args.do_df_frac:
    print(f"\nRunning data-driven fraction with {20} workers\n")

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(run_df_frac_trial, i_trial, trials_dir, ref_folder)
            for i_trial in range(len(trials))
        ]

        for f in as_completed(futures):
            f.result()  # propagate exceptions

def run_vn_vs_frac_trial(i_trial, trials_dir, trial_cfg):
    trial_dir = trials_dir / str(i_trial)

    print(f"[VN-FRAC] Trial {i_trial} starting")
    main_v2_vs_frac(
        trial_cfg,
        f"{trial_dir}/raw_yields",
        f"{trial_dir}/frac",
        False,
        True,
        False,
        ""
    )
    print(f"[VN-FRAC] Trial {i_trial} done")

if args.do_vn_vs_frac:
    print(f"\nRunning Vn vs fraction with {20} workers\n")

    trials_cfg = f"{base_dir}/config_reference.yml"
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(run_vn_vs_frac_trial, i_trial, trials_dir, trials_cfg)
            for i_trial in range(len(trials))
        ]

        for f in as_completed(futures):
            f.result()



# if args.move_files:
#     # Copy files for each trial combination
#     for i_trial, trial in enumerate(trials):
#         print(f"\nProcessing trial {i_trial + 1}/{len(trials)}")
#         os.makedirs(f"{trials_dir}/{i_trial}", exist_ok=True)
#         for i_cutset, src_dir in enumerate(trial):
#             dst_dir = f"{trials_dir}/{i_trial}"
#             for src_file in src_dir.rglob(f"*_0{i_cutset}.root"):
#                 if not src_file.is_file():
#                     continue
#                 # Preserve structure *inside* bkg_*
#                 rel_path = src_file.relative_to(src_dir)
#                 dst_file = dst_dir / rel_path
#                 dst_file.parent.mkdir(parents=True, exist_ok=True)
#                 shutil.copy2(src_file, dst_file)


# if args.do_df_frac:
#     # Run data-driven fraction extraction for each trial
#     for i_trial in range(len(trials)):
#         trial_dir = trials_dir / str(i_trial)
#         eff_path = trial_dir / "effs"

#         print(f"\nRunning data-driven fraction extraction for trial {i_trial}")
#         main_data_driven_frac(
#             cutVarFile=f"{ref_folder}/cutVar/cutVar.root",
#             effPath=eff_path,
#             batch=True,
#             outputDir=trial_dir
#         )


# if args.do_vn_vs_frac:
#     for i_trial in range(len(trials)):
#         trial_dir = trials_dir / str(i_trial)
#         print(f"\nRunning Vn vs fraction extraction for trial {i_trial}")
#         main_v2_vs_frac(
#             reference_cfg,
#             f"{trial_dir}/raw_yields",
#             f"{trial_dir}/frac",
#             False,
#             True,
#             False,
#             ""
#         )
