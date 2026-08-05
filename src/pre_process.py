'''
This script is used to pre-process a/multi large AnRes.root for the BDT training:
    - split the input by pT
    - obtain the sigma from prompt enhance sample
python3 pre_process.py config.yml AnRes_1.root AnRes_2.root --pre --sigma  
'''
import os
import sys
import yaml
import numpy as np
import array
from ROOT import TFile, TObject
import argparse
import gc
import concurrent.futures
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{script_dir}/")
sys.path.append(f"{script_dir}/../utils/")
from utils import get_centrality_bins, make_dir_root_file, logger, get_ese_band_label
from data_model import get_sparse_dict, get_tree_dict
import uproot
import pandas as pd
import awkward as ak

def get_ese_thresholds(quantile_file, det, quantile):
    """Return {cent_bin_int: threshold} for a given detector and quantile."""
    if quantile <= 0:
        return {c: 0.0 for c in range(0, 100)}
    if quantile >= 100:
        return {c: 999.0 for c in range(0, 100)}
    f = TFile.Open(quantile_file, 'read')
    h = f.Get(f'{det}/quantile_{quantile}_{det}')
    if not h:
        raise RuntimeError(f"Histogram {det}/quantile_{quantile}_{det} not found in {quantile_file}")
    thr = {}
    for c in range(0, 100):
        thr[c] = h.GetBinContent(h.FindBin(c + 0.5))
    f.Close()
    return thr

def check_existing_outputs(file_path):
    """
    Check if output file already exists, and open it accordingly.

    Args:
        file_path (str): path to the output ROOT file.

    Returns:
        tuple: (out_file, write_opt) where out_file is the opened TFile and write_opt is the write option.
    """
    if os.path.exists(file_path):
        logger(f"    Updating file: {file_path}", level='WARNING')
        out_file = TFile(file_path, 'update')
        write_opt = TObject.kOverwrite
    else:
        logger(f"    Creating file: {file_path}", level='WARNING')
        out_file = TFile.Open(file_path, 'recreate')
        write_opt = TObject.kSingleKey # Standard

    return out_file, write_opt

def get_inputs_sparse(file, full_cfg, sparse_cfg, debug=False):
    """Load a single sparse and axes info
    
    Args:
        file (TFile): input ROOT file
        full_cfg (dict): full configuration dictionary
        sparse_cfg (dict): sparse configuration dictionary
        debug (bool, optional): print debug info. Defaults to False.
    
    Returns:
        tuple: (sparse, axes) where sparse is the loaded sparse histogram and axes is the dictionary of axes information
    """

    axes = get_sparse_dict(sparse_cfg['name'], full_cfg['Dmeson'])
    sparse = file.Get(sparse_cfg['path'])
    if sparse is None:
        logger(f"Sparse {sparse_cfg['name']} not found in file {file.GetName()} at path {sparse_cfg['path']}", level='ERROR')
    else:
        logger(f"Sparse {sparse} loaded from {sparse_cfg['path']}", level='INFO')
    
    if full_cfg['Dmeson'] == 'Dzero' and sparse_cfg['name'] != "FlowSP":
        # TODO: safety checks for Dmeson reflecton and secondary peak
        if sparse_cfg['name'] == "RecoPrompt":
            sparse.GetAxis(axes['Origin']).SetRange(2, 2)       # select prompt
            sparse.GetAxis(axes['CandType']).SetRange(1, 2)     # select signal
        elif sparse_cfg['name'] == "RecoFD":
            sparse.GetAxis(axes['Origin']).SetRange(3, 3)       # select non-prompt
            sparse.GetAxis(axes['CandType']).SetRange(1, 2)     # select signal
        elif sparse_cfg['name'] == "RecoRefl":
            sparse.GetAxis(axes['CandType']).SetRange(3, 4)     # select reflection
        elif sparse_cfg['name'] == "RecoReflPrompt":
            sparse.GetAxis(axes['CandType']).SetRange(3, 4)     # select reflection
            sparse.GetAxis(axes['Origin']).SetRange(2, 2)       # select prompt
        elif sparse_cfg['name'] == "RecoReflFD":
            sparse.GetAxis(axes['CandType']).SetRange(3, 4)     # select reflection
            sparse.GetAxis(axes['Origin']).SetRange(3, 3)       # select FD
        elif sparse_cfg['name'] == "GenPrompt":
            sparse.GetAxis(axes['Origin']).SetRange(2, 2)       # select prompt
        elif sparse_cfg['name'] == "GenFD":
            sparse.GetAxis(axes['Origin']).SetRange(3, 3)       # select non-prompt
        else:
            logger(f"Unknown sparse type for Dzero {sparse_cfg['name']}", level='ERROR')

    if debug:
        print('\n')
        logger('###############################################################', level='DEBUG')
        for key, value in axes.items():
            logger(f"    {key}: {value}", level='DEBUG')
        logger('###############################################################\n', level='DEBUG')

    return sparse, axes

def process_sparse(i_file, infile_path, full_cfg, sparse_cfg, prep_out_dir, input_out_dir):
    """
    Process a single sparse from an input file for all pt bins according to the configuration.
    
    Args:
        i_file (int): index of the input file
        infile (TFile): input ROOT file
        full_cfg (dict): full configuration dictionary
        sparse_cfg (dict): sparse configuration dictionary
        prep_out_dir (str): output directory for pre-processed files
        input_out_dir (str): sub-directory for the specific input configuration
    """
    infile = TFile.Open(infile_path, 'read')
    logger(f'Processing file {i_file}, {infile.GetName()}')
    sparse, axes = get_inputs_sparse(infile, full_cfg, sparse_cfg, True)

    # Only for flow with SP, not for correlations (applied in O2Physics)
    if axes.get('Cent') is not None:
        cent_min, cent_max = get_centrality_bins(full_cfg['centrality'])[1]
        logger(f"Applying cent cut to sparse {sparse} with value {cent_min} -- {cent_max}", "INFO")
        sparse.GetAxis(axes['Cent']).SetRangeUser(cent_min, cent_max)

    pt_mins, pt_maxs = full_cfg['ptbins'][:-1], full_cfg['ptbins'][1:]
    bkg_maxs = full_cfg['preprocess']['bkg_cuts']
    mass_ranges = full_cfg['preprocess'].get('mass_ranges', None)
    axes_to_keep, rebin = sparse_cfg["axes"]['names'], sparse_cfg["axes"]['rebin']
    ese_cfg = full_cfg.get('ese') or {}
    ese_band = ese_cfg.get('band')
    do_ese = bool(ese_band) and ese_band != 'Inclusive' and sparse_cfg['name'] == 'FlowSP'
    thr_lo = thr_hi = None
    if do_ese:
        if axes.get('Qvec') is None or axes.get('Cent') is None:
            raise RuntimeError(f"ESE band {ese_band} requested but sparse {sparse_cfg['name']} "
                               f"has no Qvec and/or Cent axis")
        q_lo, q_hi = int(ese_band[0]), int(ese_band[1])
        bands = [[int(b[0]), int(b[1])] for b in (ese_cfg.get('bands') or [])]
        if bands and [q_lo, q_hi] not in bands:
            raise ValueError(f"ESE band {[q_lo, q_hi]} is not listed in ese.bands ({bands}), "
                             f"so the resolution file will not contain it")
        logger(f"Applying ESE band {get_ese_band_label(q_lo, q_hi)} "
               f"from detector {ese_cfg['det']}", "INFO")
        thr_lo = get_ese_thresholds(ese_cfg['quantile_file'], ese_cfg['det'], q_lo)
        thr_hi = get_ese_thresholds(ese_cfg['quantile_file'], ese_cfg['det'], q_hi)
    sparse_type, sparse_path = sparse_cfg['name'], sparse_cfg['path']
    sparse_dir, sparse_name = sparse_path.split('/')[0], sparse_path.split('/')[1]

    logger(f"Projecting sparse {sparse_cfg['name']} for file {i_file} into pT bins ({pt_mins} - {pt_maxs}) with bkg cuts {bkg_maxs}", level='INFO')
    for i_pt, (pt_min, pt_max, bkg_max) in enumerate(zip(pt_mins, pt_maxs, bkg_maxs)):
        logger(f"Processing pT bin {pt_min} - {pt_max} with bkg max {bkg_max}", level='INFO')
        # Create output file
        out_file_dir = f"{prep_out_dir}/preprocess/pt_{int(pt_min*10)}_{int(pt_max*10)}/{input_out_dir}"
        os.makedirs(out_file_dir, exist_ok=True)
        out_file_path = f'{out_file_dir}/AnalysisResults_{i_file}.root'
        out_file, write_opt = check_existing_outputs(out_file_path)

        sparse.GetAxis(axes.get('PtTrig', axes.get('Pt'))).SetRangeUser(pt_min, pt_max) # PtTrig for correlations, Pt for SP flow
        if axes.get('ScoreBkg') is not None: # Skip sparses for generated info
            sparse.GetAxis(axes['ScoreBkg']).SetRangeUser(0, bkg_max)
        if mass_ranges is not None:
            mass_min, mass_max = mass_ranges[i_pt]
            sparse.GetAxis(axes['Mass']).SetRangeUser(mass_min, mass_max)
        proj_axes = [axes[ax_to_keep] for ax_to_keep in axes_to_keep]

        if not do_ese:
            proj_sparse = sparse.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
        else:
            ax_cent = sparse.GetAxis(axes['Cent'])
            ax_q = sparse.GetAxis(axes['Qvec'])
            acc = None
            for c in range(int(cent_min), int(cent_max)):
                ax_cent.SetRangeUser(c + 1e-6, c + 1 - 1e-6)
                ax_q.SetRangeUser(thr_lo[c], thr_hi[c])
                h = sparse.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
                if acc is None:
                    acc = h.Clone(f'acc_{i_pt}')
                else:
                    acc.Add(h)
                h.Delete()
            ax_cent.SetRangeUser(cent_min, cent_max)
            ax_q.SetRange(0, 0)
            proj_sparse = acc

        proj_sparse.SetName(sparse.GetName())
        proj_sparse = proj_sparse.Rebin(array.array('i', rebin))
        make_dir_root_file(sparse_type, out_file)
        out_file.cd(sparse_type)
        proj_sparse.Write(f"hSparse{sparse_type}", write_opt)
        out_file.Delete(f"hSparse{sparse_type}" + ";*")
        proj_sparse.Delete()
        del proj_sparse

        gc.collect()

        out_file.Close()
        logger(f'----> Finished processing pT bin {pt_min} - {pt_max} for {i_file}, sparse: {sparse_cfg["name"]}\n', "INFO")

    del sparse
    gc.collect()
    infile.Close()

def process_tree(i_file, infile_path, full_cfg, trees_cfg, prep_out_dir, input_out_dir):
    logger(f'[Data] Processing tree file {i_file}, {infile_path}', "INFO")
    cols_dict, columns_to_keep = {}, []

    # Build columns dictionary and columns to keep
    for tree_cfg in trees_cfg['trees']:
        cols_dict.update(get_tree_dict(tree_cfg['table']))
        columns_to_keep.extend([cols_dict[col] for col in tree_cfg['cols']])

    try:
        def collect_trees(uproot_dir, tree_names, trees):
            for key in uproot_dir.keys():
                obj = uproot_dir[key]
                if '/' in key: # Else one gets duplicate entries
                    continue
                if isinstance(obj, uproot.TTree) and obj.name in tree_names:
                    # extend existing awkward array with new data
                    trees[obj.name] = ak.concatenate([trees[obj.name], obj.arrays(library="ak")])
                elif isinstance(obj, uproot.reading.ReadOnlyDirectory):
                    collect_trees(obj, tree_names, trees)  # recurse into subdirectory

        trees = {tree_cfg['table']: ak.Array([]) for tree_cfg in trees_cfg['trees']}
        tree_names = [t['table'] for t in trees_cfg['trees']]
        with uproot.open(infile_path) as f:
            collect_trees(f, tree_names, trees)

    except Exception as e:
        print(f"Failed to open file {infile_path}: {e}")
        return

    try:
        merged_tree = ak.zip(
            {field: trees[tree][field]
            for tree in trees 
            for field in trees[tree].fields}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger(f"Caught Python exception while merging trees: {e}", "ERROR")
        return

    # Loop over pt bins
    pt_mins, pt_maxs = full_cfg['ptbins'][:-1], full_cfg['ptbins'][1:]
    bkg_maxs = full_cfg['preprocess']['bkg_cuts']

    try:
        for pt_min, pt_max, bkg_max in zip(pt_mins, pt_maxs, bkg_maxs):
            out_file_dir = f"{prep_out_dir}/preprocess/pt_{int(pt_min*10)}_{int(pt_max*10)}/{input_out_dir}"
            os.makedirs(out_file_dir, exist_ok=True)
            out_file_path = f'{out_file_dir}/AO2D_{i_file}.root'

            # Selection (centrality is optional)
            if 'Cent' in cols_dict:
                cent_min, cent_max = get_centrality_bins(full_cfg['centrality'])[1]
                mask = (
                    (merged_tree[cols_dict['Pt']] >= pt_min) &
                    (merged_tree[cols_dict['Pt']] <= pt_max) &
                    (merged_tree[cols_dict['Cent']] >= cent_min) &
                    (merged_tree[cols_dict['Cent']] <= cent_max) &
                    (merged_tree[cols_dict['ScoreBkg']] >= 0) &
                    (merged_tree[cols_dict['ScoreBkg']] <= bkg_max)
                )
            else:
                mask = (
                    (merged_tree[cols_dict['Pt']] >= pt_min) &
                    (merged_tree[cols_dict['Pt']] <= pt_max) &
                    (merged_tree[cols_dict['ScoreBkg']] >= 0) &
                    (merged_tree[cols_dict['ScoreBkg']] <= bkg_max)
            )
            merged_tree_sel = merged_tree[mask]

            # Save snapshot
            uproot.recreate(out_file_path)[trees_cfg['final_table_name']] = {
                col: merged_tree_sel[col] for col in columns_to_keep
            }
            logger(f"Saved snapshot to {out_file_path}", "INFO")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger(f"Caught Python exception while saving snapshot: {e}", "ERROR")

def get_input_paths(input_cfg, input_type):
    print(f"input_cfg: {input_cfg}")
    if isinstance(input_cfg, str) and input_cfg.endswith(".txt"):
        return input_cfg
    if isinstance(input_cfg, str):
        dir_path = input_cfg
        file_paths = [f"{dir_path}/{file}" for file in os.listdir(dir_path) \
                      if file.endswith(".root") and input_type in file]
        return file_paths
    elif isinstance(input_cfg, list):
        file_paths = input_cfg
        return file_paths
    else:
        logger("Invalid type for 'files' in configuration. Must be a string (directory) or list of file paths.", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('config', metavar='text', default='config.yml', help='configuration file')
    parser.add_argument("--workers", "-w", type=int, default=1, help="number of workers")
    args = parser.parse_args()

    with open(args.config, 'r') as cfg_pre:
        full_cfg = yaml.safe_load(cfg_pre)

    output_dir = full_cfg['outdirPrep'] if full_cfg.get("outdirPrep") else full_cfg['outdir']
    for input_cfg in full_cfg['preprocess']['inputs']:
        if isinstance(input_cfg['files'], str) and not input_cfg['files'].endswith(".txt"):
            dir_path = input_cfg['files']
            file_paths = [f"{dir_path}/{file}" for file in os.listdir(dir_path) if file.endswith(".root")]
        elif isinstance(input_cfg['files'], str) and input_cfg['files'].endswith(".txt"):
            with open(input_cfg['files'], 'r') as f:
                file_paths = [line.strip() for line in f if line.strip()]
        elif isinstance(input_cfg['files'], list):
            file_paths = input_cfg['files']
        else:
            logger("Invalid type for 'files' in configuration. Must be a string (directory) or list of file paths.", "ERROR")
            continue

        if input_cfg.get('sparses'):
            if isinstance(input_cfg['files'], str) and not input_cfg['files'].endswith(".txt"):
                file_paths = get_input_paths(input_cfg['files'], "AnalysisResults")
            for sparse_cfg in input_cfg['sparses']:
                logger(f"##### Skimming {sparse_cfg['name']} #####", "WARNING")
                with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
                    tasks_sparses = [executor.submit(process_sparse, i_file, file, full_cfg, sparse_cfg, output_dir, f"{input_cfg['outdir']}/jobs") for i_file, file in enumerate(file_paths)]

        elif input_cfg.get('trees'):
            logger(f"##### Skimming {input_cfg} trees #####", "INFO")
            file_paths = get_input_paths(input_cfg['files'], "AO2D")
            with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
                tasks_trees = [executor.submit(process_tree, i_file, file_path, full_cfg, input_cfg, output_dir, f"{input_cfg['outdir']}/jobs") for i_file, file_path in enumerate(file_paths)]

        else:
            logger("No sparses or trees found in the configuration for pre-processing", "ERROR")

        # Dump file_paths used
        dump_file_paths = f"{output_dir}/preprocess/file_paths_{input_cfg['outdir']}.txt"
        os.makedirs(os.path.dirname(dump_file_paths), exist_ok=True)
        with open(dump_file_paths, "w") as dump_file:
            for file_path in file_paths:
                dump_file.write(f"{file_path}\n")

        logger(f"Finished processing {input_cfg['outdir']}\n\n", "INFO")

    # use hadd to merge the results in the directories for the different pT bins
    for pt_dir in os.listdir(f"{output_dir}/preprocess/"):
        if not os.path.isdir(f"{output_dir}/preprocess/{pt_dir}"):
            continue
        for input_type_dir in os.listdir(f"{output_dir}/preprocess/{pt_dir}"):
            prep_dir = f"{output_dir}/preprocess/{pt_dir}/{input_type_dir}"
            files_to_merge = [f"./jobs/{file}" for file in os.listdir(f"{prep_dir}/jobs") if file.endswith(".root")]
            files_to_merge_str = ' '.join(files_to_merge)
            file_name = os.path.basename(files_to_merge[0]).split('_')[0]
            merged_file = f"{prep_dir}/{file_name}_{pt_dir}.root"
            log_merge = f"{prep_dir}/log_merge.txt"
            filelist_path = f"{prep_dir}/files_to_hadd.txt"
            with open(filelist_path, "w") as f: # To avoid crash if too many files
                for fpath in files_to_merge:
                    f.write(fpath + "\n")
            hadd = f"hadd -f {merged_file} @{filelist_path} > {log_merge}"
            logger(f"Running command: {hadd}\n\n", "INFO")
            try:
                os.system(f"cd {prep_dir} && {hadd}")
            except Exception as e:
                logger(f"Error while creating hadd command: {e}", "ERROR")
                os.system(f"rm {merged_file}") # Remove the partially merged file
                os.system(f"mv {log_merge} {prep_dir}/log_merge_error.txt")
