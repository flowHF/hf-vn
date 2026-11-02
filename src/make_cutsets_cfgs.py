'''
Script to create yaml files corresponding to a set of cutset to perform
the cut variation analysis in the combined and correlated cases
python3 make_cutsets_cfgs.py config_flow.yml -o path/to/output [--correlated]
Without --correlated, the script will create yaml files for the combined case
'''

import yaml
import argparse
import os
import numpy as np
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))  # Append parent directory

def pad_to_length(list, target_len):
    '''
        Function to pad a list to a target length
        Args:
            lst (list): list to be padded
            target_len (int): target length of the list
        Returns:
            list: padded list
    '''
    list_length = len(list)
    len_offset = target_len - list_length
    return list + [list[-1]] * len_offset if list_length < target_len else list

def make_yaml_sp(flow_config, outputdir, correlated):
    '''
        Function to create a yaml file with a set of cuts for ML
        Args:
            flow_config (str): path to the flow config file
            outputdir (str): path to the output directory
    '''
    with open(flow_config, 'r') as f:
        cfg = yaml.safe_load(f)

    ptmins = cfg['ptbins'][:-1]
    ptmaxs = cfg['ptbins'][1:]
    nPtBins = len(ptmins)

    cfg_cutvar = cfg['cut_variation']
    if correlated:
        sig = cfg_cutvar['corr_bdt_cut']['sig']
        sig_cuts_lower = [list(np.arange(sig['min'][i], sig['max'][i], sig['step'][i])) for i in range(nPtBins)]
        sig_cuts_upper = [[1.0] * len(cuts) for cuts in sig_cuts_lower]
    else:
        sig = cfg_cutvar['uncorr_bdt_cut']['sig']
        sig_cuts_lower = [sig[i][:-1] for i in range(nPtBins)]
        sig_cuts_upper = [sig[i][1:]  for i in range(nPtBins)]

    # Determine the maximum number of cut sets across pt bins and pad all to uniform length
    maxCutSets = max(len(cuts) for cuts in sig_cuts_lower)
    sig_cuts_lower = [pad_to_length(cuts, maxCutSets) for cuts in sig_cuts_lower]
    sig_cuts_upper = [pad_to_length(cuts, maxCutSets) for cuts in sig_cuts_upper]
    # Transpose: convert [iPt][iCut] → [iCut][iPt]
    sig_cuts_lower = list(map(list, zip(*sig_cuts_lower)))
    sig_cuts_upper = list(map(list, zip(*sig_cuts_upper)))

    if correlated:
        bkg_cuts_upper = [[cut for iCut, cut in enumerate(cfg_cutvar['corr_bdt_cut']['bkg_max']) if iCut < nPtBins]] * maxCutSets
    else:
        bkg_cuts_upper = [pad_to_length(cuts, maxCutSets) for iCut, cuts in enumerate(cfg_cutvar['uncorr_bdt_cut']['bkg_max']) if iCut < nPtBins]
        bkg_cuts_upper  = list(map(list, zip(*bkg_cuts_upper)))

    os.makedirs(f'{outputdir}/cutsets', exist_ok=True)
    for iCut, (bkg_maxs, fd_mins, fd_maxs) in enumerate(zip(bkg_cuts_upper, sig_cuts_lower, sig_cuts_upper)):
        bkg_max = list(map(float, bkg_maxs))
        fd_min  = list(map(float, fd_mins))
        fd_max  = list(map(float, fd_maxs))

        if len(fd_min) != len(ptmins) or len(fd_max) != len(ptmins) or len(bkg_max) != len(ptmins):
            raise ValueError(f"Lengths do not match: bkg_max: {len(bkg_max)}, fd_max: {len(fd_max)}, fd_min: {len(fd_min)}, ptmins: {len(ptmins)}")

        combinations = {
            'icutset': iCut,
            'Pt': {'min': ptmins, 'max': ptmaxs},
            'ScoreBkg': {'min': [0.0] * len(ptmins), 'max': bkg_max},
            'ScoreFD': {'min': fd_min, 'max': fd_max},
        }

        with open(f'{outputdir}/cutsets/cutset_{iCut:02}.yml', 'w') as file:
            yaml.dump(combinations, file, default_flow_style=False, sort_keys=False)

    print(f'Cutsets saved in {outputdir}/cutsets')

def make_yaml_correl(flow_config, outputdir):

    with open(flow_config, 'r') as f:
        cfg = yaml.safe_load(f)

    # For each delta eta gap, generate the associated HM and LM configs
    for deta_min, deta_max in cfg['DeltaEtaRanges']:
        cfg_mod = cfg.copy()

        cfg_mod['deltaEtaBins'] = [
            [-deta_max, -deta_min],
             [deta_min,  deta_max]
        ]

        cfg_mod['suffix'] = f'{deta_min}_{deta_max}_{cfg_mod["method"]}'
        cfg_mod['outdir'] = f"{cfg_mod['outdir']}/{cfg_mod['suffix']}/"
        os.makedirs(cfg_mod['outdir'], exist_ok=True)
        deta_cfg_path = f"{cfg_mod['outdir']}/config_{cfg_mod['suffix']}.yml"
        with open(deta_cfg_path, 'w') as f:
            yaml.dump(cfg_mod, f, default_flow_style=False)
        make_yaml_sp(deta_cfg_path, cfg_mod['outdir'], correlated=False)

        if cfg["task_LM"] is not None:
            cfg_mod['outdir']        = f"{cfg_mod['outdir']}/low_mult/"
            cfg_mod['pathFileSE']    = cfg["task_LM"]['pathFileSE']
            cfg_mod['pathFileME']    = cfg["task_LM"]['pathFileME']
            cfg_mod['pathFileMass']  = cfg["task_LM"]['pathFileMass']
            cfg_mod['nDeltaPhiBins'] = cfg["task_LM"]['nDeltaPhiBins']
            cfg_mod['rebinDeltaEta'] = cfg["task_LM"]['rebinDeltaEtaLM']
            cfg_mod['rebinDeltaPhi'] = cfg["task_LM"]['rebinDeltaPhiLM']

        os.makedirs(cfg_mod['outdir'], exist_ok=True)
        deta_cfg_path = f"{cfg_mod['outdir']}/config_{cfg_mod['suffix']}_low_mult.yml"
        with open(deta_cfg_path, 'w') as f:
            yaml.dump(cfg_mod, f, default_flow_style=False)
        make_yaml_sp(deta_cfg_path, cfg_mod['outdir'], correlated=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('flow_config', metavar='text', default='config_flow.yml')
    parser.add_argument("--outputdir", "-o", metavar="text", default=".", help="output directory")
    parser.add_argument("--sp_correlated", "-spcr", action="store_true", help="Produce yml files for correlated cuts")
    parser.add_argument("--sp_combined", "-spcb", action="store_true", help="Produce yml files for correlated cuts")
    parser.add_argument("--correlations", "-2pc", action="store_true", help="Produce yml files for correlated cuts")
    args = parser.parse_args()

    if args.sp_correlated or args.sp_combined:
        print("Producing yml files for correlated cuts")
        make_yaml_sp(args.flow_config, args.outputdir, args.sp_correlated)
    elif args.correlations:
        print("Producing yml files for combined cuts")
        make_yaml_correl(args.flow_config, args.outputdir)
