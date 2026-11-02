import yaml

import copy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ROOT
ROOT.DisableImplicitMT()

import numpy as np
import argparse
import itertools
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../utils")
from utils import logger

CONFIGURABLES = {
    "BkgFunc": ["v2extraction", "BkgFunc"],
    "SgnFunc": ["v2extraction", "SgnFunc"],
    "BkgFuncVn": ["v2extraction", "BkgFuncVn"],
    "SpRanges": ["v2extraction", "SpRanges"],
    "Rebin": ["v2extraction", "Rebin"],
    "NSigma4SB": ["v2extraction", "NSigma4SB"],
    "FixVnSecPeakToSgn": ["v2extraction", "FixVnSecPeakToSgn"],
    "FixSgnFromFile": ["v2extraction", "FixSgnFromFile"],
    "InclSecPeak": ["v2extraction", "InclSecPeak"],
    "FixFromFile": ["v2extraction", "FixFromFile"],
    "FixFracSigmaSecPeak": ["v2extraction", "FixFracSigmaSecPeak"],
    "MinCountsSecPeak": ["v2extraction", "MinCountsSecPeak"],
    "MassMinSecPeak": ["v2extraction", "MassMinSecPeak"],
    "MassMaxSecPeak": ["v2extraction", "MassMaxSecPeak"],
    "FixFromConfig": ["v2extraction", "FixFromConfig"],
    "MinCountsSecPeak": ["v2extraction", "MinCountsSecPeak"],
    "MassFitRanges": ["v2extraction", "MassFitRanges"],
    "InvMassBinSteps": ["projections", "VnVsMassBins"],
    "MassMin": ["v2extraction", "MassFitRanges"],
    "MassMax": ["v2extraction", "MassFitRanges"],
    "TemplatesNorm": ["v2extraction", "TemplatesNorm"]
}

def set_config(new_cfg, ref_cfg, path, iPtBin):
    """
    path: list of nested keys
    """
    print(f"Setting {path} for pt bin index {iPtBin}")
    # Traverse reference config
    ref_sub = ref_cfg
    for key in path:
        if ref_sub.get(key) is None:
            logger(f"Key {key} not found in reference config at path {path}", "ERROR")
            return
        ref_sub = ref_sub[key]

    # Extract correct value
    if isinstance(ref_sub, list) and len(ref_sub) > iPtBin:
        print(f"Setting {path[-1]} with value {ref_sub[iPtBin]} from reference config")
        value = ref_sub[iPtBin]
    else:
        value = ref_sub

    # Traverse new config up to parent
    new_sub = new_cfg
    for key in path[:-1]:
        new_sub = new_sub[key]

    # Set final key
    new_sub[path[-1]] = value

def get_reference_config_pt_bin(cfg, iPtBin):
    # Deepcopy to ensure that we are working with a copy and not modifying the original cfg
    pt_bin_cfg = copy.deepcopy(cfg)
    pt_bin_cfg['ptbins'] = [cfg['ptbins'][iPtBin], cfg['ptbins'][iPtBin+1]]

    # Drop minimisation key
    pt_bin_cfg.pop('minimisation', None)

    # Separate iPtBin element of keys having a list value
    for key, value in pt_bin_cfg.items():
        if isinstance(value, list) and 'ptbins' not in key:
            print(f"Setting {key}")
            pt_bin_cfg[key] = value[iPtBin]

    # Cut variation
    set_config(pt_bin_cfg, cfg, ["cut_variation", "corr_bdt_cut", "bkg_max"], iPtBin)
    set_config(pt_bin_cfg, cfg, ["cut_variation", "corr_bdt_cut", "sig", "max"], iPtBin)
    set_config(pt_bin_cfg, cfg, ["cut_variation", "corr_bdt_cut", "sig", "min"], iPtBin)
    set_config(pt_bin_cfg, cfg, ["cut_variation", "corr_bdt_cut", "sig", "step"], iPtBin)
    set_config(pt_bin_cfg, cfg, ["cut_variation", "uncorr_bdt_cut", "bkg_max"], iPtBin)
    pt_bin_cfg["cut_variation"]["uncorr_bdt_cut"]["bkg_max"] = [pt_bin_cfg["cut_variation"]["uncorr_bdt_cut"]["bkg_max"]]
    set_config(pt_bin_cfg, cfg, ["cut_variation", "uncorr_bdt_cut", "sig"], iPtBin)
    pt_bin_cfg["cut_variation"]["uncorr_bdt_cut"]["sig"] = [pt_bin_cfg["cut_variation"]["uncorr_bdt_cut"]["sig"]]

    # Projections
    set_config(pt_bin_cfg, cfg, ['projections', 'VnVsMassBins'], iPtBin)

    # Fit settings
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'MassFitRanges'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'FixSgnFromFile'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'Sigma'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'Rebin'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'BkgFunc'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'SgnFunc'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'BkgFuncVn'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'NSigma4SB'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'MinCountsSecPeak'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'MassMinSecPeak'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'MassMaxSecPeak'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'InclSecPeak'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'FracSecPeak'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'FixFracSigmaSecPeak'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'SigmaSecPeak'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'FixVnSecPeakToSgn'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'FixFromFile'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'FixFromConfig'], iPtBin)
    set_config(pt_bin_cfg, cfg, ['v2extraction', 'SpRanges'], iPtBin)

    return pt_bin_cfg

def produce_trials_fit_configs(cfg_flow, ptmin, ptmax, max_trials, ref_config_ptbin, setting, multitrial_pt_dir):

    varied_configs = {}
    for cfg, values in setting["multitrial"].items():
        varied_configs[cfg] = np.unique(values).tolist()
        print(f"  Values for {cfg}: {varied_configs[cfg]}")

    # Perform itertools.product to get all combinations
    keys = list(varied_configs.keys())
    values = list(varied_configs.values())
    combinations = list(itertools.product(*values))
    config_variants = [dict(zip(keys, combination)) for combination in combinations] 

    np.random.shuffle(config_variants)

    # Print results
    trials_counter = 0
    for idx, variant in enumerate(config_variants):
        if trials_counter >= max_trials:
            print(f"Reached max trials of {max_trials}, stopping generation of further configs.")
            break
        cfg_variant = copy.deepcopy(ref_config_ptbin)
        cfg_variant["iTrial"] = idx
        cfg_variant["outdir"] = f"{multitrial_pt_dir}/trials/{idx}/"
        print(f"[{idx}]: Variant config: {variant}\n")
        # Fit quality criteria
        if setting['MaxChi2'] > 10:
            logger(f"MaxChi2 is set to {setting['MaxChi2']} for pt bin {ptmin}-{ptmax}, please consider a tight selection!", "WARNING")
        cfg_variant["MaxChi2"] = setting['MaxChi2']
        if setting['MinSignificance'] < 5:
            logger(f"MinSignificance is set to {setting['MinSignificance']}, please consider a higher lower limit!", "WARNING")
        cfg_variant["MinSignificance"] = setting['MinSignificance']
        if setting['MaxSignificance'] > 1000:
            logger(f"MaxSignificance is set to {setting['MaxSignificance']}, please consider a lower upper limit!", "WARNING")
        cfg_variant["MaxSignificance"] = setting['MaxSignificance']

        for varied_var in variant:
            if ref_config_ptbin.get(varied_var):
                if isinstance(ref_config_ptbin[varied_var], list):
                    cfg_variant[varied_var] = [variant[varied_var]]
                else:
                    cfg_variant[varied_var] = variant[varied_var]

        for configurable, path in CONFIGURABLES.items():
            print(f"Processing configurable {configurable} with path {path}, setting value {variant.get(configurable)}")
            if configurable == "InvMassBinSteps": # Not needed for CMS method, only for simfit
                cfg_variant["projections"]["VnVsMassBins"] = \
                    [np.arange(variant['MassMin'], variant['MassMax'] + variant['InvMassBinSteps'],
                               variant['InvMassBinSteps']).tolist()]
            elif configurable == "MassMin" or configurable == "MassMax":
                cfg_variant["v2extraction"]["MassFitRanges"] = \
                    [[variant["MassMin"], variant["MassMax"]]]
                if cfg_flow['Dmeson'] == 'Dplus' and (variant["MassMin"] > 1.75 or variant["MassMax"] < 1.95):
                    cfg_variant["NSigma4SB"] = [2]
                    cfg_variant["v2extraction"]["Sigma"] = [0.01]
            else:
                try:
                    cfg_variant[path[0]][path[1]] = variant[configurable]
                except KeyError:
                    logger(f"Configurable {configurable} with path {path} not found in reference config, skipping...", "WARNING")

        os.makedirs(os.path.join(f"{multitrial_pt_dir}/trials/{idx}"), exist_ok=True)
        output_file = os.path.join(f"{multitrial_pt_dir}/trials/{idx}", f"config_trial_{idx}.yml")

        # No preprocess
        cfg_variant.pop('preprocess', None)

        # Setup operations --> only fit on, so failed fits can be easily re-run with the same config
        cfg_variant["operations"]['preprocess'] = False
        cfg_variant["operations"]['make_yaml'] = True
        cfg_variant["operations"]['proj_data'] = False
        cfg_variant["operations"]['proj_mc'] = False
        cfg_variant["operations"]['efficiencies'] = False
        cfg_variant["operations"]['get_vn_vs_mass'] = True
        cfg_variant["operations"]['do_cut_variation'] = False
        cfg_variant["operations"]['data_driven_fraction'] = False
        cfg_variant["operations"]['get_v2_vs_frac'] = False

        # Cut variation is taken from reference results
        cfg_variant.pop('minimisation', None)
        print(f"MassFitRanges: {cfg_variant['v2extraction']['MassFitRanges']}")

        # Write variant to yml file
        with open(output_file.replace('config_trial', 'variants'), 'w') as variants_file:
            yaml.dump(variant, variants_file, default_flow_style=False, sort_keys=False)

        # If fit range reduced to exclude secondary peak, skip the config
        if cfg_variant["v2extraction"]["InclSecPeak"] and cfg_variant["v2extraction"].get("MassMinSecPeak"):
            if cfg_variant["v2extraction"]["MassMinSecPeak"] > cfg_variant["v2extraction"]["MassFitRanges"][0][1]:
                print(f"Skipping config {idx} because MassMinSecPeak {cfg_variant['v2extraction']['MassMinSecPeak']} is larger than MassFitRanges max {cfg_variant['v2extraction']['MassFitRanges'][0][1]}")
                # Write to file this information
                with open(output_file.replace('config_trial', 'skipped').replace('.yml', '.txt'), 'w') as skipped_file:
                    skipped_file.write(f"Skipped config {idx} because MassMinSecPeak {cfg_variant['v2extraction']['MassMinSecPeak']} is larger than MassFitRanges max {cfg_variant['v2extraction']['MassFitRanges'][0][1]}")
                continue

        with open(output_file, 'w') as out_file:
            yaml.dump(cfg_variant, out_file, default_flow_style=False, sort_keys=False)
        trials_counter += 1

def produce_trials_bdt_configs(cfg_flow, ref_config_ptbin, setting, multitrial_pt_dir):

    for i_cutset, cutset_cfg in enumerate(setting["multitrial"]["cutsets"]):
        if cutset_cfg.get("BkgScoreStep"):
            bdt_bkg_cuts = np.arange(cutset_cfg["BkgScoreRange"][0], cutset_cfg["BkgScoreRange"][1] + \
                                     cutset_cfg["BkgScoreStep"], cutset_cfg["BkgScoreStep"]).tolist()
        else:
            bdt_bkg_cuts = cutset_cfg["BkgScoreVals"]

        for i_bkg_cut, bkg_cut in enumerate(bdt_bkg_cuts):
            bkg_cut = round(bkg_cut, 4)  # To avoid floating point precision issues
            # Have equal strings, so pad 0.0x to 0.0x0 to match, e.g., 0.05 and 0.052 number of digits
            bkg_cut_str = f"{bkg_cut:.4f}"
            out_dir = f"{multitrial_pt_dir}/trials_cutset_{i_cutset}/bkg_{bkg_cut_str}/"
            os.makedirs(out_dir, exist_ok=True)

            cfg_variant = copy.deepcopy(ref_config_ptbin)
            cfg_variant["iTrial"] = i_bkg_cut
            cfg_variant["cut_variation"]["uncorr_bdt_cut"]["bkg_max"] = [[bkg_cut]]
            cfg_variant["cut_variation"]["uncorr_bdt_cut"]["sig"] = [[ref_config_ptbin["cut_variation"]["uncorr_bdt_cut"]["sig"][0][i_cutset],
                                                                      ref_config_ptbin["cut_variation"]["uncorr_bdt_cut"]["sig"][0][i_cutset+1]]]
            cfg_variant["outdir"] = out_dir
            cfg_variant['v2extraction']['MassFitRanges'] = [cfg_variant['v2extraction']['MassFitRanges']]
            cfg_variant['projections']['VnVsMassBins'] = [cfg_variant['projections']['VnVsMassBins']]

            # Setup operations
            cfg_variant["operations"]['preprocess'] = False
            cfg_variant["operations"]['make_yaml'] = True
            cfg_variant["operations"]['proj_data'] = True
            cfg_variant["operations"]['proj_mc'] = True
            cfg_variant["operations"]['efficiencies'] = True
            cfg_variant["operations"]['get_vn_vs_mass'] = True
            cfg_variant["operations"]['do_cut_variation'] = False
            cfg_variant["operations"]['data_driven_fraction'] = False
            cfg_variant["operations"]['get_v2_vs_frac'] = False

            # Pick up pre-processed files
            cfg_variant["outdirPrep"] = ref_config_ptbin["outdir"]
            with open(f"{out_dir}/config_trial_{i_bkg_cut}.yml", 'w') as out_file:
                yaml.dump(cfg_variant, out_file, default_flow_style=False, sort_keys=False)

def produce_trials_cfgs(config_flow, config_mod, output_dir, multitrial_type='fit'):

    with open(config_flow, 'r') as CfgFlow:
        cfg_flow = yaml.safe_load(CfgFlow)
    with open(config_mod, 'r') as CfgMod:
        cfg_mod = yaml.safe_load(CfgMod)

    multitrial_dir = f"{output_dir}/syst/multitrial/{multitrial_type}/"
    os.makedirs(multitrial_dir, exist_ok=True)
    os.makedirs(f"{multitrial_dir}/config_history/", exist_ok=True)

    with open(os.path.join(f"{multitrial_dir}/config_history/", f"config_reference.yml"), 'w') as out_file:
        yaml.dump(cfg_flow, out_file, default_flow_style=False, sort_keys=False)
    with open(os.path.join(f"{multitrial_dir}/config_history/", f"config_modifications.yml"), 'w') as out_file:
        yaml.dump(cfg_mod, out_file, default_flow_style=False, sort_keys=False)

    for _, setting in enumerate(cfg_mod['ptbins']):
        print(f"cfg_mod['ptbins']: {cfg_mod['ptbins']}")
        print(f"setting: {setting}")
        for ptmin, ptmax in setting['ranges']:
            logger(f"Generating multitrial configs for pt bin {ptmin}-{ptmax} GeV/c", "INFO")

            pt_bin_index = cfg_flow['ptbins'].index(ptmin)
            ref_config_ptbin = get_reference_config_pt_bin(cfg_flow, pt_bin_index)
            multitrial_pt_dir = f"{multitrial_dir}/pt_{int(ptmin*10)}_{int(ptmax*10)}/"
            
            os.makedirs(multitrial_pt_dir, exist_ok=True)
            output_file = os.path.join(multitrial_pt_dir, f"config_reference.yml")
            with open(output_file, 'w') as out_file:
                yaml.dump(ref_config_ptbin, out_file, default_flow_style=False, sort_keys=False)
                
            if multitrial_type == 'fit':
                produce_trials_fit_configs(cfg_flow, ptmin, ptmax, cfg_mod['max_trials'], ref_config_ptbin, setting, multitrial_pt_dir)
            elif multitrial_type == 'bdt':
                produce_trials_bdt_configs(cfg_flow, ref_config_ptbin, setting, multitrial_pt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('input_config', metavar='text', default='config_Ds_Fit.yml')
    parser.add_argument('--fit_multitrial', "-fm", action='store_true', default=False)
    parser.add_argument('--bdt_multitrial', "-bm", action='store_true', default=False)
    parser.add_argument('--modifications_config', "-m", metavar='text', default='')
    parser.add_argument("--outputdir", "-o", metavar="text", default=".", help="output directory")
    args = parser.parse_args()

    produce_trials_cfgs(args.input_config,
                        args.modifications_config,
                        args.outputdir,
                        'fit' if args.fit_multitrial else 'bdt')
