'''
    Normalizations are given for the fit function to use a signal PDF and
    template from hMassTotalCorrBkgs multiplied by a common normalization 
    constant.
'''

import pandas as pd
import matplotlib.pyplot as plt
import uproot
import numpy as np
import ROOT
from array import array
import os
import sys
import argparse
import yaml
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger, get_centrality_bins, make_dir_root_file
from corr_bkgs_brs import final_states
from ROOT import RooRealVar, RooDataSet, RooArgSet, RooKeysPdf, TFile, TH3F, TH1F

def get_corr_bkg(corr_bkg_file, corr_bkg_chn, sel_string, pt_label, templ_type, output_type,
                 corr_abundances=False, sgn_d_meson='Dplus', get_smoothed=True, **kwargs):
    '''
    Get correlated background template and normalization factor
    '''
    input_folder = f"{pt_label}/{corr_bkg_chn}"
    if kwargs.get("verbose", False):
        logger(f"Using correlated bkg file {corr_bkg_file.GetName()} for correlated bkg source {corr_bkg_chn} from folder {input_folder}\n", "INFO")
    try:
        hist_pdg_mc_brs = corr_bkg_file.Get(f"{input_folder}/hBRs")
    except:
        if kwargs.get("verbose", False):
            logger(f"Could not retrieve hBRs histogram from {input_folder} in file {corr_bkg_file}", "ERROR")
        sys.exit(1)
    br_pdg = hist_pdg_mc_brs.GetBinContent(2)
    br_mc = hist_pdg_mc_brs.GetBinContent(1)
    if kwargs.get("verbose", False):
        logger(f"Branching ratios: PDG = {br_pdg}, MC = {br_mc}", "INFO")
    try:
        templ_tree_mass = corr_bkg_file.Get(f"{input_folder}/{templ_type}/treeMass")
    except:
        if kwargs.get("verbose", False):
            logger(f"Could not retrieve treeMass from {input_folder}/{templ_type} in file {corr_bkg_file}", "ERROR")
        sys.exit(1)
    templ_tree_mass.SetDirectory(0)
    templ_histo_mass = corr_bkg_file.Get(f"{input_folder}/{templ_type}/hMass{'Smooth' if get_smoothed else 'Raw'}")
    templ_histo_mass.SetDirectory(0)
    full_tree = corr_bkg_file.Get(f"{input_folder}/{templ_type}/treeFracMassScoresBkgFD")
    templ_rdataframe_full = ROOT.RDataFrame(full_tree)
    n_entries = templ_rdataframe_full.Filter(sel_string).Count().GetValue()
    if kwargs.get("verbose", False):
        logger(f"Number of entries after selection '{sel_string}': {n_entries} for correlated bkg source {corr_bkg_chn}", "INFO")
    corr_abundance = 1 if not corr_abundances else final_states[corr_bkg_chn].get(f"abundance_to_{sgn_d_meson}", 1)
    if corr_abundance != 1:
        if kwargs.get("verbose", False):
            logger(f"Applying abundance correction factor of {corr_abundance} for correlated bkg source {corr_bkg_chn}", "WARNING")
    frac = (br_pdg / br_mc) * n_entries * corr_abundance
    if kwargs.get("verbose", False):
        logger(f"Returning frac {frac} for correlated bkg source {corr_bkg_chn}", "INFO")
    if kwargs.get("verbose", False):
        logger(f"Returning {output_type} for correlated bkg source {corr_bkg_chn}", "WARNING")
    if output_type == "hist":
        return templ_histo_mass, frac
    elif output_type == "tree":
        return templ_tree_mass, frac
    elif output_type == "df":
        rdf_mass = ROOT.RDataFrame(f"{input_folder}/{templ_type}/treeMass", corr_bkg_file.GetName())
        df_mass = pd.DataFrame(ROOT.RDataFrame(rdf_mass).AsNumpy())
        return df_mass, frac
    else:
        if kwargs.get("verbose", False):
            logger(f"Output type {output_type} not recognized. Choose between 'hist' or 'tree'.", "ERROR")
        sys.exit(1)

def fill_smooth_histo(df, histo, n_points_for_sample, n_points_for_kde):

    # Define the RooDataset corresponding to histogram range
    x_min = histo.GetXaxis().GetXmin()
    x_max = histo.GetXaxis().GetXmax()
    x = RooRealVar("x", "x", x_min, x_max)
    data = RooDataSet("data", "data", RooArgSet(x))

    # Fill it from DataFrame
    for i_val, val in enumerate(df['fM']):
        if i_val > n_points_for_kde:
            break
        x.setVal(val)
        data.add(RooArgSet(x))

    # Build a RooKeysPdf (kernel smoothing)
    keys_pdf = RooKeysPdf("keys", "keys", x, data, RooKeysPdf.NoMirror)
    generated = keys_pdf.generate(RooArgSet(x), n_points_for_sample)

    histo_smooth = histo.Clone(f"{histo.GetName()}")
    histo_smooth.Reset("ICESM")
    for i in range(int(generated.numEntries())):
        val = generated.get(i).getRealValue("x")
        histo_smooth.Fill(val)

    histo_smooth.Scale(len(df) / histo_smooth.Integral())
    return histo_smooth

def shift_templs(cfg_corrbkgs, cutset_sel_df, pt_min, pt_max):
    # pt-differential mass shifts
    # Copy the dataframe to avoid modifying the original one
    df = cutset_sel_df.copy(deep=True)
    
    mass_shift = 0.
    if isinstance(cfg_corrbkgs["shift_mass"], float):
        logger(f"Applying constant mass shift of {cfg_corrbkgs['shift_mass']} GeV/c^2", "INFO")
        mass_shift = cfg_corrbkgs["shift_mass"]
    else:
        logger(f"Taking mass shifts from {cfg_corrbkgs['shift_mass']}", "INFO")
        shifts_file = ROOT.TFile(cfg_corrbkgs['shift_mass'], "READ")
        shifts_histo = shifts_file.Get("delta_mean_data_mc")
        for i_bin in range(1, shifts_histo.GetNbinsX()+1):
            bin_center = shifts_histo.GetBinCenter(i_bin)
            if (bin_center > pt_min and bin_center < pt_max):
                mass_shift = shifts_histo.GetBinContent(i_bin)
                break
        shifts_histo.SetDirectory(0)
        shifts_file.Close()

    logger(f"Shifting mass by {mass_shift} GeV/c^2", "INFO")
    df.loc[:, "fM"] = df["fM"] + mass_shift
    return df

def smear_templs(cfg_corrbkgs, cutset_sel_df, pt_min, pt_max):
    # pt-differential mass smearing
    # Copy the dataframe to avoid modifying the original one
    df = cutset_sel_df.copy(deep=True)
    
    mass_smear = 0.
    if isinstance(cfg_corrbkgs["smear_mass"], float):
        sigma_smear = cfg_corrbkgs["smear_mass"]
    else:
        logger(f"Taking mass smears from {cfg_corrbkgs['smear_mass']}", "INFO")
        smear_file = ROOT.TFile(cfg_corrbkgs['smear_mass'], "READ")
        smear_histo = smear_file.Get("delta_sigma_data_mc")
        for i_bin in range(1, smear_histo.GetNbinsX()+1):
            bin_center = smear_histo.GetBinCenter(i_bin)
            if (bin_center > pt_min and bin_center < pt_max):
                sigma_smear = smear_histo.GetBinContent(i_bin)
                break
        smear_histo.SetDirectory(0)
        smear_file.Close()
        if sigma_smear > 0:
            mass_smear = np.random.normal(0.0, sigma_smear, size=len(df)).astype("float32")
            logger(f"Smearing mass by sigma = {mass_smear[:10]} GeV/c^2", "INFO")
            df.loc[:, "fM"] = df["fM"] + mass_smear
        else:
            logger(f"Mass smearing value is {sigma_smear}, no smearing applied.", "WARNING")
    return df

def produce_chn_corrbkg(cfg_corrbkgs, df, outfile, chn_dir, templ_type='raw'):

    outfile.mkdir(f'{chn_dir}/{templ_type}')
    outfile.cd(f'{chn_dir}/{templ_type}')

    histo_mass = TH1F("hMass", "hMass", 700, 1.6, 2.3)
    treeFrac = ROOT.TTree("treeFrac", "treeFrac")
    treeMass = ROOT.TTree("treeMass", "treeMass")

    # Use arrays for branches
    fM_mass = np.zeros(1, dtype=np.float32)
    fM_frac = np.zeros(1, dtype=np.float32)
    fPt = np.zeros(1, dtype=np.float32)
    fCentrality = np.zeros(1, dtype=np.float32)
    fMlScore0 = np.zeros(1, dtype=np.float32)
    fMlScore1 = np.zeros(1, dtype=np.float32)

    treeMass.Branch("fM", fM_mass, "fM/F")
    treeFrac.Branch("fM", fM_frac, "fM/F")
    treeFrac.Branch("fPt", fPt, "fPt/F")
    treeFrac.Branch("fCentrality", fCentrality, "fCentrality/F")
    treeFrac.Branch("fMlScore0", fMlScore0, "fMlScore0/F")
    treeFrac.Branch("fMlScore1", fMlScore1, "fMlScore1/F")

    # Loop only once over DataFrame length
    mass_array = df['fM'].to_numpy(dtype=np.float32)
    pt_array = df['fPt'].to_numpy(dtype=np.float32)
    centrality_array = df['fCentrality'].to_numpy(dtype=np.float32)
    score0_array = df['fMlScore0'].to_numpy(dtype=np.float32)
    score1_array = df['fMlScore1'].to_numpy(dtype=np.float32)

    n_entries = len(df)
    for i in range(n_entries):
        histo_mass.Fill(mass_array[i])
        fM_mass[0] = mass_array[i]
        fM_frac[0] = mass_array[i]
        fPt[0] = pt_array[i]
        fCentrality[0] = centrality_array[i]
        fMlScore0[0] = score0_array[i]
        fMlScore1[0] = score1_array[i]
        treeFrac.Fill()
        treeMass.Fill()

    # Smoothed histogram
    histo_mass_smooth = histo_mass.Clone("hMassSmooth")
    histo_mass_smooth.Reset("ICESM")
    histo_mass_smooth = fill_smooth_histo(df, histo_mass_smooth,
                                          cfg_corrbkgs['n_smooth_points'],
                                          cfg_corrbkgs['n_points_for_kde'])
    histo_mass_smooth.Smooth(100)

    histo_mass.Write('hMassRaw')
    histo_mass_smooth.Write('hMassSmooth')
    treeFrac.Write('treeFracMassScoresBkgFD')
    treeMass.Write('treeMass')

    return histo_mass

def produce_corr_bkgs_templs(cfg):

    full_dfs = []
    tables = [[] for table in cfg["table_names"]]
    with uproot.open(cfg["input_file"]) as f:
        for table_name, table_list, table_cols_to_keep in zip(cfg["table_names"], tables, cfg["table_cols_to_keep"]):
            for iKey, key in enumerate(f.keys()):
                if table_name in key:
                    dfData = f[key].arrays(table_cols_to_keep, library='pd')
                    table_list.append(dfData)

            full_table_df = pd.concat([df for df in table_list], ignore_index=True)
            full_dfs.append(full_table_df)
    full_df = pd.concat(full_dfs, axis=1)

    ### Centrality selection
    _, (centMin, centMax) = get_centrality_bins(config["centrality"])

    cent_sel_df = full_df.query(f"fCentrality >= {centMin} and fCentrality < {centMax}")
    logger(f"Initial candidates: {len(full_df)} ----> after cent selection: {len(cent_sel_df)}", "INFO")

    # Precompute final-state masks for all entries
    decay_masks = {}
    for fin_state, info in final_states.items():
        decay_masks[fin_state] = (abs(cent_sel_df["fFlagMcMatchRec"]) == info["flag_mc_rec"])

    # Loop over pt bins
    for pt_min, pt_max in cfg["pt_bins"]:
        pt_key = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        pt_mask = (cent_sel_df.fPt >= pt_min) & (cent_sel_df.fPt < pt_max)
        df_pt = cent_sel_df[pt_mask].reset_index(drop=True)  # pt-selected DataFrame

        logger(f"Processing pt bin: {pt_key}\n", "INFO")
        os.makedirs(os.path.dirname(cfg['outfile']), exist_ok=True)
        outfile = TFile(f"{cfg['outfile']}_{pt_key}.root", "RECREATE")

        # Precompute smeared/shifted DataFrames once per pt bin
        smeared_df = smear_templs(cfg, df_pt, pt_min, pt_max) if cfg.get("smear_mass") else None
        shifted_df = shift_templs(cfg, df_pt, pt_min, pt_max) if cfg.get("shift_mass") else None
        shifted_smeared_df = shift_templs(cfg, smeared_df, pt_min, pt_max) if cfg.get("smear_mass") and cfg.get("shift_mass") else None

        # Loop over final states using precomputed masks
        for fin_state, decay_mask_full in decay_masks.items():

            decay_pt_mask = decay_mask_full[pt_mask].reset_index(drop=True)

            # Apply pt mask
            mask = pt_mask & decay_pt_mask
            n_candidates = mask.sum()
            if n_candidates <= cfg.get("min_entries", 0):
                logger(f"----> No candidates for final state: {fin_state}!", "WARNING")
                continue

            logger(f"Found {n_candidates} candidates for final state: {fin_state}", "INFO")
            chn_dir = f"{pt_key}/{fin_state}"
            make_dir_root_file(chn_dir, outfile)
            outfile.cd(chn_dir)

            # Branching ratio histogram
            hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 2, 0, 2)
            hBRs.GetXaxis().SetBinLabel(1, "MC")
            br_mc = final_states[fin_state][f'br_sim_{cfg["coll_system"]}']
            hBRs.SetBinContent(1, br_mc)
            hBRs.GetXaxis().SetBinLabel(2, "PDG")
            br_pdg = final_states[fin_state]['br_pdg']
            hBRs.SetBinContent(2, br_pdg)
            hBRs.Write()

            # Produce correlated backgrounds in a single pass per variant
            histo_mass = produce_chn_corrbkg(cfg, df_pt[decay_pt_mask], outfile, chn_dir, templ_type='raw')
            if smeared_df is not None:
                produce_chn_corrbkg(cfg, smeared_df[decay_pt_mask], outfile, chn_dir, templ_type='smear')
            if shifted_df is not None:
                produce_chn_corrbkg(cfg, shifted_df[decay_pt_mask], outfile, chn_dir, templ_type='shift')
            if shifted_smeared_df is not None:
                produce_chn_corrbkg(cfg, shifted_smeared_df[decay_pt_mask], outfile, chn_dir, templ_type='shift_smear')

            # Scale histogram
            outfile.cd(chn_dir)
            histo_mass.Scale(br_pdg / br_mc)
            histo_mass.Write('hMassScaled')

        outfile.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger("Producing correlated backgrounds templates", "INFO")
    produce_corr_bkgs_templs(config)
