'''
    Normalizations are given for the fit function to use a signal PDF and
    template from hMassTotalCorrBkgs multiplied by a common normalization 
    constant.
'''

import pandas as pd
import uproot
import numpy as np
import ROOT
import os
import sys
import argparse
import yaml
from concurrent.futures import ProcessPoolExecutor
from ROOT import TFile, TH1F
from pathlib import Path
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger, get_centrality_bins, make_dir_root_file
from corr_bkgs_brs import final_states
from data_model import get_corr_bkg_rename
from fit_utils import RebinHisto
from ROOT import TFile, RooRealVar, RooDataSet, RooArgSet, RooKeysPdf, TH1F, TH2F


def get_rebinned_mass_sel_histo(histo, mass_min, mass_max, rebin):
    '''
    Get rebinned mass histogram and selection string
    '''
    n_bins = int((mass_max - mass_min)*10000)
    # n_bins = int((mass_max - mass_min)*10000)+1
    new_histo = TH1F(f"histo_temp_{histo.GetName()}", "histo_temp", n_bins, mass_min, mass_max)
    for i_bin_new in range(1, new_histo.GetNbinsX()+1):
        bin_center = new_histo.GetBinCenter(i_bin_new)
        for i_bin_orig in range(1, histo.GetNbinsX()+1):
            if bin_center > histo.GetBinLowEdge(i_bin_orig) and bin_center < histo.GetBinLowEdge(i_bin_orig+1):
                new_histo.SetBinContent(i_bin_new, histo.GetBinContent(i_bin_orig))
                break

    new_histo.SetDirectory(0)
    new_histo = RebinHisto(new_histo, rebin, True)
    integral = new_histo.Integral("width")
    if integral <= 0:
        logger(f"Empty template histogram {histo.GetName()} in [{mass_min}, {mass_max}], cannot normalize", "ERROR")
        sys.exit(1)
    new_histo.Scale(1.0 / integral)
    return new_histo


def get_corr_bkgs_from_reference(corr_bkgs_dict, raw_yields_file, pt_label, massMin, massMax, reb):

    ry_file = TFile(raw_yields_file, "READ")
    templ_dir = f"{pt_label}/templs"
    print(f"Reading correlated backgrounds from {raw_yields_file} in directory {templ_dir}")
    hist_fracs = ry_file.Get(f"{pt_label}/hist_templ_fracs_{pt_label}")
    hist_n_entries = ry_file.Get(f"{pt_label}/hist_templ_n_entries_{pt_label}")
    for i_bin in range(1, hist_fracs.GetNbinsX()+1):
        chn = hist_fracs.GetXaxis().GetBinLabel(i_bin)
        frac = hist_fracs.GetBinContent(i_bin)
        n_entries = hist_n_entries.GetBinContent(i_bin)
        corr_bkgs_dict[chn] = {"frac": frac, "n_entries": n_entries}
        histo_chn = ry_file.Get(f"{pt_label}/templs/{chn}/histo_{chn}")
        histo_chn.SetDirectory(0)
        corr_bkgs_dict[chn]["histo"] = histo_chn
    ry_file.Close()
    return corr_bkgs_dict


def get_corr_bkg(corr_bkg_file, corr_bkg_chn, sel_string, pt_label, templ_type, output_type,
                 project=True, corr_abundances=False, sgn_d_meson='Dplus', get_smoothed=True, **kwargs):
    '''
    Get correlated background template and normalization factor
    '''
    input_folder = f"{pt_label}/{corr_bkg_chn}"
    # Check if the input folder exists in the file
    if not corr_bkg_file.GetDirectory(input_folder):
        logger(f"Input folder {input_folder} does not exist in file {corr_bkg_file.GetName()}", "WARNING")
        return None, 0, 0, {}

    if kwargs.get("verbose", False):
        logger(f"Using correlated bkg file {corr_bkg_file.GetName()} for correlated bkg source {corr_bkg_chn} from folder {input_folder}\n", "INFO")

    # Retrieve branching ratios from the histogram
    hist_pdg_mc_brs = corr_bkg_file.Get(f"{input_folder}/hBRs")
    if not hist_pdg_mc_brs:
        logger(f"Could not retrieve hBRs histogram from {input_folder} in file {corr_bkg_file.GetName()}", "ERROR")
    br_pdg = hist_pdg_mc_brs.GetBinContent(2)
    br_mc = hist_pdg_mc_brs.GetBinContent(1)
    if kwargs.get("verbose", False):
        logger(f"Branching ratios: PDG = {br_pdg}, MC = {br_mc}", "INFO")

    # Retrieve the tree and histogram for the correlated background
    templ_tree_mass = corr_bkg_file.Get(f"{input_folder}/{templ_type}/treeMass")
    if not templ_tree_mass:
        logger(f"Could not retrieve treeMass from {input_folder}/{templ_type} in file {corr_bkg_file.GetName()}", "ERROR")
    templ_tree_mass.SetDirectory(0)
    templ_histo_mass = corr_bkg_file.Get(f"{input_folder}/{templ_type}/hMass{'Smooth' if get_smoothed else 'Raw'}")
    templ_histo_mass.SetDirectory(0)
    full_tree = corr_bkg_file.Get(f"{input_folder}/{templ_type}/treeFracMassScoresBkgFD")
    templ_rdataframe_full = ROOT.RDataFrame(full_tree)
    if kwargs.get("verbose", False):
        print(f"Entries before sel: {templ_rdataframe_full.Count().GetValue()}")

    # Project the columns of the filtered RDataFrame into histograms
    templ_rdataframe_sel = templ_rdataframe_full.Filter(sel_string)
    proj_cols = {}
    if project:
        for col in templ_rdataframe_sel.GetColumnNames():
            print(f"Projecting column {col} for correlated bkg source {corr_bkg_chn}")
            col = str(col)
            histo_proj = templ_rdataframe_sel.Histo1D(col, col)
            histo_proj.GetValue()
            histo_proj.SetDirectory(0)
            proj_cols[col] = histo_proj

    n_entries = templ_rdataframe_full.Filter(sel_string).Count().GetValue()
    if kwargs.get("verbose", False):
        logger(f"Entries after sel '{sel_string}': {n_entries} for correlated bkg source {corr_bkg_chn}", "INFO")
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
        return templ_histo_mass, frac, n_entries, proj_cols
    elif output_type == "tree":
        return templ_tree_mass, frac, n_entries, proj_cols
    elif output_type == "df":
        rdf_mass = ROOT.RDataFrame(f"{input_folder}/{templ_type}/treeMass", corr_bkg_file.GetName())
        df_mass = pd.DataFrame(rdf_mass.AsNumpy())
        return df_mass, frac, n_entries, proj_cols
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
    
    sigma_smear = 0.
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
        logger(f"Smearing mass by sigma = {sigma_smear} GeV/c^2", "INFO")
        df.loc[:, "fM"] = df["fM"] + mass_smear
    else:
        logger(f"Mass smearing value is {sigma_smear}, no smearing applied.", "WARNING")
    return df


def produce_chn_corrbkg(cfg_corrbkgs, df, outfile, chn_dir, max_bkg_score, produce_grid=False, templ_type='raw'):

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
    fOriginMcRec = np.zeros(1, dtype=np.int32)

    treeMass.Branch("fM", fM_mass, "fM/F")
    treeFrac.Branch("fM", fM_frac, "fM/F")
    treeFrac.Branch("fPt", fPt, "fPt/F")
    treeFrac.Branch("fCentrality", fCentrality, "fCentrality/F")
    treeFrac.Branch("fMlScore0", fMlScore0, "fMlScore0/F")
    treeFrac.Branch("fMlScore1", fMlScore1, "fMlScore1/F")
    treeFrac.Branch("fOriginMcRec", fOriginMcRec, "fOriginMcRec/I")

    # Loop only once over DataFrame length
    mass_array = df['fM'].to_numpy(dtype=np.float32)
    pt_array = df['fPt'].to_numpy(dtype=np.float32)
    centrality_array = df['fCentrality'].to_numpy(dtype=np.float32)
    score0_array = df['fMlScore0'].to_numpy(dtype=np.float32)
    score1_array = df['fMlScore1'].to_numpy(dtype=np.float32)
    origin_array = df['fOriginMcRec'].to_numpy(dtype=np.int32)

    n_entries = len(df)
    for i in range(n_entries):
        histo_mass.Fill(mass_array[i])
        fM_mass[0] = mass_array[i]
        fM_frac[0] = mass_array[i]
        fPt[0] = pt_array[i]
        fCentrality[0] = centrality_array[i]
        fMlScore0[0] = score0_array[i]
        fMlScore1[0] = score1_array[i]
        fOriginMcRec[0] = origin_array[i]
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

    if produce_grid:
        # Evaluate the number of entries in a grid of fMlScore0 and fMlScore1 bins
        n_score0_bins = cfg_corrbkgs.get("n_score_scan_vals", 100)
        ml_score0_min, ml_score0_max = 0, treeFrac.GetMaximum("fMlScore0") if max_bkg_score is None else max_bkg_score
        logger(f"Producing grid of fMlScore0 and fMlScore1 bins with {n_score0_bins} bins in fMlScore0 from {ml_score0_min} to {ml_score0_max}\n", "INFO")
        n_score1_bins = 10
        ml_score1_min, ml_score1_max = 0, 1
        score0_bins = np.linspace(ml_score0_min, ml_score0_max, n_score0_bins + 1)
        score1_bins = np.linspace(ml_score1_min, ml_score1_max, n_score1_bins + 1)
        hist_entries_inclusive_scan = TH2F("hEntriesInclusiveScan", "hEntriesInclusiveScan;fMlScore0;fMlScore1", 
                                           n_score0_bins, ml_score0_min, ml_score0_max, 
                                           n_score1_bins, ml_score1_min, ml_score1_max)
        hist_entries_prompt_scan = TH2F("hEntriesPromptScan", "hEntriesPromptScan;fMlScore0;fMlScore1",
                                        n_score0_bins, ml_score0_min, ml_score0_max, 
                                        n_score1_bins, ml_score1_min, ml_score1_max)
        hist_entries_fd_scan = TH2F("hEntriesFDScan", "hEntriesFDScan;fMlScore0;fMlScore1",
                                    n_score0_bins, ml_score0_min, ml_score0_max, 
                                    n_score1_bins, ml_score1_min, ml_score1_max)

        for i_score0, i_score0_max in enumerate(score0_bins[1:], 1):
            for i_score1, (i_score1_min, i_score1_max) in enumerate(zip(score1_bins[:-1], score1_bins[1:]), 1):
                ml_query = f"fMlScore0 <= {i_score0_max} and fMlScore1 >= {i_score1_min} and fMlScore1 < {i_score1_max}"

                n_inc = len(df.query(ml_query))
                n_prompt = len(df.query(f"{ml_query} and fOriginMcRec == 1"))
                n_fd = len(df.query(f"{ml_query} and fOriginMcRec == 2"))

                hist_entries_inclusive_scan.SetBinContent(i_score0, i_score1, n_inc)
                hist_entries_inclusive_scan.SetBinError(i_score0, i_score1, np.sqrt(n_inc))

                hist_entries_prompt_scan.SetBinContent(i_score0, i_score1, n_prompt)
                hist_entries_prompt_scan.SetBinError(i_score0, i_score1, np.sqrt(n_prompt))

                hist_entries_fd_scan.SetBinContent(i_score0, i_score1, n_fd)
                hist_entries_fd_scan.SetBinError(i_score0, i_score1, np.sqrt(n_fd))

        hist_entries_inclusive_scan.Write('hEntriesInclusiveScan')
        hist_entries_prompt_scan.Write('hEntriesPromptScan')
        hist_entries_fd_scan.Write('hEntriesFDScan')

    return histo_mass, ml_score0_max


def produce_df(cfg):

    if not cfg.get("is_tree_creator") and not cfg.get("is_data_creator"):
        logger("Neither is_tree_creator nor is_data_creator is set to True, please specify the input type. Exiting.", "ERROR")
        sys.exit(1)
    if cfg.get("is_tree_creator") and cfg.get("is_data_creator"):
        logger("Both is_tree_creator and is_data_creator are set to True, please specify only one input type. Exiting.", "ERROR")
        sys.exit(1)

    if cfg.get("is_tree_creator"):
        tables = [[] for _ in cfg["table_names"]]

        for file_name in cfg["input_files"]:
            logger(f"Reading file {file_name} for correlated bkg template production", "INFO")

            with uproot.open(file_name) as f:
                for table_name, table_list, table_cols_to_keep in zip(
                    cfg["table_names"], tables, cfg["table_cols_to_keep"]
                ):
                    for key in f.keys():
                        print(f"Reading table {table_name} from key {key} with columns {table_cols_to_keep}")
                        if table_name in key:
                            table_list.append(
                                f[key].arrays(table_cols_to_keep, library="pd")
                            )

        full_dfs = [
            pd.concat(table_list, ignore_index=True)
            for table_list in tables
        ]
        return pd.concat(full_dfs, axis=1)

    if cfg.get("is_data_creator"):
        tables = cfg["derived_data_creator_tables"]

        output_columns = (
            tables["cand_table"]["cols_to_keep"]
            + ["fMlScore0", "fMlScore1"]
            + tables["cent_table"]["cols_to_keep"]
            + tables["mc_truth_table"]["cols_to_keep"]
        )

        all_rows = []

        for file_name in cfg["input_files"]:
            logger(f"Reading file {file_name} for correlated bkg template production", "INFO")

            with uproot.open(file_name) as file:
                for key in file.keys():
                    key = key.split(";")[0]

                    if len(key.split("/")) != 1 or 'parentFiles' in key:
                        print(f"Skipping key {key} as it is not a top-level folder")
                        continue

                    print(f"Reading key {key}")

                    table_dfs = {
                        table_name: file[f"{key}/{table["name"]};1"].arrays(library="pd")
                        for table_name, table in tables.items()
                    }

                    cand_df = table_dfs["cand_table"]
                    print(f"Candidate DataFrame columns: {cand_df.columns}")
                    ml_df   = table_dfs["ml_table"]
                    cent_df = table_dfs["cent_table"]
                    mc_df   = table_dfs["mc_truth_table"]

                    for i_row in range(len(cand_df)):
                        cand_row = cand_df.iloc[i_row]
                        ml_row = ml_df.iloc[i_row]
                        
                        col_name_idx = f"fIndex{tables['cent_table']['name'].upper().replace('O2', '')}S"
                        cent_row = cent_df.iloc[int(cand_row[col_name_idx])]
                        mc_row = mc_df.iloc[i_row]

                        row = [cand_row[col] for col in tables["cand_table"]["cols_to_keep"]]
                        row += [
                            ml_row["fMlScores"][cfg["ml_score_bkg_idx"]],
                            ml_row["fMlScores"][cfg["ml_score_fd_idx"]],
                        ]
                        row += [cent_row[col] for col in tables["cent_table"]["cols_to_keep"]]
                        row += [mc_row[col] for col in tables["mc_truth_table"]["cols_to_keep"]]

                        all_rows.append(row)

        # Print unique entries of 'O2hfdpmc'
        unique_mc_origins = pd.DataFrame(all_rows, columns=output_columns)['fFlagMcMatchRec'].unique()
        print(f"Unique entries of 'fFlagMcMatchRec' in the combined DataFrame: {unique_mc_origins}")
        # Change the name of the centrality column to 'fCentrality'
        if "fCentrality" not in output_columns:
            for col in output_columns:
                if "Cent" in col:
                    print(f"Renaming column {col} to 'fCentrality'")
                    output_columns[output_columns.index(col)] = "fCentrality"
                    break

        return pd.DataFrame(all_rows, columns=output_columns)


def produce_corr_bkgs_templs(cfg):

    # Either from TreeCreator or DerivedDataCreator
    full_df = produce_df(cfg)
    print(f"Initial candidates: {len(full_df)}")
    unique_mc_origins = full_df['fFlagMcMatchRec'].unique()
    print(f"Unique entries of 'fFlagMcMatchRec' in the pt-selected DataFrame: {sorted(unique_mc_origins)}")

    for table_name in cfg["table_names"]:
        rename_map = get_corr_bkg_rename(table_name)
        if rename_map:
            logger(f"Renaming columns of {table_name}: {rename_map}", "INFO")
            full_df = full_df.rename(columns=rename_map)

    ### Centrality selection
    _, (centMin, centMax) = get_centrality_bins(cfg["centrality"])

    if "fCentrality" not in full_df.columns:
        logger(f"No centrality column in input, assuming the sample is already restricted "
               f"to {centMin}-{centMax}%", "WARNING")
        full_df["fCentrality"] = 0.5 * (centMin + centMax)

    cent_sel_df = full_df.query(f"fCentrality >= {centMin} and fCentrality < {centMax}")
    logger(f"Initial candidates: {len(full_df)} ----> after cent selection: {len(cent_sel_df)}", "INFO")

    # Precompute final-state masks for all entries
    decay_masks = {}
    # Bring sgn_fin_state as the first key in final_states to ensure it is processed first
    sgn_fin_state = cfg["sgn_fin_state"]
    ordered_final_states = {
        sgn_fin_state: final_states[sgn_fin_state],
        **{k: v for k, v in final_states.items() if k != sgn_fin_state}
    }

    for fin_state, info in ordered_final_states.items():
        print(f"Processing final state: {fin_state} with info: {info}")
        # check if fin_state starts for any of the hadrons in cfg['include_hadrons']
        if not any(fin_state.startswith(hadron) for hadron in cfg.get("include_hadrons", [])):
            logger(f"Skipping final state {fin_state} as it does not match any of the included hadrons: {cfg.get('include_hadrons', [])}", "INFO")
            continue
        decay_mask = (abs(cent_sel_df["fFlagMcMatchRec"]) == info["flag_mc_rec"])

        if "reflected" in info and "fCandidateSelFlag" in cent_sel_df.columns:
            hypothesis_matches = (
                ((cent_sel_df["fCandidateSelFlag"] == 1) & (cent_sel_df["fFlagMcMatchRec"] ==  1)) |
                ((cent_sel_df["fCandidateSelFlag"] == 2) & (cent_sel_df["fFlagMcMatchRec"] == -1))
            )
            decay_mask = decay_mask & (~hypothesis_matches if info["reflected"] else hypothesis_matches)

        decay_masks[fin_state] = decay_mask

    # Loop over pt bins
    os.makedirs(os.path.dirname(cfg['outfile']), exist_ok=True)
    for pt_min, pt_max in cfg["pt_bins"]:
        pt_key = f"pt_{int(pt_min*100)}_{int(pt_max*100)}"
        pt_mask = (cent_sel_df.fPt >= pt_min) & (cent_sel_df.fPt < pt_max)
        df_pt = cent_sel_df[pt_mask].reset_index(drop=True)  # pt-selected DataFrame
        # Print unique entries of 'fFlagMcMatchRec' in the pt-selected DataFrame

        logger(f"Processing pt bin: {pt_key}\n", "INFO")
        outfile = TFile(f"{cfg['outfile']}_{pt_key}.root", "RECREATE")

        # Precompute smeared/shifted DataFrames once per pt bin
        smeared_df = smear_templs(cfg, df_pt, pt_min, pt_max) if cfg.get("smear_mass") else None
        shifted_df = shift_templs(cfg, df_pt, pt_min, pt_max) if cfg.get("shift_mass") else None
        shifted_smeared_df = shift_templs(cfg, smeared_df, pt_min, pt_max) if cfg.get("smear_mass") and cfg.get("shift_mass") else None

        # Loop over final states using precomputed masks
        max_sgn_chn_bkg_score = None
        for i_state, (fin_state, decay_mask_full) in enumerate(decay_masks.items()):

            # Print entries of dataframe before
            print(f"Final state: {fin_state}, entries before pt selection: {decay_mask_full.sum()}")

            decay_pt_mask = decay_mask_full[pt_mask].reset_index(drop=True)

            # Apply pt mask
            n_candidates = decay_pt_mask.sum()
            print(f"n_candidates: {n_candidates} for final state: {fin_state} in pt bin: {pt_key}")
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
            histo_mass, max_sgn_chn_bkg_score = produce_chn_corrbkg(cfg, df_pt[decay_pt_mask], outfile, chn_dir, max_sgn_chn_bkg_score, produce_grid=True, templ_type='raw')
            if smeared_df is not None:
                produce_chn_corrbkg(cfg, smeared_df[decay_pt_mask], outfile, chn_dir, max_sgn_chn_bkg_score, produce_grid=True, templ_type='smear')
            if shifted_df is not None:
                produce_chn_corrbkg(cfg, shifted_df[decay_pt_mask], outfile, chn_dir, max_sgn_chn_bkg_score, produce_grid=True, templ_type='shift')
            if shifted_smeared_df is not None:
                produce_chn_corrbkg(cfg, shifted_smeared_df[decay_pt_mask], outfile, chn_dir, max_sgn_chn_bkg_score, produce_grid=True, templ_type='shift_smear')

            # Scale histogram
            outfile.cd(chn_dir)
            histo_mass.Scale(br_pdg / br_mc)
            histo_mass.Write('hMassScaled')

            # Scale hEntriesScan by the same factor
            outfile.cd(f"{chn_dir}/raw/")
            hist_entries_inclusive_scan = outfile.Get(f"{chn_dir}/raw/hEntriesInclusiveScan")
            hist_entries_inclusive_scan.Scale(br_pdg / br_mc)
            hist_entries_inclusive_scan.Write('hEntriesInclusiveScanScaled')
            hist_entries_prompt_scan = outfile.Get(f"{chn_dir}/raw/hEntriesPromptScan")
            hist_entries_prompt_scan.Scale(br_pdg / br_mc)
            hist_entries_prompt_scan.Write('hEntriesPromptScanScaled')
            hist_entries_fd_scan = outfile.Get(f"{chn_dir}/raw/hEntriesFDScan")
            hist_entries_fd_scan.Scale(br_pdg / br_mc)
            hist_entries_fd_scan.Write('hEntriesFDScanScaled')

        # Save also ratios of hEntriesInclusiveScanScaled to the one of the signal to obtain fractions
        hist_entries_inclusive_scan_signal = outfile.Get(f"{pt_key}/{cfg['sgn_fin_state']}/raw/hEntriesInclusiveScanScaled")
        hist_entries_prompt_scan_signal = outfile.Get(f"{pt_key}/{cfg['sgn_fin_state']}/raw/hEntriesPromptScanScaled")
        hist_entries_fd_scan_signal = outfile.Get(f"{pt_key}/{cfg['sgn_fin_state']}/raw/hEntriesFDScanScaled")
        for fin_state in final_states.keys():
            chn_dir = f"{pt_key}/{fin_state}"
            if not outfile.GetDirectory(chn_dir):
                logger(f"Directory {chn_dir} not found in output file, skipping fraction computation", "WARNING")
                continue
            outfile.cd(chn_dir)
            hist_entries_inclusive_scan_scaled = outfile.Get(f"{chn_dir}/raw/hEntriesInclusiveScanScaled")
            hist_entries_inclusive_scan_frac = hist_entries_inclusive_scan_scaled.Clone("hEntriesInclusiveScanFrac")
            hist_entries_inclusive_scan_frac.Divide(hist_entries_inclusive_scan_signal)
            hist_entries_inclusive_scan_frac.Write('hEntriesInclusiveScanFrac')
            hist_entries_prompt_scan_scaled = outfile.Get(f"{chn_dir}/raw/hEntriesPromptScanScaled")
            hist_entries_prompt_scan_frac = hist_entries_prompt_scan_scaled.Clone("hEntriesPromptScanFrac")
            hist_entries_prompt_scan_frac.Divide(hist_entries_prompt_scan_signal)
            hist_entries_prompt_scan_frac.Write('hEntriesPromptScanFrac')
            hist_entries_fd_scan_scaled = outfile.Get(f"{chn_dir}/raw/hEntriesFDScanScaled")
            hist_entries_fd_scan_frac = hist_entries_fd_scan_scaled.Clone("hEntriesFDScanFrac")
            hist_entries_fd_scan_frac.Divide(hist_entries_fd_scan_signal)
            hist_entries_fd_scan_frac.Write('hEntriesFDScanFrac')

        outfile.Close()


def _check_corr_bkgs_one_pt_bin(args):
    config, i_pt, templ_dir = args
    sgn_chn_name = config["corr_bkgs"]["sgn_chn"]
    cfg_cutsets = config["cut_variation"]["uncorr_bdt_cut"]
    prod_type_queries = {
        "Inclusive": "fOriginMcRec == 1 or fOriginMcRec == 2",
        "Prompt": "fOriginMcRec == 1",
        "FD": "fOriginMcRec == 2"
    }

    pt_min, pt_max = config["ptbins"][i_pt:i_pt + 2]
    pt_key = f"pt_{int(pt_min*100)}_{int(pt_max*100)}"
    logger(f"Processing pt bin: {pt_min} - {pt_max} GeV/c", "INFO")
    mass_min, mass_max = config["v2extraction"]["MassFitRanges"][i_pt]
    templates_file = TFile.Open(f"{templ_dir}/templs_{pt_key}.root", "READ")
    pt_dir = templates_file.GetDirectory(pt_key)
    channels = [key.GetName() for key in pt_dir.GetListOfKeys()]

    out_path = f"{templ_dir}/corr_bkgs_study_{pt_key}.root"
    out_file = TFile.Open(out_path, "RECREATE")

    for i_cutset in range(len(cfg_cutsets["bkg_max"][i_pt])):
        logger(f"   Processing cutset {i_cutset} for pt bin {pt_min} - {pt_max} GeV/c", "INFO")
        score0_max = cfg_cutsets["bkg_max"][i_pt][i_cutset]
        score1_min = cfg_cutsets["sig"][i_pt][i_cutset]
        score1_max = cfg_cutsets["sig"][i_pt][i_cutset + 1]

        for prod_type, query in prod_type_queries.items():
            logger(f"     Processing production type {prod_type} for cutset {i_cutset}, pt bin {pt_min} - {pt_max} GeV/c", "INFO")
            results = {}

            for chn_name in channels:
                sel_string = (
                    f"{query} and fMlScore0 <= {score0_max} and "
                    f"fMlScore1 >= {score1_min} and fMlScore1 < {score1_max} and "
                    f"fM >= {mass_min} and fM < {mass_max}"
                )
                _, frac, n_entries, _ = get_corr_bkg(
                    templates_file, chn_name, sel_string, pt_key, "raw", "hist"
                )
                results[chn_name] = {"n_entries": n_entries, "frac": frac}

            sorted_channels = sorted(results.items(), key=lambda x: (x[0] != sgn_chn_name, -x[1]["frac"]))

            h_n = TH1F(f"hEntriesCutset{i_cutset}{prod_type}", f"hEntriesCutset{i_cutset}{prod_type}",
                       len(sorted_channels), 0, len(sorted_channels))
            h_frac = TH1F(f"hAnchorFracCutset{i_cutset}{prod_type}", f"hAnchorFracCutset{i_cutset}{prod_type}",
                          len(sorted_channels), 0, len(sorted_channels))

            anchor_frac = results[sgn_chn_name]["frac"]

            for i_bin, (chn_name, info) in enumerate(sorted_channels, 1):
                h_n.GetXaxis().SetBinLabel(i_bin, chn_name)
                h_n.SetBinContent(i_bin, info["n_entries"])
                h_frac.GetXaxis().SetBinLabel(i_bin, chn_name)
                h_frac.SetBinContent(i_bin, info["frac"] / anchor_frac if anchor_frac else 0.)

            out_file.mkdir(f"{pt_key}/cutset_{i_cutset}/{prod_type}")
            out_file.cd(f"{pt_key}/cutset_{i_cutset}/{prod_type}")
            h_n.Write()
            h_frac.Write()

    logger(f"Finished processing pt bin: {pt_min} - {pt_max} GeV/c, output saved to {out_path}", "WARNING")
    templates_file.Close()
    out_file.Close()
    return out_path


def check_corr_bkgs_fracs(config, templ_dir, nworkers=15):
    n_pt_bins = len(config["ptbins"]) - 1
    tasks = [(config, i_pt, templ_dir) for i_pt in range(n_pt_bins)]

    if nworkers > 1:
        with ProcessPoolExecutor(max_workers=nworkers) as pool:
            out_files = list(pool.map(_check_corr_bkgs_one_pt_bin, tasks))
    else:
        out_files = [_check_corr_bkgs_one_pt_bin(task) for task in tasks]

    logger(f"Produced {len(out_files)} correlated-bkg ROOT files", "INFO")
    return out_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("-cfg_bkg", "--cfg_corr_bkgs", metavar="text", default=None, help="corr bkgs configuration file")
    parser.add_argument("-cfg_an", "--cfg_analysis", metavar="text", default=None, help="flow configuration file")
    parser.add_argument("-templ_dir", "--templates_dir", metavar="text", default=None, help="templates directory")
    args = parser.parse_args()

    if args.cfg_corr_bkgs is not None:
        logger("Producing correlated backgrounds templates", "INFO")
        with open(args.cfg_corr_bkgs, 'r') as f:
            cfg_corr_bkgs = yaml.safe_load(f)

        produce_corr_bkgs_templs(cfg_corr_bkgs)

    if args.cfg_analysis is not None:
        with open(args.cfg_analysis, 'r') as f:
            cfg_analysis = yaml.safe_load(f)

        check_corr_bkgs_fracs(cfg_analysis, args.templates_dir)
