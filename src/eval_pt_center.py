import argparse
import sys
import os
import re
import array
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position
import pandas as pd
import numpy as np
import seaborn as sns
import awkward as ak
import matplotlib.pyplot as plt
import itertools
sys.path.append("./flareflyfitter/")
from raw_yield_fitter import RawYieldFitter
import yaml
import ROOT
from ROOT import TFile, TH1F, TH1, TObject
import awkward as ak
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from utils import logger, get_centrality_bins
from load_utils import load_aod_file
from matplotlib import gridspec
import uproot
from multiprocessing import Pool, cpu_count
ROOT.gROOT.SetBatch(True)
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(20)
tf.config.threading.set_inter_op_parallelism_threads(20)

def process_pt_bin(i_pt_bin, pt_min, pt_max, cfg, cutset_files, outdir):

    logger(f"Pt bin {i_pt_bin}: {pt_min} - {pt_max}", level="INFO")
    # Load input
    pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
    has_sp_cent = True
    cfg_fit = cfg["v2extraction"]
    prep_dir = f"{cfg['ptCenter'].get('PrepDir', cfg['outdir'])}/preprocess/{pt_str}/TreesPtCenterSp"
    if not os.path.exists(prep_dir):
        prep_dir = f"{cfg['outdir']}/preprocess/{pt_str}/TreesPtCenter"
        has_sp_cent = False
        logger(f"Using tree without SP and centrality!", level="WARNING")
    downsample_frac = cfg['ptCenter']['downsample_fracs'][i_pt_bin] if cfg['ptCenter'].get('downsample_fracs') else 1.0
    df = load_aod_file(f"{prep_dir}/AO2D_{pt_str}.root", has_sp_cent, downsample_frac=downsample_frac)

    out_dir_pt = f"{outdir}/{pt_str}"
    if downsample_frac < 1.0:
        out_dir_pt += f"_downsampled_{downsample_frac}"
    os.makedirs(out_dir_pt, exist_ok=True)
    os.makedirs(out_dir_pt + "/fits", exist_ok=True)
    os.makedirs(out_dir_pt + "/vars", exist_ok=True)

    # Check if root file exists, if not create it and the histograms for the averages
    histos_avgs = {}
    print(f"Checking if root file {out_dir_pt}/pt_center.root exists ...")
    if os.path.exists(f"{out_dir_pt}/pt_center.root"):
        print(f"Updating existing root file {out_dir_pt}/pt_center.root and histograms for averages ...")
        out_file = TFile.Open(f"{out_dir_pt}/pt_center.root", "update")
    else:
        out_file = TFile.Open(f"{out_dir_pt}/pt_center.root", "recreate")

    histos_avgs[f"h_ry_{cfg_fit['SgnFuncLabel']}"] = out_file.Get(f'h_ry_{cfg_fit['SgnFuncLabel']}')
    if not isinstance(histos_avgs[f"h_ry_{cfg_fit['SgnFuncLabel']}"], TH1):
        histos_avgs[f"h_ry_{cfg_fit['SgnFuncLabel']}"] = TH1F(f"h_ry_{cfg_fit['SgnFuncLabel']}",
                                                                f"h_ry_{cfg_fit['SgnFuncLabel']}",
                                                                len(cutset_files),
                                                                array.array('d', [-0.5] + [i+0.5 for i in range(len(cutset_files))]))
    print(f"Got histogram for signal function from file: {histos_avgs[f'h_ry_{cfg_fit['SgnFuncLabel']}']}")

    sgn_funcs = {} # More info for signal functions, a dictionary is better
    sgn_funcs[cfg_fit['SgnFuncLabel']] = {
        'func': cfg_fit['SgnFunc'][i_pt_bin] if isinstance(cfg_fit['SgnFunc'], list) else cfg_fit['SgnFunc'],
        'part': cfg['Dmeson']
    }

    logger(f"Adding signal function: {sgn_funcs[cfg_fit['SgnFuncLabel']]}, {cfg_fit['SgnFuncLabel']} ... ", level="INFO")
    hasSecPeak = False
    if cfg_fit.get('InclSecPeak'):
        logger("Including second peak signal function ... ", level="INFO")
        include_sec_peak = cfg_fit['InclSecPeak'][i_pt_bin] if isinstance(cfg_fit['InclSecPeak'], list) else cfg_fit['InclSecPeak']
        if include_sec_peak:
            sgn_func_sec_peak = cfg_fit['SgnFuncSecPeak'][i_pt_bin] if isinstance(cfg_fit['SgnFuncSecPeak'], list) else cfg_fit['SgnFuncSecPeak']
            logger(f"Adding second peak signal function: {sgn_func_sec_peak} ... ", level="INFO")
            sgn_funcs[cfg_fit['SgnFuncSecPeakLabel']] = {
                'func': sgn_func_sec_peak,
                'part': 'Dplus' if cfg['Dmeson'] == 'Ds' else 'Dstar',
            }
            histos_avgs[f"h_ry_{cfg_fit['SgnFuncSecPeakLabel']}"] = out_file.Get(f'h_ry_{cfg_fit['SgnFuncSecPeakLabel']}')
            if not isinstance(histos_avgs[f"h_ry_{cfg_fit['SgnFuncSecPeakLabel']}"], TH1):
                logger(f"Could not get histogram for second peak from file, creating it ... ", level="WARNING")
                histos_avgs[f"h_ry_{cfg_fit['SgnFuncSecPeakLabel']}"] = TH1F(f"h_ry_{cfg_fit['SgnFuncSecPeakLabel']}",
                                                                                f"h_ry_{cfg_fit['SgnFuncSecPeakLabel']}",
                                                                                len(cutset_files),
                                                                                array.array('d', [-0.5] + [i+0.5 for i in range(len(cutset_files))]))

            hasSecPeak = True


    # Initialize fitter
    fitter = RawYieldFitter(cfg['Dmeson'], pt_min, pt_max, pt_str, "flarefly", verbose=True)
    fitter.set_fit_range(cfg_fit['MassFitRanges'][i_pt_bin][0], cfg_fit['MassFitRanges'][i_pt_bin][1])

    last_fd_cut = -1.0
    for i_cutset, cutset_file in enumerate(cutset_files):
        with open(cutset_file, 'r') as cs_file:
            cutset_cfg = yaml.safe_load(cs_file)
        fd_cut = cutset_cfg["ScoreFD"]["min"][i_pt_bin]
        if fd_cut == last_fd_cut:
            logger(f"Skipping cutset file {cutset_file} as it has the same FD cut as the previous one ({fd_cut}) ... ", level="INFO")
            continue
        logger(f"Processing cutset file {cutset_file} ... ", level="INFO")
        cutset_suffix = os.path.basename(cutset_file).replace('.yml', '').split('_')[-1]
        # Check if parquet exists, if yes skip the fit and go to the next cutset
        last_fd_cut = fd_cut
        if os.path.exists(f"{out_dir_pt}/sweights_cutset_{cutset_suffix}.parquet"):
            logger(f"sWeights parquet file {out_dir_pt}/sweights_cutset_{cutset_suffix}.parquet already exists, skipping fit for cutset {cutset_suffix} ... ", level="INFO")
            continue

        # Setup fitter
        fitter.add_bkg_func(cfg_fit['BkgFunc'][i_pt_bin] if isinstance(cfg_fit['BkgFunc'], list) else cfg_fit['BkgFunc'], "Comb. bkg")
        for i_sgn, (label, sgn_func) in enumerate(sgn_funcs.items()):
            logger(f"Adding signal function: {sgn_func}, {label} ... ", level="INFO")
            fitter.add_sgn_func(sgn_func['func'], label, sgn_func['part'])

        score_bkg_min = cutset_cfg["ScoreBkg"]["min"][i_pt_bin]
        score_bkg_max = cutset_cfg["ScoreBkg"]["max"][i_pt_bin]
        score_fd_min = cutset_cfg["ScoreFD"]["min"][i_pt_bin]
        score_fd_max = cutset_cfg["ScoreFD"]["max"][i_pt_bin]

        # Query the dataframe
        mass_min, mass_max = cfg_fit["MassFitRanges"][i_pt_bin]
        sel_string = f"fMlScore0 >= {score_bkg_min} and fMlScore0 < {score_bkg_max} and " \
                     f"fMlScore1 >= {score_fd_min} and fMlScore1 < {score_fd_max} and " \
                     f"fM >= {mass_min} and fM <= {mass_max}"
        logger(f"Selection string: {sel_string}", level="INFO")
        sel_df = df.query(sel_string).reset_index(drop=True)
        logger(f"Total entries: {len(df)}, selected entries: {len(sel_df)}", level="INFO")
        fitter.set_name(f"{pt_str}_{cutset_suffix}")
        fitter.set_data_to_fit_df(sel_df)
        if cfg_fit.get('Rebin'):
            fitter.set_rebin(cfg_fit['Rebin'][i_pt_bin]) if isinstance(cfg_fit['Rebin'], list) else fitter.set_rebin(cfg_fit['Rebin'])

        # Add correlated background if specified
        if cfg.get('corr_bkgs'):
            fitter.add_corr_bkgs(cfg['corr_bkgs'], f"{cfg['outdir']}/corrbkgs/", sel_string.replace(' and ', ' && '), pt_min, pt_max)

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            sel_df['fM'].hist(bins=100, alpha=0.5, range=(mass_min, mass_max))
            ax.set_xlabel('fM')
            ax.set_ylabel('Counts')
            fig.savefig(f"{out_dir_pt}/fits/fM_raw_{cutset_suffix}.pdf", dpi=300, bbox_inches="tight")

        fitter.setup()

        # Prefit the MC prompt enhanced cut to fix the tails, binned fit
        if cfg_fit.get('FixSgnFromMC'):
            fitter.set_fix_sgn_to_mc_prefit(True)
            if i_cutset == 0:
                fitter.prefit_mc(f"{cfg['outdir']}/corr_bkgs/templs_{pt_str}.root")
                fitter.plot_mc_prefit(False, True, loc=["lower left", "upper left"],
                                      path=f"{out_dir_pt}/", out_file=out_file)
                fitter.plot_raw_residuals_mc_prefit(path=f"{out_dir_pt}/fM_mc_prefit_residuals_{cutset_suffix}.pdf")

        cfg_pars_dict = {'init_pars_sgn': [], 'fix_pars_sgn': []}
        pt_center = 0.5 * (pt_min + pt_max)
        for setting in cfg_fit.get("InitFitPars", []):
            # check if this setting applies to the current pT bin
            if not any(low < pt_center < high for low, high in setting["pt_ranges"]):
                continue
            for par in setting["pars"]:
                for par_name, par_lims in par.items():
                    if par_lims[1] == par_lims[2]:  # fixed parameter
                        cfg_pars_dict['fix_pars_sgn'].append((par_lims[3], par_name, par_lims[0], [par_lims[1], par_lims[2]]))
                    else:
                        cfg_pars_dict['init_pars_sgn'].append((par_lims[3], par_name, par_lims[0], [par_lims[1], par_lims[2]]))
            break

        if cfg_fit.get('ParsFromFile'):
            do_init = True
            if isinstance(cfg_fit['ParsFromFile'], list) and not cfg_fit['ParsFromFile'][i_pt_bin]:
                do_init = False  # do not init for this pt bin
            if do_init:
                file_path = cfg_fit['ParsFromFile'][i_pt_bin] if isinstance(cfg_fit['ParsFromFile'], list) else cfg_fit['ParsFromFile']
                logger(f"Trying to fix signal parameters from file {file_path} for pt {pt_min} - {pt_max} GeV/c ...", "INFO")
                try:
                    pt_dir = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
                    fixParsFile = TFile.Open(file_path, 'r')
                    fixParsHisto = fixParsFile.Get(f'{pt_dir}/hist_signal_func')
                    fixParsHisto.SetDirectory(0)
                    for iBin in range(1, fixParsHisto.GetNbinsX()+1):
                        if iBin <= 3:    # skip integral, mean, sigma
                            continue
                        binLabel = fixParsHisto.GetXaxis().GetBinLabel(iBin)
                        binContent = fixParsHisto.GetBinContent(iBin)
                        logger(f"Appending fixed parameter from {file_path}: {binLabel} = {binContent}", level="INFO")
                        cfg_pars_dict['fix_pars_sgn'].append((0, binLabel, binContent, [binContent, binContent]))
                    fixParsFile.Close()
                except Exception as e:
                    logger(f"Exception {e} caught when trying to fix signal parameters from file {file_path}.", level='ERROR')

        if len(cfg_pars_dict['init_pars_sgn']) == 0:
            cfg_pars_dict.pop('init_pars_sgn', None)
        if len(cfg_pars_dict['fix_pars_sgn']) == 0:
            cfg_pars_dict.pop('fix_pars_sgn', None)
        if cfg_pars_dict.get('fix_pars_sgn') or cfg_pars_dict.get('init_pars_sgn'):
            fitter.set_fit_pars_from_config(cfg_pars_dict)

        valid, converged = fitter.fit()
        fit_info, sgn_pars, sgn_pars_uncs, bkg_pars, bkg_pars_uncs = fitter.get_fit_info()
        logger(f"Fit valid: {valid}, converged: {converged} for pt bin {i_pt_bin} and cutset {cutset_suffix}", level="INFO")
        removedSecPeak = False
        if hasSecPeak:
            print("Checking fit health for second peak ... ")
            print(f"sgn_pars: {sgn_pars}")

            remove_peak  = False
            if not valid:
                logger("Fit invalid → removing second peak", level="ERROR")
                remove_peak = True
            elif sgn_pars[f"sigma_{cfg_fit['SgnFuncSecPeakLabel']}"] > cfg['ptCenter']['MaxSecPeakSigma']:
                logger("Second peak sigma too large → removing peak", level="ERROR")
                remove_peak = True
            elif fit_info[cfg_fit['SgnFuncSecPeakLabel']]['ry'] < 1000:
                logger("Second peak yield too small → removing peak", level="ERROR")
                remove_peak = True
            else:
                logger("Fit with second peak is healthy", level="INFO")

            if remove_peak:
                fitter.remove_sec_peak()
                fitter.setup()
                valid, converged = fitter.fit()
                logger(
                    f"After removing second peak: valid={valid}, converged={converged}",
                    level="INFO"
                )
                removedSecPeak = True

        fitter.plot_fit(False, True, loc=["lower left", "upper left"], \
                        path=f"{out_dir_pt}/fits/fM_fit_{cutset_suffix}.pdf",
                        out_file=out_file)

        fit_info, sgn_pars, sgn_pars_uncs, bkg_pars, bkg_pars_uncs = fitter.get_fit_info()
        for label in sgn_funcs.keys():
            if hasSecPeak and removedSecPeak and label == cfg_fit['SgnFuncSecPeakLabel']:
                logger(f"Skipping filling ry for second peak for pt bin {i_pt_bin} and cutset {cutset_suffix} as the peak was removed in the fit", level="WARNING")
                continue
            print(f"histos_avgs[f'h_ry_{label}']: {histos_avgs[f'h_ry_{label}']}")
            histos_avgs[f"h_ry_{label}"].SetBinContent(i_cutset + 1, fit_info[label]["ry"])
            histos_avgs[f"h_ry_{label}"].SetBinError(i_cutset + 1, fit_info[label]["ry_unc"])

        s_weights_sgn = fitter.get_sweights_sgn(cfg_fit['SgnFuncLabel'])
        s_weights_sec_peak = fitter.get_sweights_sgn(cfg_fit['SgnFuncSecPeakLabel']) if hasSecPeak and not removedSecPeak else None

        # Build sWeights
        sgn_weights = np.asarray(s_weights_sgn)
        if s_weights_sec_peak is not None:
            bkg_weights = (np.ones(len(sgn_weights)) - sgn_weights - np.asarray(s_weights_sec_peak))
        else:
            bkg_weights = np.ones(len(sgn_weights)) - sgn_weights

        with open(f"{out_dir_pt}/fits_valid.txt", "a") as f:
            computed_sweights = True if s_weights_sgn is not None else False
            f.write(
                    f"{fitter.get_name()}: "
                    f"fit_res.valid -> {valid}, "
                    f"fit_res.converged -> {converged}, "
                    f"sweights computed -> {computed_sweights} "
                    f"sgn sweights: {len(sgn_weights)} vs {len(sel_df['fM'])}\n"
                    )

        if len(sgn_weights) != len(sel_df['fM']):
            sel_df = sel_df.query("fM > @mass_min and fM < @mass_max").reset_index(drop=True)

        # Add sWeights to the dataframe and save it
        sel_df['sgn_weight'] = sgn_weights
        sel_df['bkg_weight'] = bkg_weights
        if s_weights_sec_peak is not None:
            sel_df['sec_peak_weight'] = s_weights_sec_peak

        # Reset fitter for new cutset: correlated bkg fracs will change
        fitter.reset()
        sel_df.to_parquet(f"{out_dir_pt}/sweights_cutset_{cutset_suffix}.parquet", index=False)
        print(f"Saved dataframe with sWeights to {out_dir_pt}/sweights_cutset_{cutset_suffix}.parquet")

    out_file.cd()
    for label in sgn_funcs.keys():
        print(f"Writing histogram for {label} to file: {histos_avgs[f'h_ry_{label}'].GetName()}")
        histos_avgs[f"h_ry_{label}"].Write(histos_avgs[f"h_ry_{label}"].GetName(), TObject.kOverwrite)
    out_file.Close()

    logger(f"Finished pt bin {i_pt_bin}", level="INFO")
    return i_pt_bin


def run_fits(cfg_file_name, n_workers=1):
    # Read the configuration file
    with open(cfg_file_name, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    # Retrieve cutsets configs
    try:
        cutsets_dir = os.path.join(cfg['outdir'], f"vn_extr_{cfg['suffix']}_combined/cutsets")
        cutset_files = [os.path.join(cutsets_dir, f) for f in os.listdir(cutsets_dir) if f.endswith('.yml')]
        out_dir_type = "combined"
    except Exception as e:
        logger(f"Could not find combined cutsets, trying correlated cutsets ... ", level="WARNING")
        cutsets_dir = os.path.join(cfg['outdir'], f"vn_extr_{cfg['suffix']}_correlated/cutsets")
        cutset_files = [os.path.join(cutsets_dir, f) for f in os.listdir(cutsets_dir) if f.endswith('.yml')]
        out_dir_type = "correlated"
    cutset_files.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))

    outdir = f"{cfg['outdir']}/vn_extr_{cfg['suffix']}_{out_dir_type}/ptcenter"

    # Loop over cutset configs
    s_weights = {}

    pt_bins_to_keep = cfg['ptCenter'].get("ptBinsToKeep", None)
    jobs = []
    for i_pt_bin, (pt_min, pt_max) in enumerate(zip(cfg["ptbins"][:-1], cfg["ptbins"][1:])):
        if pt_bins_to_keep is not None and i_pt_bin not in pt_bins_to_keep:
            logger(f"Skipping pt bin {i_pt_bin} as it's not in the list of bins to keep", level="INFO")
            continue
        jobs.append((i_pt_bin, pt_min, pt_max, cfg, cutset_files, outdir))

    if n_workers == 1:
        for job in jobs:
            process_pt_bin(*job)
    else:
        with Pool(n_workers) as pool:
            results = pool.starmap(process_pt_bin, jobs)
        # Throw exceptions if any job failed
        for result in results:
            if isinstance(result, Exception):
                raise result
        # with Pool(n_workers) as pool:
        #     results = pool.starmap(process_pt_bin, jobs)


def eval_pt_center(cfg_file_name):
    # Read the configuration file
    with open(cfg_file_name, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    with TFile.Open(cfg["projections"]["Resolution"]) as reso_file:
        det_A = cfg["projections"].get('detA', 'FT0c')
        det_B = cfg["projections"].get('detB', 'FV0a')
        det_C = cfg["projections"].get('detC', 'TPCtot')
        logger(f"Getting resolution histogram from file {cfg['projections']['Resolution']} for triplet {det_A}_{det_B}_{det_C}",  "WARNING")
        reso_hist = reso_file.Get(f'{det_A}_{det_B}_{det_C}/histo_reso_delta_cent')
        resolution = reso_hist.GetBinContent(1)
        reso_hist.SetDirectory(0)

    infer_vars = ['fPt', 'fScalarProd', 'fCent']
    infer_vars_labels = ['fPt (GeV/c)', 'fScalarProd', 'fCent (%)']
    pt_bins = cfg["ptbins"]
    df_dir = f"{cfg['outdir']}/vn_extr_{cfg['suffix']}_combined/ptcenter"
    down_sample_fracs = [f"_downsampled_{str(frac)}" for frac in cfg['ptCenter']['downsample_fracs']] if cfg['ptCenter'].get('downsample_fracs') else [""] * (len(pt_bins) - 1)
    # Strip _downsampled_1.0
    down_sample_fracs = [frac.replace("_downsampled_1.0", "") for frac in down_sample_fracs]

    outfile = TFile.Open(f"{df_dir}/pt_center_summary.root", "recreate")
    hist_summary_avg_pt = TH1F("h_summary_avg_pt", "h_summary_avg_pt", len(cfg["ptbins"])-1, array.array('d', cfg["ptbins"]))
    hist_summary_pt_shifts = TH1F("h_summary_pt_shifts", "h_summary_pt_shifts", len(cfg["ptbins"])-1, array.array('d', cfg["ptbins"]))


    for i_pt_bin, (pt_min, pt_max) in enumerate(zip(cfg["ptbins"][:-1], cfg["ptbins"][1:])):
        logger(f"Processing pt bin {pt_min} - {pt_max} GeV/c ... ", level="INFO")
        pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"

        # Get raw yields histogram
        with TFile.Open(f"{df_dir}/{pt_str}{down_sample_fracs[i_pt_bin]}/pt_center.root") as f:
            h_ry_sgn = f.Get(f"h_ry_{cfg['v2extraction']['SgnFuncLabel']}")
            h_ry_sgn.SetDirectory(0)
            if cfg['v2extraction'].get('InclSecPeak') and cfg['v2extraction']['InclSecPeak'][i_pt_bin]:
                h_ry_sec_peak = f.Get(f"h_ry_{cfg['v2extraction']['SgnFuncSecPeakLabel']}")
                h_ry_sec_peak.SetDirectory(0)

        outfile.mkdir(pt_str)
        outfile.cd(pt_str)
        out_dir_pt = f"{df_dir}/{pt_str}{down_sample_fracs[i_pt_bin]}"
        sweights_dfs = [path for path in os.listdir(out_dir_pt) if path.startswith("sweights_cutset_") and path.endswith(".parquet")]
        # sort by cutset index
        sweights_dfs.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)))
        print(f"sweights_dfs: {sweights_dfs}")
        
        # Initialize histograms for averages
        histos_avgs = {}
        for var in infer_vars:
            histos_avgs[f"h_{var}_sgn"] = TH1F(f"h_{var}_sgn", f"h_{var}_sgn", len(sweights_dfs), array.array('d', [-0.5] + [i+0.5 for i in range(len(sweights_dfs))]))
            histos_avgs[f"h_{var}_bkg"] = TH1F(f"h_{var}_bkg", f"h_{var}_bkg", len(sweights_dfs), array.array('d', [-0.5] + [i+0.5 for i in range(len(sweights_dfs))]))

        for sweights_df in sweights_dfs:
            print(f"Processing sWeights dataframe {sweights_df} ...")
            suffix = sweights_df.replace("sweights_cutset_", "").replace(".parquet", "")
            i_cutset = int(suffix)
            outfile.mkdir(f"{pt_str}/Cutset_{suffix}")
            outfile.cd(f"{pt_str}/Cutset_{suffix}")
            df_with_sweights = pd.read_parquet(f"{out_dir_pt}/{sweights_df}")
            for var, label in zip(infer_vars, infer_vars_labels):
                logger(f"    Drawing {var}", level="INFO")

                # Create figure with two subplots (distros and ratio)
                fig, ax = plt.subplots(figsize=(8, 8))

                # Bins
                if var == "fCent":
                    cent_min, cent_max = get_centrality_bins(cfg['centrality'])[1]
                    nbins = cent_max - cent_min
                    var_range = (df_with_sweights[var].min()-0.5, df_with_sweights[var].max()+0.5)
                elif var == "fScalarProd":
                    nbins = 60
                    var_range = (-4, 4)
                    df_with_sweights[var] = df_with_sweights[var] / resolution
                else:
                    nbins = 20 if var == "fPt" else 60
                    var_range = (df_with_sweights[var].min(), df_with_sweights[var].max())
                bin_edges = np.linspace(var_range[0], var_range[1], nbins + 1)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                bin_widths = np.diff(bin_edges)

                values = df_with_sweights[var].to_numpy()
                # Signal histogram + errors
                logger(f"df_with_sweights['sgn_weight'].shape: {df_with_sweights['sgn_weight'].shape}, bkg_weights.shape: {df_with_sweights['bkg_weight'].shape}", level="INFO")
                sgn_bin_contents, _ = np.histogram(values, bins=bin_edges, weights=df_with_sweights['sgn_weight'])
                sgn_squared_uncs, _ = np.histogram(values, bins=bin_edges, weights=df_with_sweights['sgn_weight']**2)
                sgn_err = np.sqrt(sgn_squared_uncs)
                ax.step(bin_edges[:-1], sgn_bin_contents, where="post", label="Signal", color="#1f77b4")
                ax.errorbar(bin_centers, sgn_bin_contents, yerr=sgn_err, xerr=bin_widths / 2, fmt="o", color="#1f77b4", capsize=2)

                # Background histogram + errors
                bkg_bin_contents, _ = np.histogram(values, bins=bin_edges, weights=df_with_sweights['bkg_weight'])
                bkg_squared_uncs, _ = np.histogram(values, bins=bin_edges, weights=df_with_sweights['bkg_weight']**2)
                bkg_err = np.sqrt(bkg_squared_uncs)
                ax.step(bin_edges[:-1], bkg_bin_contents, where="post", label="Bkg", color="#ff7f0e")
                ax.errorbar(bin_centers, bkg_bin_contents, yerr=bkg_err, xerr=bin_widths / 2, fmt="o", color="#ff7f0e", capsize=2)

                # Styling
                ax.set_ylabel("Entries")
                ax.set_xlabel(label)
                ax.set_title(var)
                ax.legend()

                fig.tight_layout()
                fig.savefig(os.path.join(out_dir_pt, f"vars/{var}_{suffix}.pdf"), dpi=300, bbox_inches="tight")
                plt.close(fig)

                # Fill histograms for averages
                weighted_sum_sgn = np.sum(bin_centers * sgn_bin_contents)
                tot_sgn_weights = np.sum(sgn_bin_contents)
                avg_sgn = weighted_sum_sgn / tot_sgn_weights
                avg_sgn_unc = np.sqrt( (((bin_centers * tot_sgn_weights - weighted_sum_sgn) / tot_sgn_weights**2)**2 * sgn_err**2 ).sum() )
                histos_avgs[f"h_{var}_sgn"].SetBinContent(i_cutset + 1, avg_sgn)
                histos_avgs[f"h_{var}_sgn"].SetBinError(i_cutset + 1, avg_sgn_unc)

                weighted_sum_bkg = np.sum(bin_centers * bkg_bin_contents)
                tot_bkg_weights = np.sum(bkg_bin_contents)
                avg_bkg = weighted_sum_bkg / tot_bkg_weights
                avg_bkg_unc = np.sqrt( (((bin_centers * tot_bkg_weights - weighted_sum_bkg) / tot_bkg_weights**2)**2 * bkg_err**2 ).sum() )
                histos_avgs[f"h_{var}_bkg"].SetBinContent(i_cutset + 1, avg_bkg)
                histos_avgs[f"h_{var}_bkg"].SetBinError(i_cutset + 1, avg_bkg_unc)

                # Store the signal and bkg distributions in the output ROOT file
                sgn_hist_root = TH1F(f"h_{var}_sgn_distr", f"h_{var}_sgn_distr", nbins, var_range[0], var_range[1])
                bkg_hist_root = TH1F(f"h_{var}_bkg_distr", f"h_{var}_bkg_distr", nbins, var_range[0], var_range[1])
                for i_bin in range(nbins):
                    sgn_hist_root.SetBinContent(i_bin + 1, sgn_bin_contents[i_bin])
                    sgn_hist_root.SetBinError(i_bin + 1, sgn_err[i_bin])
                    bkg_hist_root.SetBinContent(i_bin + 1, bkg_bin_contents[i_bin])
                    bkg_hist_root.SetBinError(i_bin + 1, bkg_err[i_bin])
                sgn_hist_root.Write()
                bkg_hist_root.Write()

        # Compute average pt
        histos_avgs["h_avg_pt"] = TH1F("h_avg_pt", "h_avg_pt", 1, 0, 1)
        avg_pt = 0
        total_yield = 0
        last_ry = 0     # To skip duplicated cutsets
        for i_bin in range(h_ry_sgn.GetNbinsX()):
            ry = h_ry_sgn.GetBinContent(i_bin + 1)
            # Compare with tolerance
            logger(f"Checking cutset {i_bin} for pt bin {i_pt_bin}: ry = {h_ry_sgn.GetBinContent(i_bin + 1)}, last_ry = {last_ry}", level="INFO")
            if abs(h_ry_sgn.GetBinContent(i_bin + 1) - last_ry) < 10:
                logger(f"Skipping duplicated cutset {i_bin} for pt bin {i_pt_bin}", level="INFO")
                continue
            total_yield += ry
            last_ry = ry
            avg_pt += (histos_avgs["h_fPt_sgn"].GetBinContent(i_bin + 1) * ry)
            logger(f"Added term: {histos_avgs['h_fPt_sgn'].GetBinContent(i_bin + 1)} * {ry}", level="INFO")

        avg_pt /= total_yield
        logger(f"Computed average pt for pt bin {i_pt_bin}: {avg_pt} (total yield: {total_yield})", level="INFO")
        histos_avgs["h_avg_pt"].SetBinContent(1, avg_pt)

        hist_summary_avg_pt.SetBinContent(i_pt_bin + 1, histos_avgs['h_avg_pt'].GetBinContent(1))
        hist_summary_pt_shifts.SetBinContent(i_pt_bin + 1, histos_avgs['h_avg_pt'].GetBinContent(1) -
                                                      hist_summary_pt_shifts.GetBinCenter(i_pt_bin + 1))

        outfile.cd(pt_str)
        for hist_name, hist in histos_avgs.items():
            hist.Write(hist.GetName())

    outfile.cd()
    hist_summary_avg_pt.Write()
    hist_summary_pt_shifts.Write()
    outfile.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pt-centering with sPlot and flarefly/roofit')
    parser.add_argument('config_file', help='Path to the input configuration file')
    parser.add_argument("--minimizer", "-m", type=str, default="flarefly", help="minimizer to use")
    parser.add_argument('--do_fits', action='store_true', help='Perform fits')
    parser.add_argument("--workers", "-w", type=int, default=1, help="number of workers")
    args = parser.parse_args()

    if args.minimizer != "flarefly":
        logger("sWeights are implemented only with flarefly for now!", level="FATAL")
        sys.exit(1)

    if args.do_fits:
        run_fits(args.config_file, args.workers)

    eval_pt_center(args.config_file)
