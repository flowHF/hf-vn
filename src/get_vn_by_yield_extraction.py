import argparse
import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kWarning
import array
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "" # pylint: disable=wrong-import-position
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from ROOT import TFile, TH1F, TGraphAsymmErrors, kBlack, kFullCircle
script_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger, make_dir_root_file
from load_utils import load_aod_file
from StyleFormatter import SetGlobalStyle, SetObjectStyle
from matplotlib import gridspec
ROOT.gROOT.SetBatch(True)
import multiprocessing as mp
import sys
sys.path.append("./flareflyfitter/")
from raw_yield_fitter import RawYieldFitter
mp.set_start_method("spawn", force=True)
msg_service = ROOT.RooMsgService.instance()

def compute_avg_vn(yields, yields_uncs, bin_centers, resolution):
    weighted_avg = np.sum(np.array(yields) * (np.array(bin_centers) / resolution)) / np.sum(yields)
    weighted_avg_unc = 0
    for i_sp, (i_v2, i_ry, i_ry_unc) in enumerate(zip(bin_centers, yields, yields_uncs)):
        d_yields_dRyi = (i_v2/resolution) / np.sum(yields)
        d_v2_dRyi = (np.sum(np.array(yields) * (np.array(bin_centers) / resolution))) / (np.sum(yields)**2)
        weighted_avg_unc = weighted_avg_unc + ((d_yields_dRyi - d_v2_dRyi)**2)*i_ry_unc**2
    weighted_avg_unc = np.sqrt(weighted_avg_unc)

    return weighted_avg, weighted_avg_unc

def sp_scan_pt_bin(fitter, data_cutset, sgn_func_label, pt_label, resolution, sp_intervals,
                   outdir, mass_intervals=None, save_plots=True, is_multitrial=False):

    vals_stats = {}
    stats = [
        'Means', 'MeansUnc',
        'Sigmas', 'SigmasUnc',
        'SpRy', 'SpRyUnc',
        'Chi2', 'Chi2OverNdf',
        'VnSimFitUnc', 'SpErrorContrib',
        'SpFirstTermErrorContrib', 'SpSecondTermErrorContrib',
    ]
    if mass_intervals is not None:
        stats += [
            f'SpRyBkgMass_{mass_min:.2f}_{mass_max:.2f}'
            for mass_min, mass_max in zip(mass_intervals[:-1], mass_intervals[1:])
        ]
        stats += [
            f'SpRyBkgMass_{mass_min:.2f}_{mass_max:.2f}Unc'
            for mass_min, mass_max in zip(mass_intervals[:-1], mass_intervals[1:])
        ]

    for stat in stats:
        vals_stats[stat] = [0] * (len(sp_intervals) - 1)
    vals_stats['SpCenter'] = [0] * (len(sp_intervals) - 1)

    for isp, (sp_min, sp_max) in enumerate(zip(sp_intervals[:-1], sp_intervals[1:])):
        sp_center = (sp_min + sp_max) / 2
        sp_label = f"sp_range_{sp_min:.2f}_{sp_max:.2f}"
        if not is_multitrial:
            logger(f"Processing sp_label {sp_label}", "INFO")

        # For each sp bin, update the fitter name and reduce the dataset to fit
        fitter.reduce_dataset("fScalarProd", [sp_min, sp_max])

        fitter.set_name(sp_label)
        vals_stats['SpCenter'][isp] = sp_center
        try:
            status, converged = fitter.fit()
            if save_plots:
                fig_sp_pt_path = f"{outdir}/{sp_label}.pdf" if is_multitrial else f"{outdir}/{pt_label}/{sp_label}.pdf"
                fitter.plot_fit(False, True, loc=["lower left", "upper left"], # (log, show_extra_info)
                                path=fig_sp_pt_path)
            fit_info, sgn_pars, sgn_pars_uncs, _, _ = fitter.get_fit_info()
            vals_stats['SpRy'][isp], vals_stats['SpRyUnc'][isp] = fit_info[sgn_func_label]["ry"], fit_info[sgn_func_label]["ry_unc"]
            vals_stats['Means'][isp], vals_stats['MeansUnc'][isp] = sgn_pars[f"mu_{sgn_func_label}"], sgn_pars_uncs[f"mu_{sgn_func_label}"]
            vals_stats['Sigmas'][isp], vals_stats['SigmasUnc'][isp] = sgn_pars[f"sigma_{sgn_func_label}"], sgn_pars_uncs[f"sigma_{sgn_func_label}"]
            vals_stats['Chi2'][isp], vals_stats['Chi2OverNdf'][isp] = fit_info["chi2"], fit_info["chi2_over_ndf"]
            for mass_min, mass_max in zip(mass_intervals[:-1], mass_intervals[1:]):
                mass_str = f'SpRyBkgMass_{mass_min:.2f}_{mass_max:.2f}'
                vals_stats[mass_str][isp], vals_stats[f"{mass_str}Unc"][isp] = fitter.get_bkg_yield(mass_min, mass_max)

        except Exception as e:
            logger(f"Error fitting sp {sp_min:.2f} - {sp_max:.2f}: {e}, setting all values to 0 for this sp bin.", "WARNING")
            vals_stats['SpRy'][isp], vals_stats['SpRyUnc'][isp] = 0, 0
            vals_stats['Means'][isp], vals_stats['MeansUnc'][isp] = 0, 0
            vals_stats['Sigmas'][isp], vals_stats['SigmasUnc'][isp] = 0, 0
            vals_stats['Chi2'][isp], vals_stats['Chi2OverNdf'][isp] = -1, -1

    # Compute weighted averages and fill histograms
    vals_stats['SummedSpYields'] = np.sum(vals_stats['SpRy'])
    vals_stats['SummedSpYieldsUnc'] = np.sqrt(np.sum(np.array(vals_stats['SpRyUnc'])**2))
    vals_stats['WeightedSum'] = np.sum(np.array(vals_stats['SpRy']) * (np.array(vals_stats['SpCenter']) / resolution))
    vals_stats['VnSimFit'] = vals_stats['WeightedSum'] / vals_stats['SummedSpYields']
    weighted_avg_unc = 0
    for i_sp, (i_v2, i_ry, i_ry_unc) in enumerate(zip(vals_stats['SpCenter'], vals_stats['SpRy'], vals_stats['SpRyUnc'])):
        d_yields_dRyi = (i_v2/resolution) / vals_stats['SummedSpYields']
        d_v2_dRyi = (vals_stats['WeightedSum']) / (vals_stats['SummedSpYields']**2)
        weighted_avg_unc = weighted_avg_unc + ((d_yields_dRyi - d_v2_dRyi)**2)*i_ry_unc**2
        vals_stats['SpFirstTermErrorContrib'][i_sp] = (d_yields_dRyi)**2 * i_ry_unc**2
        vals_stats['SpSecondTermErrorContrib'][i_sp] = (d_v2_dRyi)**2 * i_ry_unc**2
        vals_stats['SpErrorContrib'][i_sp] = np.sqrt(abs(vals_stats['SpFirstTermErrorContrib'][i_sp] - \
                                                         vals_stats['SpSecondTermErrorContrib'][i_sp]))
    vals_stats['VnSimFitUnc'] = np.sqrt(weighted_avg_unc)

    if not is_multitrial:
        vals_stats["VnVsMassBkg"], vals_stats["VnVsMassBkgUnc"] = [], []
        for mass_min, mass_max in zip(mass_intervals[:-1], mass_intervals[1:]):
            mass_str = f'SpRyBkgMass_{mass_min:.2f}_{mass_max:.2f}'
            avg_vn, avg_vn_unc = compute_avg_vn(vals_stats[mass_str], vals_stats[f"{mass_str}Unc"],
                                                vals_stats['SpCenter'], resolution)
            vals_stats["VnVsMassBkg"].append(avg_vn)
            vals_stats["VnVsMassBkgUnc"].append(avg_vn_unc)

    return vals_stats

def run_trial_worker(data, cfg_trial_path, cfg_cutsets_paths, pt_min, pt_max, i_pt, resolution, is_multitrial):

    with open(cfg_trial_path, 'r') as CfgTrial:
        cfg_trial = yaml.safe_load(CfgTrial)

    # Monitor time for reading main config
    t0 = time.perf_counter()
    # Initialize the fitter for sp-integrated yield extraction
    pt_label = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
    fitter = RawYieldFitter(cfg_trial['Dmeson'], pt_min, pt_max, f"sp_integrated_{pt_label}_fit",
                            cfg_trial['v2extraction']['Minimizer'], verbose = not is_multitrial)

    stats = {}
    t1 = time.perf_counter()
    logger(f"Time to read main config {cfg_trial_path}: {t1 - t0} s", "INFO")

    fit_cfg = cfg_trial['v2extraction']
    fitter.set_fit_range(fit_cfg['MassFitRanges'][i_pt][0], fit_cfg['MassFitRanges'][i_pt][1])
    t2 = time.perf_counter()

    # Add model components
    fitter.add_bkg_func(fit_cfg['BkgFunc'][i_pt] if isinstance(fit_cfg['BkgFunc'], list) else fit_cfg['BkgFunc'], "Comb. bkg")
    sgn_funcs = {} # More info for signal functions, a dictionary is better
    sgn_funcs[fit_cfg['SgnFuncLabel']] = {
        'func': fit_cfg['SgnFunc'][i_pt] if isinstance(fit_cfg['SgnFunc'], list) else fit_cfg['SgnFunc'],
        'part': cfg_trial['Dmeson']
    }
    if fit_cfg.get('InclSecPeak'):
        include_sec_peak = fit_cfg['InclSecPeak'][i_pt] if isinstance(fit_cfg['InclSecPeak'], list) else fit_cfg['InclSecPeak']
        if include_sec_peak:
            sgn_funcs[fit_cfg['SgnFuncSecPeakLabel']] = {
                'func': fit_cfg['SgnFuncSecPeak'][i_pt] if isinstance(fit_cfg['SgnFuncSecPeak'], list) else fit_cfg['SgnFuncSecPeak'],
                'part': 'Dplus' if part == 'Ds' else 'Dstar',
            }
    for i_sgn, (label, sgn_func) in enumerate(sgn_funcs.items()):
        fitter.add_sgn_func(sgn_func['func'], label, sgn_func['part'])
    sgn_func_label = fit_cfg['SgnFuncLabel']

    if fit_cfg.get('FixParsToIntFit'):  # Fix mean to results of sp-integrated fit
        if not is_multitrial:
            logger(f"Will fix all parameters of signal functions to sp-integrated fit result", "WARNING")
        fitter.fix_sgn_pars_to_first_fit()

    t3 = time.perf_counter()
    logger(f"Time for adding model components: {t3 - t2} s", "INFO")
    for i_cutset, cutset_file in enumerate(cfg_cutsets_paths):
        t_start_cutset = time.perf_counter()
        with open(cutset_file, 'r') as CfgCutset:
            cfg_cutset = yaml.safe_load(CfgCutset)
        outdir = os.path.dirname(cutset_file).replace('cutset', "raw_yield")
        
        # Create mask and select data
        cutset_mask = (
            (data['fMlScore0'] < cfg_cutset['ScoreBkg']['max'][i_pt]) &
            (data['fMlScore0'] >= cfg_cutset['ScoreBkg']['min'][i_pt]) &
            (data['fMlScore1'] < cfg_cutset['ScoreFD']['max'][i_pt]) &
            (data['fMlScore1'] >= cfg_cutset['ScoreFD']['min'][i_pt]) &
            (data['fM'] >= cfg_trial['v2extraction']['MassFitRanges'][i_pt][0]) &
            (data['fM'] <= cfg_trial['v2extraction']['MassFitRanges'][i_pt][1])
        )
        sel_string_cutset = (
            f"fMlScore0 < {cfg_cutset['ScoreBkg']['max'][i_pt]} && "
            f"fMlScore0 >= {cfg_cutset['ScoreBkg']['min'][i_pt]} && "
            f"fMlScore1 < {cfg_cutset['ScoreFD']['max'][i_pt]} && "
            f"fMlScore1 >= {cfg_cutset['ScoreFD']['min'][i_pt]} && "
            f"fM >= {cfg_trial['v2extraction']['MassFitRanges'][i_pt][0]} && "
            f"fM <= {cfg_trial['v2extraction']['MassFitRanges'][i_pt][1]}"
        )
        t4 = time.perf_counter()
        logger(f"Time to query data for cutset {i_cutset} in pt bin {pt_label}: {t4 - t_start_cutset:.3f} s", "INFO")
        data_cutset = data[cutset_mask]
        fitter.set_data_to_fit_df(data_cutset, 'fM')
        t5 = time.perf_counter()
        logger(f"Time to set data to fit for cutset {i_cutset} in pt bin {pt_label}: {t5 - t4:.3f} s", "INFO")

        # Add correlated background if specified
        if cfg_trial.get('corr_bkgs'):
            fitter.add_corr_bkgs(cfg_trial['corr_bkgs'], sel_string, pt_min, pt_max)

        fitter.setup()
        if fit_cfg.get('InitPars'):
            fitter.set_fit_pars(fit_cfg['InitPars'], pt_min, pt_max)
        t6 = time.perf_counter()
        logger(f"Time for setting up fitter for cutset {i_cutset} in pt bin {pt_label}: {t6 - t5} s", "INFO")
        # Prefit the MC prompt enhanced cut to fix the tails, binned fit
        if fit_cfg.get('FixSgnFromMC'):
            fitter.set_fix_sgn_to_mc_prefit(True)
            if i_cutset == 0:
                fitter.prefit_mc(f"{cfg_trial['outdir'].split('cutvar')[0]}/corrbkgs/templs_{pt_label}.root")
                fitter.plot_mc_prefit(False, True, loc=["lower left", "upper left"], path=outdir)
                fitter.plot_raw_residuals_mc_prefit(path=f"{outdir}/fM_mc_prefit_residuals_{pt_label}.pdf")
        t7 = time.perf_counter()
        if i_cutset == 0:
            logger(f"Time for prefit MC for cutset {i_cutset} in pt bin {pt_label}: {t7 - t6} s", "INFO")

        t8 = time.perf_counter()
        logger(f"Time for fixing signal parameters to sp-integrated fit for cutset {i_cutset} in pt bin {pt_label}: {t8 - t7} s", "INFO")
        if i_cutset == 0:
            status, converged = fitter.fit()
            fig_int_fit_path = f"{outdir}/fit_int_{pt_label}.pdf"
            fitter.plot_fit(False, True, loc=["lower left", "upper left"],  # (log, show_extra_info)
                            path=fig_int_fit_path)
            fit_info_pt_int, sgn_pars_pt_int, sgn_pars_uncs_pt_int, bkg_pars_pt_int, bkg_pars_uncs_pt_int = fitter.get_fit_info()

        step = fit_cfg['SpWindowWidth'][i_pt][i_cutset]
        sp_abs_val_max = fit_cfg['SpRanges'][i_pt]
        sp_intervals = np.arange(-sp_abs_val_max, sp_abs_val_max + 0.5 * step, step).tolist()
        stats[i_cutset] = {}
        t9 = time.perf_counter()
        logger(f"Time for performing sp-integrated fit for cutset {i_cutset} in pt bin {pt_label}: {t9 - t8} s", "INFO")
        stats[i_cutset] = sp_scan_pt_bin(fitter, data_cutset, sgn_func_label, pt_label, resolution, sp_intervals,
                                         f"{outdir}/scan_{i_cutset:02d}", mass_intervals=cfg_trial['projections']['VnVsMassBins'][i_pt],
                                         save_plots=True, is_multitrial=is_multitrial)
        t10 = time.perf_counter()
        logger(f"Time for sp scan for cutset {i_cutset} in pt bin {pt_label}: {t10 - t9} s", "INFO")
        stats[i_cutset]['SpIntervals'] = sp_intervals
        stats[i_cutset]['RawYieldsSimFit'] = fit_info_pt_int[sgn_func_label]["ry"]
        stats[i_cutset]['RawYieldsSimFitUnc'] = fit_info_pt_int[sgn_func_label]["ry_unc"]
        stats[i_cutset]['MeanSimFit'] = sgn_pars_pt_int[f"mu_{sgn_func_label}"]
        stats[i_cutset]['MeanSimFitUnc'] = sgn_pars_uncs_pt_int[f"mu_{sgn_func_label}"]
        stats[i_cutset]['SigmaSimFit'] = sgn_pars_pt_int[f"sigma_{sgn_func_label}"]
        stats[i_cutset]['SigmaSimFitUnc'] = sgn_pars_uncs_pt_int[f"sigma_{sgn_func_label}"]
        # Clear data_cutset to save memory
        del data_cutset
        t11 = time.perf_counter()
        logger(f"Time after sp scan, cutset {i_cutset} completed: {t11 - t_start_cutset} s\n", "INFO")

    del fitter
    logger(f"Finished config file {os.path.basename(cfg_trial_path)} for pt bin {pt_label}, total time: {time.perf_counter() - t0} s\n\n", "INFO")
    return stats, cfg_trial_path

def run_pt_bin_worker(cutset_files, i_pt, pt_min, pt_max, cfg_flow, resolution, is_multitrial):

    pt_label = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
    data = load_aod_file(f"{cfg_flow['outdir']}/preprocess/{pt_label}/TreesPtCenterSp/AO2D_{pt_label}.root", True)

    stats, sp_int_fits_pars = {}, {}
    cutsets_datasets, df_cutsets = {}, {}

    trial_tasks = []
    # parallelize over cutset files
    for i_cfg, (cfg_trial, cfg_trial_cutsets) in enumerate(cutset_files.items()):
        trial_tasks.append((data, cfg_trial, cfg_trial_cutsets, pt_min, pt_max, i_pt, resolution, is_multitrial))
    with ProcessPoolExecutor(max_workers=cfg_flow.get('TrialWorkers', 1)) as executor:
        results = [executor.submit(run_trial_worker, *task) for task in trial_tasks]
        for task in as_completed(results):
            trial_stats, cfg_trial_path = task.result()
            stats[cfg_trial_path] = {}
            stats[cfg_trial_path] = trial_stats

    # Check for exceptions in workers
    for task in results:
        if task.exception() is not None:
            logger(f"Worker generated an exception: {task.exception()}", "ERROR")
            raise task.exception()

    # return ONLY python objects
    return pt_label, i_pt, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('input_config', metavar='text', default='config.yml')
    parser.add_argument('--multitrial', default=False, help='suppress prints and limit number of pdfs', action='store_true')
    parser.add_argument('--multitrial_configs', '-mult_cfgs', default='multitrial_cfgs.txt',
                        help='text file with paths to the yaml config files for multitrial', metavar='text')
    parser.add_argument('--batch', '-b', help='suppress video output', action='store_true')
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    with open(args.input_config, 'r') as CfgFlow:
        cfg_flow = yaml.safe_load(CfgFlow)

    # Retrieve cutsets configs
    cutset_files = {}
    pt_bins = cfg_flow['ptbins']
    if args.multitrial:
        for main_cfg_path in open(args.multitrial_configs, 'r'):
            main_cfg_path = main_cfg_path.strip()
            if not main_cfg_path:
                continue
            with open(main_cfg_path, 'r') as CfgFlowMultitrial:
                cfg_multitrial = yaml.safe_load(CfgFlowMultitrial)
                pt_bins = cfg_multitrial['ptbins']
            cutsets_dir = f"{os.path.dirname(main_cfg_path)}/cutsets"
            cutset_files[main_cfg_path] = [os.path.join(cutsets_dir, f) for f in os.listdir(cutsets_dir) if f.endswith('.yml')]
            cutset_files[main_cfg_path].sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
    else:
        try:
            cutsets_dir = os.path.join(cfg_flow['outdir'], f"vn_extr_{cfg_flow['suffix']}_combined/cutsets")
            cutset_files[args.input_config] = [os.path.join(cutsets_dir, f) for f in os.listdir(cutsets_dir) if f.endswith('.yml')]
        except Exception as e:
            logger(f"Could not find combined cutsets, trying correlated cutsets ... ", level="WARNING")
            cutsets_dir = os.path.join(cfg_flow['outdir'], f"vn_extr_{cfg_flow['suffix']}_correlated/cutsets")
            cutset_files[args.input_config] = [os.path.join(cutsets_dir, f) for f in os.listdir(cutsets_dir) if f.endswith('.yml')]
        cutset_files[args.input_config].sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))

    # Read the Resolution and MassSp histograms for all pt bins
    reso_file = TFile.Open(cfg_flow["projections"]["Resolution"], 'r')
    det_A = cfg_flow["projections"].get('detA', 'FT0c')
    det_B = cfg_flow["projections"].get('detB', 'FV0a')
    det_C = cfg_flow["projections"].get('detC', 'TPCtot')
    logger(f"Getting resolution histogram from file {cfg_flow['projections']['Resolution']} for triplet {det_A}_{det_B}_{det_C}",  "WARNING")
    reso_hist = reso_file.Get(f'{det_A}_{det_B}_{det_C}/histo_reso_delta_cent')
    reso_hist.SetDirectory(0)
    reso_file.Close()
    resolution = reso_hist.GetBinContent(1)

    tasks = []
    vals_stats = {}
    for i_pt, (pt_min, pt_max) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):
        tasks.append((cutset_files, i_pt, pt_min, pt_max, cfg_flow, resolution, args.multitrial))
    with ProcessPoolExecutor(max_workers=cfg_flow['v2extraction'].get('PtWorkers', 1)) as executor:
        results = [executor.submit(run_pt_bin_worker, *task) for task in tasks]
        for task in as_completed(results):
            pt_label, i_pt, stats = task.result()
            vals_stats[pt_label] = (i_pt, stats)

    # Check for exceptions in workers
    for task in results:
        if task.exception() is not None:
            logger(f"Worker generated an exception: {task.exception()}", "ERROR")
            raise task.exception()

    # Sort vals_stats by i_pt
    vals_stats = dict(sorted(vals_stats.items(), key=lambda item: item[1][0]))

    out_files, summaries = {}, {}
    for config_path, cutsets in cutset_files.items():
        out_files[config_path], summaries[config_path] = {}, {}
        for i_cutset, cutset in enumerate(cutsets):
            os.makedirs(os.path.dirname(cutset).replace('cutset', 'raw_yield'), exist_ok=True)
            out_files[config_path][i_cutset] = cutset.replace('cutset', 'raw_yield').replace('.yml', '.root')
            summaries[config_path][i_cutset] = {
                'hRawYieldsSimFit': TH1F("hRawYieldsSimFit", "hRawYieldsSimFit", len(pt_bins) - 1, np.array(pt_bins)),
                'hSummedSpYields': TH1F("hSummedSpYields", "hSummedSpYields", len(pt_bins) - 1, np.array(pt_bins)),
                'hMeanSimFit': TH1F("hMeanSimFit", "hMeanSimFit", len(pt_bins) - 1, np.array(pt_bins)),
                'hSigmaSimFit': TH1F("hSigmaSimFit", "hSigmaSimFit", len(pt_bins) - 1, np.array(pt_bins)),
                'hVnSimFit': TH1F("hVnSimFit", "hVnSimFit", len(pt_bins) - 1, np.array(pt_bins)),
                'hVnSimFitUnc': TH1F("hVnSimFitUnc", "hVnSimFitUnc", len(pt_bins) - 1, np.array(pt_bins)),
                'hWeightedSum': TH1F("hWeightedSum", "hWeightedSum", len(pt_bins) - 1, np.array(pt_bins)),
                'gVnSimFit': TGraphAsymmErrors(1),
                'gVnUnc': TGraphAsymmErrors(1),
            }
            summaries[config_path][i_cutset]['gVnSimFit'].SetName("gVnSimFit")
            summaries[config_path][i_cutset]['gVnUnc'].SetName("gVnUnc")

    for pt_label, (i_pt, pt_stats) in vals_stats.items():
        pt_min = pt_bins[i_pt]
        pt_max = pt_bins[i_pt + 1]

        for config_path, cutset_results in pt_stats.items():
            for cutset, cutset_vals in cutset_results.items():
                for var, vals in cutset_vals.items():
                    if isinstance(vals, list):
                        if "VnVsMassBkg" in var:
                            summaries[config_path][cutset][f"{pt_label}/{var}"] = TH1F(f"hist_{var}", f"hist_{var}", len(cfg_flow['projections']['VnVsMassBins'][i_pt]) - 1,
                                                                        array.array('f', cfg_flow['projections']['VnVsMassBins'][i_pt]))
                        else:
                            summaries[config_path][cutset][f"{pt_label}/{var}"] = TH1F(f"hist_{var}", f"hist_{var}", len(cutset_vals['SpIntervals']) - 1,
                                                                        array.array('f', cutset_vals['SpIntervals']))
                        summaries[config_path][cutset][f"{pt_label}/{var}"].SetDirectory(0)
                        for i_val, val in enumerate(vals):
                            summaries[config_path][cutset][f"{pt_label}/{var}"].SetBinContent(i_val + 1, val)
                            if 'Unc' in var:
                                summaries[config_path][cutset][f"{pt_label}/{var.replace('Unc', '')}"].SetBinError(i_val + 1, val)
                    else:
                        if "Unc" in var:
                            continue
                        summaries[config_path][cutset][f"h{var}"].SetBinContent(i_pt + 1, vals)
                        try:
                            summaries[config_path][cutset][f"h{var}"].SetBinError(i_pt + 1, cutset_vals[f"{var}Unc"])
                        except KeyError:
                            pass
                        if "VnSimFit" in var:
                            summaries[config_path][cutset]['gVnSimFit'].SetPoint(i_pt, (pt_min+pt_max)/2, vals)
                            summaries[config_path][cutset]['gVnSimFit'].SetPointError(i_pt, (pt_max-pt_min)/2, (pt_max-pt_min)/2, cutset_vals[f"{var}Unc"], cutset_vals[f"{var}Unc"])
                            summaries[config_path][cutset]['gVnUnc'].SetPoint(i_pt, (pt_min+pt_max)/2, cutset_vals[f"{var}Unc"])
                            summaries[config_path][cutset]['gVnUnc'].SetPointError(i_pt, (pt_max-pt_min)/2, (pt_max-pt_min)/2, 1.e-20, 1.e-20)

    logger("\n\n")
    logger("Saving results to root files ... ", "INFO")
    for main_cfg, cutset_cfgs in cutset_files.items():
        for i_cutset, cutset in enumerate(cutset_cfgs):
            out_file_name = cutset.replace('cutsets', 'raw_yields').replace('cutset', 'raw_yields').replace('.yml', '.root')
            outfile = TFile.Open(out_file_name, 'RECREATE')
            for pt_label in vals_stats.keys():
                make_dir_root_file(pt_label, outfile, verbose=False)
            for hist_name, hist in summaries[main_cfg][i_cutset].items():
                SetObjectStyle(hist, color=kBlack, markerstyle=kFullCircle)
                if "/" in hist_name:
                    pt_dir, hist_name = hist_name.split("/")
                    if "SpRyBkgMass" in hist_name:
                        make_dir_root_file(f"{pt_dir}/MassBinsBkg", outfile, verbose=False)
                        outfile.cd(f"{pt_dir}/MassBinsBkg")
                    else:
                        outfile.cd(pt_dir)
                    hist.SetName(hist_name)
                    hist.Write(hist_name)
                else:
                    outfile.cd()
                    hist.Write(hist_name)
            outfile.Close()

            logger(f"Processed config file {cutset} and saved results to {out_file_name}", "INFO")
