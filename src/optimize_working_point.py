'''
Scan ScoreBkg (0 to max) and ScoreFD (sequential or slices) and map the 
invariant-mass significance per pT bin, using a subset of preprocessed job sparses.
python3 optimize_working_point.py config.yml [-w N] [-d]
'''

import os
import sys
import glob
import multiprocessing
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import ROOT
from ROOT import TFile, TH2D
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT.gROOT.SetBatch(True)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{script_dir}/../flareflyfitter")
sys.path.append(f"{script_dir}/../utils")
from raw_yield_fitter import RawYieldFitter
from utils import logger

# Handles root files with removed Centrality axis
TITLE_TO_NAME = {
    'Inv. mass (GeV/#it{c}^{2})': 'Mass',
    '#it{p}_{T} (GeV/#it{c})':    'Pt',
    'Centrality':                 'Cent',
    'SP':                         'Sp',
    'Bkg score':                  'ScoreBkg',
    'FD score':                   'ScoreFD',
}

def build_axis_idx(merged):
    idx = {}
    for i in range(merged.GetNdimensions()):
        title = merged.GetAxis(i).GetTitle()
        idx[TITLE_TO_NAME.get(title, title)] = i
    return idx


# Gets peak mean and width from the fit result parameters
def get_mean_sigma(fitter):
    try:
        params = fitter.fit_result.params
    except Exception:
        return np.nan, np.nan
    mean, sigma = np.nan, np.nan
    for p, v in params.items():
        name = p if isinstance(p, str) else getattr(p, 'name', str(p))
        val = v['value'] if isinstance(v, dict) else float(v)
        if 'mu_signal' in name:
            mean = float(val)
        elif 'sigma_signal' in name:
            sigma = float(val)
    return mean, sigma


# Returns the merged sparse from the job files
def load_merged_sparse(job_files, obj_path, n_jobs):
    files = sorted(job_files)
    if n_jobs < len(files):
        files = files[:n_jobs]
    else:
        logger("Only a subset of the available statistics is to be used for Working Point opt!", "FATAL")
    merged = None
    for job_file in files:
        root_file = TFile.Open(job_file)
        sparse = root_file.Get(obj_path)
        if not sparse:
            root_file.Close()
            continue
        merged = sparse.Clone("merged_sparse") if merged is None else (merged.Add(sparse), merged)[1]
        root_file.Close()

    return merged, files


# Builds the selections to scan: 
# - Sequential thresholds [x, 1]
# - Slices [edges[i], edges[i+1]]
def build_scan(axis_cfg, i_pt, axis_edges):

    strategy = axis_cfg['strategy']

    if strategy == 'slices':
        edges = axis_cfg['edges']
        edges = edges[i_pt] if isinstance(edges[0], list) else edges
        return list(zip(edges[:-1], edges[1:]))

    elif strategy == 'sequential':

        low_lims = axis_cfg['min']
        low_lims = low_lims[i_pt] if isinstance(low_lims, list) else low_lims
        upper_lims = axis_cfg['max']
        upper_lims = upper_lims[i_pt] if isinstance(upper_lims, list) else upper_lims
        
        # Pick n_steps values among the axis_edges
        # Keep only edges within the requested range
        valid_edges = np.asarray(axis_edges)[
            (np.asarray(axis_edges) >= low_lims) &
            (np.asarray(axis_edges) <= upper_lims)
        ]

        n_steps = axis_cfg['nsteps'][i_pt] if isinstance(axis_cfg['nsteps'], list) else axis_cfg['nsteps']
        if n_steps == 'all':
            return valid_edges[1:]  # all bin edges, except the first one

        if n_steps >= len(valid_edges):
            return valid_edges.tolist()

        # Pick approximately equally spaced indices
        idx = np.linspace(0, len(valid_edges) - 1, n_steps)
        idx = np.round(idx).astype(int)

        return valid_edges[idx].tolist()

    else:
        logger(f"Unknown scan strategy '{strategy}'", "FATAL")
        return []


# Formats scan values
def format_scores(vals):
    if len(vals) > 1:
        span = min((b - a) for a, b in zip(sorted(vals), sorted(vals)[1:]))
    else:
        span = vals[0] if vals else 1.0
    dec = max(0, min(4, -int(np.floor(np.log10(span)))))
    return [f"{v:.{dec}f}" for v in vals]


# Scans the grid, fits the mass at each point, saves the plots (2D significance map, QA hists)
def process_pt_bin(args):
    cfg, i_pt, pt_min, pt_max, minimizer, debug = args

    wp_cfg = cfg['working_point']
    fit_cfg = cfg['v2extraction']
    dmeson = cfg['Dmeson']
    input_dir = wp_cfg.get('input_dir', 'FlowSP')
    sparse_name = wp_cfg.get('sparse_name', 'FlowSP')
    obj_path = f"{sparse_name}/hSparse{sparse_name}"
    sgn_label = fit_cfg.get('SgnFuncLabel', dmeson)

    # Retrieve input file
    pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
    job_glob = f"{cfg['outdir']}/preprocess/{pt_str}/{input_dir}/jobs/AnalysisResults_*.root"
    job_files = glob.glob(job_glob)
    if not job_files:
        logger(f"No job files at {job_glob}, skipping", "WARNING")
        return None

    n_jobs_cfg = wp_cfg.get('n_jobs')
    n_jobs = n_jobs_cfg[i_pt] if isinstance(n_jobs_cfg, list) else n_jobs_cfg
    merged, used_files = load_merged_sparse(job_files, obj_path, n_jobs)
    if merged is None:
        return None

    # Dump used files to a text file
    out_dir_pt = f"{cfg['outdir']}/working_point_{cfg['suffix']}/{pt_str}"
    os.makedirs(out_dir_pt, exist_ok=True)
    with open(f"{out_dir_pt}/used_jobs.txt", "w") as jobs_file:
        jobs_file.write("\n".join(used_files) + "\n")

    # Build lists of scan values, depending on the strategy
    axis_idx = build_axis_idx(merged)
    scan = wp_cfg['scan']
    scan_var_x, scan_var_y = list(scan.keys())
    for need in (scan_var_x, scan_var_y, 'Mass'):
        if need not in axis_idx:
            logger(f"Axis '{need}' not in sparse titles {list(axis_idx)} for {pt_str}", "ERROR")
            return None

    axis_x = merged.GetAxis(axis_idx[scan_var_x])
    edges_x = [axis_x.GetBinLowEdge(i) for i in range(1, axis_x.GetNbins() + 1)]
    edges_x.append(axis_x.GetBinUpEdge(axis_x.GetNbins()))
    scan_x = build_scan(scan[scan_var_x], i_pt, edges_x)

    axis_y = merged.GetAxis(axis_idx[scan_var_y])
    edges_y = [axis_y.GetBinLowEdge(i) for i in range(1, axis_y.GetNbins() + 1)]
    edges_y.append(axis_y.GetBinUpEdge(axis_y.GetNbins()))
    scan_y = build_scan(scan[scan_var_y], i_pt, edges_y)

    # Labels
    fd_strategy = scan[scan_var_y].get('strategy', 'sequential')
    nx, ny = len(scan_x), len(scan_y)
    xlabels = format_scores(scan_x)
    if fd_strategy == 'slices':
        ylabels = [f"[{slice_low:g}, {slice_high:g}]" for slice_low, slice_high in scan_y]
    else:
        ylabels = format_scores(scan_y)
    y_axis_label = f"{scan_var_y} slice" if fd_strategy == 'slices' else f"{scan_var_y} min"

    h_signif = TH2D(f"h_signif_{pt_str}", f"Significance {pt_str};{scan_var_x} max;{y_axis_label}", nx, 0, nx, ny, 0, ny)
    h_mean = TH2D(f"h_mean_{pt_str}", f"Mass mean {pt_str};{scan_var_x} max;{y_axis_label}", nx, 0, nx, ny, 0, ny)
    h_sigma = TH2D(f"h_sigma_{pt_str}", f"Mass sigma {pt_str};{scan_var_x} max;{y_axis_label}", nx, 0, nx, ny, 0, ny)
    h_chi2 = TH2D(f"h_chi2_{pt_str}", f"Fit chi2/ndf {pt_str};{scan_var_x} max;{y_axis_label}", nx, 0, nx, ny, 0, ny)
    for h in (h_signif, h_mean, h_sigma, h_chi2):
        for i, lab in enumerate(xlabels):
            h.GetXaxis().SetBinLabel(i + 1, lab)
        for j, lab in enumerate(ylabels):
            h.GetYaxis().SetBinLabel(j + 1, lab)
    signif_grid = np.full((ny, nx), np.nan)

    mass_min, mass_max = fit_cfg['MassFitRanges'][i_pt]
    bkg_func_cfg, sgn_func_cfg = fit_cfg['BkgFunc'], fit_cfg['SgnFunc']
    bkg_func = bkg_func_cfg[i_pt] if isinstance(bkg_func_cfg, list) else bkg_func_cfg
    sgn_func = sgn_func_cfg[i_pt] if isinstance(sgn_func_cfg, list) else sgn_func_cfg

    fitter = RawYieldFitter(dmeson, pt_min, pt_max, pt_str, minimizer, verbose=False)
    fitter.set_fit_range(mass_min, mass_max)

    scan_log_path = f"{out_dir_pt}/scan_log_{pt_str}.txt"
    scan_log = open(scan_log_path, "w")
    scan_log.write("tag signif mean sigma chi2_ndf converged\n")

    debug_pdf = PdfPages(f"{out_dir_pt}/debug_fits_{pt_str}.pdf") if debug else None

    # Scans all (ScoreBkg max, ScoreFD min or slice) combinations
    for i_x, x_max in enumerate(scan_x):
        for i_y, y_scan_val in enumerate(scan_y):
            # apply score cuts and project mass axis
            y_cut_low, y_cut_high = y_scan_val if fd_strategy == 'slices' else (y_scan_val, 1.0)
            merged.GetAxis(axis_idx[scan_var_x]).SetRangeUser(0.0, x_max)
            merged.GetAxis(axis_idx[scan_var_y]).SetRangeUser(y_cut_low, y_cut_high)
            h_mass = merged.Projection(axis_idx['Mass'])
            h_mass.SetName(f"hMass_{pt_str}_{i_x}_{i_y}")
            h_mass.SetDirectory(0)

            tag = f"{scan_var_x}{x_max:g}_{scan_var_y}{y_cut_low:g}-{y_cut_high:g}"
            fitter.add_bkg_func(bkg_func, "Comb. bkg")
            fitter.add_sgn_func(sgn_func, sgn_label, dmeson)
            fitter.set_name(f"{pt_str}_{tag}")
            fitter.set_data_to_fit_hist(h_mass)
            fitter.setup()

            try:
                fit_status, fit_converged = fitter.fit()
                info, *_ = fitter.get_fit_info()
                signif = info[sgn_label]['signif']
                mean, sigma = get_mean_sigma(fitter)
                chi2 = info.get('chi2_over_ndf', np.nan)
                chi2 = chi2 if (np.isfinite(chi2) and chi2 > 0) else np.nan
            except Exception as e:
                logger(f"    Fit failed for {tag}: {e}", "WARNING")
                signif, mean, sigma, chi2 = -1.0, np.nan, np.nan, np.nan
                fit_converged = False

            # Fills QA hists and the 2D plot grid
            h_signif.SetBinContent(i_x + 1, i_y + 1, signif if signif > 0 else 0.0)
            signif_grid[i_y, i_x] = signif if signif > 0 else np.nan
            if np.isfinite(mean):
                h_mean.SetBinContent(i_x + 1, i_y + 1, mean)
            if np.isfinite(sigma):
                h_sigma.SetBinContent(i_x + 1, i_y + 1, sigma)
            if np.isfinite(chi2):
                h_chi2.SetBinContent(i_x + 1, i_y + 1, chi2)
            logger(f"    [{pt_str}] {tag}: signif={signif:.2f} mean={mean:.4f} sigma={sigma:.4f} chi2={chi2:.2f}", "INFO")
            scan_log.write(f"{tag} {signif:.3f} {mean:.5f} {sigma:.5f} {chi2:.3f} {fit_converged}\n")

            # In debug mode, plot fits that converged, but have a bad/undefined chi2/ndf
            bad_chi2 = (not np.isfinite(chi2)) or chi2 > 3.0
            if debug_pdf is not None and fit_converged and bad_chi2:
                try:
                    fig_fit = fitter.plot_fit(logy=False, show_extra_info=True, path=f"{out_dir_pt}/_fitplot.pdf")
                    fig_fit.suptitle(f"{tag} (chi2/ndf={chi2:.2f})", fontsize=8)
                    debug_pdf.savefig(fig_fit)
                    plt.close(fig_fit)
                except Exception as e:
                    logger(f"    Could not plot fit for {tag}: {e}", "WARNING")

            fitter.reset()

    scan_log.close()
    logger(f"Wrote scan log -> {scan_log_path}", "INFO")
    if debug_pdf is not None:
        debug_pdf.close()
        logger(f"Wrote debug fit plots -> {out_dir_pt}/debug_fits_{pt_str}.pdf", "INFO")

    # Generates significance 2D plot PDF
    fig, ax = plt.subplots(figsize=(1.6 * nx + 2, 1.2 * ny + 2))
    im = ax.imshow(np.ma.masked_invalid(signif_grid), origin="lower",
                   aspect="auto", cmap="viridis")
    ax.set_xticks(range(nx), xlabels)
    ax.set_yticks(range(ny), ylabels)
    ax.set_xlabel(f"{scan_var_x} max")
    ax.set_ylabel(y_axis_label)
    ax.set_title(f"Significance {pt_str}")
    for i_y in range(ny):
        for i_x in range(nx):
            val = signif_grid[i_y, i_x]
            if np.isfinite(val):
                ax.text(i_x, i_y, f"{val:.1f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Significance")
    fig.tight_layout()
    fig.savefig(f"{out_dir_pt}/significance_map_{pt_str}.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger(f"Wrote significance map -> {out_dir_pt}/significance_map_{pt_str}.pdf", "INFO")

    out_path = f"{out_dir_pt}/working_point_{pt_str}.root"
    out_file = TFile.Open(out_path, "recreate")
    h_signif.Write()
    h_mean.Write()
    h_sigma.Write()
    h_chi2.Write()
    out_file.Close()
    logger(f"Wrote QA hists -> {out_path}", "INFO")
    return out_path

# Runs one task per pT bin, in parallel if -w > 1
def optimize_working_point(cfg_file, minimizer, workers, debug=False):
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    out_base = f"{cfg['outdir']}/working_point_{cfg['suffix']}"
    os.makedirs(out_base, exist_ok=True)

    pt_mins, pt_maxs = cfg['ptbins'][:-1], cfg['ptbins'][1:]
    tasks = [(cfg, i, pt_min, pt_max, minimizer, debug) for i, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs))]

    results = []
    if workers > 1:
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as ex:
            scans = [ex.submit(process_pt_bin, t) for t in tasks]
            for fut in as_completed(scans):
                r = fut.result()
                if r:
                    results.append(r)
    else:
        for t in tasks:
            r = process_pt_bin(t)
            if r:
                results.append(r)

    logger(f"Done. Wrote {len(results)} per-bin QA file(s) under {out_base}", "INFO")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scan BDT scores and map significance")
    parser.add_argument("config_file", help="path to the yml configuration file")
    parser.add_argument("-m", "--minimizer", default="flarefly", help="flarefly or roofit")
    parser.add_argument("-w", "--workers", type=int, default=1, help="parallel pt-bin workers")
    parser.add_argument("-d", "--debug", action="store_true", help="save PDFs of fits with bad chi2/ndf")
    args = parser.parse_args()
    optimize_working_point(args.config_file, args.minimizer, args.workers, args.debug)
