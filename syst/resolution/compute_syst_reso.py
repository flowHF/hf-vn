import os
import sys
import array
import yaml
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import argparse

from ROOT import (TFile, TCanvas, TH1F, TLegend, TGraph, TLine, TBox,
                  gStyle, gROOT, kBlack, kRed, kBlue, kGreen, kOrange, kAzure)

gROOT.SetBatch(True)

# Add custom modules
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../flareflyfitter/")
from raw_yield_fitter import RawYieldFitter
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../utils")
from utils import logger, get_centrality_bins, make_dir_root_file
from data_model import get_pt_preprocessed_sparses
from load_utils import load_aod_file

mp.set_start_method("spawn", force=True)


# ------------------ ROOT Styling ------------------
def set_root_style(line_width=3, title_size=0.05, label_size=0.045, tick_length=0.03):
    gStyle.SetOptStat(0)
    gStyle.SetLineWidth(line_width)
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
    gStyle.SetTitleSize(title_size, "XYZ")
    gStyle.SetLabelSize(label_size, "XYZ")
    gStyle.SetTickLength(tick_length, "XYZ")


# ------------------ Weighted Average ------------------
def weighted_average(values, weights):
    total_weight = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total_weight if total_weight else 0


# ------------------ Draw Histogram + Markers ------------------
def draw_hist_with_markers(name, bin_edges, values, color=kBlack, marker=21, line_width=3, draw_frame=True):
    n_bins = len(values)
    arr_bins = array.array('f', bin_edges)
    h = TH1F(name, "", n_bins, arr_bins)
    for i, val in enumerate(values):
        h.SetBinContent(i + 1, val)
    h.SetLineColor(color)
    h.SetLineWidth(line_width)
    h.SetMarkerSize(0)
    h.Draw("HIST" if draw_frame else "HIST SAME")

    # Graph markers at bin centers
    g = TGraph(n_bins)
    for i in range(n_bins):
        center = 0.5 * (bin_edges[i] + bin_edges[i + 1])
        g.SetPoint(i, center, values[i])
    g.SetMarkerStyle(marker)
    g.SetMarkerSize(1.2)
    g.SetMarkerColor(color)
    g.SetLineColor(color)
    g.SetLineWidth(line_width)
    g.Draw("P SAME")
    return h, g


# ------------------ Legend ------------------
def make_legend(graphs, labels, header=None, x1=0.65, y1=0.80, x2=0.9, y2=0.91, text_size=0.035):
    leg = TLegend(x1, y1 - 0.03*len(graphs), x2, y2)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(text_size)
    if header:
        leg.SetHeader(header)
        header_obj = leg.GetListOfPrimitives().At(0)
        header_obj.SetTextSize(text_size + 0.005)
    for g, label in zip(graphs, labels):
        leg.AddEntry(g, label, "lp")
    leg.Draw()
    return leg


# ------------------ Resolution Figure ------------------
def reso_syst_figure(pt_bins, reso_means, reso_mean_arith, syst_unc, out_file_path):
    set_root_style()
    n_bins = len(pt_bins) - 1
    c = TCanvas("c_reso", "Resolution", 800, 800)
    c.SetLeftMargin(0.20)
    c.SetBottomMargin(0.14)
    c.SetRightMargin(0.04)
    c.SetTopMargin(0.05)

    # Histogram step line
    h_reso = TH1F("h_reso", ";#it{p}_{T} (GeV/c);Resolution", n_bins, array.array('f', pt_bins))
    for i, val in enumerate(reso_means):
        h_reso.SetBinContent(i+1, val)
    h_reso.SetLineColor(kBlack)
    h_reso.SetLineWidth(3)
    h_reso.SetMarkerSize(0)

    # Set y-range with margin
    min_reso = min(reso_means + [reso_mean_arith - syst_unc])
    max_reso = max(reso_means + [reso_mean_arith + syst_unc])
    y_margin = 1.5 * (max_reso - min_reso)
    h_reso.SetMinimum(min_reso - y_margin)
    h_reso.SetMaximum(max_reso + y_margin)

    h_reso.Draw("HIST")

    # Graph markers
    g_markers = TGraph(n_bins)
    for i in range(n_bins):
        center = 0.5 * (pt_bins[i] + pt_bins[i+1])
        g_markers.SetPoint(i, center, reso_means[i])
    g_markers.SetMarkerStyle(21)
    g_markers.SetMarkerSize(1.2)
    g_markers.SetMarkerColor(kBlack)
    g_markers.SetLineColor(kBlack)
    g_markers.SetLineWidth(3)
    g_markers.Draw("P SAME")

    # Arithmetic mean line
    mean_line = TLine(pt_bins[0], reso_mean_arith, pt_bins[-1], reso_mean_arith)
    mean_line.SetLineStyle(9)
    mean_line.SetLineColor(kRed)
    mean_line.SetLineWidth(3)
    mean_line.Draw("SAME")

    # Syst. uncertainty band
    box = TBox(pt_bins[0], reso_mean_arith - syst_unc, pt_bins[-1], reso_mean_arith + syst_unc)
    box.SetFillColorAlpha(kRed, 0.35)  # 35% transparent red
    box.SetLineColor(0)                 # no border
    box.Draw("SAME")

    # Axis styling
    xaxis = h_reso.GetXaxis()
    yaxis = h_reso.GetYaxis()
    xaxis.SetTitleSize(0.05)
    yaxis.SetTitleSize(0.05)
    xaxis.SetLabelSize(0.045)
    yaxis.SetLabelSize(0.045)
    xaxis.SetTitleOffset(1.2)
    yaxis.SetTitleOffset(2.0)
    xaxis.CenterTitle()
    yaxis.CenterTitle()
    xaxis.CenterTitle()
    yaxis.CenterTitle()

    # Legend
    leg = TLegend(0.3, 0.70, 0.5, 0.85)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.04)
    leg.AddEntry(g_markers, "Reweighted means", "lp")
    leg.AddEntry(mean_line, "Arithmetic mean", "l")
    leg.AddEntry(box, "Systematic uncertainty", "f")
    leg.Draw()

    c.SaveAs(out_file_path)


# ------------------ Raw Yield vs Centrality Figure ------------------
def ry_vs_cent_figure(pt_bins, cent_intervals, pt_bins_yields, out_file_path):
    set_root_style()
    c = TCanvas("c_ry_cent", "Raw yields vs Centrality", 800, 800)
    c.SetLeftMargin(0.15)
    c.SetBottomMargin(0.14)
    c.SetRightMargin(0.04)
    c.SetTopMargin(0.05)

    colors = [kBlack, kRed, kBlue, kGreen+2, kOrange+7, kAzure+3]
    histos, graphs = [], []

    # Draw all pT bins
    for i_pt, pt_bin_yield in enumerate(pt_bins_yields):
        h, g = draw_hist_with_markers(
            f"h_ry_{i_pt}",
            cent_intervals,
            pt_bin_yield['raw_yields'],
            color=colors[i_pt % len(colors)],
            draw_frame=(i_pt==0)
        )
        if i_pt == 0:
            h.SetMinimum(0)
            h.SetMaximum(max(max(pb['raw_yields']) for pb in pt_bins_yields) * 1.2)
        histos.append(h)
        graphs.append(g)

    # Axis styling
    frame = histos[0]
    frame.GetXaxis().SetTitle("Centrality (%)")
    frame.GetYaxis().SetTitle("Raw yields")
    frame.GetXaxis().SetTitleSize(0.05)
    frame.GetYaxis().SetTitleSize(0.05)
    frame.GetXaxis().SetLabelSize(0.045)
    frame.GetYaxis().SetLabelSize(0.045)
    frame.GetXaxis().SetTitleOffset(1.2)
    frame.GetYaxis().SetTitleOffset(1.2)
    frame.GetXaxis().CenterTitle()
    frame.GetYaxis().CenterTitle()

    frame.GetYaxis().SetMaxDigits(3)
    frame.GetYaxis().SetMoreLogLabels(True)
    frame.GetYaxis().SetNoExponent(False)

    # Legend
    labels = [f"{pt_bins[i]}-{pt_bins[i+1]}" for i in range(len(pt_bins)-1)]
    legend = make_legend(graphs, labels, header="#it{p}_{T} bins (GeV/c)")
    legend.Draw()

    c.Update()
    c.SaveAs(out_file_path)


# ------------------ Compute Centrality-differential Raw Yields ------------------
def compute_cent_diff_rys(config, i_pt, pt_min, pt_max, bkg_max, cent_intervals):
    pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
    out_dir_pt = f"{config['outdir']}/vn_extr_{config['suffix']}_combined/syst/reso/{config['centrality']}/fits/{pt_str}"
    os.makedirs(out_dir_pt, exist_ok=True)
    out_file = TFile.Open(f"{out_dir_pt}/cent_diff_ry_{pt_str}.root", "RECREATE")

    # Retrieve sparse
    if config.get('UseSparses'):
        sparses_flow, _, _, axes = get_pt_preprocessed_sparses(config, pt_str)
        if 'Cent' not in axes['FlowSP']:
            logger("Centrality axis not found, re-run preprocess including it!", level='FATAL')
        sparse_flow = sparses_flow['FlowSP']
        axes_flow = axes['FlowSP']
        # Apply background cut
        sparse_flow.GetAxis(axes_flow['ScoreBkg']).SetRangeUser(0, bkg_max)
    else:
        df = load_aod_file(f"{config['outdir']}/vn_extr_{config['suffix']}_combined/syst/reso/preprocess/{pt_str}/TreesPtCenterSp/AO2D_{pt_str}.root",
                           True, downsample_frac=1.0)

    # Initialize fitter
    fitter = RawYieldFitter(config['Dmeson'], pt_min, pt_max, pt_str, 'flarefly', verbose=False)
    mass_min, mass_max = config["MassFitRanges"][i_pt]
    fitter.set_fit_range(mass_min, mass_max)
    fitter.add_bkg_func(config['BkgFunc'][i_pt] if isinstance(config['BkgFunc'], list) else config['BkgFunc'], "Comb. bkg")

    # Signal functions
    sgn_funcs = {config['SgnFuncLabel']: {'func': config['SgnFunc'][i_pt] if isinstance(config['SgnFunc'], list) else config['SgnFunc'], 'part': config['Dmeson']}}
    if config.get('InclSecPeak'):
        include_sec = config['InclSecPeak'][i_pt] if isinstance(config['InclSecPeak'], list) else config['InclSecPeak']
        if include_sec:
            sgn_funcs[config['SgnFuncSecPeakLabel']] = {
                'func': config['SgnFuncSecPeak'][i_pt] if isinstance(config['SgnFuncSecPeak'], list) else config['SgnFuncSecPeak'],
                'part': 'Dplus' if config['Dmeson'] == 'Ds' else 'Dstar'
            }
    for label, sgn in sgn_funcs.items():
        fitter.add_sgn_func(sgn['func'], label, sgn['part'])

    results = {'raw_yields': [], 'raw_yields_uncs': [], 'cent_intervals': []}
    for i_cent, (cmin, cmax) in enumerate(zip(cent_intervals[:-1], cent_intervals[1:])):
        cent_str = f"cent_{cmin}_{cmax}"
        if config.get('UseSparses'):
            sparse_flow.GetAxis(axes_flow['Cent']).SetRangeUser(cmin, cmax)
            h_mass = sparse_flow.Projection(axes_flow['Mass'])
            h_mass.SetDirectory(0)
            h_cent = sparse_flow.Projection(axes_flow['Cent'])
            out_file.cd()
            h_mass.Write(f"h_mass_{cent_str}")
            h_cent.Write(f"h_cent_{cent_str}")
            fitter.set_data_to_fit_hist(h_mass)
        else:
            sel_string = f"fCent >= {cmin} and fCent < {cmax} and fMlScore0 < {bkg_max} and fM >= {mass_min} and fM <= {mass_max}"
            logger(f"Selection string: {sel_string}", level="INFO")
            sel_df = df.query(sel_string).reset_index(drop=True)
            logger(f"Total entries: {len(df)}, selected entries: {len(sel_df)}", level="INFO")
            fitter.set_data_to_fit_df(sel_df)

        if config.get('corr_bkgs'):
            sel = f"fCent >= {cmin} and fCent < {cmax} and fMlScore0 < {bkg_max} and fM >= {mass_min} and fM <= {mass_max}"
            fitter.add_corr_bkgs(config['corr_bkgs'], sel.replace(' and ', ' && '), pt_min, pt_max)

        fitter.setup()
        if config.get('InitPars'):
            fitter.set_fit_pars(config['InitPars'], pt_min, pt_max)
        status, converged = fitter.fit()
        fitter.plot_fit(
            False, True,
            loc=["lower left", "upper left"],
            path=f"{out_dir_pt}/fM_fit_{cent_str}.pdf",
            out_file=out_file
        )
        fit_info, *_ = fitter.get_fit_info()

        ry = fit_info[config['SgnFuncLabel']]['ry']
        ry_unc = fit_info[config['SgnFuncLabel']]['ry_unc']
        results['raw_yields'].append(ry)
        results['raw_yields_uncs'].append(ry_unc)
        results['cent_intervals'].append((cmin + cmax)/2)

    hist_ry_vs_cent = TH1F("h_ry_vs_cent", ";Centrality (%);Raw Yields",
        len(cent_intervals) - 1, array.array('f', cent_intervals)
    )
    for i, ry in enumerate(results['raw_yields']):
        hist_ry_vs_cent.SetBinContent(i + 1, ry)
        hist_ry_vs_cent.SetBinError(i + 1, results['raw_yields_uncs'][i])
    out_file.cd()
    hist_ry_vs_cent.Write()

    out_file.Close()
    return results


# Wrapper for multiprocessing
def _run_pt_bin(args):
    return compute_cent_diff_rys(*args)


if __name__ == "__main__":

    # ------------------ Parse Arguments ------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to YAML config file")
    parser.add_argument('-w', '--workers', type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    # ------------------ Load Configuration ------------------
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    # ------------------ Preprocess if requested ------------------
    if config['operations'].get('preprocess', False):
        config['outdirPrep'] = f"{config['outdir']}/vn_extr_{config['suffix']}_combined/syst/reso"
        with open(args.config, 'w') as f:
            yaml.dump(config, f, sort_keys=False)

        cmd = f"python {os.path.dirname(os.path.abspath(__file__))}/../../src/pre_process.py {args.config} --workers {args.workers}"
        logger(f"Running preprocessing: {cmd}", level="INFO")
        os.system(cmd)

    # ------------------ Centrality Intervals ------------------
    _, (cent_min, cent_max) = get_centrality_bins(config['centrality'])
    cent_intervals = np.arange(cent_min, cent_max + 1, config['cent_scan_step'])

    # ------------------ Parallel or Sequential pT-bin fits ------------------
    pt_bins = config['ptbins']
    bkg_maxs = config['bkg_max']

    if config['operations'].get('perform_fits_syst_reso'):
        tasks = [
            (config, i_pt, pt_min, pt_max, bkg_max, cent_intervals)
            for i_pt, (pt_min, pt_max, bkg_max) in enumerate(zip(pt_bins[:-1], pt_bins[1:], bkg_maxs))
        ]

        if args.workers > 1:
            with Pool(args.workers) as pool:
                pt_bins_yields = pool.map(_run_pt_bin, tasks)
        else:
            pt_bins_yields = [_run_pt_bin(t) for t in tasks]
    else:
        pt_bins_yields = [None for pt_bin in pt_bins[:-1]]
        for i_pt, (pt_min, pt_max, bkg_max) in enumerate(zip(pt_bins[:-1], pt_bins[1:], bkg_maxs)):
            pt_str = f"pt_{int(10*pt_min)}_{int(10*pt_max)}"
            pt_bin_yields = {
                'cent_intervals': [],
                'raw_yields': [],
                'raw_yields_uncs': []
            }
            yields_file = TFile.Open(f"{config['outdir']}/vn_extr_{config['suffix']}_combined/syst/reso/{config['centrality']}/fits/{pt_str}/cent_diff_ry_{pt_str}.root", "r")
            yields_hist = yields_file.Get("h_ry_vs_cent")
            for i_bin in range(1, yields_hist.GetNbinsX() + 1):
                pt_bin_yields['cent_intervals'].append(yields_hist.GetBinCenter(i_bin))
                pt_bin_yields['raw_yields'].append(yields_hist.GetBinContent(i_bin))
                pt_bin_yields['raw_yields_uncs'].append(yields_hist.GetBinError(i_bin))
            yields_file.Close()
            pt_bins_yields[i_pt] = pt_bin_yields

    logger("All pT bins processed successfully", level="INFO")

    # ------------------ Resolution Values ------------------
    reso_file = TFile.Open(config["Resolution"], 'r')
    det_triplet = f"{config.get('detA', 'FT0c')}_{config.get('detB', 'FV0a')}_{config.get('detC', 'TPCtot')}"
    logger(f"Getting resolution histogram from file {config['Resolution']} for {det_triplet}", "WARNING")
    histo_reso_delta_cent = reso_file.Get(f'{det_triplet}/histo_reso_delta_cent')
    histo_reso_delta_cent.SetDirectory(0)
    reference_reso = histo_reso_delta_cent.GetBinContent(1)
    histo_reso = reso_file.Get(f'{det_triplet}/histo_reso')
    histo_reso.SetDirectory(0)
    reso_values = [histo_reso.GetBinContent(i) for i in range(1, histo_reso.GetNbinsX()+1)]
    reso_file.Close()

    # ------------------ Weighted and Arithmetic Resolutions ------------------
    avg_resos, ry_histos, single_term_histos, cent_pt_integrated_yields = [], [], [], []
    out_file = TFile.Open(f"{config['outdir']}/vn_extr_{config['suffix']}_combined/syst/reso/{config['centrality']}/syst_reso.root", "RECREATE")

    for i_pt, pt_bin_yields in enumerate(pt_bins_yields):
        tot_yield = sum(pt_bin_yields['raw_yields'])
        avg_reso = weighted_average(reso_values[:len(pt_bin_yields['raw_yields'])],
                                    pt_bin_yields['raw_yields'])
        avg_resos.append(avg_reso)

        # Fill histograms for ROOT file
        ry_histo = TH1F(f"ry_histo_{i_pt}", f";Centrality (%);Raw Yields",
                        len(pt_bin_yields['cent_intervals']), array.array('f', cent_intervals))
        single_term_histo = TH1F(f"contribution_histo_{i_pt}", ";Centrality (%);Contribution to avg. reso.",
                                 len(pt_bin_yields['cent_intervals']), array.array('f', cent_intervals))
        for i, (ry, unc, r_val) in enumerate(zip(pt_bin_yields['raw_yields'],
                                                 pt_bin_yields['raw_yields_uncs'],
                                                 reso_values[:len(pt_bin_yields['raw_yields'])])):
            ry_histo.SetBinContent(i+1, ry)
            ry_histo.SetBinError(i+1, unc)
            single_term_histo.SetBinContent(i+1, r_val * ry / tot_yield)
        ry_histos.append(ry_histo)
        single_term_histos.append(single_term_histo)
        cent_pt_integrated_yields.append(tot_yield)
        ry_histo.Write()
        single_term_histo.Write()

    # Save resolution histogram
    histo_reso.Write("reso_vs_cent")
    out_file.Write()

    # # Calculate resolution systematics as shift
    weighted_sum = sum(r * w for r, w in zip(avg_resos, cent_pt_integrated_yields))
    total_yield = sum(cent_pt_integrated_yields)
    avg_reso_trials = weighted_sum / total_yield
    syst_unc = abs(avg_reso_trials - reference_reso)
    h_syst_unc = TH1F("h_syst_unc", ";Resolution", 2, 0, 2)
    h_syst_unc.SetBinContent(1, avg_reso_trials)
    h_syst_unc.GetXaxis().SetBinLabel(1, "Weighted Mean")
    h_syst_unc.SetBinContent(2, reference_reso)
    h_syst_unc.GetXaxis().SetBinLabel(2, "Reference")
    h_syst_unc.Write('h_syst_unc')

    # ------------------ Produce Figures ------------------
    # Weighted vs Arithmetic Resolution
    reso_syst_figure(pt_bins, avg_resos, reference_reso, syst_unc,
                     f"{config['outdir']}/vn_extr_{config['suffix']}_combined/syst/reso/{config['centrality']}/reso_syst.pdf")

    # Raw Yields vs Centrality
    ry_vs_cent_figure(pt_bins, cent_intervals, pt_bins_yields,
                      f"{config['outdir']}/vn_extr_{config['suffix']}_combined/syst/reso/{config['centrality']}/reso_syst_ry_vs_cent.pdf")

    logger("All figures produced successfully", "INFO")

    out_file.Close()
    logger(f"Syst. resolution saved in {config['outdir']}", "INFO")
