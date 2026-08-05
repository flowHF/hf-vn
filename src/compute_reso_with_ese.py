import sys
import argparse
import yaml
import ROOT
from ROOT import TH1F, TH2F, TH3
import os
import numpy as np
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import get_centrality_bins, make_dir_root_file, get_ese_band_label
from load_utils import load_reso_histos, load_ese_quantiles_thresholds
from StyleFormatter import SetObjectStyle, SetGlobalStyle
SetGlobalStyle(padleftmargin=0.15, padbottommargin=0.15,
               padrightmargin=0.15, titleoffsety=1.1, maxdigits=3, titlesizex=0.03,
               labelsizey=0.04, setoptstat=0, setopttitle=0, palette=ROOT.kGreyScale)

ROOT.gROOT.SetBatch(True)

def parse_ese_bands(bands_cfg):
    """Build {label: (lo, hi)} from a list of [lo, hi] q2 percentile pairs."""
    bands = {}
    for edges in (bands_cfg or []):
        if not (isinstance(edges, (list, tuple)) and len(edges) == 2):
            raise ValueError(f"ESE band {edges} must be a [lo, hi] pair of percentiles")
        p_lo, p_hi = int(edges[0]), int(edges[1])
        bands[get_ese_band_label(p_lo, p_hi)] = (p_lo, p_hi)
    return bands

ROOT.TH1.AddDirectory(False)

# TODO: move this to the StyleFormatter
def SetFrameStyle(hFrame, xtitle, ytitle, ytitleoffset, ytitlesize, ylabelsize,
                  ylabeloffset, xticklength, yticklength, xtitlesize, xlabelsize,
                  xtitleoffset, xlabeloffset, ydivisions, xmoreloglabels, ycentertitle, ymaxdigits):
    hFrame.GetXaxis().SetTitle(xtitle)
    hFrame.GetYaxis().SetTitle(ytitle)
    hFrame.GetYaxis().SetTitleOffset(ytitleoffset)
    hFrame.GetYaxis().SetTitleSize(ytitlesize)
    hFrame.GetYaxis().SetLabelSize(ylabelsize)
    hFrame.GetYaxis().SetLabelOffset(ylabeloffset)
    hFrame.GetXaxis().SetTickLength(xticklength)
    hFrame.GetYaxis().SetTickLength(yticklength)
    hFrame.GetXaxis().SetTitleSize(xtitlesize)
    hFrame.GetXaxis().SetLabelSize(xlabelsize)
    hFrame.GetXaxis().SetTitleOffset(xtitleoffset)
    hFrame.GetXaxis().SetLabelOffset(xlabeloffset)
    hFrame.GetYaxis().SetNdivisions(ydivisions)
    hFrame.GetXaxis().SetMoreLogLabels(xmoreloglabels)
    hFrame.GetYaxis().CenterTitle(ycentertitle)
    hFrame.GetYaxis().SetMaxDigits(ymaxdigits)


def get_resolution(dets, det_lables, cent_min_max):
    '''
    Compute resolution for SP method

    Input:
        - dets:
            list of TH2D, list of TH2D objects with the SP product or EP cos(deltaphi) values vs centrality
        - det_lables:
            list of strings, list of detector labels
        - cent_min_max:
            list of floats, max and min centrality bins

    Output:
        - histo_means:
            list of TH1D, list of histograms with the mean value of the projections as a function of centrality for 1% bins
        - histo_means_deltacent:
            list of TH1D, list of histograms with the mean value of the projections as a function of centrality for CentMin-CentMax
        - histo_reso:
            TH1D, histogram with the resolution value as a function of centrality for 1% bins
        - histo_reso_delta_cent:
            TH1D, histogram with the resolution value as a function of centrality for CentMin-CentMax
    '''
    histo_projs, histo_means, histo_means_deltacent = [], [], []

    # collect the qvecs and prepare histo for mean and resolution
    for _, (det, det_label) in enumerate(zip(dets, det_lables)):
        print(f'Processing {det_label}')
        # th1 for mean 1% centrality bins
        histo_means.append(ROOT.TH1F('', '', cent_min_max[1]-cent_min_max[0], cent_min_max[0], cent_min_max[1]))
        histo_means[-1].SetDirectory(0)
        histo_means[-1].SetName(f'proj_{det_label}_mean')
        # th1 for mean CentMin-CentMax
        histo_projs.append([])
        hist_proj_dummy = det.ProjectionY(f'proj_{det.GetName()}_mean_deltacent',
                                          det.GetXaxis().FindBin(cent_min_max[0]),
                                          det.GetXaxis().FindBin(cent_min_max[1])-1)
        histo_means_deltacent.append(ROOT.TH1F('', '', 1, cent_min_max[0], cent_min_max[1]))
        histo_means_deltacent[-1].SetDirectory(0)
        histo_means_deltacent[-1].SetName(f'proj_{det_label}_mean_deltacent')

        # Set mean values for CentMin-CentMax
        histo_means_deltacent[-1].SetBinContent(1, hist_proj_dummy.GetMean())
        histo_means_deltacent[-1].SetBinError(1, hist_proj_dummy.GetMeanError())
        del hist_proj_dummy

        # collect projections 1% centrality bins
        for cent in range(cent_min_max[0], cent_min_max[1]):
            bin_cent = det.GetXaxis().FindBin(cent) # common binning
            histo_projs[-1].append(det.ProjectionY(f'proj_{det_label}_{cent}',
                                                          bin_cent, bin_cent))
        # Set mean values for 1% centrality bins
        for ihist, _ in enumerate(histo_projs[-1]):
            histo_means[-1].SetBinContent(ihist+1, histo_projs[-1][ihist].GetMean())

    # Compute resolution for 1% centrality bins
    histo_reso = ROOT.TH1F('histo_reso', 'histo_reso',
                           cent_min_max[1]-cent_min_max[0],
                           cent_min_max[0], cent_min_max[1])
    histo_reso.SetDirectory(0)
    for icent in range(cent_min_max[0], cent_min_max[1]):
        reso = compute_resolution([histo_means[i].GetBinContent(icent-cent_min_max[0]+1) for i in range(len(dets))])
        centbin = histo_reso.GetXaxis().FindBin(icent)
        histo_reso.SetBinContent(centbin, reso)

    # Compute resolution for CentMin-CentMax
    histo_reso_delta_cent = ROOT.TH1F('histo_reso_delta_cent', 'histo_reso_delta_cent',
                                      1, cent_min_max[0], cent_min_max[1])
    res_deltacent = compute_resolution([histo_means_deltacent[i].GetBinContent(1) for i in range(len(dets))])
    histo_reso_delta_cent.SetBinContent(1, res_deltacent)
    histo_reso_delta_cent.SetDirectory(0)

    return histo_means, histo_means_deltacent, histo_reso, histo_reso_delta_cent


def compute_resolution(subMean):
    '''
    Compute resolution for SP or EP method

    Input:
        - subMean:
            list of floats, list of mean values of the projections

    Output:
        - resolution:
            float, resolution value
    '''
    print(subMean)
    if len(subMean) == 1:
        resolution =  subMean[0]
        if resolution <= 0:
            return 0
        else:
            return np.sqrt(resolution)
    elif len(subMean) == 3:
        print('3 subsystems')
        resolution = (subMean[0] * subMean[1]) / subMean[2] if subMean[2] != 0 else 0
        if resolution <= 0:
            return 0
        else:
            print(resolution, np.sqrt(resolution))
            return np.sqrt(resolution)
    else:
        print('ERROR: dets must be a list of 2 or 3 subsystems')
        sys.exit(1)


def project_ese(histos_triplets, histos_triplets_labels, ese_thresholds, ese_bands,
                triplet=('FT0c', 'FV0a', 'TPCtot')):
    '''
    Project histograms for each ESE selection

    Input:
        - histos_triplets:
            list of list of TH2D, list of list of TH2D objects with the SP product or EP cos(deltaphi) values vs centrality
        - histos_triplets_labels:
            list of list of strings, list of list of detector labels
        - ese_thresholds:
            dict, dictionary with ESE thresholds for each selection
    Return:
        - histos_filtered_dict:
            dict, dictionary with the projected histograms for each ESE selection
    '''

    print("Entering project_ese")

    histos_ese_sel_dict = {}

    # Inclusive case every time
    histos_ese_sel_dict['Inclusive'] = {}
    for histo, label in zip(histos_triplets, histos_triplets_labels):
        hist_cent_vs_qvec_prod = []
        for single_hist in histo:
            if isinstance(single_hist, ROOT.TH3):
                hist_cent_vs_qvec_prod.append(single_hist.Project3D('yx'))
            else:
                hist_cent_vs_qvec_prod.append(single_hist)
        histos_ese_sel_dict['Inclusive'][label] = tuple(hist_cent_vs_qvec_prod)

    if ese_thresholds:
        cent_keys = list(next(iter(ese_thresholds.values())).keys())
        for band_label, (p_lo, p_hi) in ese_bands.items():
            for p in (p_lo, p_hi):
                if 0 < p < 100 and f'quantile_{p}' not in ese_thresholds:
                    raise KeyError(f"quantile_{p} not available in the ESE quantile file "
                                   f"(available: {sorted(ese_thresholds.keys())})")
            thr_lo_dict = (ese_thresholds[f'quantile_{p_lo}'] if p_lo > 0
                           else {c: 0.0 for c in cent_keys})
            thr_hi_dict = (ese_thresholds[f'quantile_{p_hi}'] if p_hi < 100
                           else {c: 999.0 for c in cent_keys})
            print(f'Projecting to TH2 for ESE band: {band_label} '
                  f'(quantiles {p_lo}-{p_hi})')
            histos_ese_sel_dict[band_label] = {}
            for histo, label in zip(histos_triplets, histos_triplets_labels):
                if tuple(label) != tuple(triplet):
                    continue
                hists_band = []
                for single_hist in histo:
                    h2 = TH2F(single_hist.GetName(), single_hist.GetName(),
                              single_hist.GetXaxis().GetNbins(),
                              single_hist.GetXaxis().GetXmin(),
                              single_hist.GetXaxis().GetXmax(),
                              single_hist.GetYaxis().GetNbins(),
                              single_hist.GetYaxis().GetXmin(),
                              single_hist.GetYaxis().GetXmax())
                    h2.SetDirectory(0)
                    n_matched = 0
                    for i_cent_bin in range(1, single_hist.GetXaxis().GetNbins()+1):
                        c_lo = single_hist.GetXaxis().GetBinLowEdge(i_cent_bin)
                        c_hi = single_hist.GetXaxis().GetBinUpEdge(i_cent_bin)
                        c_mid = (c_lo + c_hi) / 2
                        if c_mid not in thr_lo_dict:
                            continue
                        cent_bin = h2.GetXaxis().FindBin(c_mid)
                        single_hist.GetXaxis().SetRange(i_cent_bin, i_cent_bin)
                        single_hist.GetZaxis().SetRangeUser(thr_lo_dict[c_mid],
                                                            thr_hi_dict[c_mid])
                        hproj = single_hist.Project3D('y')
                        for i_q in range(1, single_hist.GetYaxis().GetNbins()+1):
                            h2.SetBinContent(cent_bin, i_q, hproj.GetBinContent(i_q))
                            h2.SetBinError(cent_bin, i_q, hproj.GetBinError(i_q))
                        single_hist.GetXaxis().SetRange(0, 0)
                        single_hist.GetZaxis().SetRange(0, 0)
                        n_matched += 1
                    if n_matched == 0:
                        raise RuntimeError(f"Band {band_label}: no centrality bin matched the "
                                           f"threshold map, check the centrality binning")
                    hists_band.append(h2)
                histos_ese_sel_dict[band_label][label] = tuple(hists_band)
            if not histos_ese_sel_dict[band_label]:
                raise RuntimeError(f"Band {band_label}: triplet {triplet} not found in input")

    return histos_ese_sel_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("an_res_file", metavar="text",
                        default="an_res.root", help="input ROOT file with anres")
    parser.add_argument('--centClass', '-c', metavar='text', default='k0100')
    parser.add_argument("--wagon_id", "-w", metavar="text",
                        default="", help="wagon ID", required=False)
    parser.add_argument("--outputdir", "-o", metavar="text",
                        default=".", help="output directory")
    parser.add_argument("--suffix", "-s", metavar="text",
                        default="", help="suffix for output files")
    parser.add_argument("--ese_file", "-ef", metavar="text",
                        default="", help="file with ESE percentiles")
    parser.add_argument("--ese_detector", "-ed", metavar="text",
                        default="", help="Detector for ESE quantiles")
    parser.add_argument("--config", "-cfg", metavar="text", default="",
                        help="config file, ESE bands are read from its 'ese' block")
    parser.add_argument("--batch", "-b", action='store_true', help="run in batch mode")
    args = parser.parse_args()

    _, cent_min_max = get_centrality_bins(args.centClass)

    histos_triplets, histos_triplets_labels = load_reso_histos(args.an_res_file, args.wagon_id)

    # Load EsE thresholds if provided and apply selections
    ese_cfg, proj_cfg = {}, {}
    if args.config:
        with open(args.config, 'r') as cfg_file:
            cfg = yaml.safe_load(cfg_file)
        ese_cfg = cfg.get('ese') or {}
        proj_cfg = cfg.get('projections') or {}
    ese_bands = parse_ese_bands(ese_cfg.get('bands'))
    ese_file = args.ese_file or ese_cfg.get('quantile_file', '')
    ese_det = args.ese_detector or ese_cfg.get('det', '')
    triplet = (proj_cfg.get('detA', 'FT0c'),
               proj_cfg.get('detB', 'FV0a'),
               proj_cfg.get('detC', 'TPCtot'))

    ese_thresholds = None
    if ese_file and ese_bands:
        ese_thresholds = load_ese_quantiles_thresholds(ese_file, ese_det,
                                                       cent_min_max[0], cent_min_max[1])
        if not ese_thresholds:
            raise RuntimeError(f"No quantiles loaded from {ese_file} for detector '{ese_det}'")
    elif ese_bands:
        print('WARNING: ESE bands defined but no quantile file, only Inclusive will be computed')
    histos_filtered_dict = project_ese(histos_triplets, histos_triplets_labels,
                                       ese_thresholds, ese_bands, triplet)
    # exit(1)

    # # Save histos for debugging purposes
    # debug_outfile_name = f'{args.outputdir}/debug_histo_{args.suffix}.root'
    # debug_outfile = ROOT.TFile(debug_outfile_name, 'RECREATE')
    # for ese_sel_label, histo_triplet_dict in histos_filtered_dict.items():
    #     for histo_triplet_label, histo_triplet in histo_triplet_dict.items():
    #         for hist in histo_triplet:
    #             hist.SetDirectory(debug_outfile)
    #             hist.Write()
    # debug_outfile.Close()

    print("Histograms projected for ESE selections. Starting resolution computation...")

    # prepare output file
    ytitle = 'Q^{A} Q^{B}'
    outfile_name = f'{args.outputdir}/reso_{args.suffix}.root'
    outfile = ROOT.TFile(outfile_name, 'RECREATE')

    # loop over all possible combinations of detectors
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.05)
    print('\n\n\n')
    for ese_sel_label, histo_triplet_dict in histos_filtered_dict.items():
        print(f'Processing ESE selection: {ese_sel_label}')
        print(f"histo_triplet_dict: {histo_triplet_dict}")
        for histo_triplet_label, histo_triplet in histo_triplet_dict.items():
            # histo_triplet_label = list(histo_triplet_dict.keys())[0]
            # histo_triplet = histo_triplet_dict[histo_triplet_label]
            print(f'histo_triplet_label: {histo_triplet_label}')
            print(f'histo_triplet: {histo_triplet}')
            histos_mean, histos_mean_deltacent, histo_reso, histo_reso_deltacent = get_resolution(histo_triplet,
                                                                                                histo_triplet_label,
                                                                                                cent_min_max)
            detA_label = histo_triplet_label[0]
            detB_label = histo_triplet_label[1]
            detC_label = histo_triplet_label[2]
            outfile.cd()
            make_dir_root_file(f'{ese_sel_label}/{detA_label}_{detB_label}_{detC_label}', outfile, verbose=False)
            outfile.cd(f'{ese_sel_label}/{detA_label}_{detB_label}_{detC_label}')
            print(f'Processing combination: {detA_label}, {detB_label}, {detC_label} for ESE selection: {ese_sel_label}')
            canvas = ROOT.TCanvas(f'canvas_{detA_label}_{detB_label}_{detC_label}_{ese_sel_label}',
                                f'canvas_{detA_label}_{detB_label}_{detC_label}_{ese_sel_label}',
                                2400, 800)
            print(f'Canvas created for combination: {detA_label}, {detB_label}, {detC_label} for ESE selection: {ese_sel_label}')
            canvas.Divide(3, 1)
            leg = ROOT.TLegend(0.2, 0.2, 0.5, 0.3)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetTextSize(0.03)
            print(f'Processing histograms for combination: {detA_label}, {detB_label}, {detC_label} for ESE selection: {ese_sel_label}')
            for i, (hist_det, hist_mean, histo_mean_deltacent) in enumerate(zip(histo_triplet,
                                                                                histos_mean,
                                                                                histos_mean_deltacent)):
                print(f'Processing histogram {i+1} for combination: {detA_label}, {detB_label}, {detC_label} for ESE selection: {ese_sel_label}')
                SetObjectStyle(hist_mean, color=ROOT.kRed, markerstyle=ROOT.kFullCircle,
                            markersize=1, fillstyle=0, linewidth=2)
                SetObjectStyle(histo_mean_deltacent, color=ROOT.kBlue, markerstyle=ROOT.kOpenCircle,
                            markersize=1, fillstyle=0, linestyle=2, linewidth=3)
                canvas.cd(i+1)
                canvas.cd(i+1).SetLogz()
                hFrame = canvas.cd(i+1).DrawFrame(0, -2, 100, 2)
                SetFrameStyle(hFrame,
                            xtitle='Cent. FT0c (%)',
                            ytitle=ytitle,
                            ytitleoffset=1.15,
                            ytitlesize=0.05,
                            ylabelsize=0.04,
                            ylabeloffset=0.01,
                            xticklength=0.04,
                            yticklength=0.03,
                            xtitlesize=0.05,
                            xlabelsize=0.04,
                            xtitleoffset=1.1,
                            xlabeloffset=0.020,
                            ydivisions=406,
                            xmoreloglabels=True,
                            ycentertitle=True,
                            ymaxdigits=5)
                hist_det.Draw('same colz')
                histo_mean_deltacent.Draw('same pl')
                hist_mean.Draw('same pl')
                print(f'Histograms drawn for histogram {i+1} for combination: {detA_label}, {detB_label}, {detC_label} for ESE selection: {ese_sel_label}')
                if i == 0:
                    leg.AddEntry(hist_mean, 'Average 1% centrality', 'lp')
                    leg.AddEntry(histo_mean_deltacent,
                                f'Average {cent_min_max[1]-cent_min_max[0]}% centrality', 'lp')
                    leg.Draw()
                    latex.DrawLatex(0.2, 0.85, f'A: {detA_label}, B: {detB_label}')
                elif i == 1:
                    latex.DrawLatex(0.2, 0.85, f'A: {detA_label}, B: {detC_label}')
                else:
                    latex.DrawLatex(0.2, 0.85, f'A: {detB_label}, B: {detC_label}')
                histo_mean_deltacent.Write()
                hist_mean.Write()
                hist_det.Write()
                print(f'Histograms written for histogram {i+1} for combination: {detA_label}, {detB_label}, {detC_label} for ESE selection: {ese_sel_label}')
            canvas.Update()
            canvas.Write()
            histo_reso.SetDirectory(outfile)
            histo_reso_deltacent.SetDirectory(outfile)
            histo_reso.Write()
            histo_reso_deltacent.Write()
            outfile.cd('..')
            print(f'Finished processing combination: {detA_label}, {detB_label}, {detC_label} for ESE selection: {ese_sel_label}')
    outfile.Close()

    if not args.batch:
        input('Resolutions computed. Press any key to continue')

