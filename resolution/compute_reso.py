import sys
import argparse
import ROOT
import os
import numpy as np
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import get_centrality_bins
from load_utils import load_reso_histos
from StyleFormatter import SetObjectStyle, SetGlobalStyle
SetGlobalStyle(padleftmargin=0.15, padbottommargin=0.15,
               padrightmargin=0.15, titleoffsety=1.1, maxdigits=3, titlesizex=0.03,
               labelsizey=0.04, setoptstat=0, setopttitle=0, palette=ROOT.kGreyScale)

ROOT.gROOT.SetBatch(True)

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


# def compute_reso(an_res_file, centClass, wagon_id, outputdir, suffix):


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
    parser.add_argument("--batch", "-b", action='store_true', help="run in batch mode")
    args = parser.parse_args()

    _, cent_min_max = get_centrality_bins(args.centClass)
    histos_triplets, histos_triplets_lables = load_reso_histos(args.an_res_file, args.wagon_id)

    # prepare output file
    ytitle = 'Q^{A} Q^{B}'
    outfile_name = f'{args.outputdir}/reso_{args.suffix}.root'
    outfile = ROOT.TFile(outfile_name, 'RECREATE')

    # loop over all possible combinations of detectors
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.05)
    for i, (histo_triplet, histo_triplet_label) in enumerate(zip(histos_triplets, histos_triplets_lables)):
        histos_mean, histos_mean_deltacent, histo_reso, histo_reso_deltacent = get_resolution(histo_triplet,
                                                                                              histo_triplet_label,
                                                                                              cent_min_max)
        detA_label = histo_triplet_label[0]
        detB_label = histo_triplet_label[1]
        detC_label = histo_triplet_label[2]
        outfile.cd()
        outfile.mkdir(f'{detA_label}_{detB_label}_{detC_label}')
        outfile.cd(f'{detA_label}_{detB_label}_{detC_label}')
        canvas = ROOT.TCanvas(f'canvas_{detA_label}_{detB_label}_{detC_label}',
                              f'canvas_{detA_label}_{detB_label}_{detC_label}',
                              2400, 800)
        canvas.Divide(3, 1)
        leg = ROOT.TLegend(0.2, 0.2, 0.5, 0.3)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.03)
        for i, (hist_det, hist_mean, histo_mean_deltacent) in enumerate(zip(histo_triplet,
                                                                            histos_mean,
                                                                            histos_mean_deltacent)):
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
        canvas.Update()
        canvas.Write()
        histo_reso.SetDirectory(outfile)
        histo_reso_deltacent.SetDirectory(outfile)
        histo_reso.Write()
        histo_reso_deltacent.Write()
        outfile.cd('..')
    outfile.Close()

    if not args.batch:
        input('Resolutions computed. Press any key to continue')
