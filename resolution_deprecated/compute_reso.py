import sys
import os
import argparse
import array
import ROOT
from ROOT import TH1, THnSparse, TH1F, TFile, TCanvas, TLatex, TLegend, kRed, kBlue, kOpenCircle, kFullCircle
from ROOT import kGreyScale
from ROOT import gROOT
import numpy as np
import yaml
import subprocess
work_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{work_dir}/..")
sys.path.append(f"{work_dir}/../utils")
from itertools import combinations
from utils import get_centrality_bins
from StyleFormatter import SetObjectStyle, SetGlobalStyle, SetFrameStyle
SetGlobalStyle(padleftmargin=0.15, padbottommargin=0.15,
               padrightmargin=0.15, titleoffsety=1.1, maxdigits=3, titlesizex=0.03,
               labelsizey=0.04, setoptstat=0, setopttitle=0, palette=kGreyScale)

gROOT.SetBatch(False)
TH1.AddDirectory(False)    # Avoids using SetDirectory(0) for each histogram

def get_resolution(dets, det_lables, cent_min, cent_max):
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
    histo_means, histo_means_deltacent, histo_rms = [], [], []
    delta_cent = cent_max - cent_min

    # collect the qvecs and prepare histo for mean and resolution
    for _, (det, det_label) in enumerate(zip(dets, det_lables)):

        # th1 for mean CentMin-CentMax
        hist_proj_allbins = det.ProjectionY(f'proj_{det.GetName()}_mean_deltacent',
                                            det.GetXaxis().FindBin(cent_min),
                                            det.GetXaxis().FindBin(cent_max)-1)
        histo_means_deltacent.append(TH1F('', '', 1, cent_min, cent_max))
        histo_means_deltacent[-1].SetName(f'proj_{det_label}_mean_deltacent')
        histo_means_deltacent[-1].SetBinContent(1, hist_proj_allbins.GetMean())
        histo_means_deltacent[-1].SetBinError(1, hist_proj_allbins.GetMeanError())
        del hist_proj_allbins

        # th1 for mean and rms in 1% centrality bins
        histo_means.append(TH1F(f'proj_{det_label}_mean', f'proj_{det_label}_mean', delta_cent, cent_min, cent_max))
        histo_rms.append(TH1F(f'proj_{det_label}_rms', f'proj_{det_label}_rms', delta_cent, cent_min, cent_max))
        for icent, cent in enumerate(range(cent_min, cent_max)):
            bin_cent = det.GetXaxis().FindBin(cent) # common binning
            histo_proj = det.ProjectionY(f'proj_{det_label}_{cent}_{icent}', bin_cent, bin_cent)
            histo_means[-1].SetBinContent(icent+1, histo_proj.GetMean())
            histo_rms[-1].SetBinContent(icent+1, histo_proj.GetRMS())

    # Compute resolution for 1% centrality bins
    histo_reso = TH1F('histo_reso', 'histo_reso', delta_cent, cent_min, cent_max)
    for icent in range(cent_min, cent_max):
        reso = compute_resolution([histo_means[i].GetBinContent(icent-cent_min+1) for i in range(len(dets))])
        centbin = histo_reso.GetXaxis().FindBin(icent)
        histo_reso.SetBinContent(centbin, reso)

    # Compute resolution for CentMin-CentMax
    histo_reso_delta_cent = TH1F('histo_reso_delta_cent', 'histo_reso_delta_cent', 1, cent_min, cent_max)
    res_deltacent = compute_resolution([histo_means_deltacent[i].GetBinContent(1) for i in range(len(dets))])
    histo_reso_delta_cent.SetBinContent(1, res_deltacent)

    return histo_means, histo_means_deltacent, histo_rms, histo_reso, histo_reso_delta_cent

def compute_resolution(subMean):
    '''
    Compute resolution for SP method

    Input:
        - subMean:
            list of floats, list of mean values of the projections

    Output:
        - resolution:
            float, resolution value
    '''
    resolution = (subMean[0] * subMean[1]) / subMean[2] if subMean[2] != 0 else 0
    return np.sqrt(resolution) if resolution >= 0 else 0

def getListOfHistos(an_res_list):
    '''
    Get list of histograms for SP resolution

    Input:
        - an_res_list:
            str, resolution file
        - vn_method:
            str, vn method (sp or ep)

    Output:
        - matched_triplets:
            list of TH2D, list of TH2D objects with the SP product or EP cos(deltaphi) values vs centrality
        - matched_labels:
            list of strings, list of detector labels
    '''
    # generate triplets of pairs (AB, AC, BC)
    histos = {}
    for filepath in an_res_list:
        print(f"Processing resolution file: {filepath}")
        infile = TFile(filepath, 'READ')
        dir = infile.GetDirectory('hf-task-flow-charm-hadrons/spReso')
        for hist in dir.GetListOfKeys():
            if hist.GetName().startswith('hSpReso'):
                histo = hist.ReadObj()
                if histo.GetName() not in histos:
                    histos[histo.GetName()] = histo
                else:
                    histos[histo.GetName()].Add(histo)

    pairs = [name.replace('hSpReso', '') for name, _ in histos.items()]
    triplets = list(combinations(pairs, 3))
    histo_triplets = list(combinations(list(histos.values()), 3))

    matched_triplets, matched_labels = [], []
    detsA = ['FT0c', 'FT0a', 'FV0a', 'TPCpos', 'FT0m', 'TPCneg']
    for i, triplet in enumerate(triplets):
        for detA in detsA:
            detB = triplet[0].replace(detA, '')
            detC = triplet[1].replace(detA, '')
            if (detA in triplet[0] and detA in triplet[1]) and \
               (detB in triplet[0] and detB in triplet[2]) and \
               (detC in triplet[1] and detC in triplet[2]):
                    matched_triplets.append(histo_triplets[i])
                    matched_labels.append((detA, detB, detC))

    return matched_triplets, matched_labels

def compute_reso(file_list, cent_classes, outfile):

    # loop over all possible combinations of detectors
    histos_triplets, histos_triplets_lables = getListOfHistos(file_list)
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.05)
    ytitle = 'Q^{A} Q^{B}'
    print(f"Number of triplets: {len(histos_triplets)}")
    print(f"Number of triplet labels: {len(histos_triplets_lables)}")
    print(f"triplet labels: {histos_triplets_lables}")
    # quit()
    reso_all_cents = {}
    for label in histos_triplets_lables:
        detA_label, detB_label, detC_label = label[0], label[1], label[2]
        label_string = f'{detA_label}_{detB_label}_{detC_label}'
        reso_all_cents[label_string] = TH1F(f'reso_all_cents_{label_string}', f'reso_all_cents_{label_string}', len(cent_classes)-1, array.array('d', cent_classes))
    for icent, (cent_min, cent_max) in enumerate(zip(cent_classes[:-1], cent_classes[1:])):
        for i, (triplet, labels) in enumerate(zip(histos_triplets, histos_triplets_lables)):

            histos_mean, histos_mean_deltacent, histos_rms, histo_reso, histo_reso_deltacent = get_resolution(triplet, labels, cent_min, cent_max)
            detA_label, detB_label, detC_label = labels[0], labels[1], labels[2]
            dir = f'cent_{cent_min}_{cent_max}/{detA_label}_{detB_label}_{detC_label}'
            outfile.mkdir(dir)
            outfile.cd(dir)

            canvas = TCanvas(f'c_{detA_label}_{detB_label}_{detC_label}', f'c_{detA_label}_{detB_label}_{detC_label}', 2400, 800)
            canvas.Divide(3, 1)
            leg = TLegend(0.2, 0.2, 0.5, 0.3)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetTextSize(0.03)
            for i, (hist_det, hist_mean, hist_rms, histo_mean_deltacent) in enumerate(zip(triplet,
                                                                                    histos_mean, histos_rms,
                                                                                    histos_mean_deltacent)):
                SetObjectStyle(hist_mean, color=kRed, markerstyle=kFullCircle,
                            markersize=1, fillstyle=0, linewidth=2)
                SetObjectStyle(hist_rms, color=kRed, markerstyle=kFullCircle,
                            markersize=1, fillstyle=0, linewidth=2)
                SetObjectStyle(histo_mean_deltacent, color=kBlue, markerstyle=kOpenCircle,
                            markersize=1, fillstyle=0, linestyle=2, linewidth=3)
                canvas.cd(i+1)
                canvas.cd(i+1).SetLogz()
                hFrame = canvas.cd(i+1).DrawFrame(0, -2, 100, 2)
                SetFrameStyle(hFrame, xtitle='Cent. FT0c (%)', ytitle=ytitle, ytitleoffset=1.15, ytitlesize=0.05, 
                            ylabelsize=0.04, ylabeloffset=0.01, yticklength=0.03,
                            xticklength=0.04, xtitlesize=0.05, xlabelsize=0.04,
                            xtitleoffset=1.1, xlabeloffset=0.020, ydivisions=406,
                            xmoreloglabels=True, ycentertitle=True, ymaxdigits=5)
                hist_det.Draw('same colz')
                histo_mean_deltacent.Draw('same pl')
                hist_mean.Draw('same pl')
                if i == 0:
                    leg.AddEntry(hist_mean, 'Average 1% centrality', 'lp')
                    leg.AddEntry(histo_mean_deltacent,
                                f'Average {cent_min-cent_max}% centrality', 'lp')
                    leg.Draw()
                    latex.DrawLatex(0.2, 0.85, f'A: {detA_label}, B: {detB_label}')
                elif i == 1:
                    latex.DrawLatex(0.2, 0.85, f'A: {detA_label}, B: {detC_label}')
                else:
                    latex.DrawLatex(0.2, 0.85, f'A: {detB_label}, B: {detC_label}')
                histo_mean_deltacent.Write()
                hist_mean.Write()
                hist_rms.Write()
                hist_det.Write()
            canvas.Update()
            canvas.Write()
            histo_reso.Write()
            histo_reso_deltacent.Write()
            reso_all_cents[f"{detA_label}_{detB_label}_{detC_label}"].SetBinContent(icent+1, histo_reso_deltacent.GetBinContent(1))

    outfile.cd()
    for reso_all_cent in reso_all_cents.values():
        reso_all_cent.Write()
    outfile.Close()
    print(f"Resolution files saved in {outfile.GetName()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("cfg", metavar="text", default="cfg_file.yml", help="Yaml config file")
    args = parser.parse_args()

    with open(args.cfg, 'r') as yml_file:
        config = yaml.load(yml_file, yaml.FullLoader)

    # Load the files run by run
    os.system(f"python3 {work_dir}/../utils/load_long_train_run_by_run.py -cfg {args.cfg}")

    # # Compute the resolution run by run
    file_output_dir = f"{config['output_dir']}/Train{config['train_number']}/"
    reso_output_dir = config["reso_output_dir"]
    os.makedirs(reso_output_dir, exist_ok=True)
    reso_files = []
    if config.get("grid_runs"):
        for run in config["grid_runs"]:
            print(f"Processing grid run {run} ...")
            input_file_path = f"{file_output_dir}/runs/{run}/AnalysisResults.root"
            os.makedirs(f"{reso_output_dir}/{run}", exist_ok=True)
            outfile = TFile(f'{reso_output_dir}/{run}/resosp{config["suffix"]}.root', 'RECREATE')
            compute_reso([input_file_path], config["cent_classes"], outfile)
            reso_files.append(input_file_path)
            print(f"Processed!")
        
    if config.get("single_runs"):
        for run in config["single_runs"]:
            print(f"Processing single run {run['number']}")
            input_file_path = f"{file_output_dir}/single_runs/{run['number']}/AnalysisResults.root"
            command = f'find {file_output_dir}/single_runs/{run["number"]} -wholename "*/AnalysisResults.root" | tr "\n" " "'
            print(f"command: {command}")
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            run_output_list = result.stdout.strip().split()
            print('\n')
            print(f"Files found: {len(run_output_list)}")
            print('\n')
            os.makedirs(f"{reso_output_dir}{run['number']}", exist_ok=True)
            outfile = TFile(f'{reso_output_dir}/{run["number"]}/resosp{config["suffix"]}.root', 'RECREATE')
            compute_reso(run_output_list, config["cent_classes"], outfile)
            reso_files.extend(run_output_list)
            print(run_output_list)
            print("Processed!")

    # Resolution for all runs
    print(f"Processing all runs")
    outfile = TFile(f'{reso_output_dir}resosp{config["suffix"]}_allruns.root', 'RECREATE')
    compute_reso(reso_files, config["cent_classes"], outfile)

    print(f"Resolution files saved in {reso_output_dir}/")
