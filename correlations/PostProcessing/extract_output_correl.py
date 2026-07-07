import shutil

import ROOT
from ROOT import gROOT, gStyle, TFile, TH1, TH2, TCanvas, TDirectory, kRainbow, gDirectory, gPad
import json
import os
import sys
import yaml
import argparse
from ROOT import gSystem
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../../', 'utils'))
gSystem.CompileMacro(f"{script_dir}/DhCorrelationExtraction.cxx", "kO")
from ROOT import DhCorrelationExtraction as CorrelExtractor
from alive_progress import alive_bar
from concurrent.futures import ProcessPoolExecutor, as_completed
from gc import collect
import numpy as np
import time
import threading
import psutil
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../../', 'utils'))
from utils import check_dir, logger

gROOT.SetBatch(True)
gStyle.SetOptStat(0)
gStyle.SetPalette(kRainbow)

def SetCanvasStyle():
    gStyle.SetPadLeftMargin(0.15)
    gStyle.SetPadBottomMargin(0.15)
    gStyle.SetFrameLineWidth(2)
    gStyle.SetLineWidth(2)
    gStyle.SetCanvasDefH(1126)
    gStyle.SetCanvasDefW(1840)

def get_pt_dependent_param(param, nPtBins, isList=False):
    ''' handle parameters that can be either a single value or a list of values for each pt bin.
    If isList is True, the parameter is expected to be a single value to be repeated for all pt bins.
    If isList is False, the parameter can be a list of values or a single value to be repeated.
    '''
    if isList:
        parameter = [param] * nPtBins
    else:
        if isinstance(param, list):
            if len(param) < nPtBins:
                logger(f"Length of parameter list {len(param)} does not match number of pt bins {nPtBins}", level="FATAL")
                sys.exit(1)
            parameter = param[:nPtBins]
        else:
            parameter = [param] * nPtBins
    return parameter

def process_correlation_task(tempExtractor, task):
    extractor = CorrelExtractor.CreateCopy(tempExtractor)
    ROOT.SetOwnership(extractor, True)

    results = {}
    results["task"] = task
    extractor.SetRebin2DcorrelHisto(task["rebinDeltaEta"], task["rebinDeltaPhi"])
    extractor.SetCandAndHadBins((task["ptMin"], task["ptMax"]), (task["ptHadMin"], task["ptHadMax"]))
    extractor.SetInvMassBins((task["invMassMin"], task["invMassMax"]))
    if task["method"] == "DeltaPhiBinning":
        extractor.SetMethod(CorrelExtractor.kDeltaPhiBinning)
        deltaPhiMin = task["deltaPhiMin"]
        deltaPhiMax = task["deltaPhiMax"]
        extractor.SetDeltaPhiBins((deltaPhiMin, deltaPhiMax))
    extractor.ExtractCorrelations()

    results['hCorrectedCorrel']           = extractor.GetCorrectedCorrel()
    results['hNormalizedCorrectedCorrel'] = extractor.GetNormalizedCorrectedCorrel()
    results['hCorrectedPairsMass']        = extractor.GetCorrectedPairsMass()
    results['hCorrectionRatio']           = extractor.GetCorrectionRatio()

    results['hCorrel_SE_2D']              = extractor.GetRawCorrel_SE_2D()
    results['hCorrel_ME_2D']              = extractor.GetRawCorrel_ME_2D()
    results['hCorrectedCorrel_2D']        = extractor.GetCorrectedCorrel_2D()
    results['hNormalizedCorrel_ME_2D']    = extractor.GetNormalizedCorrel_ME_2D()
    results['hOriginalCorrel_SE_2D']      = extractor.GetOriginalCorrel_SE_2D()
    results['hOriginalCorrel_ME_2D']      = extractor.GetOriginalCorrel_ME_2D()
    results['hOriginalMassVsDeltaEta_2D'] = extractor.GetOriginalMassVsDeltaEta_2D()

    results['PoolVec_OriginalCorrel_SE_2D']      = extractor.GetPoolVec_OriginalCorrel_SE_2D()
    results['PoolVec_OriginalCorrel_ME_2D']      = extractor.GetPoolVec_OriginalCorrel_ME_2D()
    results['PoolVec_RawCorrel_SE_2D']           = extractor.GetPoolVec_RawCorrel_SE_2D()
    results['PoolVec_RawCorrel_ME_2D']           = extractor.GetPoolVec_RawCorrel_ME_2D()
    results['PoolVec_NormalizedCorrel_ME_2D']    = extractor.GetPoolVec_NormalizedCorrel_ME_2D()
    results['PoolVec_CorrectedCorrel_2D']        = extractor.GetPoolVec_CorrectedCorrel_2D()
    results['PoolVec_OriginalMassVsDeltaEta_2D'] = extractor.GetPoolVec_OriginalMassVsDeltaEta_2D()
    results['PoolVec_RawMassVsDeltaEta_2D']      = extractor.GetPoolVec_RawMassVsDeltaEta_2D()
    results['PoolVec_CorrectedMass']             = extractor.GetPoolVec_CorrectedMass()
    results['PoolVec_CorrectionRatio']           = extractor.GetPoolVec_CorrectionRatio()

    del extractor
    collect()

    return results

def ExtractOutputCorrel(cfgFile):
    SetCanvasStyle()
    with open(cfgFile, 'r') as f:
        config = yaml.safe_load(f)

    # Optional settings
    debug = config.get("debug", 0)

    # Define the default extractor
    if config.get('sparseSE') or config.get('sparseME') or config.get('sparseMass'):
        logger("Custom sparse names provided in config, will use them instead of automatic settings", level="INFO")
        tempExtractor = CorrelExtractor.CreateCustom(config['sparseSE'], config['sparseME'], config['sparseMass'])
    else:
        tempExtractor = CorrelExtractor.CreateDefault() # aotumatic settings for directories and sparse names
    tempExtractor.SetInputFilenameSE(config["pathFileSE"])
    tempExtractor.SetInputFilenameME(config["pathFileME"])
    tempExtractor.SetInputFilenameMass(config["pathFileSE"]) # mass sparse is in the same file as SE

    # Set D-meson specie
    Dmeson = config["Dmeson"]
    if Dmeson == "D0" or Dmeson == "Dzero":
        tempExtractor.SetDmesonSpecie(0)
    elif Dmeson == "Dplus":
        tempExtractor.SetDmesonSpecie(1)
    elif Dmeson == "Ds":
        tempExtractor.SetDmesonSpecie(2)
    else:
        logger(f"Unknown D meson specie {Dmeson}", level="FATAL")
    
    tempExtractor.SetPoolSettings(config.get("nPools", 10), config.get("doPoolByPool", False))

    method = config.get("method", "MassBinning")
    if method == "MassBinning":
        tempExtractor.SetMethod(CorrelExtractor.kMassBinning)
    elif method == "DeltaPhiBinning":
        tempExtractor.SetMethod(CorrelExtractor.kDeltaPhiBinning)
        tempExtractor.SetDeltaEtaIntegrated(config.get("deltaEtaIntegrated", True))
        if any(len (invMassBins[i]) > 2 for i in range(len(invMassBins))):
            logger(f"Using DeltaPhiBinning method, but invMassBins has more than 2 edges, replace with 2 edges for min and max", level="WARNING")
            invMassBins = [[invMassBins[i][0], invMassBins[i][-1]] for i in range(len(config["invMassBins"]))]
    else:
        logger(f"Unknown extraction method {method}", level="FATAL")

    deltaEtaBins = config["deltaEtaBins"]
    if deltaEtaBins[0][1] > deltaEtaBins[1][0]:
        logger(f"deltaEta bins for correlation extraction overlap: {deltaEtaBins}", level="FATAL")
    tempExtractor.SetBinDeltaEtaLeft(deltaEtaBins[0][0], deltaEtaBins[0][1])
    tempExtractor.SetBinDeltaEtaRight(deltaEtaBins[1][0], deltaEtaBins[1][1])
    tempExtractor.SetDebugLevel(debug)

    outdir = config["outdir"]
    suffix = config["suffix"]
    outdirFull = os.path.join(outdir, "maps")
    logger(f"Output directory for correlation extraction: {outdirFull}", level="INFO")
    if not os.path.exists(outdirFull):
        os.makedirs(outdirFull)

    # # copy config file
    # shutil.copy(cfgFile, os.path.join(outdirFull, f"config_{suffix}.yaml"))
    outdirMass = os.path.join(outdir, "trigger_yields")
    if not os.path.exists(outdirMass):
        os.makedirs(outdirMass)

    # Produce Mass vs Pt histogram for trigger candidates
    tempExtractor.ProjMassVsPt()
    hMassVsPtTriggers = tempExtractor.GetMassVsPtHist2D()
    ## Should be 300 x 9
    print(f"Mass vs Pt histogram for trigger candidates has {hMassVsPtTriggers.GetNbinsX()} x {hMassVsPtTriggers.GetNbinsY()} bins")
    outMassVsPtFile = TFile(os.path.join(outdirMass, f"InvMassVsPt.root"), "RECREATE")
    hMassVsPtTriggers.Write()
    outMassVsPtFile.Close()
    print(f"Saved Mass vs Pt histogram for trigger candidates to {outdirMass}/InvMassVsPt.root")

    logger(f"Extracting correlations for each pt bin, associated hadron pt bin, and inv. mass bin...", level="INFO")
    # extract for all pt bins, associated hadron pt bins, and inv. mass bins
    tasks = []
    # Define binnings
    ptBinsCand = config["ptBinsCand"]
    ptBinsHad = config["ptBinsHad"]
    invMassBins = config["invMassBins"]
    deltaPhiBins = list(np.linspace(-1.5707963705062866, 4.71238911151886, config["nDeltaPhiBins"]+1))  # default 64 bins from -pi/2 to 3pi/2
    rebinsDeltaEta = get_pt_dependent_param(config.get("rebinDeltaEta", 1), len(ptBinsCand)-1, isList=False)
    rebinsDeltaPhi = get_pt_dependent_param(config.get("rebinDeltaPhi", 1), len(ptBinsCand)-1, isList=False)
    for iPtCand, (ptMin, ptMax, rebinDeltaEta, rebinDeltaPhi) in enumerate(zip(ptBinsCand[:-1], ptBinsCand[1:], rebinsDeltaEta, rebinsDeltaPhi)):
        invMassBinsPtCand = invMassBins[iPtCand]
        for iPtHad, (ptHadMin, ptHadMax) in enumerate(zip(ptBinsHad[:-1], ptBinsHad[1:])):
            for iMass, (invMassMin, invMassMax) in enumerate(zip(invMassBinsPtCand[:-1], invMassBinsPtCand[1:])):
                mass_binning_task = {
                    "iPtCand": iPtCand, "ptMin": ptMin, "ptMax": ptMax, "rebinDeltaEta": rebinDeltaEta,
                    "rebinDeltaPhi": rebinDeltaPhi, "iPtHad": iPtHad, "ptHadMin": ptHadMin, "ptHadMax": ptHadMax, 
                    "iMass": iMass, "invMassMin": invMassMin, "invMassMax": invMassMax, "method": method,
                }

                if method == "MassBinning":
                    tasks.append(mass_binning_task)

                elif method == "DeltaPhiBinning":
                    for iDeltaPhi, (deltaPhiMin, deltaPhiMax) in enumerate(zip(deltaPhiBins[:-1], deltaPhiBins[1:])):
                        tasks.append({**mass_binning_task, 
                                      "iDeltaPhi": iDeltaPhi, "deltaPhiMin": deltaPhiMin, "deltaPhiMax": deltaPhiMax})

    logger(f"[INFO] Total number of correlation extraction tasks: {len(tasks)}", level="INFO")
    all_results = []
    nWorkers = config.get("nWorkers", int(os.cpu_count()/1.6))
    with alive_bar(len(tasks), title="Processing correlation tasks") as bar:
        with ProcessPoolExecutor(max_workers=nWorkers) as executor:
            tasks_to_execute = {executor.submit(process_correlation_task, tempExtractor, task): task for task in tasks}
            for i_task in as_completed(tasks_to_execute):
                task = tasks_to_execute[i_task]
                try:
                    result = i_task.result()
                    all_results.append(result)
                except Exception as exc:
                    logger(f"Task {task} generated an exception: {exc}", level="ERROR")
                bar()

    if method == "MassBinning":
        all_results.sort(key=lambda x: (x['task']['iMass'], x['task']['iPtHad'], x['task']['iPtCand']))
    elif method == "DeltaPhiBinning":
        all_results.sort(key=lambda x: (x['task']['iMass'], x['task']['iPtHad'], x['task']['iPtCand'], x['task']['iDeltaPhi']))

    # Save outputs
    outHistFile = TFile(os.path.join(outdirFull, "CorrelationsResults.root"), "RECREATE")
    outFiles = [outHistFile] # can be extended in the future if we want to save different histograms in different files
    logger(f"Saving output histograms to {outdirFull}/{outHistFile.GetName()}", level="INFO")

    for results in all_results:
        subDirPtCand   = f"PtCand_{int(results['task']['ptMin']*10):.0f}_{int(results['task']['ptMax']*10):.0f}"
        subDirPtHad    = f"PtHad_{int(results['task']['ptHadMin']*10):.0f}_{int(results['task']['ptHadMax']*10):.0f}"
        subDirInvMass  = f"InvMassBin_{int(results['task']['invMassMin']*1000):.0f}_{int(results['task']['invMassMax']*1000):.0f}"
        subDirDeltaPhi = f"DeltaPhiBin_{int(results['task']['deltaPhiMin']*1000):.0f}_{int(results['task']['deltaPhiMax']*1000):.0f}" if results['task']['method'] == "DeltaPhiBinning" else ""

        for file in outFiles:
            if not file.GetDirectory(subDirPtCand):
                file.mkdir(subDirPtCand)
            file.cd(subDirPtCand)
            if not gDirectory.GetDirectory(subDirPtHad):
                gDirectory.mkdir(subDirPtHad)
            gDirectory.cd(subDirPtHad)
            if not gDirectory.GetDirectory(subDirInvMass if method == "MassBinning" else subDirDeltaPhi):
                gDirectory.mkdir(subDirInvMass if method == "MassBinning" else subDirDeltaPhi)

        subDir = os.path.join(subDirPtCand, subDirPtHad, subDirInvMass if method == "MassBinning" else subDirDeltaPhi)

        outHistFile.cd(subDir)
        cMainName = f"cMainSummary_{subDir.replace('/', '_')}"
        cMain = TCanvas(cMainName, "Correlation Summary", 1800, 800) if method != "DeltaPhiBinning" else TCanvas(cMainName, "Correlation Summary", 1800, 1800)

        if method == "DeltaPhiBinning" and results.get('hCorrectedPairsMass'):
            cMain.Divide(2, 2)
            has_mass_plot = True
        else:
            cMain.Divide(3, 1)
            has_mass_plot = False

        cMain.cd(1)
        if results.get('hCorrel_SE_2D'):
            results['hCorrel_SE_2D'].SetTitle("Raw SE; #Delta#eta; #Delta#phi")
            results['hCorrel_SE_2D'].Draw("SURF 1")
            gPad.SetTheta(30)
            gPad.SetPhi(40)

        cMain.cd(2)
        if results.get('hNormalizedCorrel_ME_2D'):
            results['hNormalizedCorrel_ME_2D'].SetTitle("Normalized ME; #Delta#eta; #Delta#phi")
            results['hNormalizedCorrel_ME_2D'].Draw("SURF 1")
            gPad.SetTheta(30)
            gPad.SetPhi(40)

        cMain.cd(3)
        if results.get('hCorrectedCorrel_2D'):
            results['hCorrectedCorrel_2D'].SetTitle("Corrected Correlation (Gap Applied)")
            results['hCorrectedCorrel_2D'].Draw("SURF 1")

        if has_mass_plot:
            cMain.cd(4)
            results['hCorrectedPairsMass'].SetTitle("Corrected Pairs Mass")
            results['hCorrectedPairsMass'].Draw("hist e")

        cMain.Write()

        if results.get('hCorrectedCorrel'):
            results['hCorrectedCorrel'].Write()
        if results.get('hNormalizedCorrectedCorrel'):
            results['hNormalizedCorrectedCorrel'].Write()
        if results.get('hCorrectedPairsMass'):
            results['hCorrectedPairsMass'].Write()
        if results.get('hCorrectionRatio'):
            results['hCorrectionRatio'].Write()

        if results.get('hCorrel_SE_2D'):
            results['hCorrel_SE_2D'].Write()
        if results.get('hCorrel_ME_2D'):
            results['hCorrel_ME_2D'].Write()
        if results.get('hCorrectedCorrel_2D'):
            results['hCorrectedCorrel_2D'].Write()
        if results.get('hNormalizedCorrel_ME_2D'):
            results['hNormalizedCorrel_ME_2D'].Write()

    outHistFile.Close()

def monitor_memory(pid, stop_event, time_log, mem_log):
    main_process = psutil.Process(pid)
    start_time = time.time()
    
    while not stop_event.is_set():
        current_time = time.time() - start_time
        time_log.append(current_time)
        
        total_mem = main_process.memory_info().rss
        
        for child in main_process.children(recursive=True):
            try:
                total_mem += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
                
        mem_log.append(total_mem / (1024 * 1024 * 1024))
        
        time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract output correlations")
    parser.add_argument("config", nargs="?", default="config_CorrAnalysis_v2_010_negDeta.yaml", help="Configuration file")
    args = parser.parse_args()

    print(f"Using configuration file: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    suffix = config["suffix"]
    outdir = config["outdir"]
    monitor_performance = False # set to True to enable memory and time monitoring

    check_dir(os.path.join(outdir, "trigger_yields"))

    if monitor_performance:
        time_log = []
        mem_log = []
        stop_event = threading.Event()

        monitor_thread = threading.Thread(
            target=monitor_memory, 
            args=(os.getpid(), stop_event, time_log, mem_log)
        )
        start_time = time.time()
        monitor_thread.start()

    # Call main function
    ExtractOutputCorrel(args.config)

    if monitor_performance:
        stop_event.set()
        monitor_thread.join()
        end_time = time.time()

        total_time = end_time - start_time
        logger(f"Execution time: {total_time:.2f} seconds", level="INFO")

        plt.figure(figsize=(10, 6))
        plt.plot(time_log, mem_log, color='b', linestyle='-', linewidth=2, label='Correlation Extraction')
        plt.title("Memory Consumption Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (GB)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_path = os.path.join(outdir, "maps", "Memory_Usage.png")
        plt.savefig(plot_path)
