import sys
import gc
import argparse
import yaml
import numpy as np
import pathlib as PATH
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from ROOT import TFile, TH1D
from alive_progress import alive_bar
import os
os.environ['ZFIT_DISABLE_TF_WARNINGS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../flareflyfitter/")
from raw_yield_fitter import RawYieldFitter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
from PyPDF2 import PdfMerger, PdfReader

def build_raw_yield_fitter(task, fitConfig):
    task_id = task['taskID']
    task_name = task['taskName']
    pt_min = task['ptMin']
    pt_max = task['ptMax']
    pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
    i_pt = task['iPtCand'] if 'iPtCand' in task else task['iPt']
    mass_fit_range = fitConfig['MassFitRanges'][i_pt]
    minimizer = fitConfig.get('minimizer', 'flarefly')
    
    sgn_func = {
        'func': fitConfig.get("SgnFunc")[i_pt] if isinstance(fitConfig.get("SgnFunc"), list) else fitConfig.get("SgnFunc"),
        'label': 'sgn',
        'particle': task['Dmeson']
        }
    bkg_func = {
        'func': fitConfig.get("BkgFunc")[i_pt] if isinstance(fitConfig.get("BkgFunc"), list) else fitConfig.get("BkgFunc"),
        'label': 'Comb_bkg'
    }

    # TODO: correlated background
    # if fitConfig.get('corr_bkgs')
    #     corr_bkg_funcs = 

    rebin = fitConfig.get("Rebin")[i_pt] if isinstance(fitConfig.get("Rebin"), list) else fitConfig.get("Rebin", 1)
    fit_pars = fitConfig.get('InitPars', None)

    fitter = RawYieldFitter(task['Dmeson'], pt_min, pt_max, pt_str, minimizer)
    fitter.set_fit_range(mass_fit_range[0], mass_fit_range[1])
    fitter.add_sgn_func(sgn_func["func"], sgn_func["label"], sgn_func["particle"])
    fitter.add_bkg_func(bkg_func["func"], bkg_func["label"])
    fitter.set_name(f'{task_id}_{task_name}')
    fitter.set_rebin(rebin)
    if fit_pars:
        fitter.set_fit_pars(fit_pars, pt_min, pt_max)

    return fitter

def fit_task(task, fitConfig, inFilePath, doMassFit=False):
    try:
        inFile = TFile.Open(inFilePath)
        histo_path = task['histoPath']
        raw_histo = inFile.Get(histo_path)
        if not raw_histo:
            raise ValueError(f"Histogram {histo_path} not found in file {inFilePath}")
        raw_histo.SetDirectory(0)
        inFile.Close()

        if doMassFit:
            mass_histo = raw_histo.ProjectionX(f"{raw_histo.GetName()}_MassProj", 
                                               raw_histo.GetYaxis().FindBin(task['ptMin']), 
                                               raw_histo.GetYaxis().FindBin(task['ptMax']))
            histo = mass_histo
        else:
            histo = raw_histo

        fitter = build_raw_yield_fitter(task, fitConfig)
        fitter.set_data_to_fit_hist(histo)
        fitter.setup()
        fitter.fit()

        if fitConfig.get("minimizer") == "RooFit":
            fitter.plot_fit(logy=False, path=str(task['pdfPath']), show_extra_info=False)
        else:
            fig = fitter.plot_fit(False, True, loc=["lower left", "upper left"])
            fig.savefig(task['pdfPath'], dpi=300, bbox_inches="tight")
            plt.close(fig)
            fig_residuals = fitter.plot_raw_residuals()
            residuals_path = task['pdfPath'].with_name(task['pdfPath'].stem + "_Residuals.pdf")
            fig_residuals.savefig(residuals_path, dpi=300, bbox_inches="tight")
            plt.close(fig_residuals)

        fit_info, _, _, _, _ = fitter.get_fit_info()
        result = {
            'taskID': task['taskID'],
            'iPtCand': task['iPtCand'],
            'iPtHad': task['iPtHad'],
            'ry': fit_info['sgn']['ry'],
            'ry_unc': fit_info['sgn']['ry_unc'],
            'pdfPath': task['pdfPath'],
            'pdfPathResiduals': residuals_path if fitConfig.get("minimizer") != "RooFit" else None
        }

        if doMassFit:
            result.update({
                'ptMin': task['ptMin'],
                'ptMax': task['ptMax']
            })
        else:
            result.update({
                'iDeltaPhi': task['iDeltaPhi'],
            })

        return result

    finally:
        plt.close('all')
        if 'fitter' in locals():
            del fitter
        if 'mass_histo' in locals():
            del mass_histo
        if 'histo' in locals():
            del histo
        gc.collect()

def combine_pdfs(pdf_paths, suffix=""):
    merger = PdfMerger()
    sum_pdf = pdf_paths[0].parent / f"Summary_Fits{suffix}.pdf"
    sum_pdf_combined = pdf_paths[0].parent / f"Summary_Fits_Combined{suffix}.pdf"
    for pdf_path in pdf_paths:
        merger.append(str(pdf_path))
    merger.write(str(sum_pdf))
    merger.close()

    for iTotal in range(int(np.ceil(len(pdf_paths)/15))):
        tot, axes = plt.subplots(3, 5, figsize=(20, 12))
        plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.02, right=0.98, top=0.98, bottom=0.02)
        axes = axes.flatten()
        for i, pdf_path in enumerate(pdf_paths[iTotal*15:(iTotal+1)*15]):
            doc = fitz.open(str(pdf_path))
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            axes[i].imshow(img)
            axes[i].axis('off')

            del pix
            del img
            doc.close()
        
        for j in range(i+1, 15):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(sum_pdf.parent / f"Summary_Fits_Combined_{iTotal}{suffix}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(sum_pdf.parent / f"Summary_Fits_Combined_{iTotal}{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(tot)

        del tot
        del axes
        gc.collect()

    for pdf_path in pdf_paths:
        os.remove(pdf_path)
        if pdf_path.exists():
            print(f"Warning: Failed to remove {pdf_path}")

def interface_raw_yield_fitter(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    method = config.get("method", "")
    if method != "DeltaPhiBinning":
        raise ValueError(f"Unsupported method: {method}. Only 'DeltaPhiBinning' is supported in this interface.")
    nWorkers = config.get("nWorkers", int(os.cpu_count()/0.6))
    minimizer = config["fitConfig"].get("minimizer", "flarefly")

    Dmeson = config["Dmeson"]
    ptBinsCand = config["ptBinsCand"]
    ptBinsHad = config["ptBinsHad"]
    nPtBinsCand = len(ptBinsCand) - 1
    nDeltaPhiBins = config.get("nDeltaPhiBins", 32)
    deltaPhiBins = list(np.linspace(-1.5707963705062866, 4.71238911151886, nDeltaPhiBins+1))  # default 64 bins from -pi/2 to 3pi/2

    outdir = PATH.Path(config["outdir"])
    suffix = config.get("suffix", "")
    outFilePath = outdir / f"CorrelExtract_{suffix}" / "AssociatedPairsYields"
    outFilePath.mkdir(parents=True, exist_ok=True)

    inFilePath = str(outdir / f"CorrelExtract_{suffix}" / "CorrelationsResults" / "CorrelationsResults.root")

    # build tasks
    tasks = []
    mass_tasks = []
    hPairsYields_vs_DeltaPhi = {}
    hTriggerYields = {}
    print("Building tasks...")
    for iPtCand, (ptMin, ptMax) in enumerate(zip(ptBinsCand[:-1], ptBinsCand[1:])):
        strPtCand = f"PtCandBin_{int(ptMin*10)}_{int(ptMax*10)}"

        for iPtHad, (ptHadMin, ptHadMax) in enumerate(zip(ptBinsHad[:-1], ptBinsHad[1:])):
            strPtHad = f"PtHadBin_{int(ptHadMin*10)}_{int(ptHadMax*10)}"
            hTemp = TH1D(
                f"hPairsYields_vs_DeltaPhi_{iPtCand}_{iPtHad}", 
                "Associated Pair Raw Yields vs #Delta#phi;#Delta#phi (rad); Associated pairs raw yield", 
                len(deltaPhiBins)-1, 
                np.array(deltaPhiBins, dtype='double'))
            hPairsYields_vs_DeltaPhi[(iPtCand, iPtHad)] = hTemp
            hTriggerYields[iPtCand, iPtHad] = 0
            (outFilePath / strPtCand / strPtHad).mkdir(parents=True, exist_ok=True)
            
            for iDeltaPhi, (deltaPhiMin, deltaPhiMax) in enumerate(zip(deltaPhiBins[:-1], deltaPhiBins[1:])):
                strDeltaPhi = f"DeltaPhiBin_{int(deltaPhiMin*1000)}_{int(deltaPhiMax*1000)}"
                # build task
                task_id = len(tasks)
                task = {
                    "minimizer": minimizer,
                    "taskID": task_id, 
                    "taskName": f"CandPt_{int(ptMin*10)}_{int(ptMax*10)}_HadPt_{int(ptHadMin*10)}_{int(ptHadMax*10)}_dPhi_{int(deltaPhiMin*1000)}_{int(deltaPhiMax*1000)}",
                    "iPtCand": iPtCand, "ptMin": ptMin, "ptMax": ptMax,
                    "iPtHad": iPtHad, "ptHadMin": ptHadMin, "ptHadMax": ptHadMax,
                    "iDeltaPhi": iDeltaPhi, "deltaPhiMin": deltaPhiMin, "deltaPhiMax": deltaPhiMax,
                    "Dmeson": Dmeson,
                    "pdfPath": outFilePath / strPtCand / strPtHad / f"TempFitResult_{strDeltaPhi}.pdf",
                    "histoPath": f"{strPtCand}/{strPtHad}/{strDeltaPhi}/hCorrectedPairsMass"
                }
                tasks.append(task)

            mass_task = {
                "minimizer": minimizer,
                "taskID": len(mass_tasks),
                "taskName": f"MassFit_PtCand_{int(ptMin*10)}_{int(ptMax*10)}",
                "iPtCand": iPtCand, "ptMin": ptMin, "ptMax": ptMax,
                "iPtHad": iPtHad, "ptHadMin": ptHadMin, "ptHadMax": ptHadMax,
                "Dmeson": Dmeson,
                "pdfPath": outFilePath / strPtCand / f"TempMassFitResult_{strPtCand}.pdf",
                "histoPath": f"hMassVsPt"
            }
            mass_tasks.append(mass_task)

    # build and execute fitting tasks in parallel
    pdfs = defaultdict(dict)
    pdfs_residuals = defaultdict(dict)
    with alive_bar(len(tasks), title="Fitting tasks") as bar:
        with ProcessPoolExecutor(max_workers=nWorkers) as executor:
            futures = {executor.submit(fit_task, task, config["fitConfig"], inFilePath): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                key_hPairs = (result['iPtCand'], result['iPtHad'])
                bin_idx = result['iDeltaPhi'] + 1
                hPairsYields_vs_DeltaPhi[key_hPairs].SetBinContent(bin_idx, result['ry'])
                hPairsYields_vs_DeltaPhi[key_hPairs].SetBinError(bin_idx, result['ry_unc'])
                pdfs[key_hPairs][result['iDeltaPhi']] = result['pdfPath']
                pdfs_residuals[key_hPairs][result['iDeltaPhi']] = result['pdfPathResiduals']
                bar.text = f"Completed task {result['taskID']}: PtCandBin_{result['iPtCand']}, PtHadBin_{result['iPtHad']}, DeltaPhiBin_{result['iDeltaPhi']}"
                del future
                bar()
            futures.clear()
            gc.collect()

    if minimizer == "flarefly":
    # combine PDFs for each (PtCand, PtHad) bin and their residuals in parallel
        with alive_bar(len(pdfs)+len(pdfs_residuals), title="Combining PDFs") as bar:
            with ProcessPoolExecutor(max_workers=min(len(pdfs)+len(pdfs_residuals), nWorkers)) as combiner:
                futures = []
                for key in pdfs.keys():
                    sorted_pdfs = [pdfs[key][iDeltaPhi] for iDeltaPhi in sorted(pdfs[key].keys())]
                    futures.append(combiner.submit(combine_pdfs, sorted_pdfs))
                for key in pdfs_residuals.keys():
                    sorted_pdfs = [pdfs_residuals[key][iDeltaPhi] for iDeltaPhi in sorted(pdfs_residuals[key].keys())]
                    futures.append(combiner.submit(combine_pdfs, sorted_pdfs, suffix="_residuals"))
                for future in as_completed(futures):
                    future.result()
                    del future
                    bar()
                futures.clear()
                gc.collect()

    ry_trigger = {}
    # build task and perform the fit for inv mass distribution of trigger candidates
    inFileMassPath = str(outdir / "InvMass/InvMassVsPt.root")
    with alive_bar(len(mass_tasks), title="Fitting mass distribution tasks") as bar:
        with ProcessPoolExecutor(max_workers=min(len(mass_tasks), nWorkers)) as executor:
            futures = {executor.submit(fit_task, task, config["fitConfig"], inFileMassPath, doMassFit=True): task for task in mass_tasks}
            for future in as_completed(futures):
                result = future.result()
                ry_trigger[(result['iPtCand'], result['iPtHad'])] = (result['ry'], result['ry_unc'])
                del future
                bar()
            futures.clear()
            gc.collect()

    # collect results into ROOT file
    outROOT = TFile.Open(str(outFilePath / f"PairYieldsVsPhi.root"), "RECREATE")
    for (iPtCand, iPtHad), histo in hPairsYields_vs_DeltaPhi.items():
        subdir_pt = f"PtCandBin_{int(ptBinsCand[iPtCand]*10)}_{int(ptBinsCand[iPtCand+1]*10)}"
        subdir_had = f"PtHadBin_{int(ptBinsHad[iPtHad]*10)}_{int(ptBinsHad[iPtHad+1]*10)}"
        outROOT.mkdir(subdir_pt + "/" + subdir_had)
        outROOT.cd(subdir_pt + "/" + subdir_had)
        # histo.Scale(1.0 / ry_trigger[(iPtCand, iPtHad)] if ry_trigger[(iPtCand, iPtHad)] != 0 else 1.0)
        histo.Write('hPairsYields_vs_DeltaPhi')
        # save ry_trigger as a 1-bin histogram in the same directory
        ry, ry_unc = ry_trigger.get((iPtCand, iPtHad), (0.0, 0.0))
        hRy = TH1D("ry_trigger", "ry_trigger", 1, 0., 1.)
        hRy.SetBinContent(1, ry)
        hRy.SetBinError(1, ry_unc)
        hRy.Write()
    outROOT.Close()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Raw Yield Fitter Interface")
    parser.add_argument('config', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    config_path = args.config
    interface_raw_yield_fitter(config_path)
