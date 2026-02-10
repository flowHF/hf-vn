import os
from utils import logger
from ROOT import TFile, TH1, TDirectoryFile, TList
from itertools import combinations
import time
import uproot
import pandas as pd
import numpy as np
import sys

TH1.AddDirectory(False)

def load_aod_file(aod_file, has_sp_cent, num_workers=16, chunk_size=1_000_000, downsample_frac=None):
    """
    Load AOD file using uproot with parallel decompression and chunking.
    
    Args:
        aod_file (str): Path to the AOD file.
        has_sp_cent (bool): Whether the file contains SP centrality information.
        num_workers (int): Number of workers for parallel decompression.
        chunk_size (int): Number of entries to read per chunk.
        downsample_frac (float, optional): Fraction of data to downsample. If None, no downsampling is applied.
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    start_total = time.time()
    rng = np.random.default_rng(42)
    branches = ["fPt", "fM", "fMlScore0", "fMlScore1", "fScalarProd", "fCent"] \
               if has_sp_cent else ["fPt", "fM", "fMlScore0", "fMlScore1"]
    key = "ptcentersp" if has_sp_cent else "ptcenter"

    # Open with parallel decompression
    t0 = time.time()
    f = uproot.open(aod_file, num_workers=8)
    entries = f[key].num_entries
    logger(f"Opened file in {time.time()-t0:.2f}s using {num_workers} workers, total entries: {entries}", level="INFO")

    # Iterate in chunks
    dfs = []
    logger(f"Downsampling fraction {downsample_frac}", level="INFO")
    for df_chunk in f[key].iterate(filter_name=branches, step_size=chunk_size, library="np"):
        if downsample_frac is not None:
            n = len(df_chunk[branches[0]])
            mask = rng.random(n) < downsample_frac
            df_chunk = {k: v[mask] for k, v in df_chunk.items()}
        dfs.append(pd.DataFrame(df_chunk))

    # Concatenate
    t2 = time.time()
    df = pd.concat(dfs, ignore_index=True)
    logger(f"TOTAL time: {time.time()-start_total:.2f}s", level="INFO")
    return df

def load_root_files(inputPath, prefix: str, suffix='.root') -> list[str]:
    """
    Load root files from a specified directory that match the given prefix and suffix.

    Args:
        inputPath (str): Path to the directory containing the root files.
        prefix (str): Prefix of the files to be loaded.
        suffix (str): Suffix of the files to be loaded, default is '.root'.

    Returns:
        list: List of file paths that match the criteria.
    """
    if os.path.exists(inputPath):
        return sorted(
            [f'{os.path.join(inputPath, file)}'
             for file in os.listdir(inputPath) if file.startswith(prefix) and file.endswith(suffix)]
        )
    else:
        logger(f'No folder found in {inputPath}', level='ERROR')
        raise ValueError(f'No folder found in {inputPath}')

def load_reso_histos(an_res_file, wagon_id):
    '''
    Get list of histograms for SP resolution

    Input:
        - an_res_file:
            str, resolution file
        - wagon_id:
            str, wagon ID

    Output:
        - correct_histo_triplets:
            list of TH2D, list of TH2D objects with the SP product or EP cos(deltaphi) values vs centrality
        - correct_histo_labels:
            list of strings, list of detector labels
    '''
    infile_path = f'hf-task-flow-charm-hadrons'
    if wagon_id:
        infile_path = f'{infile_path}_id{wagon_id}'

    infile_path = f'{infile_path}/spReso'  # for sp method only
    prefix = 'hSpReso'  # for sp method only
    infile = TFile(an_res_file, 'READ')
    directory = infile.GetDirectory(infile_path)
    histos = [key.ReadObj() for key in directory.GetListOfKeys()]
    for histo in histos:
        histo.SetDirectory(0)
    pairs = [key.GetName() for key in directory.GetListOfKeys()]

    # generate triplets of pairs (AB, AC, BC)
    triplets = []
    detsA = ['FT0c', 'FT0a', 'FV0a', 'TPCpos', 'FT0m', 'TPCneg']
    triplets = list(combinations(pairs, 3))
    histo_triplets = list(combinations(histos, 3))
    correct_histo_triplets = []
    correct_histo_labels = []
    for i, triplet in enumerate(triplets):
        for detA in detsA:
            detB = triplet[0].replace(prefix, '').replace(detA, '')
            detC = triplet[1].replace(prefix, '').replace(detA, '')
            if (detA in triplet[0] and detA in triplet[1]) and \
               (detB in triplet[0] and detB in triplet[2]) and \
               (detC in triplet[1] and detC in triplet[2]):
                correct_histo_triplets.append(histo_triplets[i])
                correct_histo_labels.append((detA, detB, detC))

    return correct_histo_triplets, correct_histo_labels

def load_eff_histos(effFiles) -> tuple:
    """
    Load efficiency histograms from a file or a list of files.

    Args:
        effFiles (str or list[str]): Path or list of paths to the efficiency files.

    Returns:
        tuple: If input is a single file, returns histograms for prompt and FD efficiencies, and their fractions and corrected fractions.
               If input is a list, returns lists of those histograms for each file.
    """
    def _load_single_eff_histos(effFile: str):
        f = TFile.Open(effFile)
        hEffPrompt      = f.Get('hEffPrompt')
        hEffFD          = f.Get('hEffFD')
        hPromptFrac     = hEffPrompt.Clone('hPromptFrac')
        hFDFrac         = hEffFD.Clone('hFDFrac')
        hPromptFracCorr = hEffPrompt.Clone('hPromptFracCorr')
        hFDFracCorr     = hEffFD.Clone('hFDFracCorr')
        
        hPromptFrac.SetTitle(';#it{p}_{T} (GeV/#it{c}); #it{f}_{prompt}')
        hFDFrac.SetTitle(';#it{p}_{T} (GeV/#it{c}); #it{f}_{FD}')
        hPromptFracCorr.SetTitle(';#it{p}_{T} (GeV/#it{c}); corrected #it{f}_{prompt}')
        hFDFracCorr.SetTitle(';#it{p}_{T} (GeV/#it{c}); corrected #it{f}_{FD}')
        
        f.Close()
        return hEffPrompt, hEffFD, hPromptFrac, hFDFrac, hPromptFracCorr, hFDFracCorr

    if isinstance(effFiles, str):
        return _load_single_eff_histos(effFiles)
    elif isinstance(effFiles, list):
        hEffPrompts, hEffFDs, hPromptFracs, hFDFracs, hPromptFracCorrs, hFDFracCorrs = [], [], [], [], [], []
        for effFile in effFiles:
            hEffPrompt, hEffFD, hPromptFrac, hFDFrac, hPromptFracCorr, hFDFracCorr = _load_single_eff_histos(effFile)
            hEffPrompts.append(hEffPrompt)
            hEffFDs.append(hEffFD)
            hPromptFracs.append(hPromptFrac)
            hFDFracs.append(hFDFrac)
            hPromptFracCorrs.append(hPromptFracCorr)
            hFDFracCorrs.append(hFDFracCorr)
        return (hEffPrompts, hEffFDs, 
                hPromptFracs, hFDFracs,
                hPromptFracCorrs, hFDFracCorrs)
    else:
        raise TypeError("effFiles must be a str or a list of str")

def load_cutVar_histos(cutVarFracFile: str) -> tuple:
    """
    Load histograms from a cut variation file.

    Args:
        cutVarFracFile (str): Path to the cut variation file.

    Returns:
        tuple: Contains histograms for corrected yields and covariance matrices.
    """
    cutVarFracFile   = TFile.Open(cutVarFracFile)
    hCorrYieldPrompt = cutVarFracFile.Get('hCorrYieldPrompt')
    hCorrYieldFD     = cutVarFracFile.Get('hCorrYieldFD')
    hCovPromptPrompt = cutVarFracFile.Get('hCovPromptPrompt')
    hCovPromptFD     = cutVarFracFile.Get('hCovPromptFD')
    hCovFDFD         = cutVarFracFile.Get('hCovFDFD')
    cutVarFracFile.Close()
    
    return (hCorrYieldPrompt, hCorrYieldFD, 
            hCovPromptPrompt, hCovPromptFD, hCovFDFD)

def load_object_from_file(inFile, pathToObj):
    '''
    Function to extract an object inside a root file.
    Supports nested containers with the following Data Types:
     - TFile
     - TDirecotryFile
     - TList

    Parameters
    -----------
    inFile: TFile of the input file
    pathToObj: path of the object inside the root file

    Returns:
    -----------
    outObj: target root object
    '''

    pathToObj = os.path.normpath(pathToObj)
    pathElements = pathToObj.split(os.sep)
    outObj = inFile.Get(pathElements.pop(0))

    for _, containerName in enumerate(pathElements):
        if isinstance(outObj, (TFile, TDirectoryFile)):
            outObj = outObj.Get(containerName)
        elif isinstance(outObj, TList):
            outObj = outObj.FindObject(containerName)
        else:
            print(f'\033[31mError\033[0m: instance of {type(outObj)} not implemented. Exit!')
            sys.exit()

    return outObj
