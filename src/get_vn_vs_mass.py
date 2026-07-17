'''
Script for extracting v_n vs invariant mass for D mesons
run: python get_vn_vs_mass.py fitConfigFileName.yml inFileName.root [--batch]
'''
import argparse
import numpy as np
import yaml
import os
import itertools
import ctypes
import subprocess
import sys
from ROOT import TLatex, TFile, TCanvas, TLegend, TH1D, TH1F, TGraphAsymmErrors # pylint: disable=import-error,no-name-in-module
from ROOT import gROOT, gPad, gInterpreter, kBlack, kRed, kBlue, kMagenta, kAzure, kOrange, kGreen, kFullCircle, kFullSquare, kOpenCircle # pylint: disable=import-error,no-name-in-module
script_dir = os.path.dirname(os.path.realpath(__file__))
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError  # Only show errors and above
# Only load if not already loaded
script_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, "../invmassfitter"))
lib = os.path.join(src_dir, "libvnfitter.so")

# Compile only if the shared library is missing
from ROOT import gSystem
if not os.path.exists(lib):
    print("libvnfitter.so not found — compiling...")
    subprocess.check_call([
        "rootcling", "-f",
        os.path.join(src_dir, "vnfitter_dict.cxx"),
        "-c",
        os.path.join(src_dir, "VnVsMassFitter.h"),
        os.path.join(src_dir, "LinkDefVnFitter.h"),
    ])

    cmd = [
        "g++", "-shared", "-fPIC",
        *subprocess.check_output(["root-config", "--cflags", "--libs"]).decode().split(),
        os.path.join(src_dir, "VnVsMassFitter.cxx"),
        os.path.join(src_dir, "vnfitter_dict.cxx"),
        "-o", lib,
    ]

    subprocess.check_call(cmd)
    print("Compilation done!")

# Load the library once
if not hasattr(gSystem, "_vnfitter_loaded"):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # gSystem.CompileMacro(f"{script_dir}/../invmassfitter/VnVsMassFitter.cxx", "kO")
    ret = gSystem.Load(lib)
    if ret != 0:
        raise RuntimeError(f"Failed to load {lib}")
    gSystem._vnfitter_loaded = True

from ROOT import VnVsMassFitter
os.sys.path.append(os.path.join(script_dir, '..', 'utils'))
from StyleFormatter import SetGlobalStyle, SetObjectStyle
from fit_utils import RebinHisto
from utils import logger, get_centrality_bins, get_vnfitter_results, get_refl_histo, get_particle_info, make_dir_root_file
from correlated_bkgs import get_corr_bkg, get_rebinned_mass_sel_histo, get_corr_bkgs_from_reference

def get_fit_config(cfg, cfg_cutset, config_entry, quiet):
    cfg_fit = cfg['v2extraction']
    if not config_entry in cfg_fit:
        if not quiet:
            logger(f'Config entry {config_entry} not found in fit configuration.', level='WARNING')
        return [None] * (len(cfg['ptbins'])-1)

    if isinstance(cfg_fit[config_entry], list):
        if isinstance(cfg_fit[config_entry][0], list):
            # List indicates pt- and cutset-dependent settings
            if isinstance(cfg_fit[config_entry][cfg_cutset['icutset']], list):
                setting = cfg_fit[config_entry][cfg_cutset['icutset']]
            else:
                setting = [cfg_fit[config_entry][cfg_cutset['icutset']]] * (len(cfg['ptbins'])-1)
        else:
            # List indicates pt-dependent settings
            setting = cfg_fit[config_entry]
    else:
        setting = [cfg_fit[config_entry]] * (len(cfg['ptbins'])-1)
    return setting

def get_vn_vs_mass(fitConfigFileName, cutsetFileName, inFileName, batch, isMultitrial):
    #______________________________________________________
    # Read configuration file
    with open(fitConfigFileName, 'r', encoding='utf8') as ymlfitConfigFile:
        config = yaml.load(ymlfitConfigFile, yaml.FullLoader)

    with open(cutsetFileName, 'r', encoding='utf8') as ymlcutsetFile:
        cfg_cutset = yaml.load(ymlcutsetFile, yaml.FullLoader)

    # Set outfile name
    outFileName = os.path.join(os.path.dirname(os.path.dirname(inFileName)),
                               'raw_yields',
                               os.path.basename(inFileName).replace('proj', 'raw_yields').replace('.root', ''))

    gROOT.SetBatch(batch)
    SetGlobalStyle(padleftmargin=0.14, padbottommargin=0.12, padtopmargin=0.12, opttitle=1)
    _, centMinMax = get_centrality_bins(config["centrality"])

    # Read global configuration
    ptmins = config['ptbins'][:-1]
    ptmaxs = config['ptbins'][1:]
    ptLims = list(ptmins)
    nPtBins = len(ptmins)
    ptLims.append(ptmaxs[-1])
    ptBinsArr = np.asarray(ptLims, 'd')
    ptTit = '#it{p}_{T} (GeV/#it{c})'
    particleName = config['Dmeson']
    harmonic = config.get('harmonic', 2) # default is v2

    # Read fit configuration
    configfit = config['v2extraction']
    fixSigma = get_fit_config(config, cfg_cutset, 'FixSigma', isMultitrial)
    fixSigmaFromFile = get_fit_config(config, cfg_cutset, 'FixSigmaFromFile', isMultitrial)
    fixMean = get_fit_config(config, cfg_cutset, 'FixMean', isMultitrial)
    inclSecPeak = get_fit_config(config, cfg_cutset, 'InclSecPeak', isMultitrial)
    rebins = get_fit_config(config, cfg_cutset, 'Rebin', isMultitrial)
    sigmas = get_fit_config(config, cfg_cutset, 'Sigma', isMultitrial)
    useRefl = configfit.get('enableRef', False)
    reflFile = configfit.get('ReflFile', '')
    reflFuncStr = configfit.get('ReflFunc', '2Gaus')
    fix_pars_from_file = get_fit_config(config, cfg_cutset, 'FixFromFile', isMultitrial)
    fix_pars_from_config = get_fit_config(config, cfg_cutset, 'FixFromConfig', isMultitrial)
    fracsSecPeak = get_fit_config(config, cfg_cutset, 'FracSecPeak', isMultitrial)
    massesMinSecPeak = get_fit_config(config, cfg_cutset, 'MassMinSecPeak', isMultitrial)
    massesMaxSecPeak = get_fit_config(config, cfg_cutset, 'MassMaxSecPeak', isMultitrial)
    minCountsSecPeaks = get_fit_config(config, cfg_cutset, 'MinCountsSecPeak', isMultitrial)
    fixFracSigmaSecPeaks = get_fit_config(config, cfg_cutset, 'FixFracSigmaSecPeak', isMultitrial)
    fixVnSecPeakToSgn = get_fit_config(config, cfg_cutset, 'FixVnSecPeakToSgn', isMultitrial)
    print(f"fixVnSecPeakToSgn: {fixVnSecPeakToSgn}")
    sigmasSecPeak = get_fit_config(config, cfg_cutset, 'SigmaSecPeak', isMultitrial)

    print("CIAOOOO")
    massFitRanges = configfit['MassFitRanges']
    print(f"massFitRanges: {massFitRanges}")
    massFitLows = [mass[0] for mass in massFitRanges]
    massFitHighs = [mass[1] for mass in massFitRanges]
    SgnFuncStr = get_fit_config(config, cfg_cutset, 'SgnFunc', isMultitrial)
    BkgFuncStr = get_fit_config(config, cfg_cutset, 'BkgFunc', isMultitrial)
    BkgFuncVnStr = get_fit_config(config, cfg_cutset, 'BkgFuncVn', isMultitrial)
    if not isinstance(reflFuncStr, list):
        reflFuncStr = [reflFuncStr] * nPtBins

    # Check if at least one pt bin includes second peak, and if so, if the necessary configuration entries are provided
    if any(x == 1 for x in inclSecPeak) and not configfit.get('SigmaSecPeak'):
        logger('Second peak enabled, but SigmaSecPeak not provided. Check your config file.', level='ERROR')

    # Create canvas to contain all fit plots
    nCols = 4
    if nPtBins <= 8:
        cSimFitFull = TCanvas('cSimFitFull', 'cSimFitFull', 1200, 1200)
        cSimFitFull.Divide(nCols, nCols)
    if nPtBins > 8 and nPtBins <= 16:
        cSimFitFull = TCanvas('cSimFitFull', 'cSimFitFull', 1600, 1200)
        cSimFitFull.Divide(nCols, 2*nCols)
    else:
        logger('More than 16 pt bins not supported for full fit canvas. Skip creating it.', level='WARNING')

    SgnFunc, BkgFunc, BkgFuncVn, degPol = [], [], [], []
    hVnBkgCoeffs = [[] for iPt in range(nPtBins)]
    for iPt, (bkgStr, sgnStr, bkgVnStr) in enumerate(zip(BkgFuncStr, SgnFuncStr, BkgFuncVnStr)):
        degPol.append(-1)
        if bkgStr == 'kExpo':
            BkgFunc.append(VnVsMassFitter.kExpo)
        elif bkgStr == 'kLin':
            BkgFunc.append(VnVsMassFitter.kLin)
        elif bkgStr == 'kPol2':
            BkgFunc.append(VnVsMassFitter.kPol2)
        elif bkgStr == 'kPol3':
            BkgFunc.append(6)
            degPol[-1] = 3
        elif bkgStr == 'kPol4':
            BkgFunc.append(6)
            degPol[-1] = 4
            if len(ptmins) > 1 and inclSecPeak[iPt] == 1:
                logger('kPol4 background function is not supported for second peak fit. Use kPol2 instead.', level='ERROR')
        elif bkgStr == 'kPow':
            BkgFunc.append(VnVsMassFitter.kPow)
        elif bkgStr == 'kPowEx':
            BkgFunc.append(VnVsMassFitter.kPowEx)
        else:
            logger(f'ERROR: only kExpo, kLin, kPol2, kPol3, kPol4, kPow, and kPowEx background functions supported. Exit.', level='ERROR')
        if bkgVnStr == 'kExpo':
            BkgFuncVn.append(VnVsMassFitter.kExpo)
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff0", "hVnBkgCoeff0", nPtBins, ptBinsArr))
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff1", "hVnBkgCoeff1", nPtBins, ptBinsArr))
        elif bkgVnStr == 'kConst':
            BkgFuncVn.append(VnVsMassFitter.kConst)
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff0", "hVnBkgCoeff0", nPtBins, ptBinsArr))
        elif bkgVnStr == 'kLin':
            BkgFuncVn.append(VnVsMassFitter.kLin)
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff0", "hVnBkgCoeff0", nPtBins, ptBinsArr))
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff1", "hVnBkgCoeff1", nPtBins, ptBinsArr))
        elif bkgVnStr == 'kPol2':
            BkgFuncVn.append(VnVsMassFitter.kPol2)
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff0", "hVnBkgCoeff0", nPtBins, ptBinsArr))
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff1", "hVnBkgCoeff1", nPtBins, ptBinsArr))
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff2", "hVnBkgCoeff2", nPtBins, ptBinsArr))
        else:
            logger('Only kExpo, kLin, kConst, and kPol2 background functions supported for vn. Exit.', level='ERROR')
        if sgnStr == 'kGaus':
            SgnFunc.append(VnVsMassFitter.kGaus)
        elif sgnStr == 'k2Gaus':
            SgnFunc.append(VnVsMassFitter.k2Gaus)
        elif sgnStr == 'kLeftCB':
            SgnFunc.append(VnVsMassFitter.kLeftCB)
        elif sgnStr == 'kDoubleCBAsymm':
            SgnFunc.append(VnVsMassFitter.kDoubleCBAsymm)
        elif sgnStr == 'kDoubleCBSymm':
            SgnFunc.append(VnVsMassFitter.kDoubleCBSymm)
        elif sgnStr == 'k2GausSigmaRatioPar':
            SgnFunc.append(VnVsMassFitter.k2GausSigmaRatioPar)
        else:
            logger(f'Only kGaus, k2Gaus, kLeftCB, kDoubleCBAsymm, kDoubleCBSymm and k2GausSigmaRatioPar signal functions supported! Exit.', level='ERROR')
            sys.exit()

    # Set particle configuration
    partTitle, massAxisTit, decay, massForFit, massSecPeak, secPeakLabel = get_particle_info(particleName)

    # Load histos
    infile = TFile.Open(inFileName)
    if not infile or not infile.IsOpen():
        logger(f'File "{inFileName}" cannot be opened. Exit.', level='ERROR')

    hRefl, hMass, hMassForFit, hVn, hVnForFit, fTotFuncMass,\
    fTotFuncVn, fSgnFuncMass, fBkgFuncMass, fMassBkgRflFunc,\
    fMassSecPeakFunc, fBkgFuncVn, fVnSecPeakFunc, fVnCompFuncts,\
    hMCSgn, hMCRefl, hPulls, hParsMCPrefit, hParsSignalFunc, hParsSimFit = ([] for _ in range(20))

    useTemplates = True if config.get('corr_bkgs') else False
    fMassTemplFuncts = [None]*nPtBins if useTemplates else []
    fMassTemplTotFuncts = [None]*nPtBins if useTemplates else []

    for iPt, (ptMin, ptMax) in enumerate(zip(ptmins, ptmaxs)):
        hMass.append(infile.Get(f'pt_{ptMin*10:.0f}_{ptMax*10:.0f}/hMassData'))
        hVn.append(infile.Get(f'pt_{ptMin*10:.0f}_{ptMax*10:.0f}/hVnVsMassData'))

        hMass[iPt].SetDirectory(0)
        hVn[iPt].SetDirectory(0)
        SetObjectStyle(hMass[iPt], color=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hVn[iPt], color=kBlack, markerstyle=kFullCircle)
    infile.Close()

    hSigmaToFix = None
    if configfit.get('FixSigmaRatio'):
        # Load sigma of first gaussian
        infileSigma = TFile.Open(configfit['SigmaRatioFile'])
        if not infileSigma:
            logger(f'File "{infileSigma}" cannot be opened. Exit.', level='ERROR')

        hSigmaToFix = infileSigma.Get('hRawYieldsSigma')
        hSigmaToFix.SetDirectory(0)
        if hSigmaToFix.GetNbinsX() != nPtBins:
            logger('Different number of bins for this analysis and histo for fix sigma', level='WARNING')
        infileSigma.Close()
        # Load sigma of second gaussian
        infileSigma2 = TFile.Open(configfit['SigmaRatioFile'])
        if not infileSigma2:
            logger(f'File "{infileSigma2}" cannot be opened. Exit.', level='ERROR')
        hSigmaToFix2 = infileSigma2.Get('hRawYieldsSigma2')
        hSigmaToFix2.SetDirectory(0)
        if hSigmaToFix2.GetNbinsX() != nPtBins:
            logger('Different number of bins for this analysis and histo for fix sigma', level='WARNING')
        infileSigma2.Close()

    # Check reflections
    if particleName == 'Dzero' and useRefl:
        if reflFile == '':
            reflFile = inFileName
        useRefl, hMCSgn, hMCRefl = get_refl_histo(reflFile, ptmins, ptmaxs)

    # Create histos for fit results
    hSigmaSimFit = TH1D('hSigmaSimFit', f';{ptTit};#sigma', nPtBins, ptBinsArr)
    hMeanSimFit = TH1D('hMeanSimFit', f';{ptTit};mean', nPtBins, ptBinsArr)
    hMeanSecPeakFitMass = TH1D('hMeanSecondPeakFitMass', f';{ptTit};mean second peak mass fit', nPtBins, ptBinsArr)
    hMeanSecPeakFitVn = TH1D('hMeanSecondPeakFitVn', f';{ptTit};mean second peak vn fit', nPtBins, ptBinsArr)
    hSigmaSecPeakFitMass = TH1D('hSigmaSecondPeakFitMass',
                                f';{ptTit};width second peak mass fit', nPtBins, ptBinsArr)
    hSigmaSecPeakFitVn = TH1D('hSigmaSecondPeakFitVn', f';{ptTit};width second peak vn fit', nPtBins, ptBinsArr)
    hRawYieldsSimFit = TH1D('hRawYieldsSimFit', f';{ptTit};raw yield', nPtBins, ptBinsArr)
    hRawYieldsTrueSimFit = TH1D('hRawYieldsTrueSimFit', f';{ptTit};raw yield true', nPtBins, ptBinsArr)
    hRawYieldsSecPeakSimFit = TH1D('hRawYieldsSecondPeakSimFit',
                                   f';{ptTit};raw yield second peak', nPtBins, ptBinsArr)
    hRawYieldsSignificanceSimFit = TH1D('hRawYieldsSignificanceSimFit',
                                        f';{ptTit};significance', nPtBins, ptBinsArr)
    hRawYieldsSoverBSimFit = TH1D('hRawYieldsSoverBSimFit', f';{ptTit};S/B', nPtBins, ptBinsArr)
    hRedChi2SimFit = TH1D('hRedChi2SimFit', f';{ptTit};#chi^{{2}}/#it{{ndf}}', nPtBins, ptBinsArr)
    hProbSimFit = TH1D('hProbSimFit', f';{ptTit};prob', nPtBins, ptBinsArr)
    hRedChi2SBVnPrefit = TH1D('hRedChi2SBVnPrefit', f';{ptTit};#chi^{{2}}/#it{{ndf}}', nPtBins, ptBinsArr)
    hProbSBVnPrefit = TH1D('hProbSBVnPrefit', f';{ptTit};prob', nPtBins, ptBinsArr)
    hVnSimFit = TH1D('hVnSimFit',f';{ptTit};V2 (SP)', nPtBins, ptBinsArr)
    hTemplOverSgn = TH1D('hTemplOverSgn', f';{ptTit};Templ / Sgn', nPtBins, ptBinsArr)

    SetObjectStyle(hSigmaSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hMeanSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hMeanSecPeakFitMass, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hSigmaSecPeakFitMass, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hMeanSecPeakFitVn, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hSigmaSecPeakFitVn, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hRawYieldsSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hRawYieldsTrueSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hRawYieldsSecPeakSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hRawYieldsSignificanceSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hRawYieldsSoverBSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hRedChi2SimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hProbSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(hRedChi2SBVnPrefit, color=kRed, markerstyle=kFullSquare)
    SetObjectStyle(hProbSBVnPrefit, color=kRed, markerstyle=kFullSquare)
    SetObjectStyle(hVnSimFit, color=kBlack, markerstyle=kFullCircle)

    gVnSimFit = TGraphAsymmErrors(1)
    gVnSimFit.SetName('gVnSimFit')
    gVnSimFitSecPeak = TGraphAsymmErrors(1)
    gVnSimFitSecPeak.SetName('gVnSimFitSecPeak')
    gVnUnc = TGraphAsymmErrors(1)
    gVnUnc.SetName('gVnUnc')
    gVnUncSecPeak = TGraphAsymmErrors(1)
    gVnUncSecPeak.SetName('gVnUncSecPeak')
    SetObjectStyle(gVnSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(gVnSimFitSecPeak, color=kRed, markerstyle=kOpenCircle)
    SetObjectStyle(gVnUnc, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(gVnUncSecPeak, color=kRed, markerstyle=kOpenCircle)
    hTemplFracs = {}
    hTempls = {}

    # Create canvases
    cSimFit = []
    for i in range(nPtBins):
        ptLow = ptmins[i]
        ptHigh = ptmaxs[i]
        cSimFit.append(TCanvas(f'cSimFit_pt{ptLow}_{ptHigh}', f'cSimFit_pt{ptLow}_{ptHigh}', 400, 900))
        cSimFit[-1].Divide(1, 2)
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    canvVn = TCanvas('cVn', 'cVn', 900, 900)
    canvVnUnc = TCanvas('canvVnUnc', 'canvVnUnc', 900, 900)

    #_____________________________________________________
    # Vn estimation with Scalar Product
    vnFitter, vnRes = [], []
    for iPt, (hM, hV, ptMin, ptMax, reb, sgnEnum, bkgEnum, bkgVnEnum, secPeak, massMin, massMax) in enumerate(
            zip(hMass, hVn, ptmins, ptmaxs, rebins, SgnFunc, BkgFunc, BkgFuncVn, inclSecPeak, massFitLows, massFitHighs)):
        pt_dir = f"pt_{ptMin*10:.0f}_{ptMax*10:.0f}"
        iCanv = iPt
        hMassForFit.append(TH1F())
        hVnForFit.append(TH1F())
        RebinHisto(hM, reb, not isMultitrial).Copy(hMassForFit[iPt]) #to cast TH1D to TH1F
        hMassForFit[iPt].SetDirectory(0)
        xbins = np.asarray(hV.GetXaxis().GetXbins())
        hDummy = TH1F('hDummy', '', len(xbins)-1, xbins)
        for iBin in range(1, hV.GetNbinsX()+1):
            hDummy.SetBinContent(iBin, hV.GetBinContent(iBin))
            hDummy.SetBinError(iBin, hV.GetBinError(iBin))
        hVnForFit[iPt] = hDummy
        hVnForFit[iPt].SetDirectory(0)
        hVnForFit[iPt].GetXaxis().SetTitle(massAxisTit)
        hVnForFit[iPt].GetYaxis().SetTitle(f'#it{{v}}{harmonic}')
        binWidth = hMassForFit[iPt].GetBinWidth(1)
        hMassForFit[iPt].SetTitle((f'{ptMin:0.1f} < #it{{p}}_{{T}} < {ptMax:0.1f} GeV/#it{{c}}, Cutset {cfg_cutset["icutset"]};{massAxisTit};'
                                   f'Counts per {binWidth*1000:.0f} MeV/#it{{c}}^{{2}}'))
        hMassForFit[iPt].SetName(f'MassForFit{iPt}')
        SetObjectStyle(hMassForFit[iPt], color=kBlack, markerstyle=kFullCircle, markersize=1)
        SetObjectStyle(hVnForFit[iPt], color=kBlack, markerstyle=kFullCircle, markersize=0.8)

        if not isMultitrial:
            logger(f'Processing pt {ptMin} - {ptMax} GeV/c', level='INFO')
        fitName = f'VnVsMassFit_pt_{ptMin}_{ptMax}_cutset_{cfg_cutset["icutset"]}'
        vnFitter.append(VnVsMassFitter(fitName, hMassForFit[iPt], hVnForFit[iPt],
                                       massMin, massMax, bkgEnum, sgnEnum, bkgVnEnum,
                                       isMultitrial))

        vnFitter[iPt].SetHarmonic(harmonic)

        #_____________________________________________________
        # Set the parameters for the fit
        # Mean
        vnFitter[iPt].SetInitialGaussianMean(massForFit, 1)
        if fixMean[iPt]:
            vnFitter[iPt].FixMeanFromMassFit()
        # Sigma
        if fixSigma[iPt]:
            if fixSigmaFromFile != '':
                sigmaFile = TFile.Open(fixSigmaFromFile)
                # get the sigma histo from config file
                hSigmaFromFile = sigmaFile.Get('hSigmaSimFit')
                hSigmaFromFile.SetDirectory(0)
                sigmaBin = hSigmaFromFile.FindBin((ptMin+ptMax)/2)
                if hSigmaFromFile.GetBinLowEdge(sigmaBin) != ptMin:
                    logger(f'Bin edges do not match for {fixSigmaFromFile} and pt bins {ptMin} - {ptMax}. Exit.', level='ERROR')
                vnFitter[iPt].SetInitialGaussianSigma(hSigmaFromFile.GetBinContent(sigmaBin), 2)
            else:
                vnFitter[iPt].SetInitialGaussianSigma(sigmas[iPt], 2)
        else:
            vnFitter[iPt].SetInitialGaussianSigma(sigmas[iPt], 1)
        # nSigma4SB
        if configfit.get('NSigma4SB'):
            if not isinstance(configfit['NSigma4SB'], list):
                configfit['NSigma4SB'] = [configfit['NSigma4SB']] * nPtBins
            vnFitter[iPt].SetNSigmaForVnSB(configfit['NSigma4SB'][iPt])
        # Second peak
        if secPeak:
            if not isMultitrial:
                logger(f"Including second peak with frac width {fracsSecPeak[iPt]}, fixed? {fixFracSigmaSecPeaks[iPt]}", level='INFO')
            vnFitter[iPt].IncludeSecondGausPeak(massSecPeak, False, sigmasSecPeak[iPt], False, \
                                                massesMinSecPeak[iPt], massesMaxSecPeak[iPt], minCountsSecPeaks[iPt],
                                                fracsSecPeak[iPt], fixFracSigmaSecPeaks[iPt], \
                                                True, fixVnSecPeakToSgn[iPt])
            if fixSigma[iPt]:
                vnFitter[iPt].SetInitialGaussianSigma2Gaus(sigmasSecPeak[iPt], 2)
        vnFitter[iPt].FixFrac2GausFromMassFit()
        # Reflections for D0
        if useRefl:
            Signals = hMCSgn[iPt].Integral(hMCSgn[iPt].FindBin(massMin*1.0001), hMCSgn[iPt].FindBin(massMax*0.9999))
            Reflections = hMCRefl[iPt].Integral(hMCRefl[iPt].FindBin(massMin*1.0001), hMCRefl[iPt].FindBin(massMax*0.9999))
            SoverR = Reflections / (Signals + Reflections)
            vnFitter[iPt].SetTemplateReflections(hMCRefl[iPt], reflFuncStr[iPt], massMin, massMax)
            vnFitter[iPt].SetFixReflOverS(SoverR)
            vnFitter[iPt].SetReflVnOption(0)

        # Check if to use correlated bkgs templates
        useTemplatesPtBin = False
        hTemplFracs[pt_dir] = None
        hTempls[pt_dir] = None
        cfg_corr_bkgs = config.get('corr_bkgs', None)
        if cfg_corr_bkgs:
            for cocktail in cfg_corr_bkgs['cocktails']:
                if [ptMin, ptMax] in cocktail['pt_ranges']:
                    useTemplatesPtBin = True
                    cocktail_cfg = cocktail
                    hTemplFracs[pt_dir] = TH1D('hTemplFracs', f';;Templ / Sgn', len(cocktail_cfg['channels']), 0, len(cocktail_cfg['channels']))
                    break

        if useTemplatesPtBin:
            templates = {}
            hTempls[pt_dir] = {}
            if isMultitrial or "syst/multitrial/fit" in config['outdir']:
                raw_yields_file_path_ref = f"{config['outdir'].split('syst')[0]}/raw_yields/raw_yields_0{cfg_cutset['icutset']}.root"
                get_corr_bkgs_from_reference(templates, raw_yields_file_path_ref, pt_dir, massMin, massMax, reb)
                for i_chn, (chn, chn_info) in enumerate(templates.items()):
                    hTempls[pt_dir][f"{chn}_rebin_mass_sel"] = chn_info['histo']
                    hTemplFracs[pt_dir].GetXaxis().SetBinLabel(i_chn+1, chn)
                    hTemplFracs[pt_dir].SetBinContent(i_chn+1, chn_info['frac'] * configfit.get('TemplatesNorm', 1.0))
            else:
                if 'bkg_' in config['outdir']:
                    corr_bkg_file_path = f"{config['outdir'].split('syst')[0]}../corrbkgs/templs_{pt_dir}.root"
                else:
                    corr_bkg_file_path = f"{config['outdir']}/corrbkgs/templs_{pt_dir}.root"
                logger(f"Retrieving correlated bkg templates from {corr_bkg_file_path} for pt {ptMin} - {ptMax} GeV/c ...", "INFO")
                corr_bkg_file = TFile.Open(corr_bkg_file_path, 'r')
                sel_string = f"fMlScore0 < {cfg_cutset['ScoreBkg']['max'][iPt]} && fMlScore1 > {cfg_cutset['ScoreFD']['min'][iPt]}" \
                             f" && fMlScore1 < {cfg_cutset['ScoreFD']['max'][iPt]} && fM > {massMin} && fM < {massMax}"
                sgn_hist, sgn_frac = get_corr_bkg(corr_bkg_file, cfg_corr_bkgs['sgn_chn'], sel_string, pt_dir, "raw", "hist", verbose=True)
                for i_chn, chn in enumerate(cocktail_cfg['channels']):
                    templates[chn] = {}
                    logger(f"Getting correlated bkg template for channel {chn} ...", "INFO")
                    histo, frac = get_corr_bkg(corr_bkg_file, chn, sel_string, pt_dir, "raw", "hist",
                                               config['Dmeson'], configfit.get('CorrectAbundances', False),
                                               verbose=True)
                    histo.SetDirectory(0)
                    hTempls[pt_dir][chn] = histo
                    # Restrict histogram to fit range and rebin (reb)
                    histo_rebin_mass_range = get_rebinned_mass_sel_histo(histo, massMin, massMax, reb)
                    templates[chn]['histo'] = histo_rebin_mass_range
                    hTempls[pt_dir][f"{chn}_rebin_mass_sel"] = histo_rebin_mass_range
                    logger(f"[icutset = {cfg_cutset['icutset']}] Setting channel {chn} frac to (frac / sgn_frac): {frac} / {sgn_frac}", "INFO")
                    templates[chn]['frac'] = (frac / sgn_frac)  # option to apply an additional scaling to the fraction from config, if needed for systematics
                    hTemplFracs[pt_dir].GetXaxis().SetBinLabel(i_chn+1, chn)
                    hTemplFracs[pt_dir].SetBinContent(i_chn+1, (frac / sgn_frac) * configfit.get('TemplatesNorm', 1.0))

                corr_bkg_file.Close()

            logger(f"Will further scale templates with factor {configfit.get('TemplatesNorm', 1.0)} from config for pt {ptMin} - {ptMax} GeV/c ...", "INFO")
            vnFitter[iPt].SetTemplatesHisto([templ['histo'] for templ in templates.values()],
                                            [templ['frac'] * configfit.get('TemplatesNorm', 1.0) for templ in templates.values()],
                                            cfg_corr_bkgs.get('anchor_mode', 2))

        # Initial fit parameters
        if configfit.get('InitFitPars'):
            for setting in configfit['InitFitPars']:
                pt_center = (ptMin + ptMax) / 2
                for pt_range in setting['pt_ranges']:
                    if pt_center > pt_range[0] and pt_center < pt_range[1]:
                        for par_dict in setting['pars']:
                            for par_name, par_cfg in par_dict.items():
                                if not isMultitrial:
                                    print(f"Setting initial fit parameter {par_name} to {par_cfg} "
                                          f"for pt {ptMin} - {ptMax} GeV/c ...")
                                if fix_pars_from_config[iPt]:  # fix to value from config
                                    vnFitter[iPt].SetInitPar(par_name, par_cfg[0], par_cfg[1], par_cfg[2])
                                else: # 10% variation around the value from config if not fixing, for systematics
                                    vnFitter[iPt].SetInitPar(par_name, par_cfg[0], par_cfg[0] - 0.1*par_cfg[0], par_cfg[0] + 0.1*par_cfg[0])
                        break

        # Retrieve histogram to fix signal
        if configfit.get("InitFromMC") or configfit.get("FixFromMC") and SgnFunc[iPt] in [VnVsMassFitter.kDoubleCBAsymm, VnVsMassFitter.kDoubleCBSymm]:
            if 'bkg_' in config['outdir']:
                corr_bkg_file_path = f"{config['outdir'].split('syst')[0]}/../corrbkgs/templs_{pt_dir}.root"
                logger(f"Using corr bkg file path {corr_bkg_file_path} for init/fix from MC for pt {ptMin} - {ptMax} GeV/c ...", level='INFO')
            elif isMultitrial:
                corr_bkg_file_path = f"{config['outdir'].split('syst')[0]}/../corrbkgs/templs_{pt_dir}.root"
            else:
                corr_bkg_file_path = f"{config['outdir']}/corrbkgs/templs_{pt_dir}.root"
            corr_bkg_file = TFile.Open(corr_bkg_file_path, 'r')
            prefitHisto = corr_bkg_file.Get(f"{pt_dir}/{configfit['SgnFuncLabel']}/raw/hMassSmooth")
            prefitHisto.SetDirectory(0)
            vnFitter[iPt].SetHistoPrefitSgn(prefitHisto, configfit.get("FixFromMC", False))
            corr_bkg_file.Close()

        if configfit.get('ParsFromFile'):
            do_init = True
            if isinstance(configfit['ParsFromFile'], list) and not configfit['ParsFromFile'][iPt]:
                do_init = False  # do not init for this pt bin
            if do_init:
                file_path = get_fit_config(config, cfg_cutset, 'ParsFromFile', isMultitrial)
                file_path = file_path[iPt] if isinstance(file_path, list) else file_path
                logger(f"Trying to fix signal parameters from file {file_path} for pt {ptMin} - {ptMax} GeV/c ...", "INFO")
                # quit()
                try:
                    fixParsFile = TFile.Open(file_path, 'r')
                    fixParsHisto = fixParsFile.Get(f'{pt_dir}/hist_signal_func')
                    fixParsHisto.SetDirectory(0)
                    for iBin in range(1, fixParsHisto.GetNbinsX()+1):
                        if iBin <= 3:    # skip integral, mean, sigma
                            continue
                        binLabel = fixParsHisto.GetXaxis().GetBinLabel(iBin)
                        binContent = fixParsHisto.GetBinContent(iBin)
                        if fix_pars_from_file[iPt]:  # fix to value from file
                            vnFitter[iPt].SetInitPar(binLabel, binContent, binContent, binContent)
                        else:
                            # Allow 10% variation around the value from file if not fixing, to help convergence
                            vnFitter[iPt].SetInitPar(binLabel, binContent, binContent - 0.1*binContent, binContent + 0.1*binContent)
                        # logger(f"Setting initial parameter {binLabel} to {binContent} from file {configfit['ParsFromFile']}", "INFO")
                    fixParsFile.Close()
                except Exception as e:
                    logger(f"Exception {e} caught when trying to fix signal parameters from file {file_path}.", level='ERROR')

        isFitGood = vnFitter[iPt].SimultaneousFit()
        if not isMultitrial:
            print(f"Fit status for pt {ptMin} - {ptMax} GeV/c: {isFitGood}")

        if isFitGood == 2:  # Second peak was excluded due to low counts, but fit was successful
            secPeak = False
            isFitGood = 1

        # Quality selection for systematics multitrial
        if isMultitrial:
            try:
                if vnFitter[iPt].GetReducedChiSquare() > config.get('MaxChi2PerNDF', 10):
                    print(f'[{inFileName}] Rejecting trial due to Chi2/NDF = {vnFitter[iPt].GetReducedChiSquare()} > {config.get("MaxChi2PerNDF", 1.e20)}')
                    return
            except Exception as e:
                print(f'[{inFileName}] Exception {e} caught during chi2 calculation. Rejecting trial.')
                return
            try:
                signif, signifUnc = ctypes.c_double(), ctypes.c_double()
                vnFitter[iPt].Significance(3, signif, signifUnc)
                if signif.value < config.get('MinSignificance', 0) or signif.value > config.get('MaxSignificance', 1.e20):
                    print(f'[{inFileName}] Rejecting trial due to significance = {signif.value}, (min = {config.get("MinSignificance", 0)}, max = {config.get("MaxSignificance", 1.e20)})')
                    return
            except Exception as e:
                print(f'[{inFileName}] Exception {e} caught during significance calculation. Rejecting trial.')
                return

        # isFitGood = True
        vnRes.append(None)
        hParsMCPrefit.append(None)
        hParsSignalFunc.append(None)
        hParsSimFit.append(None)
        fTotFuncMass.append(None)
        fTotFuncVn.append(None)
        fSgnFuncMass.append(None)
        fBkgFuncMass.append(None)
        fBkgFuncVn.append(None)
        if isFitGood:
            secPeakMode = None
            if secPeak:
                secPeakMode = 'VnFree'
                if fixVnSecPeakToSgn[iPt]:
                    secPeakMode = 'VnFixedToSgn'
                if fixFracSigmaSecPeaks[iPt]:
                    secPeakMode += 'FixedSigmaFrac'
            vnRes[iPt] = get_vnfitter_results(vnFitter[iPt], useRefl, useTemplatesPtBin, secPeakMode, \
                                              fracsSecPeak[iPt] if fracsSecPeak[iPt] is not None else 1)

            hSigmaSimFit.SetBinContent(iPt+1, vnRes[iPt]['sigma'])
            hSigmaSimFit.SetBinError(iPt+1, vnRes[iPt]['sigmaUnc'])
            hMeanSimFit.SetBinContent(iPt+1, vnRes[iPt]['mean'])
            hMeanSimFit.SetBinError(iPt+1, vnRes[iPt]['meanUnc'])
            hRedChi2SimFit.SetBinContent(iPt+1, vnRes[iPt]['chi2'])
            hRedChi2SimFit.SetBinError(iPt+1, 1.e-20)
            hProbSimFit.SetBinContent(iPt+1, vnRes[iPt]['prob'])
            hProbSimFit.SetBinError(iPt+1, 1.e-20)
            hRawYieldsSimFit.SetBinContent(iPt+1, vnRes[iPt]['ry'])
            hRawYieldsSimFit.SetBinError(iPt+1, vnRes[iPt]['ryUnc'])
            hRawYieldsTrueSimFit.SetBinContent(iPt+1, vnRes[iPt]['ryTrue'])
            hRawYieldsTrueSimFit.SetBinError(iPt+1, vnRes[iPt]['ryTrueUnc'])
            hRawYieldsSignificanceSimFit.SetBinContent(iPt+1, vnRes[iPt]['signif'])
            hRawYieldsSignificanceSimFit.SetBinError(iPt+1, vnRes[iPt]['signifUnc'])
            hVnSimFit.SetBinContent(iPt+1, vnRes[iPt]['vn'])
            hVnSimFit.SetBinError(iPt+1, vnRes[iPt]['vnUnc'])
            gVnSimFit.SetPoint(iPt, (ptMin+ptMax)/2, vnRes[iPt]['vn'])
            gVnSimFit.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, vnRes[iPt]['vnUnc'], vnRes[iPt]['vnUnc'])
            gVnUnc.SetPoint(iPt, (ptMin+ptMax)/2, vnRes[iPt]['vnUnc'])
            gVnUnc.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, 1.e-20, 1.e-20)
            hPulls.append(vnRes[iPt]['pulls'])
            hRawYieldsSoverBSimFit.SetBinContent(iPt+1, vnRes[iPt]["ry"]/vnRes[iPt]["bkg"])

            # Save histogram parameters
            hParsMCPrefit[iPt] = vnRes[iPt]['hParsMCPrefit']
            hParsSignalFunc[iPt] = vnRes[iPt]['hParsSignalFunc']
            hParsSimFit[iPt] = vnRes[iPt]['hParsSimFit']
            fTotFuncMass[iPt] = vnRes[iPt]['fTotFuncMass']
            fTotFuncVn[iPt] = vnRes[iPt]['fTotFuncVn']
            fSgnFuncMass[iPt] = vnRes[iPt]['fSgnFuncMass']
            fBkgFuncMass[iPt] = vnRes[iPt]['fBkgFuncMass']
            fBkgFuncVn[iPt] = vnRes[iPt]['fBkgFuncVn']

            SetObjectStyle(fTotFuncMass[iPt], color=kAzure+4, linewidth=3)
            SetObjectStyle(fSgnFuncMass[iPt], fillcolor=kAzure+4, fillstyle=1000, linewidth=0, fillalpha=0.3)
            SetObjectStyle(fBkgFuncMass[iPt], color=kOrange+1, linestyle=9, linewidth=2)
            SetObjectStyle(fBkgFuncVn[iPt], color=kOrange+1, linestyle=7, linewidth=2)
            SetObjectStyle(fTotFuncVn[iPt], color=kAzure+4, linewidth=3)

            for bkgHisto, bkgParVal, bkgParErr in zip(hVnBkgCoeffs[iPt], vnRes[iPt]['bkgPars'], vnRes[iPt]['bkgParsUncs']):
                bkgHisto.SetBinContent(iPt+1, bkgParVal)
                bkgHisto.SetBinError(iPt+1, bkgParErr)

            if secPeak:
                hMeanSecPeakFitMass.SetBinContent(iPt+1, vnRes[iPt]['secPeakMeanMass'])
                hMeanSecPeakFitMass.SetBinError(iPt+1, vnRes[iPt]['secPeakMeanMassUnc'])
                hSigmaSecPeakFitMass.SetBinContent(iPt+1, vnRes[iPt]['secPeakSigmaMass'])
                hSigmaSecPeakFitMass.SetBinError(iPt+1, vnRes[iPt]['secPeakSigmaMassUnc'])
                hMeanSecPeakFitVn.SetBinContent(iPt+1, vnRes[iPt]['secPeakMeanVn'])
                hMeanSecPeakFitVn.SetBinError(iPt+1, vnRes[iPt]['secPeakMeanVnUnc'])
                hSigmaSecPeakFitVn.SetBinContent(iPt+1, vnRes[iPt]['secPeakSigmaVn'])
                hSigmaSecPeakFitVn.SetBinError(iPt+1, vnRes[iPt]['secPeakSigmaVnUnc'])
                gVnSimFitSecPeak.SetPoint(iPt, (ptMin+ptMax)/2, vnRes[iPt]['vnSecPeak'])
                gVnSimFitSecPeak.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2,
                                               vnRes[iPt]['vnSecPeakUnc'],
                                               vnRes[iPt]['vnSecPeakUnc'])
                gVnUncSecPeak.SetPoint(iPt, (ptMin+ptMax)/2, vnRes[iPt]['vnSecPeakUnc'])
                gVnUncSecPeak.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, 1.e-20, 1.e-20)
                fMassSecPeakFunc.append(vnRes[iPt]['fMassSecPeakFunc'])
                fVnSecPeakFunc.append(vnRes[iPt]['fVnSecPeakFunct'])
                SetObjectStyle(fMassSecPeakFunc[-1], fillcolor=kGreen+1, fillstyle=1000, linewidth=0, fillalpha=0.3)

            if useRefl:
                hRefl.append(vnRes[iPt]['fMassRflFunc'])
                fMassBkgRflFunc.append(vnRes[iPt]['fMassBkgRflFunc'])
                SetObjectStyle(hRefl[iPt], fillcolor=kGreen+1, fillstyle=1000, linewidth=0, fillalpha=0.3)
                SetObjectStyle(fMassBkgRflFunc[iPt], color=kRed+1, linestyle=7, linewidth=2)

            if configfit.get('DrawVnComps'):
                fVnCompFuncts.append(vnRes[iPt]['fVnCompsFuncts'])

            # Draw upper pad
            cSimFit[iPt].cd(1)
            hMassForFit[iPt].GetYaxis().SetRangeUser(0.2*hMassForFit[iPt].GetMinimum(),
                                                     1.8*hMassForFit[iPt].GetMaximum())
            hMassForFit[iPt].GetYaxis().SetMaxDigits(3)
            hMassForFit[iPt].GetXaxis().SetRangeUser(massMin, massMax)
            hMassForFit[iPt].Draw('E')
            fSgnFuncMass[iPt].Draw('fc same')
            fBkgFuncMass[iPt].Draw('same')
            fTotFuncMass[iPt].Draw('same')
            if secPeak:
                fMassSecPeakFunc[-1].Draw('fc same')
            if useRefl:
                fMassBkgRflFunc[iPt].Draw('same')
                hRefl[iPt].Draw('same')
                latex.DrawLatex(0.18, 0.20, f'RoverS = {SoverR:.2f}')

            latex.DrawLatex(0.18, 0.80, f'#mu = {vnRes[iPt]["mean"]:.3f} #pm {vnRes[iPt]["meanUnc"]:.3f} GeV/c^{2}')
            latex.DrawLatex(0.18, 0.75, f'#sigma = {vnRes[iPt]["sigma"]:.3f} #pm {vnRes[iPt]["sigmaUnc"]:.3f} GeV/c^{2}')
            latex.DrawLatex(0.18, 0.70, f'S = {vnRes[iPt]["ry"]:.0f} #pm {vnRes[iPt]["ryUnc"]:.0f}')
            latex.DrawLatex(0.18, 0.65, f'S/B (3#sigma) = {vnRes[iPt]["ry"]/vnRes[iPt]["bkg"]:.2f}')
            latex.DrawLatex(0.18, 0.60, f'Signif. (3#sigma) = {round(vnRes[iPt]["signif"], 2)}')

            if secPeak:
                latex.DrawLatex(0.18, 0.55,
                                f'#mu ({secPeakLabel}) = {vnRes[iPt]["secPeakMeanMass"]:.3f} #pm {vnRes[iPt]["secPeakMeanMassUnc"]:.3f} GeV/c^{2}')
                latex.DrawLatex(0.18, 0.50,
                                f'#sigma ({secPeakLabel}) = {vnRes[iPt]["secPeakSigmaMass"]:.3f} #pm {vnRes[iPt]["secPeakSigmaMassUnc"]:.3f} GeV/c^{2}')

            if useTemplatesPtBin:
                fMassTemplFuncts[iPt] = vnRes[iPt]['fMassTemplFuncts']
                fMassTemplTotFuncts[iPt] = vnRes[iPt]['fMassTemplTotFunc']
                hTemplOverSgn.SetBinContent(iPt+1, vnFitter[iPt].GetTemplOverSig())
                SetObjectStyle(fMassTemplTotFuncts[iPt], color=kRed, linewidth=2)
                fMassTemplTotFuncts[iPt].Draw('same')
                cSimFit[iPt].Modified()
                cSimFit[iPt].Update()
                if configfit.get("DrawSingleTempls"):
                    for iMassTemplFunct, massTemplFunct in enumerate(fMassTemplFuncts[iPt]):
                        SetObjectStyle(massTemplFunct, color=kMagenta+2+iMassTemplFunct*2, linewidth=3)
                        massTemplFunct.Draw('same')
                        cSimFit[iPt].Modified()
                        cSimFit[iPt].Update()
            # Draw lower pad
            cSimFit[iPt].cd(2)
            hVnForFit[iPt].GetXaxis().SetRangeUser(massMin, massMax)
            hVnForFit[iPt].GetYaxis().SetDecimals()
            # delta_max_min considering uncertainties associated to points
            bin_contents = [hVnForFit[iPt].GetBinContent(bi) for bi in range(1, hVnForFit[iPt].GetNbinsX()+1)]
            bin_errors = [hVnForFit[iPt].GetBinError(bi) for bi in range(1, hVnForFit[iPt].GetNbinsX()+1)]
            bin_values_with_unc = [bc + be for bc, be in zip(bin_contents, bin_errors)] + [bc - be for bc, be in zip(bin_contents, bin_errors)]
            delta_max_min = max(bin_values_with_unc) - min(bin_values_with_unc)
            hVnForFit[iPt].GetYaxis().SetRangeUser(min(bin_values_with_unc) - 0.4*delta_max_min, max(bin_values_with_unc) + 0.4*delta_max_min)
            hVnForFit[iPt].Draw('E')
            fBkgFuncVn[iPt].Draw('same')
            fTotFuncVn[iPt].Draw('same')

            latex.DrawLatex(0.18, 0.18, f'#chi^{{2}}/ndf = {vnRes[iPt]["chi2"]:.2f}')
            latex.DrawLatex(0.18, 0.80,
                            f'#it{{v}}{harmonic}({partTitle}) = {vnRes[iPt]["vn"]:.3f} #pm {vnRes[iPt]["vnUnc"]:.3f}')
            if secPeak:
                latex.DrawLatex(0.18, 0.75, f'#it{{v}}{harmonic}({secPeakLabel}) = {vnRes[iPt]["vnSecPeak"]:.3f} #pm {vnRes[iPt]["vnSecPeakUnc"]:.3f}')

            if configfit.get('DrawVnComps'):
                legVnCompn = TLegend(0.72, 0.15, 0.9, 0.35)
                legVnCompn.SetBorderSize(0)
                legVnCompn.SetFillStyle(0)
                legVnCompn.SetTextSize(0.03)
                legVnCompn.AddEntry(fBkgFuncVn[iPt], f'#it{{v}}{harmonic} Bkg Func.', 'l')
                legVnCompn.AddEntry(fTotFuncVn[iPt], f'#it{{v}}{harmonic} Tot Func.', 'l')
                
                SetObjectStyle(fVnCompFuncts[iPt]['vnSgn'], fillcolor=kAzure+4, fillstyle=3245, linewidth=0)
                SetObjectStyle(fVnCompFuncts[iPt]['vnBkg'], color=kOrange+1, linestyle=1, linewidth=2)
                
                legVnCompn.AddEntry(fVnCompFuncts[iPt]['vnSgn'], f"Signal #it{{v}}{harmonic}", 'f')
                legVnCompn.AddEntry(fVnCompFuncts[iPt]['vnBkg'], f"Bkg #it{{v}}{harmonic}", 'l')
                if secPeak:
                    SetObjectStyle(fVnCompFuncts[iPt]['vnSecPeak'], fillcolor=kGreen+1, fillstyle=3254, linewidth=0)
                    legVnCompn.AddEntry(fVnCompFuncts[iPt]['vnSecPeak'], f"Second peak #it{{v}}{harmonic}", 'f')
                if useTemplatesPtBin:
                    for iTempl in range(len(fVnCompFuncts[iPt])-2-secPeak):
                        SetObjectStyle(fVnCompFuncts[iPt][f'vnTempl{iTempl}'], color=kMagenta+2+iTempl*2, linewidth=3)
                        legVnCompn.AddEntry(fVnCompFuncts[iPt][f'vnTempl{iTempl}'], f"Templ{iTempl} #it{{v}}{harmonic}", 'l')
                for _, vnCompFunct in fVnCompFuncts[iPt].items():
                    vnCompFunct.Draw('same')
                    cSimFit[iPt].Modified()
                    cSimFit[iPt].Update()
                legVnCompn.Draw()

            cSimFit[iPt].Modified()
            cSimFit[iPt].Update()

    canvVn.cd().SetLogx()
    hframe = canvVn.DrawFrame(0.5, -0.5, gVnSimFit.GetXaxis().GetXmax()+0.5, 0.5,
                              f';#it{{p}}_{{T}} (GeV/c); v_{{{harmonic}}} (SP)')
    hframe.GetYaxis().SetDecimals()
    hframe.GetXaxis().SetNdivisions(504)
    hframe.GetXaxis().SetMoreLogLabels()
    gPad.SetGridy()
    gVnSimFit.Draw('same pez')
    if secPeak:
        gVnSimFitSecPeak.Draw('pez same')
    latex.DrawLatexNDC(0.20, 0.80, 'This work')
    latex.DrawLatexNDC(0.20, 0.75, f'Pb#minusPb #sqrt{{#it{{s}}_{{NN}}}} = 5.36 TeV ({centMinMax[0]}#minus{centMinMax[1]}%)')
    latex.DrawLatexNDC(0.20, 0.70, decay)
    canvVn.Modified()
    canvVn.Update()
    canvVnUnc.cd()
    gVnUnc.Draw('apez same')
    if secPeak:
        gVnUncSecPeak.Draw('pez same')
    canvVnUnc.Modified()
    canvVnUnc.Update()
    if not batch:
        logger('Press Enter to continue...', level='PAUSE')

    # Save output histos
    if not isMultitrial:
        logger(f'Saving output histos to {outFileName}.root')
    os.makedirs(os.path.dirname(outFileName), exist_ok=True)
    for iPt, (ptMin, ptMax) in enumerate(zip(ptmins, ptmaxs)):
        if iPt == 0:
            suffix_pdf = '('
        elif iPt == nPtBins-1:
            suffix_pdf = ')'
        else:
            suffix_pdf = ''
        if len(ptmins)==1:
            cSimFit[iPt].SaveAs(f'{outFileName}.pdf')
        else:
            cSimFit[iPt].SaveAs(f'{outFileName}.pdf{suffix_pdf}')
    cSimFitFull.SaveAs(f'{outFileName}_AllFits.pdf')
    cSimFitFull.SaveAs(f'{outFileName}_AllFits.png')
    outFile = TFile(f'{outFileName}.root', 'recreate')

    for iPt, (ptMin, ptMax) in enumerate(zip(ptmins, ptmaxs)):
        make_dir_root_file(f'pt_{int(ptMin*10)}_{int(ptMax*10)}', outFile, not isMultitrial)
    outFile.cd()

    for canv, (pt_min, pt_max) in zip(cSimFit, zip(ptmins, ptmaxs)):
        if canv:
            outFile.cd(f'pt_{int(pt_min*10)}_{int(pt_max*10)}')
            canv.Write()
    for hist, pt_min, pt_max in zip(hParsMCPrefit, ptmins, ptmaxs):
        if hist:
            outFile.cd(f"pt_{int(pt_min*10)}_{int(pt_max*10)}")
            hist.Write('hist_mc_prefit')
    for hist, pt_min, pt_max in zip(hParsSignalFunc, ptmins, ptmaxs):
        if hist:
            outFile.cd(f"pt_{int(pt_min*10)}_{int(pt_max*10)}")
            hist.Write('hist_signal_func')
    for hist, pt_min, pt_max in zip(hParsSimFit, ptmins, ptmaxs):
        if hist:
            outFile.cd(f"pt_{int(pt_min*10)}_{int(pt_max*10)}")
            hist.Write('hist_sim_fit')
    for hist, pt_min, pt_max in zip(hMass, ptmins, ptmaxs):
        if hist:
            outFile.cd(f"pt_{int(pt_min*10)}_{int(pt_max*10)}")
            hist.Write('hist_mass')
    for hist, pt_min, pt_max in zip(hVn, ptmins, ptmaxs):
        if hist:
            outFile.cd(f"pt_{int(pt_min*10)}_{int(pt_max*10)}")
            hist.Write('hist_vn')
    for hist, pt_min, pt_max in zip(hPulls, ptmins, ptmaxs):
        if hist:
            outFile.cd(f"pt_{int(pt_min*10)}_{int(pt_max*10)}")
            hist.Write('hist_pulls')
    for ipt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs)):
        try:
            outFile.cd(f'pt_{int(ptmin*10)}_{int(ptmax*10)}')
            fTotFuncMass[ipt].Write('fTotFuncMass')
            fTotFuncVn[ipt].Write('fTotFuncVn')
            fSgnFuncMass[ipt].Write('fSgnFuncMass')
            fBkgFuncMass[ipt].Write('fBkgFuncMass')
            fBkgFuncVn[ipt].Write('fBkgFuncVn')
            for bkgHisto in hVnBkgCoeffs[ipt]:
                bkgHisto.Write(f'{bkgHisto.GetName()}')
        except:
            logger(f'Fit function for pt {ptmin*10:.0f}-{ptmax*10:.0f} not available. Skipping.', level='WARNING')

    for pt_dir, hTemplFrac in hTemplFracs.items():
        make_dir_root_file(f"{pt_dir}/templs", outFile, not isMultitrial)
        outFile.cd(f"{pt_dir}/templs/")
        if hTemplFrac:
            hTemplFrac.Write('hTemplFracs')
    for pt_dir, hTempl in hTempls.items():
        outFile.cd(f"{pt_dir}/templs/")
        if hTempl:
            for chn, histo in hTempl.items():
                histo.Write(f"hTempl_{chn}") if "rebin_mass_sel" in chn else histo.Write(f"hTempl_{chn}_raw")

    outFile.cd()
    hSigmaSimFit.Write()
    hMeanSimFit.Write()
    hMeanSecPeakFitMass.Write()
    hMeanSecPeakFitVn.Write()
    hSigmaSecPeakFitMass.Write()
    hSigmaSecPeakFitVn.Write()
    hRawYieldsSimFit.Write()
    hRawYieldsTrueSimFit.Write()
    hRawYieldsSecPeakSimFit.Write()
    hRawYieldsSignificanceSimFit.Write()
    hRawYieldsSoverBSimFit.Write()
    hRedChi2SimFit.Write()
    hProbSimFit.Write()
    hRedChi2SBVnPrefit.Write()
    hProbSBVnPrefit.Write()
    hVnSimFit.Write()
    hTemplOverSgn.Write()

    gVnSimFit.Write()
    gVnUnc.Write()
    if secPeak:
        gVnSimFitSecPeak.Write()
        gVnUncSecPeak.Write()

    outFile.Close()

    if not batch:
        logger(f'Output file saved as {outFileName}.pdf', level='INFO')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('fitConfigFileName', metavar='text', default='config_Ds_Fit.yml')
    parser.add_argument('cutset_file_name', metavar='text', default='config_Ds_Fit.yml')
    parser.add_argument('inFileName', metavar='text', default='')
    parser.add_argument('--batch', '-b', help='suppress video output', action='store_true')
    parser.add_argument('--multitrial', help='suppress redundant prints', action='store_true')
    args = parser.parse_args()

    get_vn_vs_mass(
        args.fitConfigFileName,
        args.cutset_file_name,
        args.inFileName,
        args.batch,
        args.multitrial
    )
