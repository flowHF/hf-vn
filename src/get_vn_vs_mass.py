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
from ROOT import TLatex, TFile, TCanvas, TLegend, TH1D, TH1F, TGraphAsymmErrors # pylint: disable=import-error,no-name-in-module
from ROOT import gROOT, gPad, gInterpreter, kBlack, kRed, kAzure, kOrange, kGreen, kFullCircle, kFullSquare, kOpenCircle # pylint: disable=import-error,no-name-in-module
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
        os.path.join(src_dir, "InvMassFitter.h"),
        os.path.join(src_dir, "VnVsMassFitter.h"),
        os.path.join(src_dir, "LinkDefVnFitter.h"),
    ])

    cmd = [
        "g++", "-shared", "-fPIC",
        *subprocess.check_output(["root-config", "--cflags", "--libs"]).decode().split(),
        os.path.join(src_dir, "InvMassFitter.cxx"),
        os.path.join(src_dir, "VnVsMassFitter.cxx"),
        os.path.join(src_dir, "vnfitter_dict.cxx"),
        "-o", lib,
    ]

    subprocess.check_call(cmd)
    print("Compilation done!")

# Load the library once
if not hasattr(gSystem, "_vnfitter_loaded"):
    if os.path.exists(lib):
        print(f"Loading existing library: {lib}")
        gSystem.Load(lib)
    else:
        print(f"Library {lib} not found — compiling with ACLiC...")
        gSystem.CompileMacro(os.path.join(src_dir, "VnVsMassFitter.cxx"), "kO")
        gSystem.CompileMacro(os.path.join(src_dir, "InvMassFitter.cxx"), "kO")
    gSystem._vnfitter_loaded = True

from ROOT import InvMassFitter, VnVsMassFitter
os.sys.path.append(os.path.join(script_dir, '..', 'utils'))
from StyleFormatter import SetGlobalStyle, SetObjectStyle
from fit_utils import RebinHisto
from utils import logger, get_centrality_bins, get_vnfitter_results, get_refl_histo, get_particle_info

TEMPL_COLOR_CYCLE = [ROOT.kRed+2, ROOT.kRed, ROOT.kRed-4, ROOT.kOrange+7, ROOT.kOrange-3,
                     ROOT.kMagenta+1, ROOT.kGreen+2, ROOT.kCyan+2]

def get_vn_vs_mass(fitConfigFileName, inFileName, batch, isMultitrial):
    #______________________________________________________
    # Read configuration file
    with open(fitConfigFileName, 'r', encoding='utf8') as ymlfitConfigFile:
        config = yaml.load(ymlfitConfigFile, yaml.FullLoader)

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
    
    # Read fit configuration — support both "v2extraction" and "fitConfig" keys
    configfit = config.get('v2extraction', config.get('fitConfig', {}))
    if not configfit:
        logger('ERROR: neither "v2extraction" nor "fitConfig" found in YAML.', level='ERROR')
    fixSigma = configfit.get('FixSigma', 0)
    fixSigmaFromFile = configfit.get('FixSigmaFromFile', '')
    fixMean = configfit.get('FixMean', 0)
    inclSecPeak = configfit.get('InclSecPeak', 0)
    rebins = configfit.get('Rebin', 1)
    useRefl = configfit.get('enableRef', False)
    reflFile = configfit.get('ReflFile', '')
    reflFuncStr = configfit.get('ReflFunc', '2Gaus')

    corrBkgCfg = configfit.get('CorrBkgTempls', {})
    useTempl = bool(corrBkgCfg)
    templOrder = corrBkgCfg.get('Channels', [])
    templLabels = {c: c.replace('Dzero', 'D0').replace('Dplus', 'D+') for c in templOrder}
    templColors = {c: TEMPL_COLOR_CYCLE[i % len(TEMPL_COLOR_CYCLE)] for i, c in enumerate(templOrder)}
    cutSetCfg = None
    if useTempl:
        from correlated_bkgs import get_corr_bkg, get_corr_bkg_tf1
        cutSetFileName = corrBkgCfg.get('CutSetFile', inFileName.replace('projs', 'cutsets').replace('proj_', 'cutset_').replace('.root', '.yml'))
        if os.path.exists(cutSetFileName):
            with open(cutSetFileName, 'r') as ymlCutSet:
                cutSetCfg = yaml.load(ymlCutSet, yaml.FullLoader)
        else:
            logger(f'Cutset {cutSetFileName} not found, disabling correlated bkg templates', level='WARNING')
            useTempl = False

    if not isinstance(rebins, list):
        rebins = [rebins] * len(ptmins)
    massFitRanges = configfit['MassFitRanges']
    massFitLows = [mass[0] for mass in massFitRanges]
    massFitHighs = [mass[1] for mass in massFitRanges]
    if not isinstance(fixSigma, list):
        fixSigma = [fixSigma for _ in ptmins]
    if not isinstance(fixMean, list):
        fixMean = [fixMean for _ in ptmins]
    SgnFuncStr = configfit['SgnFunc']
    if not isinstance(SgnFuncStr, list):
        SgnFuncStr = [SgnFuncStr] * nPtBins
    BkgFuncStr = configfit['BkgFunc']
    if not isinstance(BkgFuncStr, list):
        BkgFuncStr = [BkgFuncStr] * nPtBins
    BkgFuncVnStr = configfit['BkgFuncVn']
    if not isinstance(BkgFuncVnStr, list):
        BkgFuncVnStr = [BkgFuncVnStr] * nPtBins
    if not isinstance(reflFuncStr, list):
        reflFuncStr = [reflFuncStr] * nPtBins
    if not isinstance(inclSecPeak, list):
        inclSecPeak = [inclSecPeak] * nPtBins

    # Sanity check of fit configuration
    if 1 in inclSecPeak and not configfit.get('SigmaSecPeak'):
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
            BkgFunc.append(InvMassFitter.kExpo)
        elif bkgStr == 'kLin':
            BkgFunc.append(InvMassFitter.kLin)
        elif bkgStr == 'kPol2':
            BkgFunc.append(InvMassFitter.kPol2)
        elif bkgStr == 'kPol3':
            BkgFunc.append(6)
            degPol[-1] = 3
        elif bkgStr == 'kPol4':
            BkgFunc.append(6)
            degPol[-1] = 4
            if len(ptmins) > 1 and inclSecPeak[iPt] == 1:
                logger('kPol4 background function is not supported for second peak fit. Use kPol2 instead.', level='ERROR')
        elif bkgStr == 'kPow':
            BkgFunc.append(InvMassFitter.kPow)
        elif bkgStr == 'kPowEx':
            BkgFunc.append(InvMassFitter.kPowEx)
        else:
            logger(f'ERROR: only kExpo, kLin, kPol2, kPol3, kPol4, kPow, and kPowEx background functions supported. Exit.', level='ERROR')
        if bkgVnStr == 'kExpo':
            BkgFuncVn.append(InvMassFitter.kExpo)
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff0", "hVnBkgCoeff0", nPtBins, ptBinsArr))
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff1", "hVnBkgCoeff1", nPtBins, ptBinsArr))
        elif bkgVnStr == 'kConst':
            BkgFuncVn.append(InvMassFitter.kNoBk)
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff0", "hVnBkgCoeff0", nPtBins, ptBinsArr))
        elif bkgVnStr == 'kLin':
            BkgFuncVn.append(InvMassFitter.kLin)
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff0", "hVnBkgCoeff0", nPtBins, ptBinsArr))
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff1", "hVnBkgCoeff1", nPtBins, ptBinsArr))
        elif bkgVnStr == 'kPol2':
            BkgFuncVn.append(InvMassFitter.kPol2)
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff0", "hVnBkgCoeff0", nPtBins, ptBinsArr))
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff1", "hVnBkgCoeff1", nPtBins, ptBinsArr))
            hVnBkgCoeffs[iPt].append(TH1F("hVnBkgCoeff2", "hVnBkgCoeff2", nPtBins, ptBinsArr))
        else:
            logger('Only kExpo, kLin, and kPol2 background functions supported for vn. Exit.', level='ERROR')
        if sgnStr == 'kGaus':
            SgnFunc.append(InvMassFitter.kGaus)
        elif sgnStr == 'k2Gaus':
            SgnFunc.append(InvMassFitter.k2Gaus)
        elif sgnStr == 'k2GausSigmaRatioPar':
            SgnFunc.append(InvMassFitter.k2GausSigmaRatioPar)
        else:
            logger('Only kGaus, k2Gaus and k2GausSigmaRatioPar signal functions supported! Falling back to gaussian!', level='ERROR')
            SgnFunc.append(InvMassFitter.kGaus)

    # Set particle configuration
    partTitle, massAxisTit, decay, massForFit, massSecPeak, secPeakLabel = get_particle_info(particleName)

    # Load histos
    infile = TFile.Open(inFileName)
    if not infile or not infile.IsOpen():
        logger(f'File "{inFileName}" cannot be opened. Exit.', level='ERROR')
    
    hRefl, hMass, hMassForFit, hVn, hVnForFit, fTotFuncMass,\
    fTotFuncVn, fSgnFuncMass, fBkgFuncMass, fMassBkgRflFunc,\
    fMassSecPeakFunc, fBkgFuncVn, fVnSecPeakFunc, fVnCompFuncts,\
    hMCSgn, hMCRefl = ([] for _ in range(16))
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
            logger('DDifferent number of bins for this analysis and histo for fix sigma', level='WARNING')
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
    vnFitter = []
    templKdesAll = []
    templNamesPerPt = []
    legends = []
    hPulls = []
    pullLines = []
    pads = []
    hMassDraws = []
    for iPt, (hM, hV, ptMin, ptMax, reb, sgnEnum, bkgEnum, bkgVnEnum, secPeak, massMin, massMax) in enumerate(
            zip(hMass, hVn, ptmins, ptmaxs, rebins, SgnFunc, BkgFunc, BkgFuncVn, inclSecPeak, massFitLows, massFitHighs)):
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
        hMassForFit[iPt].SetTitle((f'{ptMin:0.1f} < #it{{p}}_{{T}} < {ptMax:0.1f} GeV/#it{{c}};{massAxisTit};'
                                   f'Counts per {binWidth*1000:.0f} MeV/#it{{c}}^{{2}}'))
        hMassForFit[iPt].SetName(f'MassForFit{iPt}')
        SetObjectStyle(hMassForFit[iPt], color=kBlack, markerstyle=kFullCircle, markersize=1)
        SetObjectStyle(hVnForFit[iPt], color=kBlack, markerstyle=kFullCircle, markersize=0.8)

        if not isMultitrial:
            logger(f'Processing pt {ptMin} - {ptMax} GeV/c', level='INFO')
        vnFitter.append(VnVsMassFitter(hMassForFit[iPt], hVnForFit[iPt],
                                       massMin, massMax, bkgEnum, sgnEnum, bkgVnEnum))
        vnFitter[iPt].SetHarmonic(harmonic)
        vnFitter[iPt].SetSuppressOutput(isMultitrial)

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
                vnFitter[iPt].SetInitialGaussianSigma(configfit['Sigma'][iPt], 2)
        else:
            vnFitter[iPt].SetInitialGaussianSigma(configfit['Sigma'][iPt], 1)
        # nSigma4SB
        if configfit.get('NSigma4SB'):
            if not isinstance(configfit['NSigma4SB'], list):
                configfit['NSigma4SB'] = [configfit['NSigma4SB']] * nPtBins
            vnFitter[iPt].SetNSigmaForVnSB(configfit['NSigma4SB'][iPt])
        # Second peak
        if secPeak:
            vnFitter[iPt].IncludeSecondGausPeak(massSecPeak, False, configfit['SigmaSecPeak'][iPt], False, 1, configfit.get('FixVnSecPeakToSgn', False))
            if fixSigma[iPt]:
                vnFitter[iPt].SetInitialGaussianSigma2Gaus(configfit['SigmaSecPeak'][iPt], 2)
        vnFitter[iPt].FixFrac2GausFromMassFit()
        # Reflections for D0
        if useRefl:
            Signals = hMCSgn[iPt].Integral(hMCSgn[iPt].FindBin(massMin*1.0001), hMCSgn[iPt].FindBin(massMax*0.9999))
            Reflections = hMCRefl[iPt].Integral(hMCRefl[iPt].FindBin(massMin*1.0001), hMCRefl[iPt].FindBin(massMax*0.9999))
            SoverR = Reflections / (Signals + Reflections)
            vnFitter[iPt].SetTemplateReflections(hMCRefl[iPt], reflFuncStr[iPt], massMin, massMax)
            vnFitter[iPt].SetFixReflOverS(SoverR)
            vnFitter[iPt].SetReflVnOption(0)
        # TODO: add correlated bkgs
        if configfit.get('InitBkg'):
            if configfit['InitBkg'][iPt] != []:
                vnFitter[iPt].SetBkgPars(list(itertools.chain(*configfit['InitBkg'][iPt])))

        # Correlated bkg templates
        templTf1s, templRatios, templKdes, templNames = [], [], [], []
        if useTempl:
            bkgMin, bkgMax = cutSetCfg['ScoreBkg']['min'][iPt], cutSetCfg['ScoreBkg']['max'][iPt]
            fdMin, fdMax = cutSetCfg['ScoreFD']['min'][iPt], cutSetCfg['ScoreFD']['max'][iPt]
            selString = (f"fMlScore0 > {bkgMin} && fMlScore0 < {bkgMax} && "
                         f"fMlScore1 > {fdMin} && fMlScore1 < {fdMax}")
            selStringMass = f"{selString} && fM > {massMin} && fM < {massMax}"
            ptLabel = f"pt_{int(ptMin*10)}_{int(ptMax*10)}"
            corrBkgFile = TFile.Open(f"{corrBkgCfg['InputFiles']}_{ptLabel}.root", "READ")
            _, fracSgn = get_corr_bkg(corrBkgFile, corrBkgCfg['SgnFinState'], selStringMass, ptLabel, 'raw', 'hist')
            for chn in corrBkgCfg['Channels']:
                kde, tf1, frac, integ = get_corr_bkg_tf1(corrBkgFile, chn, selString, selStringMass, ptLabel,
                                                         massMin, massMax,
                                                         min_entries=corrBkgCfg.get('MinEntries', 50))
                if tf1 is None:
                    continue
                templKdes.append(kde)
                templKdesAll.append(kde)
                templTf1s.append(tf1)
                templNames.append(chn)
                templRatios.append(frac / fracSgn / integ)
            corrBkgFile.Close()
            logger(f"pt {ptMin}-{ptMax}: built {len(templTf1s)} templates, ratios {[f'{r:.4f}' for r in templRatios]}", level='INFO')
        templNamesPerPt.append(templNames)

        # Collect fit results
        isfitGood = vnFitter[iPt].SimultaneousFit(False)

        # Second pass with correlated bkg templates anchored to the prefit signal yield
        if useTempl and isfitGood and templTf1s:
            rawYieldPrefit = vnFitter[iPt].GetRawYield()
            binWidth = hMassForFit[iPt].GetBinWidth(1)
            vecTempls = ROOT.std.vector('TF1')()
            for tf1 in templTf1s:
                vecTempls.push_back(tf1)
            vecNames = ROOT.std.vector('int')(range(len(templTf1s)))
            initW = ROOT.std.vector('double')([r * rawYieldPrefit * binWidth for r in templRatios])
            minW = ROOT.std.vector('double')([1.] * len(templTf1s))   # min > max -> fixed
            maxW = ROOT.std.vector('double')([0.] * len(templTf1s))
            vnFitter[iPt].SetKDETemplates(vecTempls, vecNames, initW, minW, maxW,
                                          initW, minW, maxW, corrBkgCfg.get('SameVnAsSignal', True))
            logger(f"pt {ptMin}-{ptMax}: refit with templates, prefit S = {rawYieldPrefit:.0f}, weights {[f'{w:.1f}' for w in initW]}", level='INFO')
            isfitGood = vnFitter[iPt].SimultaneousFit(False)

        # Try recovering fit if it failed for disappearing second peak
        if not isfitGood and secPeak:
            logger(f'Fit failed in pt bin {iPt+1}/{nPtBins}: {ptMin} - {ptMax} GeV/c. Try recovering by disabling second peak fit.', level='WARNING')
            vnFitter[iPt].ExcludeSecondGausPeak()
            isfitGood = vnFitter[iPt].SimultaneousFit(False)
            secPeak = False

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

        if isfitGood:
            vnResults = get_vnfitter_results(vnFitter[iPt], secPeak, useRefl, useTempl and bool(templTf1s))
            hSigmaSimFit.SetBinContent(iPt+1, vnResults['sigma'])
            hSigmaSimFit.SetBinError(iPt+1, vnResults['sigmaUnc'])
            hMeanSimFit.SetBinContent(iPt+1, vnResults['mean'])
            hMeanSimFit.SetBinError(iPt+1, vnResults['meanUnc'])
            hRedChi2SimFit.SetBinContent(iPt+1, vnResults['chi2'])
            hRedChi2SimFit.SetBinError(iPt+1, 1.e-20)
            hProbSimFit.SetBinContent(iPt+1, vnResults['prob'])
            hProbSimFit.SetBinError(iPt+1, 1.e-20)
            hRawYieldsSimFit.SetBinContent(iPt+1, vnResults['ry'])
            hRawYieldsSimFit.SetBinError(iPt+1, vnResults['ryUnc'])
            hRawYieldsTrueSimFit.SetBinContent(iPt+1, vnResults['ryTrue'])
            hRawYieldsTrueSimFit.SetBinError(iPt+1, vnResults['ryTrueUnc'])
            hRawYieldsSignificanceSimFit.SetBinContent(iPt+1, vnResults['signif'])
            hRawYieldsSignificanceSimFit.SetBinError(iPt+1, vnResults['signifUnc'])
            hVnSimFit.SetBinContent(iPt+1, vnResults['vn'])
            hVnSimFit.SetBinError(iPt+1, vnResults['vnUnc'])
            gVnSimFit.SetPoint(iPt, (ptMin+ptMax)/2, vnResults['vn'])
            gVnSimFit.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, vnResults['vnUnc'], vnResults['vnUnc'])
            gVnUnc.SetPoint(iPt, (ptMin+ptMax)/2, vnResults['vnUnc'])
            gVnUnc.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, 1.e-20, 1.e-20)
            hRawYieldsSoverBSimFit.SetBinContent(iPt+1, vnResults['ry']/vnResults['bkg'])

            for bkgHisto, bkgParVal, bkgParErr in zip(hVnBkgCoeffs[iPt], vnResults['bkgPars'], vnResults['bkgParsUncs']):
                bkgHisto.SetBinContent(iPt+1, bkgParVal)
                bkgHisto.SetBinError(iPt+1, bkgParErr)

            fTotFuncMass.append(vnResults['fTotFuncMass'])
            fTotFuncVn.append(vnResults['fTotFuncVn'])
            fSgnFuncMass.append(vnResults['fSgnFuncMass'])
            fBkgFuncMass.append(vnResults['fBkgFuncMass'])
            fBkgFuncVn.append(vnResults['fBkgFuncVn'])
            
            SetObjectStyle(fTotFuncMass[iPt], color=kAzure+4, linewidth=3)
            SetObjectStyle(fSgnFuncMass[iPt], fillcolor=kAzure+4, fillstyle=1000, linewidth=0, fillalpha=0.3)
            SetObjectStyle(fBkgFuncMass[iPt], color=kOrange+1, linestyle=9, linewidth=2)
            SetObjectStyle(fBkgFuncVn[iPt], color=kOrange+1, linestyle=7, linewidth=2)
            SetObjectStyle(fTotFuncVn[iPt], color=kAzure+4, linewidth=3)

            if secPeak:
                hMeanSecPeakFitMass.SetBinContent(iPt+1, vnResults['secPeakMeanMass'])
                hMeanSecPeakFitMass.SetBinError(iPt+1, vnResults['secPeakMeanMassUnc'])
                hSigmaSecPeakFitMass.SetBinContent(iPt+1, vnResults['secPeakSigmaMass'])
                hSigmaSecPeakFitMass.SetBinError(iPt+1, vnResults['secPeakSigmaMassUnc'])
                hMeanSecPeakFitVn.SetBinContent(iPt+1, vnResults['secPeakMeanVn'])
                hMeanSecPeakFitVn.SetBinError(iPt+1, vnResults['secPeakMeanVnUnc'])
                hSigmaSecPeakFitVn.SetBinContent(iPt+1, vnResults['secPeakSigmaVn'])
                hSigmaSecPeakFitVn.SetBinError(iPt+1, vnResults['secPeakSigmaVnUnc'])
                gVnSimFitSecPeak.SetPoint(iPt, (ptMin+ptMax)/2, vnResults['vnSecPeak'])
                gVnSimFitSecPeak.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2,
                                               vnResults['vnSecPeakUnc'],
                                               vnResults['vnSecPeakUnc'])
                gVnUncSecPeak.SetPoint(iPt, (ptMin+ptMax)/2, vnResults['vnSecPeakUnc'])
                gVnUncSecPeak.SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, 1.e-20, 1.e-20)
                fMassSecPeakFunc.append(vnResults['fMassSecPeakFunc'])
                fVnSecPeakFunc.append(vnResults['fVnSecPeakFunct'])
                SetObjectStyle(fMassSecPeakFunc[-1], fillcolor=kGreen+1, fillstyle=1000, linewidth=0, fillalpha=0.3)

            if useRefl:
                hRefl.append(vnResults['fMassRflFunc'])
                fMassBkgRflFunc.append(vnResults['fMassBkgRflFunc'])
                SetObjectStyle(hRefl[iPt], fillcolor=kGreen+1, fillstyle=1000, linewidth=0, fillalpha=0.3)
                SetObjectStyle(fMassBkgRflFunc[iPt], color=kRed+1, linestyle=7, linewidth=2)

            if configfit.get('DrawVnComps'):
                fVnCompFuncts.append(vnResults['fVnCompsFuncts'])

            # Draw upper pad
            cSimFit[iPt].cd(1)
            padMass = ROOT.TPad(f'padMass_{iPt}', '', 0., 0.22, 1., 1.)
            padMass.SetBottomMargin(0.02)
            padMass.Draw()
            padPull = ROOT.TPad(f'padPull_{iPt}', '', 0., 0., 1., 0.22)
            padPull.SetTopMargin(0.015)
            padPull.SetBottomMargin(0.32)
            padPull.Draw()
            pads += [padMass, padPull]

            padMass.cd()
            hMassDraw = hMassForFit[iPt].Clone(f'hMassDraw_{iPt}')
            hMassDraw.SetDirectory(0)
            hMassDraws.append(hMassDraw)
            hMassDraw.GetYaxis().SetRangeUser(0., 1.8*hMassDraw.GetMaximum())
            hMassDraw.GetYaxis().SetMaxDigits(3)
            hMassDraw.GetXaxis().SetRangeUser(massMin, massMax)
            hMassDraw.GetXaxis().SetLabelSize(0)
            hMassDraw.GetXaxis().SetTitleSize(0)
            hMassDraw.Draw('E')
            fSgnFuncMass[iPt].Draw('fc same')
            fBkgFuncMass[iPt].Draw('same')
            fTotFuncMass[iPt].Draw('same')
            legend = ROOT.TLegend(0.64, 0.46, 0.94, 0.84)
            legend.SetBorderSize(0)
            legend.SetFillStyle(0)
            legend.SetTextSize(0.028)
            legend.SetMargin(0.25)
            templFuncts = vnResults.get('fMassTemplFuncts', [])
            templsByName = {templNamesPerPt[iPt][i]: f for i, f in enumerate(templFuncts)}
            for templName in templOrder:
                templFunc = templsByName.get(templName)
                if templFunc is None:
                    continue
                templFunc.SetLineColor(templColors.get(templName, ROOT.kMagenta))
                templFunc.SetLineStyle(2)
                templFunc.SetLineWidth(2)
                templFunc.Draw('same')
                legend.AddEntry(templFunc, templLabels.get(templName, templName), 'l')
            vnResults['fBkgFuncMass'].SetLineStyle(2)
            vnResults['fBkgFuncMass'].SetLineColor(ROOT.kOrange-4)
            legend.AddEntry(vnResults['fBkgFuncMass'], 'Comb. background', 'l')
            vnResults['fSgnFuncMass'].SetFillColorAlpha(ROOT.kAzure-9, 0.7)
            vnResults['fSgnFuncMass'].SetFillStyle(1001)
            legend.AddEntry(vnResults['fSgnFuncMass'], decay, 'f')
            legend.AddEntry(fTotFuncMass[iPt], 'Total fit function', 'l')
            legend.AddEntry(hMassForFit[iPt], 'Data', 'pe')
            legend.Draw()
            legends.append(legend)
            if secPeak:
                fMassSecPeakFunc[-1].Draw('fc same')
            if useRefl:
                fMassBkgRflFunc[iPt].Draw('same')
                hRefl[iPt].Draw('same')
                latex.DrawLatex(0.18, 0.20, f'RoverS = {SoverR:.2f}')

            latex.DrawLatex(0.18, 0.80, f'#mu = {vnResults["mean"]:.3f} #pm {vnResults["meanUnc"]:.3f} GeV/c^{2}')
            latex.DrawLatex(0.18, 0.75, f'#sigma = {vnResults["sigma"]:.3f} #pm {vnResults["sigmaUnc"]:.3f} GeV/c^{2}')
            latex.DrawLatex(0.18, 0.70, f'S = {vnResults["ry"]:.0f} #pm {vnResults["ryUnc"]:.0f}')
            latex.DrawLatex(0.18, 0.65, f'S/B (3#sigma) = {vnResults["ry"]/vnResults["bkg"]:.2f}')
            latex.DrawLatex(0.18, 0.60, f'Signif. (3#sigma) = {round(vnResults["signif"], 2)}')

            if secPeak:
                latex.DrawLatex(0.18, 0.55,
                                f'#mu ({secPeakLabel}) = {vnResults["secPeakMeanMass"]:.3f} #pm {vnResults["secPeakMeanMassUnc"]:.3f} GeV/c^{2}')
                latex.DrawLatex(0.18, 0.50,
                                f'#sigma ({secPeakLabel}) = {vnResults["secPeakSigmaMass"]:.3f} #pm {vnResults["secPeakSigmaMassUnc"]:.3f} GeV/c^{2}')

            # Draw lower pad
            padPull.cd()
            hPull = hMassForFit[iPt].Clone(f'hPull_{iPt}')
            hPull.SetDirectory(0)
            hPull.Reset()
            hPull.SetStats(0)
            hPull.SetTitle('')
            for iBin in range(1, hMassForFit[iPt].GetNbinsX()+1):
                binCent = hMassForFit[iPt].GetBinCenter(iBin)
                unc = hMassForFit[iPt].GetBinError(iBin)
                if binCent < massMin or binCent > massMax or unc <= 0:
                    continue
                hPull.SetBinContent(iBin, (hMassForFit[iPt].GetBinContent(iBin)
                                           - fTotFuncMass[iPt].Eval(binCent)) / unc)
                hPull.SetBinError(iBin, 0)
            hPull.GetXaxis().SetRangeUser(massMin, massMax)
            hPull.GetYaxis().SetRangeUser(-6, 6)
            hPull.GetYaxis().SetTitle('Pull')
            hPull.GetYaxis().SetNdivisions(505)
            hPull.GetYaxis().SetTitleSize(0.15)
            hPull.GetYaxis().SetTitleOffset(0.35)
            hPull.GetYaxis().CenterTitle()
            hPull.GetYaxis().SetLabelSize(0.13)
            hPull.GetXaxis().SetTitleSize(0.15)
            hPull.GetXaxis().SetTitleOffset(1.0)
            hPull.GetXaxis().SetLabelSize(0.13)
            hPull.SetFillColor(kAzure+4)
            hPull.SetLineColor(kAzure+4)
            hPull.Draw('HIST')
            pullLine = ROOT.TLine(massMin, 0., massMax, 0.)
            pullLine.SetLineColor(ROOT.kBlack)
            pullLine.SetLineWidth(1)
            pullLine.Draw()
            hPulls.append(hPull)
            pullLines.append(pullLine)

            cSimFit[iPt].cd(2)
            hVnForFit[iPt].GetXaxis().SetRangeUser(massMin, massMax)
            hVnForFit[iPt].GetYaxis().SetTitle(f'#it{{v}}_{{{harmonic}}} (SP)')
            hVnForFit[iPt].GetYaxis().SetTitleOffset(1.4)
            hVnForFit[iPt].GetYaxis().SetDecimals()
            hVnForFit[iPt].GetYaxis().SetRangeUser(0.5*hVnForFit[iPt].GetMinimum(),
                                                    1.5*hVnForFit[iPt].GetMaximum())
            hVnForFit[iPt].Draw('E')
            fBkgFuncVn[iPt].Draw('same')
            fTotFuncVn[iPt].Draw('same')
            
            latex.DrawLatex(0.18, 0.18, f'#chi^{{2}}/ndf = {vnResults["chi2"]:.2f}')
            latex.DrawLatex(0.18, 0.80,
                            f'#it{{v}}{harmonic}({partTitle}) = {vnResults["vn"]:.3f} #pm {vnResults["vnUnc"]:.3f}')
                
            if secPeak:
                latex.DrawLatex(0.18, 0.75,
                                f'#it{{v}}{harmonic}({secPeakLabel}) = {vnResults["vnSecPeak"]:.3f} #pm {vnResults["vnSecPeakUnc"]:.3f}')
            
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
                for _, vnCompFunct in fVnCompFuncts[iPt].items():
                    vnCompFunct.Draw('same')
                    cSimFit[iCanv].Modified()
                    cSimFit[iCanv].Update()
                legVnCompn.Draw()

            cSimFit[iCanv].Modified()
            cSimFit[iCanv].Update()

            # Draw on full canvas
            massPadIndex = iPt + 1 + nCols*(iPt // nCols)
            vnPadIndex = massPadIndex + nCols
            cSimFitFull.cd(massPadIndex)
            hMassForFit[iPt].Draw('E')
            fSgnFuncMass[iPt].Draw('fc same')
            fBkgFuncMass[iPt].Draw('same')
            fTotFuncMass[iPt].Draw('same')
            if secPeak:
                fMassSecPeakFunc[-1].Draw('fc same')
                latex.DrawLatex(0.18, 0.55,
                                f'#mu ({secPeakLabel}) = {vnResults["secPeakMeanMass"]:.3f} #pm {vnResults["secPeakMeanMassUnc"]:.3f} GeV/c^{2}')
                latex.DrawLatex(0.18, 0.50,
                                f'#sigma ({secPeakLabel}) = {vnResults["secPeakSigmaMass"]:.3f} #pm {vnResults["secPeakSigmaMassUnc"]:.3f} GeV/c^{2}')
            if useRefl:
                fMassBkgRflFunc[iPt].Draw('same')
                hRefl[iPt].Draw('same')
                latex.DrawLatex(0.18, 0.20, f'RoverS = {SoverR:.2f}')
            latex.DrawLatex(0.18, 0.80, f'#mu = {vnResults["mean"]:.3f} #pm {vnResults["meanUnc"]:.3f} GeV/c^{2}')
            latex.DrawLatex(0.18, 0.75, f'#sigma = {vnResults["sigma"]:.3f} #pm {vnResults["sigmaUnc"]:.3f} GeV/c^{2}')
            latex.DrawLatex(0.18, 0.70, f'S = {vnResults["ry"]:.0f} #pm {vnResults["ryUnc"]:.0f}')
            latex.DrawLatex(0.18, 0.65, f'S/B (3#sigma) = {vnResults["ry"]/vnResults["bkg"]:.2f}')
            latex.DrawLatex(0.18, 0.60, f'Signif. (3#sigma) = {round(vnResults["signif"], 2)}')
            cSimFitFull.cd(vnPadIndex)
            hVnForFit[iPt].Draw('E')
            fBkgFuncVn[iPt].Draw('same')
            fTotFuncVn[iPt].Draw('same')
            latex.DrawLatex(0.18, 0.18, f'#chi^{{2}}/ndf = {vnResults["chi2"]:.2f}')
            latex.DrawLatex(0.18, 0.80,
                            f'#it{{v}}_{{{harmonic}}}({partTitle}) = {vnResults["vn"]:.3f} #pm {vnResults["vnUnc"]:.3f}')
            if secPeak:
                latex.DrawLatex(0.18, 0.75,
                                f'#it{{v}}_{{{harmonic}}}({secPeakLabel}) = {vnResults["vnSecPeak"]:.3f} #pm {vnResults["vnSecPeakUnc"]:.3f}')
            cSimFitFull.Modified()
            cSimFitFull.Update()

        else:
            fTotFuncMass.append(None)
            fTotFuncVn.append(None)
            fSgnFuncMass.append(None)
            fBkgFuncMass.append(None)
            fMassBkgRflFunc.append(None)
            fMassSecPeakFunc.append(None)
            fBkgFuncVn.append(None)
            fVnSecPeakFunc.append(None)
            fVnCompFuncts.append(None)

            logger(f'Fit failed for pt bin {ptMin} - {ptMax} GeV/c. Skipping.', level='WARNING')

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

    for canv in cSimFit:
        canv.Write()
    for ih, hist in enumerate(hMass):
        hist.Write(f'hist_mass_pt{ptmins[ih]*10:.0f}_{ptmaxs[ih]*10:.0f}')
    for ih, hist in enumerate(hVn):
        hist.Write(f'hist_vn_pt{ptmins[ih]*10:.0f}_{ptmaxs[ih]*10:.0f}')
    for ipt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs)):
        try:
            fTotFuncMass[ipt].Write(f'fTotFuncMass_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fTotFuncVn[ipt].Write(f'fTotFuncVn_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fSgnFuncMass[ipt].Write(f'fSgnFuncMass_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fBkgFuncMass[ipt].Write(f'fBkgFuncMass_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fBkgFuncVn[ipt].Write(f'fBkgFuncVn_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            for bkgHisto in hVnBkgCoeffs[ipt]:
                bkgHisto.Write(f'{bkgHisto.GetName()}_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
        except:
            logger(f'Fit function for pt {ptmin*10:.0f}-{ptmax*10:.0f} not available. Skipping.', level='WARNING')

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
    parser.add_argument('inFileName', metavar='text', default='')
    parser.add_argument('--batch', '-b', help='suppress video output', action='store_true')
    parser.add_argument('--multitrial', help='suppress redundant prints', action='store_true')
    args = parser.parse_args()

    get_vn_vs_mass(
        args.fitConfigFileName,
        args.inFileName,
        args.batch,
        args.multitrial
    )
