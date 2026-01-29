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
    script_dir = os.path.dirname(os.path.realpath(__file__))
    gSystem.CompileMacro(f"{script_dir}/../invmassfitter/VnVsMassFitter.cxx", "kO")
    gSystem.CompileMacro(f"{script_dir}/../invmassfitter/InvMassFitter.cxx", "kO")
    gSystem._vnfitter_loaded = True

from ROOT import InvMassFitter, VnVsMassFitter
os.sys.path.append(os.path.join(script_dir, '..', 'utils'))
from StyleFormatter import SetGlobalStyle, SetObjectStyle
from fit_utils import RebinHisto
from utils import logger, get_centrality_bins, get_vnfitter_results, get_refl_histo, get_particle_info
from correlated_bkgs import get_corr_bkg

def get_vn_vs_mass(fitConfigFileName, cutsetFileName, inFileName, batch, isMultitrial):
    #______________________________________________________
    # Read configuration file
    with open(fitConfigFileName, 'r', encoding='utf8') as ymlfitConfigFile:
        config = yaml.load(ymlfitConfigFile, yaml.FullLoader)

    print(f"Loading cutset configuration from {cutsetFileName}")
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
    fixSigma = configfit.get('FixSigma', 0)
    fixSigmaFromFile = configfit.get('FixSigmaFromFile', '')
    fixMean = configfit.get('FixMean', 0)
    inclSecPeak = configfit.get('InclSecPeak', 0)
    rebins = configfit.get('Rebin', 1)
    useRefl = configfit.get('enableRef', False)
    reflFile = configfit.get('ReflFile', '')
    reflFuncStr = configfit.get('ReflFunc', '2Gaus')

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
        elif sgnStr == 'kDoubleCBAsymm':
            SgnFunc.append(InvMassFitter.kDoubleCBAsymm)
        elif sgnStr == 'kDoubleCBSymm':
            SgnFunc.append(InvMassFitter.kDoubleCBSymm)
        elif sgnStr == 'k2GausSigmaRatioPar':
            SgnFunc.append(InvMassFitter.k2GausSigmaRatioPar)
        else:
            print('ERROR: only kGaus, k2Gaus, kDoubleCBAsymm, kDoubleCBSymm and k2GausSigmaRatioPar signal functions supported! Exit!')
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
    hMCSgn, hMCRefl, hPulls, hPullsPrefit = ([] for _ in range(18))

    useTemplates = True if config.get('corr_bkgs') else False
    fMassTemplFuncts = [[None]*nPtBins] if useTemplates else []
    fMassTemplTotFuncts = [None]*nPtBins if useTemplates and (particleName == 'Dplus' or particleName == 'Ds') else []

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
    # gVnTempls = [TGraphAsymmErrors(1) for _ in range(len(corrBkgsTemplates))] if useTemplates else []
    # gVnTemplsUncs = [TGraphAsymmErrors(1) for _ in range(len(corrBkgsTemplates))] if useTemplates else []
    SetObjectStyle(gVnSimFit, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(gVnSimFitSecPeak, color=kRed, markerstyle=kOpenCircle)
    SetObjectStyle(gVnUnc, color=kBlack, markerstyle=kFullCircle)
    SetObjectStyle(gVnUncSecPeak, color=kRed, markerstyle=kOpenCircle)
    # for gvnTempl, gvnTemplUnc in zip(gVnTempls, gVnTemplsUncs):
    #     SetObjectStyle(gvnTempl, color=kBlack, markerstyle=kFullCircle)
    #     SetObjectStyle(gvnTemplUnc, color=kRed, markerstyle=kOpenCircle)

    # Create canvases
    cSimFit, cInvMassPrefits = [], []
    for i in range(nPtBins):
        ptLow = ptmins[i]
        ptHigh = ptmaxs[i]
        cSimFit.append(TCanvas(f'cSimFit_pt{ptLow}_{ptHigh}', f'cSimFit_pt{ptLow}_{ptHigh}', 400, 900))
        cInvMassPrefits.append(TCanvas(f'cMassPrefit_Pt{ptLow}_{ptHigh}', f'cMassPrefit_Pt{ptLow}_{ptHigh}', 400, 900))
        cSimFit[-1].Divide(1, 2)
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    canvVn = TCanvas('cVn', 'cVn', 900, 900)
    canvVnUnc = TCanvas('canvVnUnc', 'canvVnUnc', 900, 900)

    #_____________________________________________________
    # Vn estimation with Scalar Product
    vnFitter = []
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

        # Check if to use correlated bkgs templates
        cfg_corr_bkgs = config.get('corr_bkgs', None)
        if cfg_corr_bkgs:
            for cocktail in cfg_corr_bkgs['cocktails']:
                if [ptMin, ptMax] in cocktail['pt_ranges']:
                    useTemplatesPtBin = True
                    cocktail_cfg = cocktail
                    break

        if useTemplatesPtBin:
            pt_dir = f"pt_{ptMin*10:.0f}_{ptMax*10:.0f}"
            corr_bkg_file = TFile.Open(f"{config['outdir']}/corrbkgs/templs_{pt_dir}.root", 'r')
            sel_string = f"fMlScore0 < {cfg_cutset['ScoreBkg']['max'][iPt]} && fMlScore1 > {cfg_cutset['ScoreFD']['min'][iPt]}" \
                         f" && fMlScore1 < {cfg_cutset['ScoreFD']['max'][iPt]}"
            sgn_hist, sgn_frac = get_corr_bkg(corr_bkg_file, cfg_corr_bkgs['sgn_chn'], sel_string, pt_dir, "raw", "hist", verbose=True)
            templates = {}
            for chn in cocktail_cfg['channels']:
                templates[chn] = {}
                print(f"Getting correlated bkg template for channel {chn} .....")
                histo, frac = get_corr_bkg(corr_bkg_file, chn, sel_string, pt_dir, "raw", "hist",
                                                         config['Dmeson'], configfit.get('CorrectAbundances', False),
                                                         verbose=True)
                histo.SetDirectory(0)
                templates[chn]['histo'] = histo
                templates[chn]['frac'] = frac
            
            # Normalize fractions with respect to sgn chn
            print(f"templates: {templates}")
            for templ_name, templ_dict in templates.items():
                templ_dict['frac'] /= sgn_frac

            print(f"Using correlated bkgs templates for pt {ptMin} - {ptMax} GeV/c:")
            print(f"templates: {templates}")

            print(f"Setting template parameters .....")
            vnFitter[iPt].SetTemplatesHisto([templ['histo'] for templ in templates.values()],
                                            [templ['frac'] for templ in templates.values()],
                                            cfg_corr_bkgs.get('anchor_mode', 2))
            print("Histo templates set!")
            corr_bkg_file.Close()

        # Initial fit parameters
        if configfit.get('InitFitPars') and configfit['InitFitPars'][iPt] != []:
            initPars = configfit['InitFitPars'][iPt]
        else:
            initPars = []

        # Retrieve histogram to fix signal
        if configfit.get("FixSgnFromMC") or configfit.get("InitSgnFromMC"):
            parFile = TFile.Open(configfit['FixSgnFromMC'], 'r')
            sgnParsFromHisto = []
            for key in parFile.GetListOfKeys():
                # pick histograms containing pt-dependent parameters
                if "hist_" in key.GetName():
                    parHist = key.ReadObj()
                    parValue = parHist.GetBinContent(iPt+1)
                    parName = convert_flarefly_par_name(key.GetName().replace("hist_", ""))
                    if configfit.get("FixSgnFromMC") and parName is not None:
                        lowLim = 0 if parName not in ["Mean", "Sigma"] else parValue-(0.3*parValue)
                        upLim = -1 if parName not in ["Mean", "Sigma"] else parValue+(0.3*parValue)
                        sgnParsFromHisto.append([parName, parValue, lowLim, upLim])
                    elif configfit.get("InitSgnFromMC") and parName is not None: 
                        lowLim = 0 if parName not in ["Mean", "Sigma"] else parValue-(configfit["InitSgnFromMC"]*parValue)
                        upLim = -1 if parName not in ["Mean", "Sigma"] else parValue+(configfit["InitSgnFromMC"]*parValue)
                        sgnParsFromHisto.append([parName, parValue, lowLim, upLim])
                    else:
                        logger(f'Parameter name {parName} not recognized for signal fit. Skip it.', level='WARNING')
            initPars.extend(sgnParsFromHisto)

        print(f"\ninit pars: {initPars}\n")
        # Collect fit results

        # if configfit.get('InitFitPars') and configfit['InitFitPars'][iPt] != []:
            # vnFitter[iPt].SetInitPars(configfit['InitFitPars'][iPt])
        vnFitter[iPt].SetInitPars(initPars)

        isfitGood = vnFitter[iPt].SimultaneousFit(False)
        print(f"Fit status for pt {ptMin} - {ptMax} GeV/c: {isfitGood}")

        # Try recovering fit if it failed for disappearing second peak
        if not isfitGood and secPeak:
            logger(f'Fit failed in pt bin {iPt+1}/{nPtBins}: {ptMin} - {ptMax} GeV/c. Try recovering by disabling second peak fit.', level='WARNING')
            vnFitter[iPt].ExcludeSecondGausPeak()
            isfitGood = vnFitter[iPt].SimultaneousFit(False)
            secPeak = False

        print(f"Fit status for pt {ptMin} - {ptMax} GeV/c after recovery: {isfitGood}")
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
        print("Fit completed.")
        if isfitGood:
            print("Getting fit results ...")
            vnResults = get_vnfitter_results(vnFitter[iPt], secPeak, useRefl, useTemplatesPtBin)
            print(f"vnResults: {vnResults}")
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
            hPulls.append(vnResults['pulls'])

            print("Appending total mass function ...")
            fTotFuncMass.append(vnResults['fTotFuncMass'])
            fTotFuncVn.append(vnResults['fTotFuncVn'])
            fSgnFuncMass.append(vnResults['fSgnFuncMass'])
            fBkgFuncMass.append(vnResults['fBkgFuncMass'])
            fBkgFuncVn.append(vnResults['fBkgFuncVn'])

            print(f"fTotFuncMass: {fTotFuncMass}")
            print(f"iPt: {iPt}")
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

            print("Drawing fit results ...")
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

            print("Drawing templates ...")
            if useTemplatesPtBin:
                fMassTemplFuncts[iPt] = vnResults['fMassTemplFuncts']
                fMassTemplTotFuncts[iPt] = vnResults['fMassTemplTotFunc']
                # REVIEW: I would suggest to use the append here

                print(f"Filling TemplOverSgn: {vnFitter[iPt].GetTemplOverSig()}")
                hTemplOverSgn.SetBinContent(iPt+1, vnFitter[iPt].GetTemplOverSig())
                print(f"vnResults['vnTemplates']: {vnResults['vnTemplates']}")
                # for iTempl, (templVn, templVnUnc) in enumerate(zip(vnResults["vnTemplates"], vnResults["vnTemplatesUncs"])):
                #     gVnTempls[iTempl].SetPoint(iPt, (ptMin+ptMax)/2, templVn)
                #     gVnTempls[iTempl].SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, 1.e-20, 1.e-20)
                #     gVnTemplsUncs[iTempl].SetPoint(iPt, (ptMin+ptMax)/2, templVnUnc)
                #     gVnTemplsUncs[iTempl].SetPointError(iPt, (ptMax-ptMin)/2, (ptMax-ptMin)/2, 1.e-20, 1.e-20)

                SetObjectStyle(fMassTemplTotFuncts[iPt], color=kRed, linewidth=2)
                fMassTemplTotFuncts[iPt].Draw('same')
                cSimFit[iCanv].Modified()
                cSimFit[iCanv].Update()
                if configfit.get("DrawSingleTempls"):
                    for iMassTemplFunct, massTemplFunct in enumerate(fMassTemplFuncts[iPt]):
                        SetObjectStyle(massTemplFunct, color=kMagenta+2+iMassTemplFunct*2, linewidth=3)
                        massTemplFunct.Draw('same')
                        cSimFit[iCanv].Modified()
                        cSimFit[iCanv].Update()
            # Draw lower pad
            cSimFit[iPt].cd(2)
            hVnForFit[iPt].GetXaxis().SetRangeUser(massMin, massMax)
            hVnForFit[iPt].GetYaxis().SetTitle(f'#it{{v}}_{{{harmonic}}} (SP)')
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
            
            if useTemplatesPtBin:
                # for iVnTempl, (vnCoeff, vnCoeffUnc) in enumerate(zip(vnResults["vnTemplates"], vnResults["vnTemplatesUncs"])):
                #     latex.DrawLatex(0.18, 0.70-iVnTempl*0.05,
                #                 f'#it{{v}}{harmonic}(Templ{iVnTempl}) = {vnCoeff:.3f} #pm {vnCoeffUnc:.3f}')
                    cSimFit[iCanv].Modified()
                    cSimFit[iCanv].Update()
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
                    cSimFit[iCanv].Modified()
                    cSimFit[iCanv].Update()
                legVnCompn.Draw()

            cSimFit[iCanv].Modified()
            cSimFit[iCanv].Update()

        invMassPrefit = vnFitter[iPt].GetMassPrefitObject()
        hPullsPrefit.append(invMassPrefit.GetPullDistribution())
        histoMassPrefit = invMassPrefit.GetHistoClone()
        totFuncMassPrefit = invMassPrefit.GetMassFunc()
        bkgFuncMassPrefit = invMassPrefit.GetBackgroundRecalcFunc()
        sgnFuncMassPrefit = invMassPrefit.GetSignalFunc()
        cInvMassPrefits[iPt] = TCanvas(f"cMass_{ptMin*10:.0f}_{ptMax*10:.0f}", f"Mass Fit {ptMin}-{ptMax} GeV/c", 800, 600)
        histoMassPrefit.SetStats(0)
        histoMassPrefit.Draw("E")
        bkgFuncMassPrefit.SetLineColor(kGreen+2)
        bkgFuncMassPrefit.SetLineWidth(2)
        bkgFuncMassPrefit.SetLineWidth(3)
        bkgFuncMassPrefit.Draw("same")
        sgnFuncMassPrefit.SetLineColor(kBlue)
        sgnFuncMassPrefit.SetLineWidth(2)
        sgnFuncMassPrefit.SetLineWidth(3)
        sgnFuncMassPrefit.Draw("same")
        if useTemplatesPtBin:
            templFuncMassPrefit = invMassPrefit.GetTemplFunc()
            print(f"invMassPrefit.GetTemplOverSig(): {invMassPrefit.GetTemplOverSig()}")
            print(f"vnFitter[iPt].GetTemplOverSig(): {vnFitter[iPt].GetTemplOverSig()}")
            templFuncMassPrefit.SetLineColor(kMagenta)
            templFuncMassPrefit.SetLineWidth(2)
            templFuncMassPrefit.SetLineWidth(3)
            templFuncMassPrefit.Draw("same")

        totFuncMassPrefit.SetLineColor(kRed)
        totFuncMassPrefit.SetLineWidth(2)
        totFuncMassPrefit.SetLineWidth(3)
        totFuncMassPrefit.Draw("same")

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
    for hist in hPulls:
        hist.Write('hist_pulls')
    # for hist in hPullsPrefit:
    #     hist.Write('hist_pulls_prefit')
    for ipt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs)):
        try:
            fTotFuncMass[ipt].Write(f'fTotFuncMass_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fTotFuncVn[ipt].Write(f'fTotFuncVn_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fSgnFuncMass[ipt].Write(f'fSgnFuncMass_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fBkgFuncMass[ipt].Write(f'fBkgFuncMass_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
            fBkgFuncVn[ipt].Write(f'fBkgFuncVn_pt{ptmin*10:.0f}_{ptmax*10:.0f}')
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
    hTemplOverSgn.Write()

    gVnSimFit.Write()
    gVnUnc.Write()
    if secPeak:
        gVnSimFitSecPeak.Write()
        gVnUncSecPeak.Write()
    # if useTemplates:
    #     for gvnTempl, gvnTemplUnc in zip(gVnTempls, gVnTemplsUncs):
    #         gvnTempl.Write('gvnTempl')
    #         gvnTemplUnc.Write('gvnTemplUnc')

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
