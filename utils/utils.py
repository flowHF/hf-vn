import shutil
import os
import sys
import ROOT
import ctypes
from itertools import combinations
from PIL import Image
import math
import glob
import re
from ROOT import TH1, TH2, TH3
import numpy as np

def check_dir(dir):

	if not os.path.exists(dir):
		print(f"\033[32m{dir} does not exist, it will be created\033[0m")
		os.makedirs(dir)
	else:
		print(f"\033[33m{dir} already exists, it will be overwritten\033[0m")
		shutil.rmtree(dir)
		os.makedirs(dir)

	return

def logger(message, level='INFO'):
	"""
	Function to log messages with different levels.
	Args:
		message (str): The message to log.
		level (str): The level of the message ('INFO', 'WARNING', 'ERROR').
	"""
	message = f"[{level}] {message}"
	if level == 'INFO':
		print(f"\033[32m{message}\033[0m")
	elif level == 'WARNING':
		print(f"\033[33m{message}\033[0m")
	elif level == 'ERROR':
		print(f"\033[31m{message}\033[0m")
		sys.exit(1)
	elif level == 'COMMAND':
		print(f"\033[35m{message}\033[0m")
	elif level == 'DEBUG':
		print(f"\033[34m{message}\033[0m")
	elif level == 'PAUSE':
		input(f"\033[36m{message}\n{level}: Press Enter to continue.\033[0m")
	else:
		print(f"\033[37m{message}\033[0m")  # Default to white for unknown levels

def make_dir_root_file(dir, file):
    if not file.GetDirectory(dir):
        file.mkdir(dir)
		
def make_dir_root_file(directory, file):
    if not file.GetDirectory(directory):
        file.mkdir(directory)
        logger(f"Created directory {directory} in file {file.GetName()}", level='WARNING')
    else:
        logger(f"Directory {directory} already exists in file {file.GetName()}", level='WARNING')

def profile_mass_sp(hist_mass_sp, inv_mass_bins, resolution):
    '''
    Profile the mass sparse to get vn versus mass
    Input:
        - hist_mass_sp:
            THnSparse, input THnSparse object (already projected in centrality and pt)
        - inv_mass_bins:
            list of floats, bin edges for the mass axis
        - resolution:
            float, resolution to normalize the vn values
    Output:
        - hist_vn_vs_mass:
            TH1D, histogram with vn as a function of mass
    '''
    hist_vn_vs_mass = ROOT.TH1D('hist_vn_vs_mass', 'hist_vn_vs_mass', len(inv_mass_bins)-1, np.array(inv_mass_bins))
    hist_vn_vs_mass.SetDirectory(0)
    for i in range(hist_vn_vs_mass.GetNbinsX()):
        bin_low = hist_mass_sp.GetXaxis().FindBin(inv_mass_bins[i])
        bin_high = hist_mass_sp.GetXaxis().FindBin(inv_mass_bins[i+1])
        profile = hist_mass_sp.ProfileY(f'profile_{bin_low}_{bin_high}', bin_low, bin_high)
        mean_sp = profile.GetMean()
        mean_sp_err = profile.GetMeanError()
        hist_vn_vs_mass.SetBinContent(i+1, mean_sp / resolution)
        hist_vn_vs_mass.SetBinError(i+1, mean_sp_err / resolution)
    return hist_vn_vs_mass

def get_vn_versus_mass(sparse, inv_mass_bins, mass_axis, vn_axis, debug=False):
    '''
    Project vn versus mass

    Input:
        - sparse:
            THnSparse, input THnSparse object (already projected in centrality and pt)
        - inv_mass_bins:
            list of floats, bin edges for the mass axis
        - mass_axis:
            int, axis number for mass
        - vn_axis:
            int, axis number for vn
        - debug:
            bool, if True, create a debug file with the projections (default: False)

    Output:
        - hist_mass_proj:
            TH1D, histogram with vn as a function of mass
    '''
    hist_mass_sp_proj = sparse.Projection(vn_axis, mass_axis)
    hist_mass_sp_proj.SetName('hist_mass_sp_proj')
    hist_mass_sp_proj.SetDirectory(0)

    hist_mass_proj = sparse.Projection(mass_axis)
    hist_mass_proj.Reset()
    vn_vs_mass_bins = np.array(inv_mass_bins)
    hist_mass_proj = ROOT.TH1D('hist_mass_proj', 'hist_mass_proj', len(vn_vs_mass_bins)-1, vn_vs_mass_bins)

    for i in range(hist_mass_proj.GetNbinsX()):
        bin_low = hist_mass_sp_proj.GetXaxis().FindBin(vn_vs_mass_bins[i])
        bin_high = hist_mass_sp_proj.GetXaxis().FindBin(vn_vs_mass_bins[i+1])
        profile = hist_mass_sp_proj.ProfileY(f'profile_{bin_low}_{bin_high}', bin_low, bin_high)
        mean_sp = profile.GetMean()
        mean_sp_err = profile.GetMeanError()
        hist_mass_proj.SetBinContent(i+1, mean_sp)
        hist_mass_proj.SetBinError(i+1, mean_sp_err)

    if debug:
        outfile = ROOT.TFile('debug.root', 'RECREATE')
        hist_mass_sp_proj.Write()
        hist_mass_proj.Write()
        outfile.Close()

    return hist_mass_proj

def get_vnfitter_results(vnFitter, secPeak, useRefl, useTempl):
    '''
    Get vn fitter results:
    0: BkgInt
    1: BkgSlope
    2: SgnInt
    3: Mean
    4: Sigma
    5: SecPeakInt
    6: SecPeakMean
    7: SecPeakSigma
    8: ConstVnBkg
    9: SlopeVnBkg
    10: v2Sgn
    11: v2SecPeak
    12: reflection

    Input:
        - vnfitter:
            VnVsMassFitter, vn fitter object
        - secPeak:
            bool, if True, save secondary peak results
        - useRefl:
            bool, if True, save the results with reflection

    Output:
        - vn_results:
            dict, dictionary with vn results
            vn: vn value
            vnUnc: uncertainty of vn value
            mean: mean value
            meanUnc: uncertainty of mean value
            sigma: sigma value
            sigmaUnc: uncertainty of sigma value
            ry: raw yield
            ryUnc: uncertainty of raw yield
            ryTrue: true raw yield
            ryTrueUnc: uncertainty of true raw yield
            signif: significance
            signifUnc: uncertainty of significance
            chi2: reduced chi2
            prob: fit probability
            fTotFuncMass: total fit function for mass
            fTotFuncVn: total fit function for vn
            secPeakMeanMass: secondary peak mean mass
            secPeakMeanMassUnc: uncertainty of secondary peak mean mass
            secPeakSigmaMass: secondary peak sigma mass
            secPeakSigmaMassUnc: uncertainty of secondary peak sigma mass
            secPeakMeanVn: secondary peak mean vn
            secPeakMeanVnUnc: uncertainty of secondary peak mean vn
            secPeakSigmaVn: secondary peak sigma vn
            secPeakSigmaVnUnc: uncertainty of secondary peak sigma vn
            vnSecPeak: vn secondary peak
            vnSecPeakUnc: uncertainty of vn secondary peak
            fMassRflFunc: mass reflection function
            fMassBkgRflFunc: mass background reflection function
            fVnSecPeakFunct: vn secondary peak function
            fVnCompsFuncts: dictionary with vn components functions
            fMassTemplFuncts: dictionary with mass template functions
            vnTemplates: list of vn templates
            vnTemplatesUncs: list of vn templates uncertainties
    '''
    vn_results = {}
    vn_results['vn'] = vnFitter.GetVn()
    vn_results['vnUnc'] = vnFitter.GetVnUncertainty()
    vn_results['mean'] = vnFitter.GetMean()
    vn_results['meanUnc'] = vnFitter.GetMeanUncertainty()
    vn_results['sigma'] = vnFitter.GetSigma()
    vn_results['sigmaUnc'] = vnFitter.GetSigmaUncertainty()
    vn_results['ry'] = vnFitter.GetRawYield()
    vn_results['ryUnc'] = vnFitter.GetRawYieldUncertainty()
    vn_results['chi2'] = vnFitter.GetReducedChiSquare()
    vn_results['prob'] = vnFitter.GetFitProbability()
    vn_results['fTotFuncMass'] = vnFitter.GetMassTotFitFunc()
    vn_results['fTotFuncVn'] = vnFitter.GetVnVsMassTotFitFunc()
    vn_results['fBkgFuncMass'] = vnFitter.GetMassBkgFitFunc()
    vn_results['fBkgFuncVn'] = vnFitter.GetVnVsMassBkgFitFunc()
    vn_results['fSgnFuncMass'] = vnFitter.GetMassSignalFitFunc()

    vn_results['fVnCompsFuncts'] = {}
    vn_comps = vnFitter.GetVnCompsFuncts()
    vn_results['fVnCompsFuncts']['vnSgn'] = vn_comps[0]
    vn_results['fVnCompsFuncts']['vnBkg'] = vn_comps[1]
    if secPeak:
        vn_results['fVnCompsFuncts']['vnSecPeak'] = vn_comps[2]

    bkg, bkgUnc = ctypes.c_double(), ctypes.c_double()
    vnFitter.Background(3, bkg, bkgUnc)
    vn_results['bkg'] = bkg.value
    vn_results['bkgUnc'] = bkgUnc.value
    sgn, sgnUnc = ctypes.c_double(), ctypes.c_double()
    vnFitter.Signal(3, sgn, sgnUnc)
    vn_results['ryTrue'] = sgn.value
    vn_results['ryTrueUnc'] = sgnUnc.value
    signif, signifUnc = ctypes.c_double(), ctypes.c_double()
    vnFitter.Significance(3, signif, signifUnc)
    vn_results['signif'] = signif.value
    vn_results['signifUnc'] = signifUnc.value

    massSgnPars = vnFitter.GetNMassSgnPars()
    massBkgPars = vnFitter.GetNMassBkgPars()
    massSecPeakPars = vnFitter.GetNMassSecPeakPars()
    massReflPars = vnFitter.GetNMassReflPars()
    totMassPars = massSgnPars + massBkgPars + massSecPeakPars +  massReflPars
    vnSgnPars = vnFitter.GetNVnSgnPars()
    vnBkgPars = vnFitter.GetNVnBkgPars()

    if secPeak:
        vn_results['fMassSecPeakFunc'] = vnFitter.GetMassSecPeakFunc()
        vn_results['fVnSecPeakFunct'] = vnFitter.GetVnSecPeakFunc()
        vn_results['secPeakMeanMass'] = vn_results['fTotFuncMass'].GetParameter(vn_results['fTotFuncMass'].GetParName(massSgnPars + massBkgPars + 1))
        vn_results['secPeakMeanMassUnc'] = vn_results['fTotFuncMass'].GetParError(massSgnPars + massBkgPars + 1)
        vn_results['secPeakSigmaMass'] = vn_results['fTotFuncMass'].GetParameter(vn_results['fTotFuncMass'].GetParName(massSgnPars + massBkgPars + 2))
        vn_results['secPeakSigmaMassUnc'] = vn_results['fTotFuncMass'].GetParError(massSgnPars + massBkgPars + 2)
        vn_results['secPeakMeanVn'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(totMassPars + vnSgnPars + vnBkgPars + 1))
        vn_results['secPeakMeanVnUnc'] = vn_results['fTotFuncVn'].GetParError(vnSgnPars + vnBkgPars + 1)
        vn_results['secPeakSigmaVn'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(totMassPars + vnSgnPars + vnBkgPars + 2))
        vn_results['secPeakSigmaVnUnc'] = vn_results['fTotFuncVn'].GetParError(vnSgnPars + vnBkgPars + 2)
        vn_results['vnSecPeak'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(totMassPars + vnSgnPars + vnBkgPars))
        vn_results['vnSecPeakUnc'] = vn_results['fTotFuncVn'].GetParError(totMassPars + vnSgnPars + vnBkgPars)

    if useRefl:
        vn_results['fMassRflFunc'] = vnFitter.GetMassRflFunc()
        vn_results['fMassBkgRflFunc'] = vnFitter.GetMassBkgRflFunc()
    
    if useTempl:
        vn_results['vnTemplates'] = list(vnFitter.GetVnTemplates())
        vn_results['vnTemplatesUncs'] = list(vnFitter.GetVnTemplatesUncertainties())

    return vn_results

def get_particle_info(particleName):
    '''
    Get particle information

    Input:
        - particleName: 
            the name of the particle

    Output:
        - particleTit: 
            the title of the particle
        - massAxisTit: 
            the title of the mass axis
        - decay: 
            the decay of the particle
        - massForFit: 
            float, the mass of the particle
    '''

    if particleName == 'Dplus':
        particleTit = 'D^{+}'
        massAxisTit = '#it{M}(K#pi#pi) (GeV/#it{c}^{2})'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(411).Mass()
        decay = 'D^{+} #rightarrow K^{#minus}#pi^{+}#pi^{+}'
        massSecPeak = ROOT.TDatabasePDG.Instance().GetParticle(413).Mass() # D* mass
        secPeakLabel = 'D^{*+}'
    elif particleName == 'Ds':
        particleTit = 'D_{s}^{+}'
        massAxisTit = '#it{M}(KK#pi) (GeV/#it{c}^{2})'
        decay = 'D_{s}^{+} #rightarrow #phi#pi^{+} #rightarrow K^{+}K^{#minus}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(431).Mass()
        massSecPeak = ROOT.TDatabasePDG.Instance().GetParticle(411).Mass() # D+ mass
        secPeakLabel = 'D^{+}'
    elif particleName == 'LctopKpi':
        particleTit = '#Lambda_{c}^{+}'
        massAxisTit = '#it{M}(pK#pi) (GeV/#it{c}^{2})'
        decay = '#Lambda_{c}^{+} #rightarrow pK^{#minus}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(4122).Mass()
    elif particleName == 'LctopK0s':
        massAxisTit = '#it{M}(pK^{0}_{s}) (GeV/#it{c}^{2})'
        decay = '#Lambda_{c}^{+} #rightarrow pK^{0}_{s}'
        massForFit = 2.25 # please calfully check the mass of Lc->pK0s, it is constant
        # massForFit = ROOT.TDatabasePDG.Instance().GetParticle(4122).Mass()
    elif particleName == 'Dstar':
        particleTit = 'D^{*+}'
        massAxisTit = '#it{M}(K#pi#pi) - #it{M}(K#pi) (GeV/#it{c}^{2})'
        decay = 'D^{*+} #rightarrow D^{0}#pi^{+} #rightarrow K^{#minus}#pi^{+}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(413).Mass() - ROOT.TDatabasePDG.Instance().GetParticle(421).Mass()
    elif particleName == 'Dzero':
        particleTit = 'D^{0}'
        massAxisTit = '#it{M}(K#pi) (GeV/#it{c}^{2})'
        decay = 'D^{0} #rightarrow K^{#minus}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(421).Mass()
    elif particleName == 'Xic':
        particleTit = 'X_{c}^{+}'
        massAxisTit = '#it{M}(pK#pi) (GeV/#it{c}^{2})'
        decay = 'X_{c}^{+} #rightarrow pK^{#minus}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(4132).Mass()
        massSecPeak = ROOT.TDatabasePDG.Instance().GetParticle(4122).Mass() # Lc mass
        secPeakLabel = '#Lambda_{c}^{+}'
    else:
        print(f'ERROR: the particle "{particleName}" is not supported! Choose between Dzero, Dplus, Ds, Dstar, and Lc. Exit!')
        sys.exit()

    return particleTit, massAxisTit, decay, massForFit, massSecPeak if 'massSecPeak' in locals() else None, secPeakLabel if 'secPeakLabel' in locals() else None

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

def check_file_exists(file_path):
    '''
    Check if file exists

    Input:
        - file_path:
            str, file path

    Output:
        - file_exists:
            bool, if True, file exists
    '''
    file_exists = False
    if os.path.exists(file_path):
        file_exists = True
    return file_exists

def check_histo_exists(file, histo_name):
    '''
    Check if histogram exists in file

    Input:
        - file:
            TFile, ROOT file
        - histo_name:
            str, histogram name

    Output:
        - histo_exists:
            bool, if True, histogram exists
    '''
    if not check_file_exists(file):
        return False
    file = ROOT.TFile(file, 'READ')
    histo_exists = False
    if file.Get(histo_name):
        histo_exists = True
    return histo_exists
def get_centrality_bins(centrality):
    '''
    Get centrality bins

    Input:
        - centrality:
            str, centrality class (e.g. 'k3050')

    Output:
        - cent_bins:
            list of floats, centrality bins
        - cent_label:
            str, centrality label
    '''
    if centrality == 'k05':
        return '0_5', [0, 5]
    if centrality == 'k510':
        return '5_10', [5, 10]
    if centrality == 'k010':
        return '0_10', [0, 10]
    if centrality == 'k1015':
        return '10_15', [10, 15]
    if centrality == 'k1520':
        return '15_20', [15, 20]
    if centrality == 'k1020':
        return '10_20', [10, 20]
    if centrality == 'k1030':
        return '10_30', [10, 30]
    if centrality == 'k020':
        return '0_20', [0, 20]
    if centrality == 'k1030':
        return '10_30', [10, 30]
    if centrality == 'k2030':
        return '20_30', [20, 30]
    elif centrality == 'k2050':
        return '20_50', [20, 50]
    elif centrality == 'k3040':
        return '30_40', [30, 40]
    elif centrality == 'k3050':
        return '30_50', [30, 50]
    elif centrality == 'k4050':
        return '40_50', [40, 50]
    elif centrality == 'k2060':
        return '20_60', [20, 60]
    elif centrality == 'k4060':
        return '40_60', [40, 60]
    elif centrality == 'k4080':
        return '40_80', [40, 80]
    elif centrality == 'k5060':
        return '50_60', [50, 60]
    elif centrality == 'k5080':
        return '50_80', [50, 80]
    elif centrality == 'k50100':
        return '50_100', [50, 100]
    elif centrality == 'k6070':
        return '60_70', [60, 70]
    elif centrality == 'k6080':
        return '60_80', [60, 80]
    elif centrality == 'k7080':
        return '70_80', [70, 80]
    elif centrality == 'k0100':
        return '0_100', [0, 100]
    else:
        print(f"ERROR: cent class \'{centrality}\' is not supported! Exit")
    sys.exit()




def get_resolution(dets, det_lables, cent_min_max):
    '''
    Compute resolution for SP method
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

def getListOfHisots(an_res_file, wagon_id):
    '''
    Get list of histograms for SP resolution
    Now only SP method
    '''
    # Build the path - only SP method now
    infile_path = f'hf-task-flow-charm-hadrons'
    if wagon_id:
        infile_path = f'{infile_path}_id{wagon_id}'
    infile_path = f'{infile_path}/spReso'
    prefix = 'hSpReso'

    print(f"🔍 Looking for directory in ROOT file: {infile_path}")

    infile = ROOT.TFile(an_res_file, 'READ')
    directory = infile.GetDirectory(infile_path)

    if not directory:
        print(f"❌ ERROR: Directory '{infile_path}' not found in file {an_res_file}")
        print("📁 Available top-level keys:")
        infile.ls()
        return [], []

    histos = [key.ReadObj() for key in directory.GetListOfKeys()]
    for histo in histos:
        histo.SetDirectory(0)
    pairs = [key.GetName() for key in directory.GetListOfKeys()]

    triplets = list(combinations(pairs, 3))
    histo_triplets = list(combinations(histos, 3))
    correct_histo_triplets = []
    correct_histo_labels = []
    detsA = ['FT0c', 'FT0a', 'FV0a', 'TPCpos', 'FT0m', 'TPCneg']

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

def compute_resolution(subMean):
    '''
    Compute resolution for SP method
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

# FIXED: Removed compute_r2 function as it was EP-specific

# FIXED: Removed get_invmass_vs_deltaphi function as it was EP-specific

# FIXED: Removed get_ep_vn function as it was EP-specific

def get_cut_sets(npt_bins, sig_cut, bkg_cut_maxs, correlated_cuts=True):
    '''
    Get cut sets
    '''
    nCutSets = []
    sig_cuts_lower, sig_cuts_upper, bkg_cuts_lower, bkg_cuts_upper = {}, {}, {}, {}
    if correlated_cuts:
        sig_cut_mins = sig_cut['min']
        sig_cut_maxs = sig_cut['max']
        sig_cut_steps = sig_cut['step']

        # compute the signal cutsets for each pt bin
        sig_cuts_lower = [list(np.arange(sig_cut_mins[iPt], sig_cut_maxs[iPt], sig_cut_steps[iPt])) for iPt in range(npt_bins)]
        sig_cuts_upper = [[1.0 for _ in range(len(sig_cuts_lower[iPt]))] for iPt in range(npt_bins)]

        # compute the ncutsets by signal cut for each pt bin
        nCutSets = [len(sig_cuts_lower[iPt]) for iPt in range(npt_bins)]

        # bkg cuts lower edge should always be 0
        bkg_cuts_lower = [[0. for _ in range(nCutSets[iPt])] for iPt in range(npt_bins)]
        bkg_cuts_upper = [[bkg_cut_maxs[iPt] for _ in range(nCutSets[iPt])] for iPt in range(npt_bins)]

    else:
        # load the signal cut
        sig_cuts_lower = [sig_cut[iPt]['min'] for iPt in range(npt_bins)]
        sig_cuts_upper = [sig_cut[iPt]['max'] for iPt in range(npt_bins)]
        
        # compute the ncutsets by the signal cut for each pt bin
        nCutSets = [len(sig_cuts_lower[iPt]) for iPt in range(npt_bins)]
        
        # load the background cut
        bkg_cuts_lower = [[0. for _ in range(nCutSets[iPt])] for iPt in range(npt_bins)]
        # different max bkg cuts for different cutsets, list of the max bkg cuts given
        bkg_cuts_upper = [bkg_cut_maxs[iPt] for iPt in range(npt_bins)]
        
    # safety check

    for iPt in range(npt_bins):
        assert len(sig_cuts_lower[iPt]) == len(sig_cuts_upper[iPt]) == len(bkg_cuts_lower[iPt]) == len(bkg_cuts_upper[iPt]) == nCutSets[iPt], (
            f"Mismatch in lengths for pt bin {iPt}: \n"
            f"sig_low:{len(sig_cuts_lower[iPt])}, \n"
            f"sig_up: {len(sig_cuts_upper[iPt])}, \n"
            f"bkg_low: {len(bkg_cuts_lower[iPt])}, \n"
            f"bkg_up: {len(bkg_cuts_upper[iPt])}, \n"
            f"nCutSets: {nCutSets[iPt]}"
        )

    return nCutSets, sig_cuts_lower, sig_cuts_upper, bkg_cuts_lower, bkg_cuts_upper

def get_cut_sets_config(config):
    '''
    Get cut sets from configuration file
    '''
    import yaml
    with open(config, 'r') as ymlCfgFile:
        config = yaml.load(ymlCfgFile, yaml.FullLoader)

    ptmins = config['ptmins']
    ptmaxs = config['ptmaxs']
    correlated_cuts = config['minimisation']['correlated']
    if correlated_cuts:
        sig_cut = config['cut_variation']['corr_bdt_cut']['sig']
        bkg_cut_maxs = config['cut_variation']['corr_bdt_cut']['bkg_max']
    else:
        sig_cut = config['cut_variation']['uncorr_bdt_cut']['sig']
        bkg_cut_maxs = config['cut_variation']['uncorr_bdt_cut']['bkg_max']

    return get_cut_sets(len(ptmins), sig_cut, bkg_cut_maxs, correlated_cuts)

def cut_var_image_merger(config, cut_var_dir, suffix):
    '''
    Merge cut variation images
    '''
    def pdf_to_images(pdf_path, dpi=100):
        """Extract high-quality images from a PDF."""
        import fitz
        doc = fitz.open(pdf_path)
        images = []
        
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        return images

    def create_multipanel(images, iImage, images_per_row=None, images_per_col=None, bg_color="white"):
        """Combine multiple images into a grid layout with customizable rows and columns."""
        if not images:
            raise ValueError("At least one image is required.")

        num_images = len(images)
        # Auto-calculate rows and columns if not specified
        if images_per_row is None and images_per_col is None:
            images_per_row = math.ceil(math.sqrt(num_images))
        if images_per_row is None:
            images_per_row = math.ceil(num_images / images_per_col)
        if images_per_col is None:
            images_per_col = math.ceil(num_images / images_per_row)

        # Resize images to match the smallest width and height
        min_width = min(img[iImage].width for img in images)
        min_height = min(img[iImage].height for img in images)
        resized_images = [img[iImage].resize((min_width, min_height), Image.LANCZOS) for img in images]

        # Create a blank canvas
        combined_width = min_width * images_per_row
        combined_height = min_height * images_per_col
        combined = Image.new("RGB", (combined_width, combined_height), bg_color)

        # Paste images into the grid
        for index, img in enumerate(resized_images):
            row, col = divmod(index, images_per_row)
            x_offset = col * min_width
            y_offset = row * min_height
            combined.paste(img, (x_offset, y_offset))

        return combined

    def process_pdfs(config, pdf_list, output_folder, images_names, images_per_row=None, images_per_col=None):
        """Processes PDFs and saves multipanel images with high-quality settings."""
        if len(pdf_list) == 0:
            raise ValueError("No PDF provided!")

        images = [pdf_to_images(pdf, dpi=100) for pdf in pdf_list]
        num_pages = min(len(imgs) for imgs in images)

        os.makedirs(output_folder, exist_ok=True)

        for i in range(num_pages):
            panel = create_multipanel(images, i, images_per_row, images_per_col)
            output_path = os.path.join(output_folder, f"{images_names}_pt_{int(config['ptmins'][i]*10)}_{int(config['ptmaxs'][i]*10)}.png")
            panel.save(output_path, format="PNG", compression_level=1)

        print(f"Saved {num_pages} high-quality multipanel images in '{output_folder}'.")

    cutvar_files = [
        f"{cut_var_dir}/CutVarFrac/CutVarFrac_{suffix}_CorrMatrix.pdf", 
        f"{cut_var_dir}/CutVarFrac/CutVarFrac_{suffix}_Distr.pdf", 
        f"{cut_var_dir}/CutVarFrac/CutVarFrac_{suffix}_Eff.pdf", 
        f"{cut_var_dir}/CutVarFrac/CutVarFrac_{suffix}_Frac.pdf",
        f"{cut_var_dir}/V2VsFrac/FracV2_{suffix}.pdf"
    ]
    
    try:
        process_pdfs(config, cutvar_files, f"{cut_var_dir}/merged_images/cutvar", 'cutvar_summary')
    except:
        print("Error in merging cut variation files")
    
    try:
        fit_files = glob.glob(f"{cut_var_dir}/ry/*.pdf")
        fit_files_sorted = sorted(fit_files, key=lambda x: int(re.search(r"_(\d+)_D\w*.pdf$", x).group(1)))
        process_pdfs(config, fit_files_sorted, f"{cut_var_dir}/merged_images/fits/", 'fit_summary', 5)
    except:
        print("Error in merging fit files")