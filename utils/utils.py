import shutil
import os
import sys
import ROOT
import ctypes
from ROOT import TH1, TH1D, TFile
import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation
import yaml


def merge_cutsets_fits(fits_dir, verbose=True):

    all_fit_pdfs = sorted(
        [p for p in fits_dir.glob("*.pdf")],
        key=lambda p: int(p.stem.split('_')[2])
    )

    if not all_fit_pdfs:
        print("No fit PDFs found.")
        return

    # Load all PDFs
    readers = [PdfReader(p) for p in all_fit_pdfs]

    writer = PdfWriter()

    # Determine grouping
    group_size = 5 if len(all_fit_pdfs) > 5 else len(all_fit_pdfs)

    max_pages = max(len(r.pages) for r in readers)

    # ---- FIRST LOOP OVER PT BINS ----
    for page_idx in range(max_pages):

        # ---- THEN LOOP OVER FIT CHUNKS ----
        for batch_start in range(0, len(readers), group_size):

            batch_readers = readers[batch_start:batch_start + group_size]

            pages_to_merge = [
                r.pages[page_idx]
                for r in batch_readers
                if page_idx < len(r.pages)
            ]

            if not pages_to_merge:
                continue

            widths = [
                float(p.cropbox.right) - float(p.cropbox.left)
                for p in pages_to_merge
            ]

            heights = [
                float(p.cropbox.top) - float(p.cropbox.bottom)
                for p in pages_to_merge
            ]

            total_width = sum(widths)
            max_height = max(heights)

            new_page = writer.add_blank_page(
                width=total_width,
                height=max_height
            )

            current_x = 0

            for page, width in zip(pages_to_merge, widths):

                box = page.cropbox

                shift_x = -float(box.left)
                shift_y = -float(box.bottom)

                transform = Transformation().translate(
                    tx=shift_x + current_x,
                    ty=shift_y
                )

                new_page.merge_transformed_page(page, transform)

                current_x += width

        if verbose:
            print(f"Merged pt bin idx {page_idx+1}")

    output_path = fits_dir / "AllCutsetsFits.pdf"

    with open(output_path, "wb") as f:
        writer.write(f)

    if verbose:
        print(f"All fit cutsets grouped into: {output_path}")

def produce_pt_bins_fit_summary(fits_dir, config_path):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    pt_bins = cfg['ptbins']

    all_fit_root_files = sorted(
        [p for p in fits_dir.glob("*.root")],
        key=lambda p: int(p.stem.split('_')[2])
    )

    total_cutsets = len(all_fit_root_files)

    summary = {}
    for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
        pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        summary[pt_str] = {}

    # Pick first ROOT file, so that the structure of the dictionaries
    # can be defined (the fit function can change)

    # print(f"Opened first fit file: {first_fit_file}")
    for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
        pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        try:
            input_file = TFile(str(all_fit_root_files[0]), "READ")
            hist_fit_pars = input_file.Get(f"{pt_str}/hist_sim_fit")
        except:
            print(f"Could not retrieve histogram from first fit file for pt bin {pt_str}, trying the others ...")
            for fit_file_path in all_fit_root_files[1:]:
                try:
                    input_file = TFile(str(fit_file_path), "READ")
                    hist_fit_pars = input_file.Get(f"{pt_str}/hist_sim_fit")
                    break
                except:
                    print(f"Could not retrieve histogram from fit file {fit_file_path}")
            else:
                print(f"Could not retrieve histogram for pt bin {pt_str} from any fit file")
                continue

        hist_fit_pars.SetDirectory(0)
        for i_bin in range(hist_fit_pars.GetNbinsX()):
            par_name = hist_fit_pars.GetXaxis().GetBinLabel(i_bin + 1)
            summary[pt_str][f"Hist{par_name}"] = \
                TH1D(f"{pt_str}_{par_name}", f"{pt_str}_{par_name}", total_cutsets + 1, -0.5, total_cutsets + 0.5)
            summary[pt_str][f"Hist{par_name}"].SetDirectory(0)
            summary[pt_str][f"Hist{par_name}Unc"] = \
                TH1D(f"{pt_str}_{par_name}Unc", f"{pt_str}_{par_name} Uncertainty;Cutset;Fit Unc.", total_cutsets + 1, -0.5, total_cutsets + 0.5)
            summary[pt_str][f"Hist{par_name}Unc"].SetDirectory(0)
        input_file.Close()

    for i_cutset, fit_file_path in enumerate(all_fit_root_files):
        fit_file = TFile(str(fit_file_path), "READ")
        for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
            pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
            try:
                hist_fit_pars = fit_file.Get(f"{pt_str}/hist_sim_fit")
                for i_bin in range(hist_fit_pars.GetNbinsX()):
                    par_name = hist_fit_pars.GetXaxis().GetBinLabel(i_bin + 1)
                    par_value = hist_fit_pars.GetBinContent(i_bin + 1)
                    par_unc = hist_fit_pars.GetBinError(i_bin + 1)
                    summary[pt_str][f"Hist{par_name}"].SetBinContent(i_cutset + 1, par_value)
                    summary[pt_str][f"Hist{par_name}"].SetBinError(i_cutset + 1, par_unc)
                    summary[pt_str][f"Hist{par_name}Unc"].SetBinContent(i_cutset + 1, par_unc)
            except Exception as e:
                print(f"[{fit_file_path}] Pt bin {pt_str} did not converge!")
                summary[pt_str][f"Hist{par_name}"].SetBinContent(i_cutset + 1, -1)
                summary[pt_str][f"Hist{par_name}"].SetBinError(i_cutset + 1, 0)
                summary[pt_str][f"Hist{par_name}Unc"].SetBinContent(i_cutset + 1, 0)

    summary_file = TFile.Open(f"{fits_dir}/FitSummary.root", "RECREATE")
    for pt_bin, pt_summary in summary.items():
        summary_file.mkdir(pt_bin)
        summary_file.cd(pt_bin)
        for hist_name, hist in pt_summary.items():
            hist.Write(hist_name)

    summary_file.Close()

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
	elif level == 'FATAL':
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

def make_dir_root_file(directory, file, verbose=True):
    if not file.GetDirectory(directory):
        file.mkdir(directory)
        if verbose:
            logger(f"Created directory {directory} in file {file.GetName()}", level='WARNING')
    else:
        if verbose:
            logger(f"Directory {directory} already exists in file {file.GetName()}", level='WARNING')

def profile_mass_sp(hist_mass_sp, vn_vs_mass_bins, resolution):
    '''
    Profile the mass sparse to get vn versus mass
    Input:
        - hist_mass_sp:
            THnSparse, input THnSparse object (already projected in centrality and pt)
        - vn_vs_mass_bins:
            list of floats, bin edges for the mass axis
        - resolution:
            float, resolution to normalize the vn values
    Output:
        - hist_vn_vs_mass:
            TH1D, histogram with vn as a function of mass
    '''
    hist_vn_vs_mass = ROOT.TH1D('hist_vn_vs_mass', 'hist_vn_vs_mass', len(vn_vs_mass_bins)-1, np.array(vn_vs_mass_bins))
    hist_vn_vs_mass.SetDirectory(0)
    for i in range(hist_vn_vs_mass.GetNbinsX()):
        bin_low = hist_mass_sp.GetXaxis().FindBin(vn_vs_mass_bins[i])
        bin_high = hist_mass_sp.GetXaxis().FindBin(vn_vs_mass_bins[i+1])
        profile = hist_mass_sp.ProfileY(f'profile_{bin_low}_{bin_high}', bin_low, bin_high)
        mean_sp = profile.GetMean()
        mean_sp_err = profile.GetMeanError()
        hist_vn_vs_mass.SetBinContent(i+1, mean_sp / resolution)
        hist_vn_vs_mass.SetBinError(i+1, mean_sp_err / resolution)
    return hist_vn_vs_mass

def get_vn_versus_mass(sparse, vn_vs_mass_bins, mass_axis, vn_axis, debug=False):
    '''
    Project vn versus mass

    Input:
        - sparse:
            THnSparse, input THnSparse object (already projected in centrality and pt)
        - vn_vs_mass_bins:
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
    vn_vs_mass_bins = np.array(vn_vs_mass_bins)
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

def get_vnfitter_results(vnFitter, useRefl, useTempl, secPeak, secPeakWidthFrac=None):
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
    vn_results['pulls'] = vnFitter.GetPullDistribution()
    try:
        vn_results['hParsMCPrefit'] = vnFitter.GetPrefitParsHisto()
    except:
        vn_results['hParsMCPrefit'] = None
    try:
        vn_results['hParsSignalFunc'] = vnFitter.GetSignalParsHisto()
    except:
        vn_results['hParsSignalFunc'] = None
    try:
        vn_results['hParsSimFit'] = vnFitter.GetSimFitParsHisto()
    except:
        vn_results['hParsSimFit'] = None

    vn_results['fVnCompsFuncts'] = {}
    vnComps = vnFitter.GetVnCompsFuncts()
    vn_results['fVnCompsFuncts']['vnSgn'] = vnComps[0]
    vn_results['fVnCompsFuncts']['vnBkg'] = vnComps[1]
    if secPeak is not None:
        vn_results['fVnCompsFuncts']['vnSecPeak'] = vnComps[2]
    vn_results['fMassTemplTotFunc'] = vnFitter.GetMassTemplFitFunc()
    vn_results['fMassTemplFuncts'] = vnFitter.GetMassTemplFuncts()
    if useTempl:
        for iTempl in range(len(vn_results['fMassTemplFuncts'])):
            vn_results['fVnCompsFuncts'][f'vnTempl{iTempl}'] = vnComps[2+1+iTempl] if secPeak is not None \
                                                               else vnComps[2+iTempl]

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
    bkg_pars_with_uncs = vnFitter.GetBkgPars()
    vn_results['bkgPars'] = bkg_pars_with_uncs[:len(bkg_pars_with_uncs)//2]
    vn_results['bkgParsUncs'] = bkg_pars_with_uncs[len(bkg_pars_with_uncs)//2:]

    massSgnPars = vnFitter.GetNMassSgnPars()
    massBkgPars = vnFitter.GetNMassBkgPars()
    massSecPeakPars = vnFitter.GetNMassSecPeakPars()
    massReflPars = vnFitter.GetNMassReflPars()
    massTemplPars = 0 # len(vn_results['fMassTemplFuncts'])
    totMassPars = massSgnPars + massBkgPars + massSecPeakPars + massReflPars + massTemplPars
    vnSgnPars = vnFitter.GetNVnSgnPars()
    vnBkgPars = vnFitter.GetNVnBkgPars()

    if secPeak is not None:
        vn_results['fMassSecPeakFunc'] = vnFitter.GetMassSecPeakFunc()
        vn_results['fVnSecPeakFunct'] = vnFitter.GetVnSecPeakFunc()
        vn_results['secPeakMeanMass'] = vn_results['fTotFuncMass'].GetParameter(vn_results['fTotFuncMass'].GetParName(massSgnPars + massBkgPars + 1))
        vn_results['secPeakMeanMassUnc'] = vn_results['fTotFuncMass'].GetParError(massSgnPars + massBkgPars + 1)
        if 'FixedSigmaFrac' in secPeak:
            vn_results['secPeakSigmaMass'] = secPeakWidthFrac*vn_results['fTotFuncMass'].GetParameter(vn_results['fTotFuncMass'].GetParName(massBkgPars + 2))
            vn_results['secPeakSigmaMassUnc'] = secPeakWidthFrac*vn_results['fTotFuncMass'].GetParError(massBkgPars + 2)
        else:
            vn_results['secPeakSigmaMass'] = vn_results['fTotFuncMass'].GetParameter(vn_results['fTotFuncMass'].GetParName(massSgnPars + massBkgPars + 2))
            vn_results['secPeakSigmaMassUnc'] = vn_results['fTotFuncMass'].GetParError(massSgnPars + massBkgPars + 2)
        if 'VnFree' in secPeak:
            print("Getting vn of secondary peak free")
            vn_results['secPeakMeanVn'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(totMassPars + vnSgnPars + vnBkgPars + 1))
            vn_results['secPeakMeanVnUnc'] = vn_results['fTotFuncVn'].GetParError(vnSgnPars + vnBkgPars + 1)
            vn_results['secPeakSigmaVn'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(totMassPars + vnSgnPars + vnBkgPars + 2))
            vn_results['secPeakSigmaVnUnc'] = vn_results['fTotFuncVn'].GetParError(vnSgnPars + vnBkgPars + 2)
            # vn_results['vnSecPeak'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(vnBkgPars + 1))
            vn_results['vnSecPeak'] = vn_results['fTotFuncVn'].GetParameter(totMassPars + vnBkgPars)
            vn_results['vnSecPeakUnc'] = vn_results['fTotFuncVn'].GetParError(totMassPars + vnBkgPars)
        if 'VnFixedToSgn' in secPeak:
            print("Getting vn of secondary peak fixed to signal vn")
            vn_results['secPeakMeanVn'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(totMassPars + 1))
            vn_results['secPeakMeanVnUnc'] = vn_results['fTotFuncVn'].GetParError(vnSgnPars + 1)
            vn_results['secPeakSigmaVn'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(totMassPars + 2))
            vn_results['secPeakSigmaVnUnc'] = vn_results['fTotFuncVn'].GetParError(vnSgnPars + 2)
            vn_results['vnSecPeak'] = vn_results['fTotFuncVn'].GetParameter(vn_results['fTotFuncVn'].GetParName(totMassPars))
            vn_results['vnSecPeakUnc'] = vn_results['fTotFuncVn'].GetParError(totMassPars)
        print(f"vnSecPeak: {vn_results['vnSecPeak']} +/- {vn_results['vnSecPeakUnc']}, iPar: {totMassPars}")

    if useRefl:
        vn_results['fMassRflFunc'] = vnFitter.GetMassRflFunc()
        vn_results['fMassBkgRflFunc'] = vnFitter.GetMassBkgRflFunc()
    
    if useTempl:
        vn_results['fMassTemplFuncts'] = list(vnFitter.GetMassTemplFuncts())
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

def get_refl_histo(reflFile, ptMins, ptMaxs):
    '''
    Method that loads MC histograms for the reflections of D0

    Input:
        - reflFile:
           TFile, ROOT file, include reflections of D0
        - centMinMax:
            list, min and max centrality
        - ptMins:
            list, min pt bins
        - ptMaxs:
            list, max pt bins
    
    Output:
        - useRefl:
            bool, if True, MC histograms for the reflections of D0 exists
        - hMCSgn:
            lsit, signal histograms of D0
        - hMCRefl:
            list, reflection histograms of D0
    '''
    hMCSgn, hMCRefl = [], []
    if not check_file_exists(reflFile):
        logger(f'Reflections file {reflFile} does not exist! Turning off reflections usage', level='ERROR')
        return False
    
    reflFile = TFile(reflFile, 'READ')
    for iPt, (ptMin, ptMax) in enumerate(zip(ptMins, ptMaxs)):
        ptMinSuf, ptMaxSuf = int(ptMin*10), int(ptMax*10)
        dirName = f'pt_{ptMinSuf}_{ptMaxSuf}'
        if not reflFile.GetDirectory(dirName):
            logger(f'No directory {dirName} found! Turning off reflections usage', level='ERROR')
            return False

        hMCSgn.append(reflFile.Get(f'{dirName}/hFDMass'))
        hMCSgn[iPt].Add(reflFile.Get(f'{dirName}/hPromptMass'), 1)
        if not isinstance(hMCSgn[iPt], TH1) or hMCSgn[iPt] == None:
            logger(f'In directory {dirName}, hFDMass/hPromptMass_{ptMinSuf}_{ptMaxSuf} not found! Turning off reflections usage', level='ERROR')
            return False
        hMCSgn[iPt].SetName(f'histSgn_{iPt}')
        hMCSgn[iPt].SetDirectory(0)

        hMCRefl.append(reflFile.Get(f'{dirName}/hRecoReflMass'))
        if not isinstance(hMCRefl[iPt], TH1) or hMCRefl[iPt] == None:
            logger(f'In directory {dirName}, hRecoReflMass not found! Turning off reflections usage', level='ERROR')
            return False
        hMCRefl[iPt].SetName(f'histRfl_{iPt}')
        hMCRefl[iPt].SetDirectory(0)

        if hMCRefl[iPt].Integral() <= 0:
            logger(f'Error: Empty reflection template for pt bin {ptMin}-{ptMax}! Turning off reflections usage', level='ERROR')
            return False

    reflFile.Close()

    return True, hMCSgn, hMCRefl

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

def get_ese_band_label(p_lo, p_hi):
    """Label for an ESE q2 percentile band. Shared by all ESE scripts."""
    p_lo, p_hi = int(p_lo), int(p_hi)
    if not 0 <= p_lo < p_hi <= 100:
        raise ValueError(f"Invalid ESE percentiles [{p_lo}, {p_hi}]: need 0 <= lo < hi <= 100")
    return f'q2_{p_lo}_{p_hi}'

def suggest_skip_cuts(hRawYields, hEffPrompt, hEffFD, nPtBins):
    """Suggest cuts to skip based on zero or negative efficiencies or raw yields"""
    nCuts = len(hRawYields)
    suggested_skipped_cuts_pts = []
    for iPt in range(nPtBins):
        rys = [hRawYields[iCut].GetBinContent(iPt+1) for iCut in range(nCuts)]
        effPs = [hEffPrompt[iCut].GetBinContent(iPt+1) for iCut in range(nCuts)]
        effFs = [hEffFD[iCut].GetBinContent(iPt+1) for iCut in range(nCuts)]
        suggested_skipped_cuts = []

        for iCut in range(nCuts):
            # zero or negative efficiencies or raw yields
            if any([rys[iCut] < 0, effPs[iCut] < 0, effFs[iCut] < 0]):
                logger(f'Cut {iCut} for pt bin {iPt+1} has negative value, prompt eff={effPs[iCut]}, FD eff={effFs[iCut]}, ry={rys[iCut]}', level='ERROR')
            elif any ([rys[iCut] == 0, effPs[iCut] == 0, effFs[iCut] == 0]):
                logger(f'Suggested to skip cut {iCut} for pt bin {iPt+1} due to zero or negative efficiency/raw yield', level='WARNING')
                suggested_skipped_cuts.append(iCut)
            elif iCut == 0 and nCuts > 1 and rys[0] == rys[1]:
                logger(f'Suggested to skip cut {iCut} for pt bin {iPt+1} ({rys[iCut]}) due to identical raw yield as next cut', level='WARNING')
                suggested_skipped_cuts.append(iCut)
            # raw yield differs from previous cut by less than 0.01%
            elif iCut == nCuts - 1 and abs(rys[iCut] - rys[iCut-1]) / abs(rys[iCut]) < 0.0001:
                logger(f'Suggested to skip cut {iCut} for pt bin {iPt+1} ({rys[iCut]}) due to negligible change in raw yield', level='WARNING')
                suggested_skipped_cuts.append(iCut)
            elif iCut > 0 and iCut < nCuts - 1 and abs(rys[iCut] - rys[iCut-1]) / abs(rys[iCut]) < 0.0001 and abs(rys[iCut] - rys[iCut+1]) / abs(rys[iCut]) < 0.0001: 
                logger(f'Suggested to skip cut {iCut} for pt bin {iPt+1} ({rys[iCut]}) due to negligible change in raw yield', level='WARNING')
                suggested_skipped_cuts.append(iCut)
        suggested_skipped_cuts_pts.append(suggested_skipped_cuts)
    for iPt, cuts in enumerate(suggested_skipped_cuts_pts):
        print(f'\t\t{cuts}, # suggested cuts to skip for pt {iPt+1}')
    return suggested_skipped_cuts_pts

def reweight_histo_1D(histo, weights, binned=False):
    for iBin in range(1, histo.GetNbinsX()+1):
        ptCent = histo.GetBinCenter(iBin)
        weight = weights[iBin-1] if binned else weights(ptCent) if weights(ptCent) > 0 else 0
        histo.SetBinContent(iBin, histo.GetBinContent(iBin) * weight)
        histo.SetBinError(iBin, histo.GetBinError(iBin) * weight)
    proj_hist = histo.Clone(histo.GetName())
    return proj_hist

def reweight_histo_2D(histo, weights, binned=False):
    for iBinX in range(1, histo.GetXaxis().GetNbins()+1):
        for iBinY in range(1, histo.GetYaxis().GetNbins()+1):
            if binned:
                weight = weights[iBinY-1] if weights[iBinY-1] > 0 else 0
            else:
                binCentVal = histo.GetYaxis().GetBinCenter(iBinY)
                weight = weights(binCentVal) if weights(binCentVal) > 0 else 0
            weighted_content = histo.GetBinContent(iBinX, iBinY) * weight
            weighted_error = histo.GetBinError(iBinX, iBinY) * weight
            histo.SetBinContent(iBinX, iBinY, weighted_content)
            histo.SetBinError(iBinX, iBinY, weighted_error)
    proj_hist = histo.ProjectionX(histo.GetName(), 0, histo.GetYaxis().GetNbins()+1, 'e')
    return proj_hist

def reweight_histo_3D(histo, weightsY, weightsZ):
    for iBinX in range(1, histo.GetXaxis().GetNbins()+1):
        for iBinY in range(1, histo.GetYaxis().GetNbins()+1):
            for iBinZ in range(1, histo.GetZaxis().GetNbins()+1):
                binCenterY = histo.GetYaxis().GetBinCenter(iBinY)
                weight = weightsZ[iBinZ-1]*weightsY(binCenterY) if weightsY(binCenterY) > 0 else weightsZ[iBinZ-1] 
                weighted_content = histo.GetBinContent(iBinX, iBinY, iBinZ) * weight
                weighted_error = histo.GetBinError(iBinX, iBinY, iBinZ) * weight if weight > 0 else 0
                histo.SetBinContent(iBinX, iBinY, iBinZ, weighted_content)
                histo.SetBinError(iBinX, iBinY, iBinZ, weighted_error)
    proj_hist = histo.ProjectionX(histo.GetName(), 0, histo.GetYaxis().GetNbins()+1,
                                  0, histo.GetZaxis().GetNbins()+1, 'e')
    return proj_hist
