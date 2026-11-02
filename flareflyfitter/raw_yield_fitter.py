import argparse
import yaml
import os
import sys
import ROOT
from ROOT import gStyle, TFile, TH1, TH1D, TH1F, TCanvas, TLegend, TLine, TBox, kDashed, kGray, kRed, kBlue # pylint: disable=import-error,no-name-in-module
from ROOT import RooFit
from ROOT import RooRealProxy, RooChi2Var, RooMinimizer, RooRealVar, RooFormulaVar, RooRealSumFunc, RooDataHist, RooHistPdf, RooAddPdf, RooArgList, RooArgSet, RooExtendPdf, RooMsgService
from ROOT import RooAbsTestStatistic, RooChebychev, RooGaussian, RooPolynomial, RooExponential, RooDataSet
gStyle.SetEndErrorSize(0)
script_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.join(script_dir, '..', 'utils'))
os.sys.path.append(os.path.join(script_dir, '..', 'src'))
from correlated_bkgs import get_corr_bkg
from utils import logger, get_centrality_bins
import zfit
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import uproot
import copy
import numpy as np
msg_service = RooMsgService.instance()
msg_service.setGlobalKillBelow(RooFit.FATAL)  # Only show FATAL errors (you can also use RooFit.ERROR or INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position
ROOT.gROOT.SetBatch(True)  # Run in batch mode to avoid opening canvases

class RawYieldFitter:
    """ 
    Fitter of invariant mass spectra to extract raw yields using the flarefly package
    """

    def __init__(self, particle, pt_min, pt_max, label, minimizer, fit_vn_vs_mass, verbose=True):
        logger(f"Initializing RawYieldFitter, verbosity: {verbose}", "INFO")
        self.verbose = verbose
        self.fit_vn_vs_mass = fit_vn_vs_mass
        if minimizer == 'flarefly':
            self.minimize_flarefly = True
            self.minimize_roofit = False
        else:
            self.minimize_flarefly = False
            self.minimize_roofit = True
        self.vn_func_denominator = None
        self.vn_func_numerator = None
        self.rebin = 1
        self.vn_terms = []
        self.mass_fit_range_min = None
        self.mass_fit_range_max = None
        self.sp_range_min = None
        self.sp_range_max = None
        self.mass_sgn_templ_frac = None
        self.mass_sgn_templ_name = None
        self.fitter = None
        self.mass_data = None
        self.vn_vs_mass_data = None
        self.fix_sgn_to_mc_prefit = False
        self.hist = None
        self.mass_fit_result = None
        self.particle = None
        self.x_axis_label = None
        self.fit_name = f"{particle}_{label}"
        self.particle_pdg = None
        self.mass_fit_var = None
        self.roofit_sp_var = None
        self.fit_model = {}
        self.set_particle(particle)
        self.mass_sgn_pdfs = None
        self.mass_sgn_pdfs_labels = None
        self.mass_bkg_pdfs = None
        self.mass_bkg_pdfs_labels = None
        self.cfg_pars_init = None
        self.signal_pdg_masses = []
        self.n_pdfs_bkg = 0
        self.n_pdfs_sgn = 0
        self.mass_model = None
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.pdfs = RooArgList()
        self.mc_pars = {}
        self.fit_counter = 0
        self.fix_sgn_to_first_fit = False
        self.first_fit_pars = None
        self.vn_mass_var = None

    def set_particle(self, particle_name):
        if self.verbose:
            logger(f"Setting particle to fit: {particle_name}\n")
        self.particle = particle_name
        if particle_name == "Dplus":
            self.x_axis_label = r"$M(\mathrm{\pi^+ K^- \pi^+})\ \mathrm{(GeV/}c^2)$"
            self.sgn_main_label = "DplusToPiKPi"
            self.particle_pdg = 411
        elif particle_name == "D0" or particle_name == "Dzero":
            self.x_axis_label = r"$M(\mathrm{K^- \pi^+})\ \mathrm{(GeV/}c^2)$"
            self.sgn_main_label = "D0ToKPi"
            self.particle_pdg = 421
        elif particle_name == "Ds":
            self.x_axis_label = r"$M(\mathrm{K^+ K^- \pi^+})\ \mathrm{(GeV/}c^2)$"
            self.sgn_main_label = "DsToKKPi"
            self.particle_pdg = 431
        else:
            logger(f"Particle {particle_name} not recognized!")
            sys.exit(1)

    def set_name(self, fit_name):
        if self.verbose:
            logger(f"Setting fit name: {fit_name}\n", "INFO")
        self.fit_name = fit_name

    def set_fit_range(self, fit_range_min, fit_range_max):
        if self.verbose:
            logger(f"Setting fit range: {fit_range_min} - {fit_range_max} GeV/c\n", "INFO")
        self.mass_fit_range_min = fit_range_min
        self.mass_fit_range_max = fit_range_max
        if self.minimize_roofit: # Very loose range, to be constrained specifically for each fit
            self.mass_fit_var = RooRealVar("fM", "Invariant Mass", self.mass_fit_range_min, self.mass_fit_range_max)
            self.vn_mass_var = RooRealVar("fM_vn", "Invariant Mass",self.mass_fit_range_min,self.mass_fit_range_max)
            # self.vn_vs_mass_fit_var = RooRealVar("fM", "Invariant Mass", self.mass_fit_range_min, self.mass_fit_range_max)

    def set_rebin(self, rebin):
        if self.verbose:
            logger(f"Setting rebin: {rebin}\n", "INFO")
        self.rebin = rebin

    def set_vn_vs_mass_data_to_fit_hist(self, data):
        print(f"Setting vn vs mass data to fit from histogram with limits {self.mass_fit_range_min} - {self.mass_fit_range_max} GeV/c")
        if self.verbose:
            logger(f"Setting data to fit from histogram with limits {self.mass_fit_range_min} - {self.mass_fit_range_max} GeV/c"
                   f" for fitter with name {self.fit_name}\n", "INFO")
        if self.minimize_flarefly:
            self.vn_vs_mass_data = DataHandler(data, limits=[self.mass_fit_range_min, self.mass_fit_range_max])
        else:
            self.vn_vs_mass_data = RooDataHist("vn_vs_mass_hist", "vn_vs_mass_hist", RooArgList(self.vn_mass_var), data)
            # self.vn_vs_mass_data = RooDataHist("vn_vs_mass_hist", "vn_vs_mass_hist", RooArgList(self.vn_vs_mass_fit_var), data)

    def set_mass_data_to_fit_hist(self, data):
        if self.verbose:
            logger(f"Setting data to fit from histogram with limits {self.mass_fit_range_min} - {self.mass_fit_range_max} GeV/c"
                   f" for fitter with name {self.fit_name}\n", "INFO")
        self.hist = data
        if self.minimize_flarefly:
            self.mass_data = DataHandler(data, limits=[self.mass_fit_range_min, self.mass_fit_range_max])
            # self.mass_data = DataHandler(data, limits=[self.mass_fit_range_min, self.mass_fit_range_max], rebin=self.rebin) CRASHES DUE TO NON-ALIGNED BINS WITH FIT RANGE
        else:
            self.mass_data = RooDataHist("data_hist", "data_hist", RooArgList(self.mass_fit_var), data)

    def set_fix_sgn_to_mc_prefit(self, fix):
        if self.verbose:
            logger(f"Setting fix signal to MC prefit: {fix}\n", "INFO")
        self.fix_sgn_to_mc_prefit = fix

    def set_fixed_sigma(self, sigma_val):
        if self.verbose:
            logger(f"Fixing signal sigma to value {sigma_val}", "INFO")
        self.fixed_sigma = sigma_val
        if self.minimize_roofit:
            for name, comp in self.fit_model.items():
                if comp['type'] == 'sgn':
                    comp['par_sigma'].setVal(sigma_val)
                    comp['par_sigma'].setConstant(True)

    def set_sigma_init(self, sigma_val):
        if self.verbose:
            logger(f"Setting initial signal sigma to value {sigma_val}", "INFO")
        self.sigma_init = sigma_val
        if self.minimize_roofit:
            for name, comp in self.fit_model.items():
                if comp['type'] == 'sgn':
                    comp['par_sigma'].setVal(sigma_val)

    def prefit_mc(self, input_path):
        if self.verbose:
            logger(f"Performing MC prefit on data with fit range {self.mass_fit_range_min} - {self.mass_fit_range_max} GeV/c\n", "INFO")
        # Setup a temporary fitter for the prefit
        for name, sgn_func in self.fit_model.items():
            if sgn_func['type'] != 'sgn':
                continue
            try:
                pt_label = f"pt_{int(self.pt_min*10)}_{int(self.pt_max*10)}"
                corr_bkg_file = TFile.Open(input_path, "READ")
                if self.verbose:
                    logger(f"Adding correlated backgrounds to the fitter for pt range {self.pt_min} - {self.pt_max} GeV/c", level="INFO")
                sel_string = f"fM >= {self.mass_fit_range_min} && fM < {self.mass_fit_range_max}"
                hist_mc, _ = get_corr_bkg(corr_bkg_file, name, sel_string, pt_label, "raw", "hist", get_smoothed=False)
                corr_bkg_file.Close()
            except Exception as e:
                if self.verbose:
                    logger(f"Prefit not available for signal function {name}: {e}", level="ERROR")
                continue

            init_mass = self.get_particle_mass(name)
            if self.minimize_flarefly:
                self.fit_model[name]['mcmassvar'] = [init_mass-0.2, init_mass+0.2]
                self.fit_model[name]['mchist'] = DataHandler(hist_mc, limits=self.fit_model[name]['mcmassvar'], rebin=self.rebin)
                self.fit_model[name]['mcfitter'] = F2MassFitter(self.fit_model[name]['mchist'], name=f"{self.fit_name}_mc_prefit_{self.fit_model[name]['label']}",
                                                                label_signal_pdf=[self.fit_model[name]['label']], name_signal_pdf=[self.fit_model[name]['pdf']],
                                                                name_background_pdf=["nobkg"], label_bkg_pdf=["nobkg"])
                # Set particle mass and sigma initial par for the main signal
                self.fit_model[name]['mcfitter'].set_particle_mass(0, pdg_id=self.particle_pdg)
                self.fit_model[name]['mcfitter'].set_signal_initpar(0, "sigma", 0.015, limits=[0., 0.1])
                fit_result = self.fit_model[name]['mcfitter'].mass_zfit()
                self.mc_pars[name] = self.fit_model[name]['mcfitter'].get_signal_pars()[0]
            else:
                self.mass_fit_var.setRange(f"mc_fit_{name}", init_mass-0.2, init_mass+0.2)
                self.fit_model[name]['mchist'] = RooDataHist(f"mc_dataset_{name}", f"mc_dataset_{name}",
                                                 RooArgList(self.mass_fit_var), hist_mc)
                fit_result = self.fit_model[name]['pdf'].fitTo(self.fit_model[name]['mchist'], RooFit.Range(f"mc_fit_{name}"),
                                                               RooFit.Save(), RooFit.PrintLevel(1 if self.verbose else -1),
                                                               RooFit.PrintEvalErrors(1 if self.verbose else 0))
                self.mc_pars[name] = self.fit_model[name]['pdf'].getParameters(self.fit_model[name]['mchist'])

    def set_data_to_fit_df(self, data_df, var_name='fM'):
        if self.verbose:
            logger(f"Setting data to fit from dataframe of length {len(data_df)} with variable {var_name} "
                   f"and limits {self.mass_fit_range_min} - {self.mass_fit_range_max} GeV/c\n", "INFO")
        if self.minimize_flarefly:
            self.mass_data = DataHandler(data_df, var_name=var_name, limits=[self.mass_fit_range_min, self.mass_fit_range_max])
        else:
            self.sp_range_min = -4.
            self.sp_range_max = 4.
            self.roofit_sp_var = RooRealVar("fScalarProd", "Scalar Product", self.sp_range_min, self.sp_range_max)

            # Temporary TTree
            tmp_tree = ROOT.TTree("tmp_tree", "temporary tree")

            buf_m  = np.zeros(1, dtype="float64")
            buf_sp = np.zeros(1, dtype="float64")

            tmp_tree.Branch("fM", buf_m,  "fM/D")
            tmp_tree.Branch("fScalarProd", buf_sp, "fScalarProd/D")

            # Fill tree (single loop, cache-friendly)
            m_vals  = data_df["fM"].to_numpy()
            sp_vals = data_df["fScalarProd"].to_numpy()

            for m, sp in zip(m_vals, sp_vals):
                if m < self.mass_fit_range_min or m >= self.mass_fit_range_max:
                    continue
                buf_m[0]  = float(m)
                buf_sp[0] = float(sp)
                tmp_tree.Fill()

            self.mass_data = RooDataSet("data", "dataset from dataframe", tmp_tree,
                                        RooArgList(self.mass_fit_var, self.roofit_sp_var))

    def set_data_to_fit_tree(self, data_tree, var_name):
        if self.minimize_flarefly:
            self.mass_data = DataHandler(data_tree, var_name=var_name, limits=[self.mass_fit_range_min, self.mass_fit_range_max])
        else:
            # RooFit data structure
            self.mass_data = RooDataSet("data", "dataset with fM", data_tree, RooArgSet(RooRealVar(var_name, var_name, self.mass_fit_range_min, self.mass_fit_range_max)))

    def reduce_dataset(self, var_name, var_range):
        if self.minimize_flarefly:
            logger(f"Dataset reductions are not implemented in flarefly!", "FATAL")
        if self.verbose:
            logger(f"Reducing RooDataSet with selection {var_range} on variable {var_name}\n", "INFO")
        self.sp_range_min = var_range[0]
        self.sp_range_max = var_range[1]
        self.roofit_sp_var.setRange("sp", self.sp_range_min, self.sp_range_max)

    def add_sgn_func(self, sgn_func, label, particle, vn_func=None):
        if self.fit_vn_vs_mass and vn_func is None:
            logger(f"vn_func must be provided when fitting vn vs mass!", "FATAL")
        # Add the parameters to the dictionary
        self.add_func_to_model('sgn', sgn_func, label, vn_func, particle)

    def add_bkg_func(self, bkg_func, label, vn_func=None):
        if self.fit_vn_vs_mass and vn_func is None:
            logger(f"vn_func must be provided when fitting vn vs mass!", "FATAL")
        # Add the parameters to the dictionary
        logger(f"Adding background function {label} to the fit model", "INFO")
        if vn_func is not None:
            logger(f"With vn function {vn_func}\n", "INFO")
        self.add_func_to_model('bkg', bkg_func, label, vn_func)

    def set_fit_pars(self, cfg, pt_min, pt_max):
        for setting in cfg:
            for pt_range in setting['pt_ranges']:
                pt_center = (pt_range[0] + pt_range[1]) / 2
                if (pt_center >= pt_min and pt_center < pt_max):
                    if self.verbose:
                        logger(f"Found init pars fit cfg for pt range {pt_min} - {pt_max} GeV/c\n", level="INFO")
                    self.cfg_pars_init = setting
                    break

    def set_fit_pars_from_config(self, cfg_pars_dict):
        if self.verbose:
            logger(f"Setting fit parameters from config dict: {cfg_pars_dict}\n", level="INFO")
        self.cfg_pars_init = cfg_pars_dict

    def add_corr_bkgs(self, cfg, corr_bkgs_dir, sel_string, pt_min, pt_max, var_name='fM'):
        # find correlated bkg cocktail associated to this pt-bin
        cocktail_cfg = None
        for cocktail in cfg['cocktails']:
            for pt_range in cocktail['pt_ranges']:
                pt_center = (pt_range[0] + pt_range[1]) / 2
                if (pt_center >= pt_min and pt_center < pt_max):
                    cocktail_cfg = cocktail
                    break
            if cocktail_cfg is not None:
                break

        if cocktail_cfg is None:
            if self.verbose:
                logger(f"No correlated background cocktail found for pt range {pt_min} - {pt_max} GeV/c", level="WARNING")
            return

        pt_label = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        corr_bkg_file = TFile.Open(f"{corr_bkgs_dir}/templs_{pt_label}.root", "READ")
        if self.verbose:
            logger(f"Adding correlated backgrounds to the fitter for pt range {pt_min} - {pt_max} GeV/c", level="INFO")
        self.mass_sgn_templ_name = cfg['sgn_fin_state']
        _, self.mass_sgn_templ_frac = get_corr_bkg(corr_bkg_file, self.mass_sgn_templ_name, sel_string, pt_label, cfg['templ_type'], cfg['output_type'])
        count_sgn_templs, count_bkg_templs = 0, 0
        for i_chn, chn in enumerate(cocktail_cfg['channels']):

            if self.verbose:
                logger(f"Setting correlated bkg source {chn}\n", "INFO")
            # Correlated bkgs that are functions
            if cocktail_cfg.get('sgn_funcs'):
                self.add_func_to_model('sgn', cocktail_cfg['sgn_funcs'][i_chn], chn)
                continue

            name = chn['name']
            output_type = 'hist'
            if self.minimize_flarefly and not self.mass_data.get_is_binned():
                output_type = 'df'
            templ, frac = get_corr_bkg(corr_bkg_file, name, sel_string, pt_label, cfg['templ_type'], output_type)
            if frac < 1e-10:
                if self.verbose:
                    logger(f"Skipping correlated bkg source {chn} with negligible fraction {frac}", level="WARNING")
                continue

            self.fit_model[chn] = {}
            if self.minimize_flarefly:
                self.fit_model[chn]['frac'] = frac
            else:
                self.fit_model[name]['frac'] = RooRealVar(f"frac_{name}", f"frac_{name}", frac)
            self.fit_model[name]['type'] = 'bkg'
            self.fit_model[name]['idx'] = self.n_pdfs_bkg
            self.n_pdfs_bkg += 1
            self.fit_model[chn]['data'] = templ
            self.fit_model[chn]['label'] = chn
            self.fit_model[chn]['chn'] = chn
            if cocktail_cfg.get('fix_to'):
                self.fit_model[chn]['anchor_fix'] = cocktail_cfg['fix_to'][i_chn]
            elif cocktail_cfg.get('init_to'):
                self.fit_model[chn]['anchor_init'] = cocktail_cfg['init_to'][i_chn]
            else:
                if self.verbose:
                    logger(f"Correlated bkg source {self.fit_model[chn]['label']} without 'fix_to' or 'init_to' key", level="WARNING")

            if self.minimize_roofit:
                self.fit_model[name]['RooDataSet'] = RooDataHist(f"dataset_{name}", f"dataset_{name}",
                                                     RooArgList(self.mass_fit_var), self.fit_model[name]['data'])
                self.fit_model[name]['pdf'] = RooHistPdf(name, name, RooArgSet(self.mass_fit_var), self.fit_model[name]['RooDataSet'])
                self.fit_model[name]['yield'] = RooRealVar(f"yield_{name}", f"yield_{name}", 5000, 0, 1e7)
            else:
                self.fit_model[name]['pdf'] = 'hist' if self.mass_data.get_is_binned() else 'kde_grid'

        corr_bkg_file.Close()

    def get_particle_mass(self, part_name):
        if part_name == 'Dplus' or part_name == 'DplusToPiKPi':
            return 1.869
        elif part_name == 'D0':
            return 1.865
        elif part_name == 'Ds':
            return 1.968
        elif part_name == 'Dstar' or part_name == 'DstarD0ToPiKPi':
            return 2.010
        else:
            return 1.869  # default D+ mass

    def fix_sgn_pars_to_first_fit(self):
        self.fix_sgn_to_first_fit = True

    def fix_sgn_par(self, sgn_func_idx, par_name, par_val):
        if self.verbose:
            logger(f"Fixing signal parameter {par_name} of function index {sgn_func_idx} to value {par_val}\n")
        self.fitter.set_signal_initpar(sgn_func_idx, par_name, par_val, fix=True)

    def fix_bkg_par(self, bkg_func_idx, par_name, par_val):
        if self.verbose:
            logger(f"Fixing background parameter {par_name} of function index {bkg_func_idx} to value {par_val}\n")
        self.fitter.set_background_initpar(bkg_func_idx, par_name, par_val, fix=True)

    def set_sgn_par(self, sgn_func_idx, par_name, par_val, lims):
        if self.verbose:
            logger(f"Setting signal parameter {par_name} of function index {sgn_func_idx} to value {par_val}\n")
        self.fitter.set_signal_initpar(sgn_func_idx, par_name, par_val, limits=lims)

    def set_bkg_par(self, bkg_func_idx, par_name, par_val, lims):
        if self.verbose:
            logger(f"Setting background parameter {par_name} of function index {bkg_func_idx} to value {par_val}\n")
        self.fitter.set_background_initpar(bkg_func_idx, par_name, par_val, limits=lims)

    def set_pdf_frac(self, pdf_idx, frac, pdf_type):
        if self.verbose:
            logger(f"Setting fraction of function index {pdf_idx} and type {pdf_type} to {frac}\n")
        if pdf_type == 'sgn':
            self.fitter.set_signal_initpar(pdf_idx, "frac", frac, limits=[0., 1.])
        else:
            self.fitter.set_background_initpar(pdf_idx, "frac", frac, limits=[0., 1.])

    def setup(self):
        if self.minimize_flarefly:
            self.setup_flarefly()
        else:
            self.setup_roofit()

    def fit(self):
        if self.minimize_flarefly:
            return self.perform_fit_flarefly()
        else:
            return self.perform_fit_roofit()

    def setup_flarefly(self):

        self.mass_sgn_pdfs = [v['pdf'] for k, v in self.fit_model.items() if v['type'] == 'sgn']
        self.mass_sgn_pdfs_labels = [k for k, v in self.fit_model.items() if v['type'] == 'sgn']
        self.mass_bkg_pdfs = [v['pdf'] for k, v in self.fit_model.items() if v['type'] == 'bkg']
        self.mass_bkg_pdfs_labels = [k for k, v in self.fit_model.items() if v['type'] == 'bkg']

        # Revert bkg lists to have the comb bkg at the end and adjust indices in fit model accordingly
        self.mass_bkg_pdfs = self.mass_bkg_pdfs[::-1]
        self.mass_bkg_pdfs_labels = self.mass_bkg_pdfs_labels[::-1]
        for label, comp in self.fit_model.items():
            print(f"Component {label} with type {comp['type']} and idx {comp['idx']}")
            if comp['type'] != 'bkg':
                continue
            if 'data' in comp:
                comp['idx'] = bkg_idx
                bkg_idx += 1
            else:  # combinatorial bkg
                comb_key = label
        # put combinatorial background last
        if comb_key is not None:
            self.fit_model[comb_key]['idx'] = bkg_idx # self.n_pdfs_bkg - 1

        self.fitter = F2MassFitter(self.mass_data, name=self.fit_name,
                                   label_signal_pdf=self.mass_sgn_pdfs_labels, name_signal_pdf=self.mass_sgn_pdfs,
                                   name_background_pdf=self.mass_bkg_pdfs, label_bkg_pdf=self.mass_bkg_pdfs_labels,
                                   extended=True if not self.mass_data.get_is_binned() else False)

        # Setup templates data handlers and fractions
        for i_templ, (name, templ) in enumerate(self.fit_model.items()):
            if templ.get('data') is None:
                continue

            if self.verbose:
                logger(f"Setting up template for correlated source {name} with template {templ}\n", "INFO")

            # Create data handler
            if self.mass_data.get_is_binned():
                if self.verbose:
                    logger(f"Setting binned template histogram for source {name}")
                data_hdl = DataHandler(templ['data'], limits=(self.mass_fit_range_min, self.mass_fit_range_max), \
                                       rebin=self.rebin)
            else:
                if self.verbose:
                    logger(f"Setting unbinned template KDE for source {name}")
                data_hdl = DataHandler(templ['data'], limits=(self.mass_fit_range_min, self.mass_fit_range_max), \
                                       nbins=100, var_name="fM")

            # Set template
            if self.mass_data.get_is_binned():
                if self.verbose:
                    logger(f"Setting background template for source {name}, idx {templ['idx']}")
                self.fitter.set_background_template(templ['idx'], data_hdl)
            else:
                if self.verbose:
                    logger(f"Setting background KDE for source {name}, idx {templ['idx']}")
                self.fitter.set_background_kde(templ['idx'], data_hdl)

    def set_fit_pars_flarefly(self):
        # First init, then eventually override with fix
        if self.cfg_pars_init.get("init_pars_sgn"):
            for sett in self.cfg_pars_init["init_pars_sgn"]:
                sgn_func_idx, par_name, par_val, par_lims = sett[0], sett[1], sett[2], sett[3]
                self.set_sgn_par(sgn_func_idx, self.parameter_map(par_name), par_val, par_lims)
                if self.verbose:
                    logger(f"---> setting sgn par {par_name} to value {par_val}, limits {par_lims}\n", "INFO")

        if self.cfg_pars_init.get("init_pars_bkg"):
            for sett in self.cfg_pars_init["init_pars_bkg"]:
                par_name, par_val, par_lims = sett[0], sett[1], sett[2]
                self.set_bkg_par(len(self.mass_bkg_pdfs)-1, par_name, par_val, par_lims)
                if self.verbose:
                    logger(f"---> setting bkg par {par_name} to value {par_val}, limits {par_lims}\n", "INFO")

        if self.cfg_pars_init.get("fix_pars_sgn"):
            for sett in self.cfg_pars_init["fix_pars_sgn"]:
                sgn_func_idx, par_name, par_val = sett[0], sett[1], sett[2]
                self.fix_sgn_par(sgn_func_idx, self.parameter_map(par_name), par_val)
                if self.verbose:
                    logger(f"---> fixing sgn par {par_name} to value {par_val}", "INFO")

        if self.cfg_pars_init.get("fix_pars_bkg"):
            for sett in self.cfg_pars_init["fix_pars_bkg"]:
                par_name, par_val = sett[0], sett[1]
                self.set_bkg_par(len(self.mass_bkg_pdfs)-1, par_name, par_val, fix=True)
                if self.verbose:
                    logger(f"---> fixing bkg par {par_name} to value {par_val}", "INFO")

        if self.cfg_pars_init.get("fix_sgn_from_file"):
            # Initialization from MC fits from file
            for sett in self.cfg_pars_init["fix_sgn_from_file"]:
                sgn_func_idx, par_names, file_pars = sett[0], sett[1], sett[2]
                if self.verbose:
                    logger(f"Opening file {file_pars} to fix signal parameters {par_names}", "INFO")
                par_file = TFile.Open(file_pars, "READ")
                for par_name in par_names:
                    try:
                        histo_par = par_file.Get(f"hist_{par_name}")
                        for i_bin in range(histo_par.GetNbinsX()+1):
                            bin_center = histo_par.GetBinCenter(i_bin)
                            if bin_center > pt_min and bin_center < pt_max:
                                par_val = histo_par.GetBinContent(i_bin)
                                break
                        # TODO: Shift the mean or add smearing to compensate data-MC discrepancies
                        if self.verbose:
                            logger(f"---> fixing signal parameter {par_name} to value {par_val}, shift {shift}, smear {smear}", "INFO")
                        self.fix_sgn_par(sgn_func_idx, self.parameter_map(par_name), par_val)
                    except Exception as e:
                        if self.verbose:
                            logger(f"        Parameter {par_name} not present!", "WARNING")

                par_file.Close()

    def perform_fit_flarefly(self):

        # Set particle mass and sigma initial par for the main signal
        # function here, so they can be overridden later if needed
        self.fitter.set_particle_mass(len(self.mass_sgn_pdfs)-1, pdg_id=self.particle_pdg)
        self.fitter.set_signal_initpar(len(self.mass_sgn_pdfs)-1, "sigma", 0.015, limits=[0.005, 0.05])
        self.fitter.set_background_initpar(len(self.mass_bkg_pdfs)-1, "c0", 1000.0, limits=[0.0, 1e6])         # Resonable value for c0
        self.fitter.set_background_initpar(len(self.mass_bkg_pdfs)-1, "c1", 0.0, limits=[-1000.0, 1000.0])     # Resonable value for c1
        self.fitter.set_background_initpar(len(self.mass_bkg_pdfs)-1, "c2", 0.0, limits=[-1000.0, 1000.0])     # Resonable value for c2

        # Setup templates data handlers and fractions
        for i_templ, (name, templ) in enumerate(self.fit_model.items()):
            if templ.get('data') is None:
                continue

            # Set fraction
            anchor_mode = 'anchor_fix' if templ.get('anchor_fix') else 'anchor_init' if templ.get('anchor_init') else None
            if anchor_mode is None:
                if self.verbose:
                    logger(f"Correlated bkg source {name} without 'fix_to' or 'init_to' key", level="WARNING")
                continue
            anchor_func = templ[anchor_mode]

            # Retrieve type and index of anchor function
            anchor_pdf_frac = self.mass_sgn_templ_frac if anchor_func == "DplusToPiKPi" else self.fit_model[anchor_func]['frac']
            anchor_pdf_idx = self.fit_model[anchor_func]['idx']
            frac = templ['frac'] / anchor_pdf_frac
            if self.verbose:
                logger(f"frac of chn {name} wrt anchor {templ[anchor_mode]}: " \
                    f"{templ['frac']} / {anchor_pdf_frac} = {frac}", "INFO")

            if templ.get('anchor_fix'):
                if self.fit_model[anchor_func]['type'] == 'bkg':
                    self.fitter.fix_bkg_frac_to_bkg_pdf(templ['idx'], anchor_pdf_idx, frac)
                else:
                    self.fitter.fix_bkg_frac_to_signal_pdf(templ['idx'], anchor_pdf_idx, frac)
            else:
                self.set_pdf_frac(templ['idx'], frac, templ['type'])

        if self.verbose:
            logger(f"Performing flarefly fit on data with fit range {self.mass_fit_range_min} - {self.mass_fit_range_max} GeV/c\n", "INFO")

        for i_sgn_func, (name, sgn_func) in enumerate(self.fit_model.items()):
            if name not in self.mc_pars:
                continue
            for par_name, par_val in self.mc_pars[name].items():
                if par_name == "mu" or par_name == "sigma" or par_name == "frac":
                    continue
                if self.fix_sgn_to_mc_prefit:
                    if self.verbose:
                        logger(f"Fixing prefit MC parameter {par_name} to value {par_val}", "WARNING")
                    self.fix_sgn_par(self.fit_model[name]['idx'], par_name, par_val)
                else:
                    if self.verbose:
                        logger(f"Setting prefit MC parameter {par_name} to value {par_val}", "WARNING")
                    self.set_sgn_par(self.fit_model[name]['idx'], par_name, par_val, lims=[par_val*0.1, par_val*10])

        if self.cfg_pars_init is not None:
            self.set_fit_pars_flarefly()

        if self.fit_counter > 0 and self.fix_sgn_to_first_fit:
            for i_sgn_func, sgn_pars_dict in enumerate(self.first_fit_pars):
                for par_name, par_val in sgn_pars_dict.items():
                    if "frac" in par_name:
                        if i_sgn_func == 0:
                            continue
                        else:
                            frac_first = self.first_fit_pars[0]['frac']
                            frac_this = par_val
                            frac_ratio = frac_this / frac_first
                            if self.verbose:
                                logger(f"Fixing fraction of signal function index {i_sgn_func} to first signal function with ratio {frac_ratio}")
                            self.fitter.fix_signal_frac_to_signal_pdf(i_sgn_func, 0, frac_ratio)
                            continue
                    if self.verbose:
                        logger(f"Fixing signal parameter {par_name} of function index {i_sgn_func} to first fit value {par_val}\n")
                    self.fitter.set_signal_initpar(i_sgn_func, par_name, par_val, fix=True)

        self.mass_fit_result = self.fitter.mass_zfit()

        if self.fit_counter <= 0 and self.fix_sgn_to_first_fit:
            self.first_fit_pars = copy.deepcopy(self.fitter.get_signal_pars())
            if self.verbose:
                logger(f"Stored signal params of the first fit!\n", "WARNING")
        self.fit_counter += 1
        return self.mass_fit_result.status, self.mass_fit_result.converged

    def plot_mc_prefit(self, logy, show_extra_info, loc=None, path=None, out_file=None):
        os.makedirs(path, exist_ok=True)
        for i_sgn_func, (name, sgn_func) in enumerate(self.fit_model.items()):
            if sgn_func['type'] != 'sgn':
                continue
            fig_path = f"{path}/{self.fit_model[name]['label']}_mc_prefit_pt_{int(self.pt_min*10)}_{int(self.pt_max*10)}.pdf"
            if self.minimize_flarefly:
                fig, axs = self.fit_model[name]['mcfitter'].plot_mass_fit(style="ATLAS",
                                                            figsize=(8, 8),
                                                            axis_title=self.x_axis_label,
                                                            show_extra_info=show_extra_info,
                                                            logy=logy,
                                                            extra_info_loc=loc if loc is not None else ["lower right", "lower left"]
                                                            )
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            else:
                # Set range
                frame = self.mass_fit_var.frame(RooFit.Title(f"{self.fit_model[name]['label']} MC Prefit"))
                self.fit_model[name]['mchist'].plotOn(frame)
                self.fit_model[name]['pdf'].plotOn(
                    frame,
                    RooFit.Range(f"mc_fit_{name}"),
                    RooFit.Normalization(
                        self.fit_model[name]['mchist'].sumEntries(),
                        RooAbsReal.NumEvent
                    )
                )
                canvas = ROOT.TCanvas("mc_prefit_canvas", "MC Prefit Canvas", 800, 600)
                frame.Draw()
                canvas.Update()
                canvas.SaveAs(fig_path)
                if out_file is not None:
                    out_file.cd()
                    canvas.Write(f"{self.fit_model[name]['label']}_mc_prefit")

    def plot_raw_residuals_mc_prefit(self, path):
        for i_sgn_func, (name, sgn_func) in enumerate(self.fit_model.items()):
            if sgn_func['type'] != 'sgn':
                continue
            if self.minimize_flarefly:
                fig_res = self.fit_model[name]['mcfitter'].plot_raw_residuals(style="ATLAS",
                                                        figsize=(8, 8),
                                                        axis_title=self.x_axis_label)
                fig_res.savefig(path, dpi=300, bbox_inches="tight")
            else:
                if self.verbose:
                    logger("Raw residuals plot for RooFit MC prefit not implemented yet", "WARNING")

    def plot_std_residuals_mc_prefit(self, path):
        if self.minimize_flarefly:
            fig_pulls = self.fit_model[name]['mcfitter'].plot_std_residuals(style="ATLAS",
                                                    figsize=(8, 8),
                                                    axis_title=self.x_axis_label)
            fig_pulls.savefig(path, dpi=300, bbox_inches="tight")
        else:
            if self.verbose:
                logger("Pulls plot for RooFit MC prefit not implemented yet", "WARNING")

    def plot_mass_fit(self, logy, show_extra_info, loc=None, path=None, out_file=None):
        if self.verbose:
            logger(f"Plotting fit to {path}\n", "INFO")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.minimize_flarefly:
            fig, axs = self.fitter.plot_mass_fit(style="ATLAS",
                                                figsize=(8, 8),
                                                axis_title=self.x_axis_label,
                                                show_extra_info=show_extra_info,
                                                logy=logy,
                                                extra_info_loc=loc if loc is not None else ["lower right", "lower left"]
                                                )
            # fig.savefig(path, dpi=300, bbox_inches="tight")
            return fig
        else:
            # --- Bin setup ---
            nbins = int((self.mass_fit_range_max - self.mass_fit_range_min) * 1000 / self.rebin) - 1

            # --- RooPlot frame ---
            frame = self.mass_fit_var.frame(
                RooFit.Bins(nbins),
                RooFit.Title(
                    f";M(#pi K#pi) (GeV/#it{{c}}^{{2}});Counts per {self.rebin} MeV/#it{{c}}^{{2}}"
                )
            )

            # --- Legend ---
            self.legend = TLegend(0.20, 0.77 - 0.05 * len(self.fit_model), 0.45, 0.82)
            self.legend.SetBorderSize(0)
            self.legend.SetFillStyle(0)
            self.legend.SetTextSize(0.035)

            # --- Plot data ---
            self.mass_data_fit.plotOn(
                frame,
                RooFit.Range("fit"),
                # RooFit.Binning(nbins),
                RooFit.MarkerStyle(ROOT.kFullCircle),
                RooFit.MarkerSize(0.8),
                RooFit.LineColor(ROOT.kBlack),
                RooFit.DrawOption("PE0")  # vertical error bars, no endcaps
            )

            legend_dummies = []
            # Create a dummy TGraph with one point
            dummy_data = ROOT.TGraph(1)
            dummy_data.SetMarkerStyle(ROOT.kFullCircle)
            dummy_data.SetMarkerColor(ROOT.kBlack)
            dummy_data.SetMarkerSize(0.8)
            self.legend.AddEntry(dummy_data, "Data", "pe")
            legend_dummies.append(dummy_data)  # keep reference

            # --- Add entries for model components ---
            for label, pdf_dict in self.fit_model.items():
                info = pdf_dict.get("plot_info", {})

                # Plot the PDF on the frame
                args = [frame, RooFit.Components(pdf_dict["label"]), RooFit.Range("fit"), RooFit.Binning(nbins)]
                if "line_color" in info: args.append(RooFit.LineColor(info["line_color"]))
                if "line_width" in info: args.append(RooFit.LineWidth(info["line_width"]))
                if "line_style" in info: args.append(RooFit.LineStyle(info["line_style"]))
                if "fill_color" in info: args.append(RooFit.FillColor(info["fill_color"]))
                if "fill_style" in info: args.append(RooFit.FillStyle(info["fill_style"]))
                if "draw_option" in info: args.append(RooFit.DrawOption(info["draw_option"]))

                self.mass_model.plotOn(*args)

                # Create persistent dummy for legend
                if info.get("draw_option", "L") == "F":
                    dummy = TBox(0,0,1,1)
                    dummy.SetFillColor(info["fill_color"])
                    dummy.SetFillStyle(info["fill_style"])
                    self.legend.AddEntry(dummy, label, "f")
                else:
                    dummy = TLine(0,0,1,1)
                    dummy.SetLineColor(info["line_color"])
                    dummy.SetLineWidth(info.get("line_width", 2))
                    dummy.SetLineStyle(info.get("line_style", 1))
                    self.legend.AddEntry(dummy, label, "l")

                legend_dummies.append(dummy)  # keep reference

            # --- Total fit curve ---
            total_curve = self.mass_model.plotOn(
                frame,
                RooFit.Binning(nbins),
                RooFit.Range("fit"),
                RooFit.LineColor(ROOT.kAzure + 4),
                RooFit.LineWidth(4)
            )
            dummy = TLine(0,0,1,1)
            dummy.SetLineColor(ROOT.kAzure + 4)
            dummy.SetLineWidth(4)
            self.legend.AddEntry(dummy, "Total fit", "l")
            legend_dummies.append(dummy)  # keep reference

            # --- Canvas ---
            canvas = TCanvas("fit_canvas", "Fit Canvas", 600, 600)
            canvas.SetLeftMargin(0.14)
            canvas.SetTopMargin(0.12)
            canvas.SetBottomMargin(0.12)
            canvas.SetTicks(1, 1)  # ticks on all sides

            # --- Axis formatting ---
            frame.GetXaxis().SetTitleOffset(1.20)
            frame.GetYaxis().SetTitleOffset(1.35)
            frame.GetXaxis().SetTitleSize(0.042)
            frame.GetYaxis().SetTitleSize(0.042)
            frame.GetYaxis().SetMoreLogLabels()
            frame.GetYaxis().SetNoExponent(False)
            frame.GetYaxis().SetLabelSize(0.04)
            frame.GetYaxis().SetLabelFont(42)
            frame.GetYaxis().SetMaxDigits(3)

            # --- Draw frame and legend ---
            frame.Draw()
            self.legend.Draw()

            # --- Optional: Canvas title using TLatex ---
            canva_title = f"{self.pt_min} < #it{{p}}_{{T}} < {self.pt_max} GeV/#it{{c}}, " \
                          f"{self.sp_range_min:.2f} < SP < {self.sp_range_max:.2f}" \
                          if self.sp_range_min is not None and self.sp_range_max is not None \
                          else f"{self.pt_min} < #it{{p}}_{{T}} < {self.pt_max} GeV/#it{{c}}"
            latex = ROOT.TLatex()
            latex.SetNDC()
            latex.SetTextAlign(22)
            latex.SetTextFont(42)
            latex.SetTextSize(0.045)
            latex.DrawLatex(0.5, 0.94, canva_title)
            fit_info, _, _, _, _ = self.get_fit_info()
            ry_text = f"Signal: {fit_info['sgn']['ry']:.2f} #pm {fit_info['sgn']['ry_unc']:.2f}"
            print(f"Fit info[Debug]: {fit_info}")
            latex.DrawLatex(0.5, 0.88, ry_text)

            canvas.Update()
            canvas.SaveAs(path)
            if out_file is not None:
                if self.verbose:
                    logger(f"Writing fit canvas to output file with name fit_canvas_{self.fit_name}", "INFO")
                out_file.cd()
                canvas.Write(f"fit_canvas_{self.fit_name}")

            frame.Print("v")
            if self.verbose:
                logger(f"Plot saved to {path}", "INFO")


    def plot_vn_vs_mass_fit(self, logy, show_extra_info, loc=None, path=None, out_file=None):
        if self.verbose:
            logger(f"Plotting fit to {path}\n", "INFO")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 1. Setup the frame with explicit range and binning for the data points
        logger("Creating RooPlot frame for vn vs mass fit", "INFO")
        frame = self.vn_mass_var.frame(
            RooFit.Range(self.mass_fit_range_min, self.mass_fit_range_max),
            RooFit.Title(f";M(#pi K#pi) (GeV/#it{{c}}^{{2}});Cand. #it{{v}}_{{2}}")
        )

        # 2. Plot Data
        logger("Plotting vn vs mass data points", "INFO")
        self.vn_vs_mass_data.plotOn(
            frame,
            RooFit.MarkerStyle(ROOT.kFullCircle),
            RooFit.MarkerSize(0.8),
            RooFit.LineColor(ROOT.kBlack),
            RooFit.DrawOption("PE0"),
            RooFit.Name("data_points") # Named for legend potential
        )

        # # 3. Setup Legend
        # logger("Setting up legend for vn vs mass fit", "INFO")
        # self.legend = ROOT.TLegend(0.20, 0.77 - 0.05 * len(self.fit_model), 0.45, 0.85)
        # self.legend.SetBorderSize(0)
        # self.legend.SetFillStyle(0)
        # self.legend.AddEntry("data_points", "Data", "pe")

        # 4. Plot Components
        # We plot the 'vn_func' from your dictionary, NOT the total model with RooFit.Components
        logger("Plotting vn vs mass fit components", "INFO")
        for label, pdf_dict in self.fit_model.items():
            logger(f"Plotting vn component for {label}", "INFO")

            info = pdf_dict.get("plot_info", {})

            # THIS is the correct object
            vn_comp_func = pdf_dict['vn_term']

            plot_args = [
                RooFit.LineColor(info.get("line_color", ROOT.kRed)),
                RooFit.LineWidth(info.get("line_width", 2)),
                RooFit.LineStyle(info.get("line_style", 1)),
                RooFit.Name(f"curve_{label}")
            ]

            vn_comp_func.plotOn(frame, *plot_args)

        # logger("Plotting vn vs mass fit components", "INFO")
        # for label, pdf_dict in self.fit_model.items():
        #     logger(f"Plotting vn component for {label}", "INFO")
        #     info = pdf_dict.get("plot_info", {})
        #     vn_comp_func = pdf_dict['vn_func'] # This is the RooAbsReal for this component's v2
            
        #     # Define plot arguments
        #     logger(f"Getting plot_args", "INFO")
        #     plot_args = [RooFit.LineColor(info.get("line_color", ROOT.kRed)),
        #                  RooFit.LineWidth(info.get("line_width", 2)),
        #                  RooFit.LineStyle(info.get("line_style", 2)),
        #                  RooFit.Name(f"curve_{label}")]

        #     logger(f"Plotting on frame", "INFO")
        #     vn_comp_func.plotOn(frame, *plot_args)
        #     logger(f"Plotted vn component for {label}", "INFO")
        #     # Add to legend
        #     # self.legend.AddEntry(f"curve_{label}", label, "l")
        #     logger(f"Added legend entry for {label}", "INFO")

        # 5. Plot Total Model (The Ratio)
        logger("Plotting total vn vs mass fit model", "INFO")
        print(f"self.vn_vs_mass_model: {self.vn_vs_mass_model}")
        self.vn_vs_mass_model.plotOn(
            frame,
            RooFit.LineColor(ROOT.kAzure + 4),
            RooFit.LineWidth(4),
            RooFit.Name("total_fit")
        )
        logger("Plotted total vn vs mass fit model", "INFO")
        # self.legend.AddEntry("total_fit", "Total fit", "l")
        logger("Added legend entry for total fit", "INFO")

        # 6. Final Drawing and Canvas formatting
        canvas = ROOT.TCanvas(f"c_{self.fit_name}", "Fit Canvas", 700, 600)
        canvas.SetLeftMargin(0.15)

        # Set Y-axis range manually if needed (v2 is usually small)
        frame.SetMinimum(-0.05)
        frame.SetMaximum(0.35)

        logger("Formatting axes", "INFO")
        frame.Draw()
        # self.legend.Draw()

        logger("Adding canvas title", "INFO")
        # --- Optional: Canvas title using TLatex ---
        canva_title = f"{self.pt_min} < #it{{p}}_{{T}} < {self.pt_max} GeV/#it{{c}}, " \
                      f"{self.sp_range_min:.2f} < SP < {self.sp_range_max:.2f}" \
                      if self.sp_range_min is not None and self.sp_range_max is not None \
                      else f"{self.pt_min} < #it{{p}}_{{T}} < {self.pt_max} GeV/#it{{c}}"
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextAlign(22)
        latex.SetTextFont(42)
        latex.SetTextSize(0.045)
        latex.DrawLatex(0.5, 0.94, canva_title)

        canvas.Update()
        canvas.SaveAs(path)
        if out_file is not None:
            if self.verbose:
                logger(f"Writing fit canvas to output file with name fit_canvas_{self.fit_name}", "INFO")
            out_file.cd()
            canvas.Write(f"fit_canvas_{self.fit_name}")

        if self.verbose:
            logger(f"Plot saved to {path}", "INFO")


    def plot_raw_residuals(self):
        fig_res = self.fitter.plot_raw_residuals(style="ATLAS",
                                                 figsize=(8, 8),
                                                 axis_title=self.x_axis_label)
        return fig_res

    def plot_std_residuals(self):
        fig_pulls = self.fitter.plot_std_residuals(style="ATLAS",
                                                   figsize=(8, 8),
                                                   axis_title=self.x_axis_label)
        return fig_pulls

    def get_data(self):
        if self.minimize_flarefly:
            return self.mass_data.to_pandas()['fM'].to_numpy()

    def get_sweights_sgn(self, label):

        if self.minimize_flarefly:
            for i_sgn_func, (name, pdf_dict) in enumerate(self.fit_model.items()):
                if pdf_dict['type'] != 'sgn':
                    continue
                print(f"Checking signal function {name} with label {pdf_dict['label']} vs requested {label}")
                if self.fit_model[name]['label'] == label:
                    sgn_sw_idx = self.fit_model[name]['idx']
            return self.fitter.get_sweights()[f"signal{sgn_sw_idx}"]
        else:
            if self.verbose:
                logger("SWeights extraction for RooFit not implemented yet", "ERROR")
            sys.exit(1)

    def get_fitter(self):
        return self.fitter

    def get_name(self):
        return self.fit_name

    def get_bkg_yield(self, mass_min, mass_max):
        if self.minimize_flarefly:
            if self.verbose:
                logger("Getting background yield with flarefly not implemented yet", "WARNING")
            return 0.0
        else:
            # define the range on the PDF variable
            self.mass_fit_var.setRange("subrange", mass_min, mass_max)

            # fraction of PDF in that range
            frac = self.fit_model['Comb_Bkg']['pdf'].createIntegral(
                RooArgSet(self.mass_fit_var),
                RooFit.NormSet(self.mass_fit_var),
                RooFit.Range("subrange")
            ).getVal()
            nbkg = self.fit_model['Comb_Bkg']['yield'].getVal()  # fitted yield
            bkg_yield = frac * nbkg
            # Define the formula: fraction * yield
            bkg_in_range = RooFormulaVar(
                "bkg_in_range",
                "@0 * @1",                    # formula: fraction * yield
                RooArgList(
                    RooFit.RooConst(frac),   # frac is constant, no error
                    self.fit_model['Comb_Bkg']['yield']  # RooRealVar with fitted error
                )
            )

            # propagate uncertainty from the fit
            if hasattr(self, "fit_result") and self.mass_fit_result:
                err = bkg_in_range.getPropagatedError(self.mass_fit_result)
            else:
                err = 0.0

            return bkg_yield, err

    def get_fit_info(self):

        fit_info = {}
        try:
            if self.minimize_roofit:
                # RooFit binned fit: chi2 from model vs data (createChi2)
                chi2_obj = self.model.createChi2(self.data, ROOT.RooFit.Range("fit"), ROOT.RooFit.NumCPU(1))
                chi2_val = float(chi2_obj.getVal())
                n_bins = self.hist.GetXaxis().FindBin(self.fit_range_max) - self.hist.GetXaxis().FindBin(self.fit_range_min) + 1
                n_free = int(self.fit_result.floatParsFinal().getSize())
                ndf = max(n_bins - n_free, 1)
                fit_info['chi2'] = chi2_val
                fit_info['chi2_over_ndf'] = chi2_val / ndf
            else:
                fit_info['chi2'] = float(self.fitter.get_chi2())
                fit_info['chi2_over_ndf'] = float(self.fitter.get_chi2())/self.fitter.get_ndf()
        except Exception as e:
            if self.verbose:
                logger(f"Could not get chi2: {e}", "WARNING")
            fit_info['chi2'] = -1.
            fit_info['chi2_over_ndf'] = -1.

        for i_sgn_func, (name, sgn_func) in enumerate(self.fit_model.items()):
            if sgn_func['type'] != 'sgn':
                continue
            fit_info[name] = {}
            sgn_func_idx = self.fit_model[name]['idx']
            try:
                if self.minimize_flarefly:
                    fit_info[name]['ry'] = self.fitter.get_raw_yield(sgn_func_idx)[0]
                    fit_info[name]['ry_unc'] = self.fitter.get_raw_yield(sgn_func_idx)[1]
                else:
                    fit_info[name]['ry'] = self.fit_model[name]['yield'].getVal()
                    fit_info[name]['ry_unc'] = self.fit_model[name]['yield'].getError()
            except Exception as e:
                if self.verbose:
                    logger(f"Could not get raw yield: {e}", "WARNING")
                fit_info[name]['ry'] = -1.
                fit_info[name]['ry_unc'] = -1.
            try:
                fit_info[name]['ry_bin_counting'] = self.fitter.get_raw_yield_bincounting(sgn_func_idx, nsigma=5)[0]
                fit_info[name]['ry_bin_counting_unc'] = self.fitter.get_raw_yield_bincounting(sgn_func_idx, nsigma=5)[1]
            except Exception as e:
                if self.verbose:
                    logger(f"Could not get raw yield from bin counting: {e}", "WARNING")
                fit_info[name]['ry_bin_counting'] = -1.
                fit_info[name]['ry_bin_counting_unc'] = -1.
            try:
                fit_info[name]['signif'] = self.fitter.get_significance(sgn_func_idx)[0]
                fit_info[name]['signif_unc'] = self.fitter.get_significance(sgn_func_idx)[1]
            except Exception as e:
                if self.verbose:
                    logger(f"Could not get significance: {e}", "WARNING")
                fit_info[name]['signif'] = -1.
                fit_info[name]['signif_unc'] = -1.
            try:
                fit_info[name]['s_over_b'] = self.fitter.get_signal_over_background(sgn_func_idx)[0]
                fit_info[name]['s_over_b_unc'] = self.fitter.get_signal_over_background(sgn_func_idx)[1]
            except Exception as e:
                if self.verbose:
                    logger(f"Could not get signal over background: {e}", "WARNING")
                fit_info[name]['s_over_b'] = -1.
                fit_info[name]['s_over_b_unc'] = -1.
            try:
                if self.minimize_flarefly:
                    fit_info[name]['sigma'] = self.fitter.get_signal_pars()[sgn_func_idx]['sigma']
                    fit_info[name]['sigma_unc'] = self.fitter.get_signal_pars_uncs()[sgn_func_idx]['sigma']
                else:
                    fit_info[name]['sigma'] = self.fit_model[name]['par_sigma'].getVal()
                    fit_info[name]['sigma_unc'] = self.fit_model[name]['par_sigma'].getError()
            except Exception as e:
                if self.verbose:
                    logger(f"Could not get sigma: {e}", "WARNING")
                fit_info[name]['sigma'] = -1.
                fit_info[name]['sigma_unc'] = -1.

            signal_pars = {}
            signal_pars_uncs = {}
            bkg_pars = {}
            bkg_pars_uncs = {}
            if self.minimize_flarefly:
                signal_pars_list = self.fitter.get_signal_pars()
                signal_pars_uncs_list = self.fitter.get_signal_pars_uncs()
                bkg_pars_list = self.fitter.get_bkg_pars()
                bkg_pars_uncs_list = self.fitter.get_bkg_pars_uncs()
                signal_pars
                for i_comp, (name, comp) in enumerate(self.fit_model.items()):
                    if comp['type'] == 'sgn':
                        for par_name, par_val in signal_pars_list[comp['idx']].items():
                            signal_pars[f"{par_name}_{name}"] = par_val
                            signal_pars_uncs[f"{par_name}_{name}"] = signal_pars_uncs_list[comp['idx']][par_name]
                    else:
                        for par_name, par_val in bkg_pars_list[comp['idx']].items():
                            bkg_pars[f"{par_name}_{name}"] = par_val
                            bkg_pars_uncs[f"{par_name}_{name}"] = bkg_pars_uncs_list[comp['idx']][par_name]
            else:
                for name, comp in self.fit_model.items():
                    if comp['type'] == 'sgn':
                        for par in comp['pdf'].getParameters(self.mass_data):
                            signal_pars[par.GetName()] = par.getVal()
                            signal_pars_uncs[par.GetName()] = par.getError()
                    else:
                        for par in comp['pdf'].getParameters(self.mass_data):
                            bkg_pars[par.GetName()] = par.getVal()
                            bkg_pars_uncs[par.GetName()] = par.getError()

        return fit_info, signal_pars, signal_pars_uncs, bkg_pars, bkg_pars_uncs

    def reset(self):
        logger(f"############## Resetting fitter ##############", "INFO")
        self.mass_sgn_templ_frac = None
        self.fit_model = {}
        self.mass_sgn_pdfs = None
        self.mass_sgn_pdfs_labels = None
        self.mass_bkg_pdfs = None
        self.mass_bkg_pdfs_labels = None
        self.cfg_pars_init = None
        self.n_pdfs_bkg = 0
        self.n_pdfs_sgn = 0
        self.mass_model = None

    def init_sgn_pars(self, pars_dict, sgn_func_label):
        # Init mean and sigma, tail parameters are always taken from MC prefit
        if self.verbose:
            logger(f"Initializing mean to {pars_dict['mu']} and sigma to {pars_dict['sigma']} for signal function {sgn_func_label}", "INFO")
        sgn_func_idx = self.fit_model[sgn_func_label]['idx']
        if self.minimize_flarefly:
            self.fitter.set_signal_initpar(sgn_func_idx, "mu", pars_dict["mu"], limits=[1.8, 2.0])
            self.fitter.set_signal_initpar(sgn_func_idx, "sigma", pars_dict["sigma"], limits=[0.005, 0.1])
        else:
            self.fit_model[sgn_func_label]["par_mu"].setVal(pars_dict["mu"])
            self.fit_model[sgn_func_label]["par_sigma"].setVal(pars_dict["sigma"])

    def init_comb_bkg_pars(self, pars_dict):
        for par_name, par_val in pars_dict.items():
            par_name = par_name.split("_")[0]
            if self.verbose:
                logger(f"Initializing comb bkg parameter {par_name} to value {par_val}", "INFO")
            if self.minimize_flarefly:
                self.fitter.set_background_initpar(len(self.mass_bkg_pdfs)-1, par_name, par_val, limits=[-1000.0, 1000.0])     # Resonable value for c1
            else:
                self.fit_model["Comb_Bkg"][f"par_{par_name}"].setVal(par_val)
            if self.verbose:
                logger(f"Set comb bkg parameter {par_name} to value {par_val}", "INFO")

    def setup_roofit(self):

        if self.verbose:
            logger(f"Setting up RooFit PDFs for fit range {self.mass_fit_range_min} - {self.mass_fit_range_max} GeV/c\n", "INFO")

        # Create extended PDFs for signal and background
        for i_comp, (name, comp) in enumerate(self.fit_model.items()):
            if comp.get('data'): # correlated bkgs
                if comp.get('anchor_fix'):
                    anchor_mode = 'anchor_fix'
                elif comp.get('anchor_init'):
                    anchor_mode = 'anchor_init'
                else:
                    if self.verbose:
                        logger(f"Correlated bkg source {name} without 'fix_to' or 'init_to' key", level="WARNING")
                    continue
                anchor_func = comp[anchor_mode]
                anchor_pdf_idx = self.fit_model[anchor_func]['idx']
                anchor_pdf_frac = self.mass_sgn_templ_frac if anchor_func == self.mass_sgn_templ_name else self.fit_model[anchor_func]['frac']
                frac = comp['frac'].getVal() / anchor_pdf_frac
                logger(f"frac of chn {comp['label']} wrt anchor {comp[anchor_mode]}: {comp['frac'].getVal()} / {anchor_pdf_frac} = {frac}", "WARNING")
                comp['frac'].setVal(frac)
                comp['frac'].setMin(0.)
                comp['frac'].setMax(1.)
                if anchor_mode == "anchor_fix":
                    if self.verbose:
                        logger(f"Fixing fraction of component {comp['label']} to value {frac}", "WARNING")
                    comp['frac'].setConstant(True)
                comp['yieldRooLinearVar'] = RooLinearVar(    # a * X + b = frac * anchor_yield + 0
                    f"yield_{name}_lin",
                    f"yield_{name}_lin",
                    self.fit_model[anchor_func]['yield'],         # X  (the anchor yield)
                    RooFit.RooConst(comp['frac'].getVal()),  # constant scale
                    RooFit.RooConst(0.0)                     # offset
                )
                if self.verbose:
                    logger(f"Created RooLinearVar for yield of component {comp['label']}: {comp['yieldRooLinearVar'].GetName()} "
                           f"with value {comp['yieldRooLinearVar'].getVal()}", "INFO")

            if self.verbose:
                logger(f"Creating extended RooFit PDFs for component: {comp}", "INFO")
            comp['ext_pdf'] = RooExtendPdf(f"ext_{comp['label']}", f"extended_{comp['label']}",
                                                comp['pdf'], comp.get('yieldRooLinearVar', comp['yield']))

    def perform_fit_roofit(self):
        if self.verbose:
            logger(f"Performing RooFit fit on data with fit range {self.mass_fit_range_min} - {self.mass_fit_range_max} GeV/c", "INFO")

        if self.fit_counter == 0:
            if self.fix_sgn_to_mc_prefit:
                for i_sgn_func, (name, sgn_func) in enumerate(self.fit_model.items()):
                    if sgn_func['type'] != 'sgn':
                        continue

                    for par in self.mc_pars[name]:
                        par_name = par.GetName().split("_")[0]
                        if "mu" in par_name or "sigma" in par_name or "yield" in par_name:
                            continue
                        par_val = par.getVal()
                        if self.verbose:
                            logger(f"Fixing parameter {par_name} of last signal pdf to MC prefit value {par_val}", "WARNING")
                        self.fit_model[name][f"par_{par_name}"].setConstant(True)

            self.mass_model = RooAddPdf(" + ".join([comp['label'] for comp in self.fit_model.values()]),
                                        " + ".join([comp['label'] for comp in self.fit_model.values()]),
                                        RooArgList([comp['pdf'] for comp in self.fit_model.values()]),
                                        RooArgList([comp.get('yieldRooLinearVar', comp['yield']) for comp in self.fit_model.values()]))
                                        # RooArgList([comp['ext_pdf'] for comp in self.fit_model.values()]))
            # Check the yield vars of the single components
            if self.verbose:
                logger("=== Yield variables of the fit components ===", "WARNING")
                for comp in self.fit_model.values():
                    yield_var = comp.get('yieldRooLinearVar', comp['yield'])
                    logger(f"Component {comp['label']}: yield variable = {yield_var.GetName()}, value = {yield_var.getVal()}", "INFO")
            self.mass_fit_var.setRange("fit", self.mass_fit_range_min, self.mass_fit_range_max)
            # self.mass_data = self.mass_data.reduce(RooFit.Range(self.mass_fit_range_min, self.mass_fit_range_max))

        # Dataset with sp cut
        if not isinstance(self.hist, ROOT.TH1):
            self.mass_data_fit = self.mass_data.reduce(
                RooFit.Cut(f"fScalarProd >= {self.sp_range_min} && fScalarProd < {self.sp_range_max}"),
                # RooFit.Range(self.sp_range_min, self.sp_range_max)
            )
        if isinstance(self.hist, ROOT.TH1):
            self.hist.Rebin(self.rebin)
            print(f"Rebinned histogram with factor {self.rebin}")
            self.mass_data_fit = RooDataHist(
                "data_rebinned",
                "data_rebinned",
                RooArgSet(self.mass_fit_var),
                self.hist
            )
        print(f"self.mass_data_fit.numEntries() = {self.mass_data_fit.numEntries()}")
        self.mass_fit_result = self.mass_model.fitTo(
            self.mass_data_fit,
            RooFit.Extended(True),
            RooFit.Range("fit"),
            RooFit.SumW2Error(True),
            RooFit.Save(True),
            RooFit.PrintLevel(1 if self.verbose else -1),
            RooFit.PrintEvalErrors(1 if self.verbose else 0)
        )

        if self.verbose:
            logger("=== Fit status ===", "WARNING")
            logger(f"status      = {self.mass_fit_result.status()}", "INFO")
            logger(f"covQual     = {self.mass_fit_result.covQual()}", "INFO")
            logger(f"edm         = {self.mass_fit_result.edm()}", "INFO")
            logger(f"minNll      = {self.mass_fit_result.minNll()}", "INFO")
            logger("=== Mass Fit Results Floating parameters ===", "WARNING")
            self.mass_fit_result.floatParsFinal().Print("v")
            logger("=== Constant parameters ===", "WARNING")
            self.mass_fit_result.constPars().Print("v")
            logger("=== Correlation matrix ===", "WARNING")
            self.mass_fit_result.correlationMatrix().Print()

        if self.fit_counter <= 0 and self.fix_sgn_to_first_fit:
            for name, sgn_func in self.fit_model.items():
                if sgn_func['type'] != 'sgn':
                    continue
                if self.verbose:
                    logger(f"Fixing signal parameters of function {name} to first fit results\n")
                for par_name, par_var in self.fit_model[name].items():
                    if not par_name.startswith("par_"):
                        continue
                    self.fit_model[name][par_name].setConstant(True)

        self.fit_counter += 1
        return self.mass_fit_result.status(), self.mass_fit_result.covQual()

    def perform_vn_vs_mass_fit_roofit(self):
        # Perform vn vs mass fit if enabled

        self._vn_internal_objects = []

        # # Freeze all parameters of the mass fit
        # for par in self.mass_fit_result.floatParsFinal():
        #     par.setConstant(True)

        # Build the vn vs mass fit function
        logger("Build normalized shape functions", "INFO")
        # Build normalized shape functions (RooAbsReal!)
        for comp_name, comp in self.fit_model.items():
            comp['shape_val'] = comp['vn_pdf']
            self._vn_internal_objects.append(comp['shape_val'])

        logger("Build yield * shape functions", "INFO")
        for comp_name, comp in self.fit_model.items():
            yield_var = comp.get('yieldRooLinearVar', comp['yield'])

            comp['yield_shape'] = RooFormulaVar(
                f"yield_shape_{comp_name}",
                "@0 * @1",
                RooArgList(yield_var, comp['shape_val'])
            )
            self._vn_internal_objects.append(comp['yield_shape'])

        # Debug: plot yield shapes
        frame = self.vn_mass_var.frame(
            RooFit.Title(";M(#pi K#pi) (GeV/c^{2});Mass fraction"),
            RooFit.Range(self.mass_fit_range_min, self.mass_fit_range_max)
        )
        for comp_name, comp in self.fit_model.items():
            ys = comp['yield_shape']
            ys.plotOn(
                frame,
                RooFit.LineColor(ROOT.kRed),  # choose a unique color per component
                RooFit.LineWidth(2),
                RooFit.Name(comp_name)
            )
        # Draw
        canvas_ys = TCanvas("c_yield_shape", "Yield * Shape functions", 700, 500)
        frame.Draw()
        canvas_ys.Update()
        canvas_ys.Draw()
        canvas_ys.SaveAs("yield_shape_debug.png")

        logger("Build total yield * shape function", "INFO")
        yield_shape_list = RooArgList()
        for comp in self.fit_model.values():
            yield_shape_list.add(comp['yield_shape'])

        self.total_yield_shape = RooFormulaVar(
            "total_yield_shape",
            "+".join(f"@{i}" for i in range(yield_shape_list.getSize())),
            yield_shape_list
        )
        
        # Debug total yield shape
        frame_total_ys = self.vn_mass_var.frame(
            RooFit.Title(";M(#pi K#pi) (GeV/c^{2});Total Yield * Shape function"),
            RooFit.Range(self.mass_fit_range_min, self.mass_fit_range_max)
        )
        self.total_yield_shape.plotOn(frame_total_ys,
                RooFit.LineColor(ROOT.kBlue),
                RooFit.LineWidth(2),
                RooFit.Name("total_yield_shape")
        )
        # Draw
        canvas_total_ys = TCanvas("c_total_yield_shape", "Total Yield * Shape function", 700, 500)
        frame_total_ys.Draw()
        canvas_total_ys.Update()
        canvas_total_ys.Draw()
        canvas_total_ys.SaveAs("total_yield_shape_debug.png")

        logger("Build mass fraction functions", "INFO")
        for comp_name, comp in self.fit_model.items():
            comp['mass_frac'] = RooFormulaVar(
                f"mass_frac_{comp_name}",
                "@0 / @1",
                RooArgList(comp['yield_shape'], self.total_yield_shape)
            )
            self._vn_internal_objects.append(comp['mass_frac'])

        # Create a frame over the mass variable
        frame = self.vn_mass_var.frame(
            RooFit.Title(";M(#pi K#pi) (GeV/c^{2});Mass fraction"),
            RooFit.Range(self.mass_fit_range_min, self.mass_fit_range_max)
        )

        # Plot each component's mass fraction
        for comp_name, comp in self.fit_model.items():
            mf = comp['mass_frac']
            mf.plotOn(
                frame,
                RooFit.LineColor(ROOT.kRed),  # choose a unique color per component
                RooFit.LineWidth(2),
                RooFit.Name(comp_name)
            )

        # Draw
        canvas = TCanvas("c_mass_frac", "Mass fractions", 700, 500)
        frame.Draw()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("mass_fractions_debug.png")

        # Debug vn_funcs
        frame_vnfunc = self.vn_mass_var.frame(
            RooFit.Title(";M(#pi K#pi) (GeV/c^{2});vn functions"),
            RooFit.Range(self.mass_fit_range_min, self.mass_fit_range_max)
        )
        colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kCyan+2]
        for i, (comp_name, comp) in enumerate(self.fit_model.items()):
            vnf = comp['vn_func']
            vnf.plotOn(frame_vnfunc,
                    RooFit.LineColor(colors[i % len(colors)]),  # choose a unique color per component
                    RooFit.LineWidth(2),
                    RooFit.Name(comp_name)
            )
        # Draw
        canvas_vnfunc = TCanvas("c_vn_func", "vn functions", 700, 500)
        frame_vnfunc.Draw()
        canvas_vnfunc.Update()
        canvas_vnfunc.Draw()
        canvas_vnfunc.SaveAs("vn_funcs_debug.png")

        logger("Building vn components", "INFO")
        self.vn_comps = RooArgList()
        for comp_name, comp in self.fit_model.items():
            vn_term = RooFormulaVar(
                f"vn_term_{comp_name}",
                "@0 * @1",
                RooArgList(comp['vn_func'], comp['mass_frac'])
            )
            comp['vn_term'] = vn_term      # <-- REQUIRED
            self.vn_comps.add(vn_term)
            self._vn_internal_objects.append(vn_term)

        logger("vn components list:", "INFO")
        self.vn_comps.Print()
        logger(f"Number of vn components: {self.vn_comps.getSize()}")

        # Final vn vs mass: numerator / denominator
        self.vn_vs_mass_model = RooFormulaVar(
            "vn_vs_mass",
            "+".join(f"@{i}" for i in range(self.vn_comps.getSize())),
            self.vn_comps
        )

        # Debug vn components
        frame_vncomps = self.vn_mass_var.frame(
            RooFit.Title(";M(#pi K#pi) (GeV/c^{2});vn components"),
            RooFit.Range(self.mass_fit_range_min, self.mass_fit_range_max)
        )
        for i in range(self.vn_comps.getSize()):
            vn_comp = self.vn_comps.at(i)
            vn_comp.plotOn(frame_vncomps,
                    RooFit.LineColor(ROOT.kRed),  # choose a unique color per component
                    RooFit.LineWidth(2),
                    RooFit.Name(vn_comp.GetName())
            )
        self.vn_vs_mass_model.plotOn(frame_vncomps,
                RooFit.LineColor(ROOT.kBlue),
                RooFit.LineWidth(2),
                RooFit.Name("vn_vs_mass_model")
        )
        # Draw
        canvas_vncomps = TCanvas("c_vn_components", "vn components", 700, 500)
        # Plot data points
        self.vn_vs_mass_data.plotOn(
            frame_vncomps,
            RooFit.MarkerStyle(ROOT.kFullCircle),
            RooFit.MarkerSize(0.8),
            RooFit.LineColor(ROOT.kBlack),
            RooFit.DrawOption("PE0"),
            RooFit.Name("data_points") # Named for legend potential
        )
        frame_vncomps.Draw()
        canvas_vncomps.Update()
        canvas_vncomps.Draw()
        canvas_vncomps.SaveAs("vn_components_debug.png")

        logger("Finished setting up vn vs mass fit function", "INFO")

        cfg = RooAbsTestStatistic.Configuration()
        cfg.integrateBins = True
        cfg.integrationEps = 1e-4   # optional, but recommended

        chi2 = RooChi2Var(
            "vn_chi2",
            "chi2(vn vs mass)",
            self.vn_vs_mass_model,
            self.vn_vs_mass_data,
            False,
            RooDataHist.SumW2,
            cfg
        )

        # # Now build chi2
        # chi2 = RooChi2Var(
        #     "vn_chi2",
        #     "chi2(vn vs mass)",
        #     self.vn_vs_mass_model,
        #     self.vn_vs_mass_data,       # histogram with arbitrary errors
        #     False,              # extended must be False
        #     RooDataHist.SumW2,   # use bin errors
        #     RooFit.IntegrateBins(1e-4),
        # )

        logger("Chi2 variable created successfully", "INFO")

        # 1. Get the list of ALL parameters actually used by the Chi2
        vn_params = chi2.getParameters(self.vn_vs_mass_data)
        print(f"vn_params: {vn_params.Print()}")
        for vn_par in vn_params:
            # if vn_par.GetName() == "vn_c0_Comb_Bkg":
            #     vn_par.setVal(0.05)
            #     vn_par.setConstant(True)
            # if vn_par.GetName() == "vn_c1_Comb_Bkg":
            #     vn_par.setVal(0.0)
            #     vn_par.setConstant(True)
            # if vn_par.GetName() == "vn_c0_DplusToPiKPi":
            #     vn_par.setVal(0.17)
            #     vn_par.setConstant(True)
            for mass_par in self.mass_fit_result.floatParsFinal():
                if vn_par.GetName() == mass_par.GetName():
                    vn_par.setVal(mass_par.getVal())
                    vn_par.setConstant(True)
                    logger(f"Fixing parameter {mass_par.GetName()} to {mass_par.getVal()}", "INFO")

        # Print fit parameters before minimization
        logger("=== vn vs mass fit parameters before minimization ===", "WARNING")
        for vn_par in chi2.getParameters(self.vn_vs_mass_data):
            logger(f"{vn_par.GetName()}: {vn_par.getVal()} +/- {vn_par.getError()}", "INFO")

        # Minimization
        minimizer = RooMinimizer(chi2)
        minimizer.setStrategy(2)       # More robust for complex formulas
        minimizer.setPrintLevel(5)
        minimizer.migrad()
        minimizer.hesse()
        logger("Minimization done", "INFO")

        self.vn_vs_mass_fit_result = minimizer.save()
        logger("=== Vn Vs Mass Fit Results Floating parameters ===", "WARNING")
        self.vn_vs_mass_fit_result.floatParsFinal().Print("v")


        debug_file = TFile("vn_vs_mass_debug.root", "RECREATE")
        debug_file.cd()
        # Write all canvas
        canvas_ys.Write()
        canvas_total_ys.Write()
        canvas_vnfunc.Write()
        canvas_vncomps.Write()
        debug_file.Close()

        return self.vn_vs_mass_fit_result.status(), self.vn_vs_mass_fit_result.covQual()
        # print("Vn vs mass fit performed successfully")
        

        
        # return None, None

    def add_func_to_model(self, sgn_or_bkg, func, label, vn_func=None, particle=None):

        label = label.replace('Comb. bkg', 'Comb_Bkg')

        name = f"pdf_{func}_{sgn_or_bkg}_idx_{self.n_pdfs_bkg}_{label}" if sgn_or_bkg == 'bkg' \
                else f"pdf_{func}_{sgn_or_bkg}_idx_{self.n_pdfs_sgn}_{label}"
        self.fit_model[label] = {}
        fit_model_entry = self.fit_model[label]
        fit_model_entry['name'] = name
        fit_model_entry['label'] = label
        fit_model_entry['type'] = sgn_or_bkg
        fit_model_entry['idx'] = self.n_pdfs_sgn if sgn_or_bkg == 'sgn' else self.n_pdfs_bkg

        if sgn_or_bkg == 'sgn':
            init_mass = self.get_particle_mass(particle)

        if func == 'kGaus':
            if self.verbose:
                logger(f"Adding Gaussian signal function", "INFO")
            if self.minimize_roofit:
                mu = RooRealVar(f"mu_{label}", f"mean_{label}", init_mass, init_mass - 0.02, init_mass + 0.02)
                sigma = RooRealVar(f"sigma_{label}", f"sigma_{label}", 0.015, 0.005, 0.05)
                fit_model_entry['par_mu'] = mu
                fit_model_entry['par_sigma'] = sigma
                fit_model_entry['pdf'] = RooGaussian(label, label, self.mass_fit_var, mu, sigma)
                fit_model_entry['vn_pdf'] = RooGaussian(label, label, self.vn_mass_var, mu, sigma)
                fit_model_entry['yield'] = RooRealVar(f"yield_{label}", f"yield_{label}", 5000, 0, 1e7)
            else:
                fit_model_entry['par_mu'] = init_mass
                fit_model_entry['par_sigma'] = 0.015
                fit_model_entry['pdf'] = "gaussian"
        elif func == 'kDoubleSidedAsymmCB':
            if self.verbose:
                logger(f"Adding Double-Sided Asymm CB signal function", "INFO")
            if self.minimize_roofit:
                mu = RooRealVar(f"mu_{label}", f"mean_{label}", init_mass, init_mass - 0.02, init_mass + 0.02)
                sigma = RooRealVar(f"sigma_{label}", f"sigma_{label}", 0.015, 0.005, 0.05)
                alphaL = RooRealVar(f"alphaL_{label}", f"alphaL_{label}", 1.885, 0.5, 5.0)
                nL = RooRealVar(f"nL_{label}", f"nL_{label}", 1.90, 0.5, 10.0)
                alphaR = RooRealVar(f"alphaR_{label}", f"alphaR_{label}", 1.391, 0.5, 5.0)
                nR = RooRealVar(f"nR_{label}", f"nR_{label}", 8.188, 0.5, 20.0)
                fit_model_entry['par_mu'] = mu
                fit_model_entry['par_sigma'] = sigma
                fit_model_entry['par_alphaL'] = alphaL
                fit_model_entry['par_nL'] = nL
                fit_model_entry['par_alphaR'] = alphaR
                fit_model_entry['par_nR'] = nR
                fit_model_entry['pdf'] = RooCrystalBall(label, label, self.mass_fit_var, mu, sigma, alphaL, nL, alphaR, nR)
                fit_model_entry['vn_pdf'] = RooCrystalBall(label, label, self.vn_mass_var, mu, sigma, alphaL, nL, alphaR, nR)
                fit_model_entry['yield'] = RooRealVar(f"yield_{label}", f"yield_{label}", 5000, 0, 1e7)
            else:
                fit_model_entry['par_mu'] = init_mass
                fit_model_entry['par_sigma'] = 0.015
                fit_model_entry['par_alphaL'] = 1.885
                fit_model_entry['par_nL'] = 1.90
                fit_model_entry['par_alphaR'] = 1.391
                fit_model_entry['par_nR'] = 8.188
                fit_model_entry['pdf'] = "doublecb"
        elif func == 'kConst':
            if self.verbose:
                logger(f"Adding Constant background function", "INFO")
            if self.minimize_roofit:
                fit_model_entry['pdf'] = RooChebychev(label, label, self.mass_fit_var, RooArgList())
                fit_model_entry['vn_pdf'] = RooChebychev(label, label, self.vn_mass_var, RooArgList())
                fit_model_entry['yield'] = RooRealVar(f"yield_{label}", f"yield_{label}", 100000, 0, 1e8)
            else:
                fit_model_entry['par_c0'] = 1.0
                fit_model_entry['pdf'] = "chebpol0"
        elif func == 'kExpo':
            if self.verbose:
                logger(f"Adding Exponential background function", "INFO")
            if self.minimize_roofit:
                lambd = RooRealVar(f"lambda_{label}", f"exponential lambda_{label}", -1.0, -5.0, 0.0)
                fit_model_entry['par_lambda'] = lambd
                fit_model_entry['pdf'] = RooExponential(label, label, self.mass_fit_var, lambd)
                fit_model_entry['vn_pdf'] = RooExponential(label, label, self.vn_mass_var, lambd)
                fit_model_entry['yield'] = RooRealVar(f"yield_{label}", f"yield_{label}", 100000, 0, 1e8)
            else:
                fit_model_entry['par_lambda'] = 1.0
                fit_model_entry['pdf'] = "expo"
        elif func == 'kLin':
            if self.verbose:
                logger(f"Adding Chebyshev Polynomial of degree 1 background function", "INFO")
            if self.minimize_roofit:
                c1 = RooRealVar(f"c1_{label}", f"c1_{label}", 0.0, -1.0, 1.0)
                fit_model_entry['par_c1'] = c1
                fit_model_entry['pdf'] = RooChebychev(label, label, self.mass_fit_var, RooArgList(c1))
                fit_model_entry['vn_pdf'] = RooChebychev(label, label, self.vn_mass_var, RooArgList(c1))
                fit_model_entry['yield'] = RooRealVar(f"yield_{label}", f"yield_{label}", 100000, 0, 1e8)
            else:
                fit_model_entry['par_c0'] = 1.0
                fit_model_entry['par_c1'] = 0.0
                fit_model_entry['pdf'] = "chebpol1"
        elif func == 'kPol2':
            if self.verbose:
                logger(f"Adding Chebyshev Polynomial of degree 2 background function", "INFO")
            if self.minimize_roofit:
                c1 = RooRealVar(f"c1_{label}", f"c1_{label}", 0.0, -1.0, 1.0)
                c2 = RooRealVar(f"c2_{label}", f"c2_{label}", 0.0, -1.0, 1.0)
                fit_model_entry['par_c1'] = c1
                fit_model_entry['par_c2'] = c2
                fit_model_entry['pdf'] = RooChebychev(label, label, self.mass_fit_var, RooArgList(c1, c2))
                fit_model_entry['vn_pdf'] = RooChebychev(label, label, self.vn_mass_var, RooArgList(c1, c2))
                fit_model_entry['yield'] = RooRealVar(f"yield_{label}", f"yield_{label}", 100000, 0, 1e8)
            else:
                fit_model_entry['par_c0'] = 1.0
                fit_model_entry['par_c1'] = 0.0
                fit_model_entry['par_c2'] = 0.0
                fit_model_entry['pdf'] = "chebpol2"
        else:
            if self.verbose:
                logger(f"Function {func} not recognized for mass component!", "ERROR")
            sys.exit(1)

        # Add vn function if provided
        if vn_func is not None:
            if vn_func == 'kConst':
                if self.verbose:
                    logger(f"Adding constant vn background function", "INFO")
                if self.minimize_roofit:
                    vn_c0 = RooRealVar(f"vn_c0_{label}", f"vn_c0_{label}", 0.1, -100.0, 100.0)
                    fit_model_entry['vn_par_c0'] = vn_c0
                    fit_model_entry['vn_func'] = RooFormulaVar(
                        f"vn_func_{label}",
                        "@0",
                        RooArgList(vn_c0) #, self.mass_fit_var)
                    )
                else:
                    fit_model_entry['vn_par_c0'] = 1.0
                    fit_model_entry['vn_func'] = "chebpol0"
            elif vn_func == 'kExpo':
                if self.verbose:
                    logger(f"Adding exponential vn background function", "INFO")
                if self.minimize_roofit:
                    vn_lambd = RooRealVar(f"vn_lambda_{label}", f"vn_lambda_{label}", -1.0, -5.0, 0.0)
                    fit_model_entry['vn_par_lambda'] = vn_lambd
                    fit_model_entry['vn_func'] = RooFormulaVar(
                        f"vn_func_{label}", "exp(@0 * @1)",
                        RooArgList(vn_lambd, self.vn_mass_var)
                    )
                else:
                    fit_model_entry['vn_par_lambda'] = 1.0
                    fit_model_entry['vn_func'] = "expo"
            elif vn_func == 'kLin':
                if self.verbose:
                    logger(f"Adding Chebyshev Polynomial of degree 1 vn background function", "INFO")
                if self.minimize_roofit:
                    vn_c0 = RooRealVar(f"vn_c0_{label}", f"vn_c0_{label}", 0.035, -100.0, 100.0)
                    vn_c1 = RooRealVar(f"vn_c1_{label}", f"vn_c1_{label}", -0.05, -100.0, 100.0)
                    fit_model_entry['vn_par_c0'] = vn_c0
                    fit_model_entry['vn_par_c1'] = vn_c1
                    fit_model_entry['vn_func'] = RooFormulaVar(
                        f"vn_func_{label}",
                        # "@0 + @1 * (@4 - (@2 + @3) / 2)",
                        "(@0 - @1/2 * (@3*@3 - @2*@2)) / (@3 - @2) + @1 * @4",
                        # "(@0 - (@1 / (2 * (@3 * @3) - (@2 * @2) ) ) / (@3 - @2) + @1 * @4",
                        RooArgList(vn_c0, vn_c1, self.mass_fit_range_min, self.mass_fit_range_max, self.vn_mass_var)
                    )
                else:
                    fit_model_entry['vn_par_c0'] = 1.0
                    fit_model_entry['vn_par_c1'] = 0.0
                    fit_model_entry['vn_func'] = "chebpol1"
            elif vn_func == 'kPol2':
                if self.verbose:
                    logger(f"Adding Chebyshev Polynomial of degree 2 vn background function", "INFO")
                if self.minimize_roofit:
                    vn_c0 = RooRealVar(f"vn_c0_{label}", f"vn_c0_{label}", 0.1, -100.0, 100.0)
                    vn_c1 = RooRealVar(f"vn_c1_{label}", f"vn_c1_{label}", 0.1, -100.0, 100.0)
                    vn_c2 = RooRealVar(f"vn_c2_{label}", f"vn_c2_{label}", 0.1, -100.0, 100.0)
                    fit_model_entry['vn_par_c0'] = vn_c0
                    fit_model_entry['vn_par_c1'] = vn_c1
                    fit_model_entry['vn_par_c2'] = vn_c2
                    fit_model_entry['vn_func'] = RooFormulaVar(
                        f"vn_func_{label}",
                        "@0 + @1 * (@5 - (@3 + @4) / 2) + @2 * (@5 - (@3 + @4) / 2) * (@5 - (@3 + @4) / 2)",
                        RooArgList(vn_c0, vn_c1, vn_c2, self.mass_fit_range_min, self.mass_fit_range_max, self.vn_mass_var)
                    )
                else:
                    fit_model_entry['vn_par_c0'] = 1.0
                    fit_model_entry['vn_par_c1'] = 0.0
                    fit_model_entry['vn_par_c2'] = 0.0
                    fit_model_entry['vn_func'] = "chebpol2"
            else:
                if self.verbose:
                    logger(f"Function {vn_func} not recognized for vn component!", "FATAL")
                sys.exit(1)

        plot_info = {}
        if sgn_or_bkg == "sgn":
            # e.g., signal: filled area
            color = ROOT.TColor.GetColorTransparent(ROOT.kAzure + 4 + 2*self.n_pdfs_sgn, 0.6)
            plot_info["fill_color"] = color
            plot_info["fill_style"] = 3145
            plot_info["line_color"] = ROOT.kAzure + 4
            plot_info["line_width"] = 2
            plot_info["draw_option"] = "F"  # filled
        else:
            # e.g., background: lines
            line_idx = self.n_pdfs_bkg
            color = ROOT.kOrange + 1 + 2*line_idx  # for variation
            plot_info["line_color"] = color
            plot_info["line_width"] = 4
            plot_info["line_style"] = 9 if label == "Comb_Bkg" else 1
            plot_info["draw_option"] = "L"

        fit_model_entry["plot_info"] = plot_info

        if sgn_or_bkg == 'sgn':
            self.n_pdfs_sgn += 1
        else:
            self.n_pdfs_bkg += 1
