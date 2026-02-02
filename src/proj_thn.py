'''
Script to project the MC distributions and apply the pt weights from the AnRes.root of Dtask
python3 proj_thn.py config_flow.yml --cutsetConfig config_cutset.yml [-c --correlated]
If the last argument is not provided, the script will project the combined cutsets.
'''
import ROOT
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import yaml
import argparse
import sys
import os
import glob
import numpy as np
from pathlib import Path
from functools import partial
from ROOT import TFile, TObject, TH1F
from alive_progress import alive_bar
from scipy.interpolate import make_interp_spline
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from data_model import get_pt_preprocessed_sparses
from utils import reweight_histo_1D, reweight_histo_2D, reweight_histo_3D, get_vn_versus_mass, profile_mass_sp, make_dir_root_file, logger, get_centrality_bins

ROOT.TH1.AddDirectory(False)

def proj_multitrial(config, multitrial_folder, workers, resolution):

    pt_bin_label = Path(multitrial_folder).name
    logger(f"Processing multitrial projections for pt bin {pt_bin_label} ...", level='INFO')

    # Load default cutsets
    cutset_dir = f"{config['outdir']}/cutvar_{config['suffix']}_combined/cutsets"
    default_cutsets = [f"{cutset_dir}/{f}" for f in os.listdir(cutset_dir) if f.endswith('.yml')]
    # Load Mass and MassSp histos from the default cases
    default_histos = {}
    for default_cutset in default_cutsets:
        suffix = os.path.basename(default_cutset).replace(".yml", "").replace("cutset_", "")
        default_proj = TFile.Open(default_cutset.replace(".yml", ".root").replace("cutset", "proj"), "READ")
        default_histos[suffix] = {}
        default_histos[suffix]['Mass'] = default_proj.Get(f"{pt_bin_label}/hMassData")
        default_histos[suffix]['MassSp'] = default_proj.Get(f"{pt_bin_label}/hMassSpData")
        default_proj.Close()

    def process_cutset(multitrial_dir, default_histos):
        trial_number = Path(multitrial_dir).name.replace("trial_", "")
        try:
            with open(f"{multitrial_dir}/config_trial_{trial_number}.yml", 'r') as ymlCutSetFile:
                config_trial = yaml.safe_load(ymlCutSetFile)
        except Exception as e:
            logger(f"Error opening or reading config file for trial {trial_number}: {e}", level='ERROR')
            return

        for suffix, histo in default_histos.items():
            output_dir = f"{multitrial_dir}/projs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/proj_{suffix}.root"
            output_file = TFile.Open(output_path, "RECREATE")
            output_file.mkdir(pt_bin_label)
            output_file.cd(pt_bin_label)
            default_histos[suffix]['Mass'].Write("hMassData")
            hist_vn_vs_mass = profile_mass_sp(default_histos[suffix]['MassSp'], config_trial['projections']['inv_mass_bins'][0], resolution)
            hist_vn_vs_mass.Write("hVnVsMassData")
            output_file.Close()

        logger(f"[{trial_number}] Completed projections!", level='INFO')

    # Parallel execution
    multitrial_dirs = [f for f in glob.glob(f"{multitrial_folder}/trials/*") if os.path.isdir(f)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(partial(process_cutset, default_histos=default_histos), multitrial_dirs)

def proj_data(i_bin, sparse, axes, resolution, proj_cfg, writeopt):

    proj_vars = proj_cfg.get('ProjVars', [])
    proj_vars += ['Mass', 'Sp']
    proj_axes = [axes['FlowSP'][var] for var in proj_vars]

    for var, ax in zip(proj_vars, proj_axes):
        hist_var = sparse.Projection(ax)
        hist_var.Write(f'h{var.capitalize()}Data', writeopt)

    hist_vn_sp = get_vn_versus_mass(sparse, proj_cfg['inv_mass_bins'][i_bin], axes['FlowSP']['Mass'], axes['FlowSP']['Sp'])
    hist_vn_sp.Scale(1/resolution) # Correct for resolution
    hist_vn_sp.Write('hVnVsMassData', writeopt)

    # Save a TH2 of (Mass, Sp) for the multitrial systematic
    mass_lowest_bin, mass_highest_bin, sp_lowest_bin, sp_highest_bin = -1, 1e10, -1, 1e10
    hist_mass_sp = sparse.Projection(axes['FlowSP']['Sp'], axes['FlowSP']['Mass'])
    sp_lowest_bin = hist_mass_sp.ProjectionY().FindFirstBinAbove(0)
    sp_highest_bin = hist_mass_sp.ProjectionY().FindLastBinAbove(0)
    mass_lowest_bin = hist_mass_sp.ProjectionX().FindFirstBinAbove(0)
    mass_highest_bin = hist_mass_sp.ProjectionX().FindLastBinAbove(0)
    hist_mass_sp.GetXaxis().SetRange(mass_lowest_bin, mass_highest_bin)
    hist_mass_sp.GetYaxis().SetRange(sp_lowest_bin, sp_highest_bin)
    hist_mass_sp.Write('hMassSpData', writeopt)

def proj_mc_reco(sparses_reco, sPtWeightsD, sPtWeightsB, Bspeciesweights, writeopt, save_centrality=False):

    for key, sparse in sparses_reco.items():
        if key != 'RecoPrompt' and key != 'RecoFD':
            sparse.Projection(axes[key]['Mass']).Write(f'h{key}Mass')
            sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    hMassPrompt = sparses_reco['RecoPrompt'].Projection(axes['RecoPrompt']['Mass'])
    hMassPrompt.SetName(f'hPromptMass_{pt_min}_{pt_max}')
    hMassFD = sparses_reco['RecoFD'].Projection(axes['RecoFD']['Mass'])
    hMassFD.SetName(f'hFDMass_{pt_min}_{pt_max}')

    ### project pt prompt
    hPtPrompt = sparses_reco['RecoPrompt'].Projection(axes['RecoPrompt']['Pt'])
    if sPtWeightsD:
        hPtPrompt = reweight_histo_1D(hPtPrompt, sPtWeightsD, binned=False)

    ### project pt FD
    if sPtWeightsD:
        hPtFD = reweight_histo_1D(sparses_reco['RecoFD'].Projection(axes['RecoFD']['Pt']), sPtWeightsD, binned=False)
    elif sPtWeightsB:
        if Bspeciesweights:
            hPtFD = reweight_histo_3D(
                sparses_reco['RecoFD'].Projection(axes['RecoFD']['Pt'], axes['RecoFD']['PtBMoth'], axes['RecoFD']['FlagBHad']),
                sPtWeightsB, Bspeciesweights
            )
        else:
            hPtFD = reweight_histo_2D(
                sparses_reco['RecoFD'].Projection(axes['RecoFD']['PtBMoth'], axes['RecoFD']['Pt']),          # 2D projection: Projection(ydim, xdim)
                sPtWeightsB, binned=False
            )
    elif Bspeciesweights:
        hPtFD = reweight_histo_2D(
            sparses_reco['RecoFD'].Projection(axes['RecoFD']['FlagBHad'], axes['RecoFD']['Pt']),             # 2D projection: Projection(ydim, xdim)
            Bspeciesweights, binned=True
        )
    else:
        hPtFD = sparses_reco['RecoFD'].Projection(axes['RecoFD']['Pt'])

    ## write the output
    hMassPrompt.Write('hPromptMass', writeopt)
    hMassFD.Write('hFDMass', writeopt)
    hPtPrompt.Write('hPromptPt', writeopt)
    hPtFD.Write('hFDPt', writeopt)

    # Store also centrality of prompt, if available
    if save_centrality and 'Cent' in axes['RecoPrompt']:
        hRecoCentPrompt = sparses_reco['RecoPrompt'].Projection(axes['RecoPrompt']['Cent'])
        hRecoCentPrompt.Write('hPromptRecoCent', writeopt)

    return hPtPrompt, hPtFD

def proj_mc_gen(sparses_gen, sPtWeightsD, sPtWeightsB, Bspeciesweights, writeopt, save_centrality=False):

    for key, sparse in sparses_gen.items():
        if key != 'GenPrompt' and key != 'GenFD':
            sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    ### prompt
    hGenPtPrompt = sparses_gen['GenPrompt'].Projection(axes['GenPrompt']['Pt'])
    if sPtWeightsD:
        hGenPtPrompt = reweight_histo_1D(hGenPtPrompt, sPtWeightsD, binned=False)

    ### FD
    if sPtWeightsD:
        hGenPtFD = reweight_histo_1D(sparses_gen['GenFD'].Projection(axes['GenFD']['Pt']), sPtWeightsD, binned=False)
    elif sPtWeightsB:
        if Bspeciesweights:
            hGenPtFD = reweight_histo_3D(
                sparses_gen['GenFD'].Projection(axes['GenFD']['Pt'], axes['GenFD']['PtBMoth'], axes['GenFD']['FlagBHad']),
                sPtWeightsB, Bspeciesweights
            )
        else:
            hGenPtFD = reweight_histo_2D(
                sparses_gen['GenFD'].Projection(axes['GenFD']['PtBMoth'], axes['GenFD']['Pt']),         # 2D projection: Projection(ydim, xdim)
                sPtWeightsB, binned=False
            )
    elif Bspeciesweights:
        hGenPtFD = reweight_histo_2D(
            sparses_gen['GenFD'].Projection(axes['GenFD']['FlagBHad'], axes['GenFD']['Pt']),            # 2D projection: Projection(ydim, xdim)
            Bspeciesweights, binned=True
        )
    else:
        hGenPtFD = sparses_gen['GenFD'].Projection(axes['GenFD']['Pt'])

    ## write the output
    hGenPtPrompt.Write('hPromptGenPt', writeopt)
    hGenPtFD.Write('hFDGenPt', writeopt)

    # Store also centrality of prompt, if available
    if save_centrality and 'Cent' in axes['GenPrompt']:
        hGenCentPrompt = sparses_gen['GenPrompt'].Projection(axes['GenPrompt']['Cent'])
        hGenCentPrompt.Write('hPromptGenCent', writeopt)

    return hGenPtPrompt, hGenPtFD

def get_pt_weights(cfgProj):
    """Get pt weights and return weights flags with spline

    Args:
        cfgProj (dict): Configuration dictionary for projections

    Outputs:
        sPtWeights (spline): Spline for ptWeights interpolation
        sPtWeightsB (spline): Spline for ptWeightsB weights interpolation
        Bspeciesweights (str): B species weights # TODO
    """

    # REVIEW: the ptWeights inputed is a list, but the ptWeights outputed is a TH1D object
    # and actually ptweights is used as a flag
        # compute info for pt weights
    if not cfgProj.get('PtWeightsFile'):
        logger('No pt weights for D and B mesons provided in the config file!', level='WARNING')
        return None, None, None
        
    ptWeightsFile = TFile.Open(cfgProj["PtWeightsFile"], 'r')

    if cfgProj.get('ApplyPtWeightsD'):
        hPtWeightsD = ptWeightsFile.Get('hPtWeightsFONLLtimesTAMUDcent')
        ptBinCentersD = [ (hPtWeightsD.GetBinLowEdge(i)+hPtWeightsD.GetBinLowEdge(i+1))/2 for i in range(1, hPtWeightsD.GetNbinsX()+1)]
        ptBinContentsD = [hPtWeightsD.GetBinContent(i) for i in range(1, hPtWeightsD.GetNbinsX()+1)]
        sPtWeights = make_interp_spline(ptBinCentersD, ptBinContentsD)
    else:
        logger('pt weights for D mesons will not be provided!', level='WARNING')
        sPtWeights = None

    if cfgProj.get('ApplyPtWeightsB'):
        hPtWeightsB = ptWeightsFile.Get('hPtWeightsFONLLtimesTAMUBcent')
        ptBinCentersB = [ (hPtWeightsB.GetBinLowEdge(i)+hPtWeightsB.GetBinLowEdge(i+1))/2 for i in range(1, hPtWeightsB.GetNbinsX()+1)]
        ptBinContentsB = [hPtWeightsB.GetBinContent(i) for i in range(1, hPtWeightsB.GetNbinsX()+1)]
        sPtWeightsB = make_interp_spline(ptBinCentersB, ptBinContentsB)
    else:
        logger('pt weights for B mesons will not be provided!', level='WARNING')
        sPtWeightsB = None

    if cfgProj.get('ApplyBSpeciesWeights'):
        Bspeciesweights = config['Bspeciesweights']
    else:
        logger('B species weights will not be provided!', level='WARNING')
        Bspeciesweights = None
    
    return sPtWeights, sPtWeightsB, Bspeciesweights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    parser.add_argument('--cutsetConfig', "-cc", metavar='text', type=str, nargs='?',
                        const=None, default='cutsetConfig.yaml',
                        help='Optional cutset configuration file (default: cutsetConfig.yaml)')
    parser.add_argument("--multitrial_folder", "-multfolder", metavar="text",
                        default="", help="Produce projection files for multitrial systematics")
    parser.add_argument("--multitrial_workers", "-multworkers", metavar="int",
                        type=int, default=1, help="Number of workers for multitrial projections")
    parser.add_argument("--outputDir", "-o", metavar="text",
                        default="", help="output directory, used only for directly running the script")
    args = parser.parse_args()

    with open(args.config, 'r') as ymlCfgFile:
        config = yaml.load(ymlCfgFile, yaml.FullLoader)
    operations = config["operations"]

    proj_mc_cent_diff = True if operations.get("proj_mc") and config['projections'].get('CentDiffBinsMCYieldsStep', False) else False
    if config['projections'].get('CentDiffBinsMCYieldsStep'):
        _, (centLowLim, centMaxLim) = get_centrality_bins(config['centrality'])

    if operations.get("proj_data") or args.multitrial_folder != "":
        reso_file = TFile.Open(config["projections"]["Resolution"], 'r')
        det_A = config["projections"].get('detA', 'FT0c')
        det_B = config["projections"].get('detB', 'FV0a')
        det_C = config["projections"].get('detC', 'TPCtot')
        logger(f"Getting resolution histogram from file {config['projections']['Resolution']} for triplet {det_A}_{det_B}_{det_C}",  "WARNING")
        reso_hist = reso_file.Get(f'{det_A}_{det_B}_{det_C}/histo_reso_delta_cent')
        resolution = reso_hist.GetBinContent(1)
        reso_hist.SetDirectory(0)
        reso_file.Close()

    if args.multitrial_folder != "":
        logger(f"Running multitrial projections with config: {args.config}", level='INFO')
        proj_multitrial(config, args.multitrial_folder, args.multitrial_workers, resolution)
        sys.exit(0)

    with open(args.cutsetConfig, 'r') as ymlCutSetFile:
        cutSetCfg = yaml.load(ymlCutSetFile, yaml.FullLoader)
        iCut = f"{int(cutSetCfg['icutset']):02d}"

    outDir = os.path.join(os.path.dirname(os.path.dirname(args.cutsetConfig)), 'projs') if args.outputDir == "" else args.outputDir
    outfilePath = os.path.join(outDir, f"proj_{iCut}.root")
    os.makedirs(outDir, exist_ok=True)

    if operations.get("proj_data") or operations.get("proj_mc"):
        if os.path.exists(outfilePath):
            logger(f"Found previous projection file {outfilePath}, will update it", level='INFO')
            outfile = TFile.Open(outfilePath, 'UPDATE')
        else:
            logger(f"No previous projection file found, will create a new one at {outfilePath}", level='INFO')
            outfile = TFile(outfilePath, 'RECREATE')
    else:
        sys.exit(0)

    write_opt_data = TObject.kOverwrite if operations.get("proj_data") else 0
    write_opt_mc = TObject.kOverwrite if operations.get("proj_mc") else 0

    # compute info for pt weights
    if operations.get("proj_mc"):
        sPtWeightsD, sPtWeightsB, Bspeciesweights = get_pt_weights(config["projections"]) if config['projections'].get('PtWeightsFile') else (None, None, None)

    if operations.get("proj_data"):
        reso_hist.Write("hResolution", write_opt_data)
        outfile.cd()
        resolution = reso_hist.GetBinContent(1)

    with alive_bar(len(cutSetCfg['Pt']['min']), title='Processing pT bins') as bar:
        for i_pt, (pt_min, pt_max, bkg_min, bkg_max, fd_min, fd_max) in enumerate(zip(cutSetCfg['Pt']['min'], cutSetCfg['Pt']['max'],
                                                                                   cutSetCfg['ScoreBkg']['min'], cutSetCfg['ScoreBkg']['max'],
                                                                                   cutSetCfg['ScoreFD']['min'], cutSetCfg['ScoreFD']['max'])):

            # Cut on centrality and pt on data applied in the preprocessing
            logger(f'Projecting distributions for {pt_min:.1f} < pT < {pt_max:.1f} GeV/c')
            pt_label = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
            make_dir_root_file(pt_label, outfile)
            sparse_flow, sparses_reco, sparses_gen, axes = get_pt_preprocessed_sparses(config, pt_label)
            outfile.cd(pt_label)
            if operations.get("proj_data"):
                sparse_flow["FlowSP"].GetAxis(axes['FlowSP']['ScoreFD']).SetRangeUser(fd_min, fd_max)
                sparse_flow["FlowSP"].GetAxis(axes['FlowSP']['ScoreBkg']).SetRangeUser(bkg_min, bkg_max)
                proj_data(i_pt, sparse_flow["FlowSP"], axes, resolution, config["projections"], write_opt_data)
                logger("Projected data!")

            if operations.get("proj_mc"):
                for key, i_sparse in sparses_reco.items():
                    i_sparse.GetAxis(axes[key]['ScoreBkg']).SetRangeUser(bkg_min, bkg_max)
                    i_sparse.GetAxis(axes[key]['ScoreFD']).SetRangeUser(fd_min, fd_max)

                proj_mc_reco(sparses_reco, sPtWeightsD, sPtWeightsB, Bspeciesweights, write_opt_mc, save_centrality=proj_mc_cent_diff)
                logger("Projected mc reco!")
                proj_mc_gen(sparses_gen, sPtWeightsD, sPtWeightsB, Bspeciesweights, write_opt_mc, save_centrality=proj_mc_cent_diff)
                logger("Projected mc gen!\n\n")

                if proj_mc_cent_diff:

                    centStep = config['projections']['CentDiffBinsMCYieldsStep']
                    centBins = list(np.arange(centLowLim, centMaxLim + centStep, centStep))
                    hCentDiffYieldsRecoPrompt = TH1F("hCentDiffYieldsRecoPrompt", ";Centrality (%);Yield", len(centBins)-1, np.asarray(centBins, 'd'))
                    hCentDiffYieldsRecoFD = TH1F("hCentDiffYieldsRecoFD", ";Centrality (%);Yield", len(centBins)-1, np.asarray(centBins, 'd'))
                    hCentDiffYieldsGenPrompt = TH1F("hCentDiffYieldsGenPrompt", ";Centrality (%);Yield", len(centBins)-1, np.asarray(centBins, 'd'))
                    hCentDiffYieldsGenFD = TH1F("hCentDiffYieldsGenFD", ";Centrality (%);Yield", len(centBins)-1, np.asarray(centBins, 'd'))
                    for i_cent_bin, (cent_min, cent_max) in enumerate(zip(centBins[:-1], centBins[1:])):
                        cent_label = f'cent_{int(cent_min)}_{int(cent_max)}'
                        make_dir_root_file(f"{pt_label}/{cent_label}", outfile)
                        outfile.cd(f"{pt_label}/{cent_label}")
                        for key, i_sparse in sparses_reco.items():
                            i_sparse.GetAxis(axes[key]['Cent']).SetRangeUser(cent_min, cent_max)
                        for key, i_sparse in sparses_gen.items():
                            i_sparse.GetAxis(axes[key]['Cent']).SetRangeUser(cent_min, cent_max)

                        hPtPrompt, hPtFD = proj_mc_reco(sparses_reco, sPtWeightsD, sPtWeightsB, Bspeciesweights, write_opt_mc, save_centrality=True)
                        hCentDiffYieldsRecoPrompt.SetBinContent(i_cent_bin+1, hPtPrompt.Integral())
                        hCentDiffYieldsRecoFD.SetBinContent(i_cent_bin+1, hPtFD.Integral())
                        hGenPtPrompt, hGenPtFD = proj_mc_gen(sparses_gen, sPtWeightsD, sPtWeightsB, Bspeciesweights, write_opt_mc, save_centrality=True)
                        hCentDiffYieldsGenPrompt.SetBinContent(i_cent_bin+1, hGenPtPrompt.Integral())
                        hCentDiffYieldsGenFD.SetBinContent(i_cent_bin+1, hGenPtFD.Integral())
                        logger(f"Projected mc reco and gen for cent {cent_min}-{cent_max}!", "INFO")

                    logger("\n\n")
                    outfile.cd(pt_label)
                    hCentDiffYieldsRecoPrompt.Sumw2()
                    hCentDiffYieldsRecoPrompt.Write("hCentDiffYieldsRecoPrompt", write_opt_mc)
                    hCentDiffYieldsRecoFD.Sumw2()
                    hCentDiffYieldsRecoFD.Write("hCentDiffYieldsRecoFD", write_opt_mc)
                    hCentDiffYieldsGenPrompt.Sumw2()
                    hCentDiffYieldsGenPrompt.Write("hCentDiffYieldsGenPrompt", write_opt_mc)
                    hCentDiffYieldsGenFD.Sumw2()
                    hCentDiffYieldsGenFD.Write("hCentDiffYieldsGenFD", write_opt_mc)

                    # Restore full cent range
                    for key, i_sparse in sparses_reco.items():
                        i_sparse.GetAxis(axes[key]['Cent']).SetRangeUser(centLowLim, centMaxLim)
                    for key, i_sparse in sparses_gen.items():
                        i_sparse.GetAxis(axes[key]['Cent']).SetRangeUser(centLowLim, centMaxLim)

            bar()

    outfile.Close()
