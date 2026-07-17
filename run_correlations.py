import yaml, os
from itertools import product
import pathlib as PATH
from tqdm.notebook import tqdm
import ROOT
import time
import argparse

HFVN_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
HFVN_POSTDIR = f"{HFVN_DIR}/correlations/PostProcessing"
print(f"[INFO] hf-vn directory: {HFVN_DIR}")

import sys
work_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{work_dir}/utils")
from utils import logger

paths = {
	"Preprocess": os.path.join(work_dir, "./src/pre_process.py"),
	"YamlCuts": os.path.join(work_dir, "./src/make_cutsets_cfgs.py"),
	"CorrelMaps": os.path.join(work_dir, "./correlations/PostProcessing/extract_output_correl.py"),
	"RyTriggerFit": os.path.join(work_dir, "./correlations/PostProcessing/extract_ry_triggers.py"),
	"FitCorrelations": os.path.join(work_dir, "./correlations/PostProcessing/fit_correl.py"),
	"GetVnVsMass": os.path.join(work_dir, "./src/get_vn_vs_mass.py"),
	"CutVariation": os.path.join(work_dir, "./src/cut_variation.py"),
	"DataDrivenFraction": os.path.join(work_dir, "./src/data_driven_fraction.py"),
	"GetV2VsFrac": os.path.join(work_dir, "./src/get_v2_vs_frac.py"),
}


def make_yaml(flow_config, outdir):
	logger("YAML file will be created", level="INFO")

	cmd = (
		f'python3 {paths["YamlCuts"]} {flow_config} -o {outdir} -2pc'
	)
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)


def run_command(cmd, log, label):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] {cmd} > {log} 2>&1")
    out = os.system(f"{cmd} > {log} 2>&1")
    if out != 0:
        print(f"[{label}] WARNING: Command exited with code {out}")


def produce_correl_maps(flow_config):
	logger("Producing correlation maps", level="INFO")

	cmd = (
		f'python3 {paths['CorrelMaps']} {flow_config}'
	)
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)


def task_fit_mass(flow_config):
	print(f"output log: {flow_config.parent / f"log_fit_mass_{flow_config.stem}.txt"}")
	run_command(f"python3 {HFVN_DIR}/src/ry_interface.py {flow_config}", flow_config.parent / f"log_fit_mass_{flow_config.stem}.txt", "FitMass")


def fit_correlations(flow_config):
	logger("Producing correlation maps", level="INFO")

	cmd = (
		f'python3 {paths['FitCorrelations']} {flow_config}'
	)
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)


def extract_ry_trigger(flow_config):
	logger("Extracting raw yield trigger", level="INFO")

	cmd = (
		f'python3 {paths['RyTriggerFit']} extract-ry-trigger {flow_config}'
	)
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)


def make_cutsets_cfgs(flow_config, outdir):
    print(f"output log: {flow_config.parent / f"log_cutsets_{flow_config.stem}.txt"}")
    print(f"Producing cutset configs in {outdir}/CUTSETS/")
    run_command(f"python3 {HFVN_DIR}/src/make_cutsets_cfgs.py {flow_config} --outputdir {outdir}", flow_config.parent / f"log_cutsets_{flow_config.stem}.txt", "Cutsets")


def perform_sim_fit(flow_config):
    logger("Performing simultaneous fit", level="INFO")

    with open(flow_config) as f:
        config = yaml.safe_load(f)

    # make_cutsets_cfgs(flow_config, config['outdir'])

    mass_v2_dir = PATH.Path(config['outdir']) / "vn_vs_mass"
    pt_bins_cand = [float(x) for x in config['ptBinsCand']]
    pt_bins_had = [float(x) for x in config['ptBinsHad']]

    fit_config = config.get('fitConfig', {})
    simfit_config = {
        "Dmeson": config.get("Dmeson", "Dzero"),
        "ptbins": pt_bins_cand,
        "centrality": config.get("centrality", "k020"),
        "v2extraction": {
            "SgnFunc": fit_config.get("SgnFunc", "kGaus"),
            "BkgFunc": fit_config.get("BkgFunc", "kExpo"),
            "BkgFuncVn": fit_config.get("BkgFuncVn", "kLin"),
            "Sigma": fit_config.get("Sigma", [0.02] * (len(pt_bins_cand)-1)),
            "MassFitRanges": fit_config.get("MassFitRanges", [[pt_bins_cand[0], pt_bins_cand[-1]]] * (len(pt_bins_cand)-1)),
            "Rebin": fit_config.get("Rebin", 4),
            "FixSigma": fit_config.get("FixSigma", 0),
            "FixMean": fit_config.get("FixMean", 0),
            "InclSecPeak": fit_config.get("InclSecPeak", 0),
            "enableRef": fit_config.get("enableRef", False),
            "ReflFunc": fit_config.get("ReflFunc", "2gaus")
        }
    }

    cutsets_cfgs = [f"{config['outdir']}/cutsets/{file}" for file in os.listdir(f"{config['outdir']}/cutsets/") if file.startswith("cutset_")]
    for i_pt_had in range(len(pt_bins_had)-1):

        pt_had_label = f"PtAssoc{int(pt_bins_had[i_pt_had]*10):02d}to{int(pt_bins_had[i_pt_had+1]*10):02d}"
        mass_v2_file = mass_v2_dir / f"InvMassVsV2_{pt_had_label}.root"

        sim_fit_cfg = mass_v2_dir / f"simfit_config_{pt_had_label}.yml"
        with open(sim_fit_cfg, 'w') as f:
            yaml.dump(simfit_config, f, default_flow_style=False)
        for cutset_cfg in cutsets_cfgs:
            logger(f"Running sim fit for hadron pt bin {pt_had_label} with cutset {cutset_cfg}", level="INFO")
            os.system(f"python3 {paths['GetVnVsMass']} {sim_fit_cfg} {cutset_cfg} {mass_v2_file} -b")


def produce_final_results(flow_config, vn_hadron):

    with open(flow_config) as f:
        config = yaml.safe_load(f)

    input_file = PATH.Path(config['outdir']) / f"CorrelExtract_{config['suffix']}" / "CorrelationFitResults" / "Output_CorrelationFitting_Root" / "CorrPhiD0.root"
    output_file = PATH.Path(config['outdir']) / f"CorrelExtract_{config['suffix']}" / "final_results.root"
    print(f"[FinalResults] Processing {input_file} -> {output_file}")

    if not input_file.exists():
        print(f"[FinalResults] WARNING: Input file {input_file} does not exist, skipping")
        return

    f = ROOT.TFile.Open(str(input_file))
    for k in f.GetListOfKeys():
        print(f"[FinalResults] Processing histogram: {k.GetName()}")
        h = f.Get(k.GetName())
        h.SetDirectory(0)
        h.Scale(1/vn_hadron)
        o = ROOT.TFile.Open(str(output_file), "UPDATE")
        o.cd()
        h.Write(k.GetName(), ROOT.TObject.kOverwrite)
        o.Close()
    f.Close()


def run_delta_phi(cfg_path, cfg_lm_path):

    # Launch analysis
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    if operations["do_maps"]:
        produce_correl_maps(cfg_path)

    if operations["do_trigger_ry"]:
        task_fit_mass(cfg_path)

    if config.get('task_LM', {}).get('do', False):
        if operations["do_lm_maps"]:
            produce_correl_maps(cfg_lm_path)
        if operations["do_lm_trigger_ry"]:
            task_fit_mass(cfg_lm_path)

    if operations["do_fit_correl"]:
        fit_correlations(cfg_path)

    if operations["do_final_results"]:
        produce_final_results(cfg_path, config["vnHadron"])


def run_vn_vs_mass(cfg_path, cfg_lm_path):

    # Launch analysis
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    if operations["do_maps"]:
        produce_correl_maps(cfg_path)

    if operations["do_trigger_ry"]:
        extract_ry_trigger(cfg_path)

    if config.get('task_LM', {}).get('do', False):
        if operations["do_lm_maps"]:
            produce_correl_maps(cfg_lm_path)
        if operations["do_lm_trigger_ry"]:
            extract_ry_trigger(cfg_lm_path)

    if operations["do_fit_correl"]:
        fit_correlations(cfg_path)

    if operations["do_simfit"]:
        perform_sim_fit(cfg_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('flow_config', metavar='text', default='config_flow_d0.yml', help='configuration file')
    parser.add_argument("--workers", "-w", type=int, default=1, help="number of workers")
    parser.add_argument("--vn_vs_mass", "-vm", action="store_true", help="Vn vs. mass workflow")
    parser.add_argument("--delta_phi", "-dphi", action="store_true", help="Delta phi workflow")
    args = parser.parse_args()

    start_time = time.time()

    # Launch analysis
    with open(args.flow_config, 'r') as f:
        config = yaml.safe_load(f)
    method = config.get("method", "DeltaPhiBinning")
    logger(f"Using method: {method}", level="INFO")

    operations = config["operations"]
    if operations['do_make_yaml']:
        make_yaml(args.flow_config, config['outdir'])

    delta_eta_ranges = config.get("DeltaEtaRanges", [])
    for inner_edge, outer_edge in tqdm(delta_eta_ranges, desc="Processing delta eta ranges", unit="configs"):
        print(f"\n{'='*60}\n  inner={inner_edge}, outer={outer_edge}\n{'='*60}")

        base_cfg_str = f"{inner_edge}_{outer_edge}_{config['method']}"
        base_dir = f"{config['outdir']}/{base_cfg_str}/"
        deta_cfg_path = f"{base_dir}/config_{base_cfg_str}.yml"
        deta_cfg_path_lm = f"{base_dir}/low_mult/config_{base_cfg_str}_low_mult.yml"
        if args.vn_vs_mass:
            run_vn_vs_mass(deta_cfg_path, deta_cfg_path_lm)
        if args.delta_phi:
            run_delta_phi(deta_cfg_path, deta_cfg_path_lm)
