import os
import sys
import argparse
import yaml
import concurrent.futures
import time
import subprocess
# from concurrent.futures import ProcessPoolExecutor
work_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{work_dir}/utils")
from utils import check_dir, logger, merge_cutsets_fits, produce_pt_bins_fit_summary
from pathlib import Path

paths = {
	"Preprocess": os.path.join(work_dir, "./src/pre_process.py"),
	"YamlCuts": os.path.join(work_dir, "./src/make_cutsets_cfgs.py"),
	"Projections": os.path.join(work_dir, "./src/proj_thn.py"),
	"Efficiencies": os.path.join(work_dir, "./src/compute_efficiencies.py"),
	"GetVnVsMass": os.path.join(work_dir, "./src/get_vn_vs_mass.py"),
	"GetVnByYieldExtraction": os.path.join(work_dir, "./src/get_vn_by_yield_extraction.py"),
	"CutVariation": os.path.join(work_dir, "./src/cut_variation.py"),
	"DataDrivenFraction": os.path.join(work_dir, "./src/data_driven_fraction.py"),
	"GetV2VsFrac": os.path.join(work_dir, "./src/get_v2_vs_frac.py"),
}

def make_yaml(flow_config, outdir, correlated=False):
	logger("YAML file will be created", level="INFO")
	check_dir(f"{outdir}/cutsets")

	method = "--sp_correlated" if correlated else "--sp_combined"
	cmd = (
		f'python3 {paths["YamlCuts"]} {flow_config} -o {outdir} {method}'
	)

	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)


def project(flow_config, outdir, nworkers, mCutSets):
	logger("Projections will be performed", level="INFO")
	os.makedirs(f"{outdir}/projs", exist_ok=True)

	def run_projections(i):
		"""Run sparse projection for a given cutset index."""
		iCutSets = f"{i:02d}"
		logger(f"Processing cutset {iCutSets}...", level="INFO")

		config_cutset = f"{outdir}/cutsets/cutset_{iCutSets}.yml"
		cmd = (
			f"python3 {paths['Projections']} {flow_config} -cc {config_cutset}"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)

	with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
		results_proj = list(executor.map(run_projections, range(mCutSets)))


def efficiencies(flow_config, outdir, nworkers, mCutSets):
	logger("Efficiencies will be computed", level="INFO")
	check_dir(f"{outdir}/effs")

	def run_efficiency(i):
		"""Run efficiency computation for a given cutset index."""
		iCutSet = f"{i:02d}"
		print(f"\033[32mProcessing cutset {iCutSet}...\033[0m")

		proj_cutset = f"{outdir}/projs/proj_{iCutSet}.root"
		cmd = (
			f"python3 {paths['Efficiencies']} {flow_config} {proj_cutset} -b"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)

	with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
		results_eff = list(executor.map(run_efficiency, range(mCutSets)))


def get_vn(flow_config, outdir, nworkers, mCutSets, extraction_type):
	logger("Fit v2 vs mass will be performed", level="INFO")
	check_dir(f"{outdir}/raw_yields")

	if extraction_type != 'simfit':
		proj_cutset = f"{outdir}/projs/proj_00.root"
		cmd = (
			f"python3 {paths['GetVnByYieldExtraction']} {flow_config} -b"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)
	else:
		def run_fit(i):
			"""Run simultaneous fit for a given cutset index."""
			iCutSets = f"{i:02d}"
			cutset_cfg = f"{outdir}/cutsets/cutset_{iCutSets}.yml"
			print(f"\033[32mProcessing cutset {iCutSets}...\033[0m")

			proj_cutset = f"{outdir}/projs/proj_{iCutSets}.root"
			cmd = (
				f"python3 {paths['GetVnVsMass']} {flow_config} {cutset_cfg} {proj_cutset} -b"
			)
			logger(f"{cmd}", level="COMMAND")
			os.system(cmd)

		with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
			results_fit = list(executor.map(run_fit, range(mCutSets)))

		merge_cutsets_fits(Path(f"{outdir}/raw_yields"))
		produce_pt_bins_fit_summary(Path(f"{outdir}/raw_yields"), flow_config)


def cut_variation(flow_config, outdir, correlated, combined=False, operations=None):
	check_dir(f"{outdir}/cutVar")

	if correlated:
		logger("Cut variation will be performed", level="INFO")

		ry_path = f"{outdir}/raw_yields"
		eff_path = f"{outdir}/effs"

		cmd = (
			f"python3 {paths['CutVariation']} {flow_config} {ry_path} {eff_path} -b"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)

	if combined:
		outdirCorr = os.path.join(os.path.dirname(outdir), os.path.basename(outdir).replace("_combined", "_correlated"))
		if os.path.exists(f"{outdirCorr}/cutVar/cutVar.root"):
			logger("Cut variation with correlated cuts found!", level="INFO")
			logger(f"The cut variation file {outdirCorr}/cutVar/cutVar.root will be used", level="WARNING")
			cmd = (
				f"cp {outdirCorr}/cutVar/cutVar.root {outdir}/cutVar/cutVar.root"
			)
			logger(f"{cmd}", level="COMMAND")
			os.system(cmd)
		else:
			logger(f"Cut variation with correlated cuts not found in {outdirCorr}, running correlated cut variation", level="WARNING")

			logger("Running analysis with correlated cut variation", level="WARNING")
			logger("========================================================================================", level="WARNING")
			import copy
			operationsCorr = copy.deepcopy(operations)
			for key in operationsCorr:
				if key in ['do_cut_variation', 'make_yaml', 'proj_mc', 'proj_data', 'efficiencies', 'get_vn_vs_mass']:
					operationsCorr[key] = True
				else:
					operationsCorr[key] = False
			if 'get_vn_yield_extraction' in operationsCorr:
				operationsCorr['get_vn_vs_mass'] = True
			os.makedirs(os.path.join(outdir, "config_flow"), exist_ok=True)
			flow_configCorr = os.path.join(outdir, "config_flow", f"corr_{os.path.basename(flow_config)}")
			with open(flow_config, 'r') as cfgFlow:
				config = yaml.safe_load(cfgFlow)
				config['operations'] = operationsCorr
				config['outdir'] = os.path.dirname(outdir)
				with open(flow_configCorr, 'w') as cfgFlow:
					yaml.dump(config, cfgFlow, default_flow_style=True)
			logger(f"Using temporary flow config: {flow_configCorr}", level="WARNING")
			run_correlated_cut_variation(flow_configCorr, operationsCorr, nworkers, outdirCorr)
			logger("========================================================================================", level="WARNING")

			logger("Copying cut variation with correlated cuts to combined cut variation", level="WARNING")
			cmd = (
				f"cp {outdirCorr}/cutVar/cutVar.root {outdir}/cutVar/cutVar.root"
			)
			logger(f"{cmd}", level="COMMAND")
			os.system(cmd)


def data_driven_fraction(outdir, combined=False):

	if combined:
		correlated_path = os.path.join(os.path.dirname(outdir), os.path.basename(outdir).replace("_combined", "_correlated"))
		logger(f"Using cut variation from correlated analysis at {correlated_path}", level="INFO")
		vn_extr_file = f"{correlated_path}/cutVar/cutVar.root"
	else:
		logger("Data driven fraction should be performed only for combined analysis. Are you sure you want to continue?", level="WARNING")
		vn_extr_file = f"{outdir}/cutVar/cutVar.root"
	
	eff_path = f"{outdir}/effs"

	cmd = (
		f"python3 {paths['DataDrivenFraction']} {vn_extr_file} {eff_path} -b"
	)
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)

 
def get_v2_vs_frac(flow_config, outdir, correlated=False, batch=False):
	logger("Fit v2 vs fd fraction will be performed", level="INFO")
	check_dir(f"{outdir}/v2")

	ry_path = f"{outdir}/raw_yields"
	frac_path = f"{outdir}/frac"

	cmd = (
		f"python3 {paths['GetV2VsFrac']} {flow_config} {ry_path} {frac_path} -b"
	)
	if correlated:
		cmd += " --correlated"
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)


def run_correlated_cut_variation(flow_config, operations, nworkers, outdir):

	#___________________________________________________________________________________________________________________________
	# make yaml file
	if operations.get('make_yaml', False):
		make_yaml(flow_config, outdir, correlated=True)
	else:
		logger("Make yaml will not be performed", level="WARNING")

	mCutSets = len([f for f in os.listdir(f"{outdir}/cutsets") if os.path.isfile(os.path.join(f"{outdir}/cutsets", f))])
	logger(f"mCutSets: {mCutSets}", level="INFO")

	#___________________________________________________________________________________________________________________________
	# Projection for MC and apply the ptweights
	if operations.get('proj_mc', False) or operations.get('proj_data', False):
		project(flow_config, outdir, nworkers, mCutSets)
	else:
		logger("Projections will not be performed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Efficiencies
	if operations.get('efficiencies', False):
		efficiencies(flow_config, outdir, nworkers, mCutSets)
	else:
		logger("Efficiencies will not be computed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Simultaneous fit
	if operations.get('get_vn_vs_mass', False):
		get_vn(flow_config, outdir, nworkers, mCutSets, 'simfit')
	elif operations.get('get_vn_yield_extraction', False):
		get_vn(flow_config, outdir, nworkers, mCutSets, 'yield_extraction')
	else:
		logger("v2 signal will not be extracted", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Cut variation
	if operations.get('do_cut_variation'):
		cut_variation(flow_config, outdir, correlated=True, combined=False, operations=operations)
	else:
		logger("Cut variation will not be performed", level="WARNING")
  
	#___________________________________________________________________________________________________________________________
	 # Data driven fraction
	if operations.get('data_driven_fraction', False):
		data_driven_fraction(outdir)
	else:
		logger("Data driven fraction will not be performed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	 # linear fit of vn vs fd fraction
	if operations.get('get_v2_vs_frac'):
		get_v2_vs_frac(flow_config, outdir, correlated=True)
	else:
		logger("Fit v2 vs fd fraction will not be performed", level="WARNING")


def run_combined_cut_variation(flow_config, operations, nworkers, outdir):

	#___________________________________________________________________________________________________________________________
	# make yaml file
	if operations.get('make_yaml', False):
		make_yaml(flow_config, outdir, correlated=False)
	else:
		logger("Make yaml will not be performed", level="WARNING")

	mCutSets = len([f for f in os.listdir(f"{outdir}/cutsets") if os.path.isfile(os.path.join(f"{outdir}/cutsets", f))])
	logger(f"mCutSets: {mCutSets}", level="INFO")

	#___________________________________________________________________________________________________________________________
	# Projection for MC and apply the ptweights
	if operations.get('proj_mc', False) or operations.get('proj_data', False):
		project(flow_config, outdir, nworkers, mCutSets)
	else:
		logger("Projections will not be performed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Efficiencies
	if operations.get('efficiencies', False):
		efficiencies(flow_config, outdir, nworkers, mCutSets)
	else:
		logger("Efficiencies will not be computed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Simultaneous fit
	if operations.get('get_vn_vs_mass', False):
		get_vn(flow_config, outdir, nworkers, mCutSets, 'simfit')
	elif operations.get('get_vn_yield_extraction', False):
		get_vn(flow_config, outdir, nworkers, mCutSets, 'yield_extraction')
	else:
		logger("v2 signal will not be extracted", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Cut variation
	if operations.get('do_cut_variation', False):
		cut_variation(flow_config, outdir, correlated=False, combined=True, operations=operations)
	else:
		logger("Cut variation will not be performed", level="WARNING")
  
	#___________________________________________________________________________________________________________________________
	# Data driven fraction
	if operations.get('data_driven_fraction', False):
		data_driven_fraction(outdir, combined=True)
	else:
		logger("Data driven fraction will not be performed", level="WARNING")
  
	#___________________________________________________________________________________________________________________________
	# linear fit of vn vs fd fraction
	if operations.get('get_v2_vs_frac'):
		get_v2_vs_frac(flow_config, outdir, correlated=False)
	else:
		logger("Fit v2 vs fd fraction will not be performed", level="WARNING")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('flow_config', metavar='text', default='config_flow_d0.yml', help='configuration file')
	parser.add_argument("--workers", "-w", type=int, default=1, help="number of workers")
	parser.add_argument("--correlated", "-corr", action="store_true", help="perform correlated analysis")
	parser.add_argument("--combined", "-comb", action="store_true", help="perform combined analysis")
	args = parser.parse_args()

	start_time = time.time()

	# Load and copy the configuration file
	with open(args.flow_config, 'r') as cfgFlow:
		config = yaml.safe_load(cfgFlow)

	operations = config['operations']
	nworkers = args.workers
	outdir = config['outdir']
 
	# For bdt bkg cut scan
	if "bkg_" not in config['outdir'] and "syst" not in config['outdir']:
		if args.correlated:
			outdir = f"{config['outdir']}/vn_extr_{config['suffix']}" + "_correlated"
		else:
			outdir = f"{config['outdir']}/vn_extr_{config['suffix']}" + "_combined"
	os.system(f"mkdir -p {outdir}")

	# copy the configuration file
	nfile = 0
	os.makedirs(f'{outdir}/config_flow', exist_ok=True)
	while os.path.exists(f'{outdir}/config_flow/{os.path.splitext(os.path.basename(args.flow_config))[0]}_{config["suffix"]}_{nfile}.yml'):
		nfile = nfile + 1
	os.system(f'cp {args.flow_config} {outdir}/config_flow/{os.path.splitext(os.path.basename(args.flow_config))[0]}_{config["suffix"]}_{nfile}.yml')

	if operations.get('preprocess'):
		logger("Preprocess will be performed", level="INFO")
		os.system(f"python3 {paths['Preprocess']} {args.flow_config} --workers {nworkers}")
	else:
		logger("Preprocess will not be performed", level="WARNING")

	if not args.correlated and not args.combined:
		logger("No cut variation will be performed, please specify --correlated or --combined", level="WARNING")
		sys.exit(0)
	if args.correlated:
		run_correlated_cut_variation(args.flow_config, operations, nworkers, outdir)
	if args.combined:
		run_combined_cut_variation(args.flow_config, operations, nworkers, outdir)

	end_time = time.time()
	execution_time = end_time - start_time
	logger("Analysis completed, total execution time: {:.2f} seconds".format(execution_time), level="INFO")
