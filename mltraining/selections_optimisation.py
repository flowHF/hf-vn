import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
from concurrent.futures import ThreadPoolExecutor
import itertools

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import uproot
import yaml
import seaborn as sns

from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import zfit

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger

def __draw_1d(ax, df, cols_to_plot, cols_names, cfg, i_pt):
    x_axis = "ML_output_Prompt" #"ML_output_Bkg"
    x_axis_name = "BDT background score <"

    for i, (col, col_name) in enumerate(zip(cols_to_plot, cols_names)):
        # Draw the col_name as a function of the x_axis
        ax[i // 2, i % 2].plot(
            df[x_axis],
            df[col],
            marker='o',
            linestyle='None',
            label=col_name,
            markersize=5
        )
        # Get the actual x and y labels
        x_labels = ['' for _ in range(len(df))]
        x_labels[0] = f"{df[x_axis].iloc[0]:.2f}"
        x_labels[-1] = f"{df[x_axis].iloc[-1]:.2f}"
        
        ax[i // 2, i % 2].set_title(col_name, fontsize=25)
        ax[i // 2, i % 2].set_xlabel(x_axis_name, fontsize=20)
        ax[i // 2, i % 2].tick_params(axis='x', which='both')

        if cfg["working_points"] is not None:
            ax[i // 2, i % 2].axvline(
                x=cfg["working_points"][x_axis][i_pt],
                color='black', linestyle='--', linewidth=2
            )


def __draw_2d(ax, df, cols_to_plot, cols_names, cfg, i_pt):
    x_axis = "ML_output_Bkg"
    x_axis_name = "BDT background score <"
    y_axis = "ML_output_Prompt"
    y_axis_name = "BDT prompt score >"

    for i, (col, col_name) in enumerate(zip(cols_to_plot, cols_names)):
        df_pivot = df.pivot(index=y_axis, columns=x_axis, values=col)
        sns.heatmap(
            df_pivot, ax=ax[i // 2, i % 2], cmap='coolwarm', cbar=True,
            norm=matplotlib.colors.LogNorm() if 'efficiencies' in col else None)
        # Get the actual x and y labels
        x_labels = ['' for _ in df_pivot.columns]
        y_labels = ['' for _ in df_pivot.index]
        x_labels[0] = f"{df_pivot.columns[0]:.2f}"
        x_labels[-1] = f"{df_pivot.columns[-1]:.2f}"
        y_labels[0] = f"{df_pivot.index[0]:.2f}"
        y_labels[-1] = f"{df_pivot.index[-1]:.2f}"

        # Set the x and y tick labels
        ax[i // 2, i % 2].invert_yaxis()
        ax[i // 2, i % 2].set_xticks(np.arange(len(x_labels)) + 0.5)  # Set x ticks to center the labels
        ax[i // 2, i % 2].set_xticklabels(x_labels)
        ax[i // 2, i % 2].set_yticks(np.arange(len(y_labels)) + 0.5)  # Set y ticks to center the labels
        ax[i // 2, i % 2].set_yticklabels(y_labels)
        
        ax[i // 2, i % 2].set_title(col_name, fontsize=25)
        ax[i // 2, i % 2].set_xlabel(x_axis_name, fontsize=20)
        ax[i // 2, i % 2].set_ylabel(y_axis_name, fontsize=20)
        ax[i // 2, i % 2].tick_params(axis='both', which='both', labelsize=20)
        ax[i // 2, i % 2].tick_params(axis='x', which='both')
        cbar = ax[i // 2, i % 2].collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

        if cfg["working_points"] is not None:
            x_on_plot = (cfg["working_points"][x_axis][i_pt] - df_pivot.columns.min()) / (df_pivot.columns.max() - df_pivot.columns.min())
            x_on_plot *= len(df_pivot.columns)
            y_on_plot = (cfg["working_points"][y_axis][i_pt] - df_pivot.index.min()) / (df_pivot.index.max() - df_pivot.index.min())
            y_on_plot *= len(df_pivot.index)
            ax[i // 2, i % 2].scatter(
                x_on_plot,
                y_on_plot,
                color='black', 
                marker='X', s=100
            )


def draw_selection_scan(df, cfg, i_pt):
    pt_min = cfg['pt_mins'][i_pt]
    pt_max = cfg['pt_maxs'][i_pt]

    fig, ax = plt.subplots(3, 2, figsize=(16, 16))

    cols_to_plot = [
        "efficiencies_prompt", "efficiencies_fd",
        "expected_signals_fd", "expected_bkgs_prompt",
        "expected_significances_fd", "fracs_fd"
    ]

    cols_names = [
        r"Prompt efficiency", r"Non-prompt efficiency",
        r"Expected signal non-prompt", r"Expected background",
        r"Expected significance non-prompt", r"Non-prompt fraction"
    ]

    dim = len(df.columns) - list(df.columns).index("fracs_fd") - 1

    if dim == 1:
        __draw_1d(ax, df, cols_to_plot, cols_names, cfg, i_pt)
    else:
        __draw_2d(ax, df, cols_to_plot, cols_names, cfg, i_pt)


    fig.tight_layout()
    fig.savefig(
        os.path.join(
            cfg['output_dir'],
            f"selection_scan_pt_{pt_min * 10:.0f}_{pt_max * 10:.0f}.pdf"
        )
    )
    plt.close(fig)


def dump_results(results, cfg, pt_min, pt_max):

    print(f"Dumping results for pT bin {pt_min}-{pt_max} GeV/c")
    print(f"Number of results: {len(results)}")
    print(f"results: {results}")

    out_dicts = []
    for future, selection in results:
        try:
            results_dict = future.result()  # re-raises the real exception
        except Exception:
            logger(
                f"Error in task for selection '{selection}':\n{traceback.format_exc()}",
                "ERROR"
            )
            raise  # stop everything immediately

        if future.exception() is not None:
            logger(f"Error in task: {future.exception()}", "ERROR")

        results_dict = future.result()
        out_dict = {}
        print(results_dict)
        for var in results_dict:
            for origin in results_dict[var]:
                out_dict[f"{var}_{origin}"] = results_dict[var][origin]
    
        # selection = result[1]
        selections = selection.split(" and ")
        for sel in selections:        
            out_dict[sel.split(" ")[0]] = float(sel.split(" ")[2])

        out_dicts.append(out_dict)
        
    out_df = pd.DataFrame(out_dicts)
    out_file = os.path.join(
        cfg['output_dir'],
        f"selection_scan_pt_{pt_min * 10:.0f}_{pt_max * 10:.0f}.parquet"
    )
    out_df.to_parquet(out_file)

    return out_df


def load_predictions(cfg_predictions):
    """
    Load predictions from a ROOT file.

    Parameters:
    - cfg_predictions (dict): Configuration dictionary.

    Returns:
    - dict: A dictionary with the following keys:
        - 'prompt': The differential cross-section 
            uproot.Model_TH1D_v3 histogram for prompt.
        - 'feeddown': The differential cross-section 
            uproot.Model_TH1D_v3 histogram for feeddown.
    """
    with uproot.open(cfg_predictions['crosssec']['filename']) as f:
        prediction_prompt = f[cfg_predictions['crosssec']['histonames']['prompt']]
        prediction_fd = f[cfg_predictions['crosssec']['histonames']['feeddown']]
    return {
        'prompt': prediction_prompt,
        'feeddown': prediction_fd,
    }


def __get_prediction(predictions_hist, origin, pt_min, pt_max, br_corr):
    pt_bins = predictions_hist[origin].axis().edges()
    pt_min_idx = np.nonzero(np.isclose(pt_bins, pt_min))[0][0]
    pt_max_idx = np.nonzero(np.isclose(pt_bins, pt_max))[0][0]

    dsigma_dpt = 0.
    for i_pt in range(pt_min_idx, pt_max_idx):
        d_pt = pt_bins[i_pt + 1] - pt_bins[i_pt]
        dsigma_dpt += predictions_hist[origin].values()[i_pt] * d_pt

    return {origin: dsigma_dpt / (pt_max - pt_min) * br_corr}


def get_pt_predictions(predictions_histos, pt_min, pt_max, cfg):
    """
    Calculate the differential cross-section (dsigma/dpt x BR) for prompt and feeddown
    components of specified particles within a specified transverse momentum (pt) range.

    Parameters:
    - predictions_histos (dict): A dictionary containing 'prompt' and 'feeddown' keys, 
        each associated with a uproot.Model_TH1D_v3 object containing the predictions.
    - pt_min (float): The minimum pt value of the range.
    - pt_max (float): The maximum pt value of the range.

    Returns:
        dict: A dictionary containing pT predictions for the specified particles and origins.
    """
    origins = ['prompt', 'feeddown']

    predictions_pt = {}
    for origin in origins:
        br_corr = cfg['predictions']['crosssec']['histonames']['br_corr']
        predictions_pt.update(__get_prediction(predictions_histos, origin, pt_min, pt_max, br_corr))

    return predictions_pt


def load_data(cfg):
    cfg_input = cfg['infiles']
    data_files = [f"{cfg_input['model_dir']}/pt_{pt_min}_{pt_max}/application/{cfg_input['data_label']}_pT_{pt_min}_{pt_max}_ModelApplied.parquet.gzip" \
                  for pt_min, pt_max in zip(cfg['pt_mins'], cfg['pt_maxs'])]
    mc_prompt_files = [f"{cfg_input['model_dir']}/pt_{pt_min}_{pt_max}/application/{cfg_input['prompt_label']}_pT_{pt_min}_{pt_max}_ModelApplied.parquet.gzip" \
                       for pt_min, pt_max in zip(cfg['pt_mins'], cfg['pt_maxs'])]
    mc_fd_files = [f"{cfg_input['model_dir']}/pt_{pt_min}_{pt_max}/application/{cfg_input['feeddown_label']}_pT_{pt_min}_{pt_max}_ModelApplied.parquet.gzip" \
                   for pt_min, pt_max in zip(cfg['pt_mins'], cfg['pt_maxs'])]

    df_data = pd.concat([pd.read_parquet(f) for f in data_files])
    df_mc_prompt = pd.concat([pd.read_parquet(f) for f in mc_prompt_files])
    df_mc_fd = pd.concat([pd.read_parquet(f) for f in mc_fd_files])

    return {
        'data': df_data, 'mc_prompt': df_mc_prompt, 'mc_fd': df_mc_fd
    }


def load_mc_fitters(trial, data_handlers, cfg):
    """
    Load and configure fitters for different datasets based on the provided trial, data handlers, and configuration.

    Parameters:
    - trial (dict): A dictionary containing trial information.
    - data_handlers (dict): A dictionary of data handlers for different datasets.
    - cfg (dict): Configuration dictionary containing information about the background fit function.

    Returns:
    dict: A dictionary of configured fitters for each dataset.
    """
    fitters = {}
    for df in trial['dfs']:
        if df == 'data':
            continue # we will set it after the MC fits
        else:
            fitters[df] = F2MassFitter(
                data_handlers[df], name_signal_pdf=['gaussian'],
                name_background_pdf=["nobkg"],
                name=f"fit_{df}_pt_{trial['pt_min'] * 10:.0f}_{trial['pt_max'] * 10:.0f}",
                verbosity=1, tol=1.e-1
            )

    fitters['mc_prompt'].set_particle_mass(0, pdg_id=411)
    fitters['mc_prompt'].set_signal_initpar(0, "sigma", 0.008, limits=[0., 0.1])
    fitters['mc_prompt'].set_signal_initpar(0, "frac", 0.1, limits=[0., 1.])

    fitters['mc_fd'].set_particle_mass(0, pdg_id=411)
    fitters['mc_fd'].set_signal_initpar(0, "sigma", 0.008, limits=[0., 0.1])
    fitters['mc_fd'].set_signal_initpar(0, "frac", 0.1, limits=[0., 1.])

    return fitters


def get_efficiencies(trial):
    """
    Calculate the efficiencies.

    Parameters:
    - trial (dict): The considerd trial.

    Returns:
    - efficiencies (dict): A dictionary with the calculated efficiencies.
    """
    efficiencies = {}
    for df in trial['dfs']:
        if "mc" in df:
            sel_cands = len(trial['dfs'][df].query(trial['selection']))
            tot_cands = len(trial['dfs'][df])
            origin = df.replace("mc_", "")
            efficiencies[origin] = sel_cands / tot_cands

    return efficiencies


def get_expected_signals(trial, cfg, efficiencies):
    """
    Calculate the expected signals.

    Parameters:
    - trial (dict): The considered trial.
    - cfg (dict): The configuration dictionary.
    - efficiencies (dict): The calculated efficiencies.

    Returns:
    - expected_signals (dict): A dictionary with the calculated expected signals.
    """
    expected_signals = {}
    int_lumi = cfg['n_expected_events'] / cfg['sigma_mb']
    d_pt = trial['pt_max'] - trial['pt_min']
    corr_factors = 2 * int_lumi * d_pt
    expected_signals['prompt'] = efficiencies['prompt'] * trial['predictions']['prompt'] * corr_factors
    expected_signals['fd'] = efficiencies['fd'] * trial['predictions']['feeddown'] * corr_factors

    return expected_signals

def get_expected_bkgs(trial, cfg):
    dfs_sel = {
        df: trial['dfs'][df].query(trial['selection'])
        for df in trial['dfs']
    }

    data_handlers = load_data_handlers(
        dfs_sel,
        trial['min_mass'],
        trial['max_mass'],
        trial['fraction_to_keep']
    )
    # dfs_sel = {}
    # trial['sel_bkg_dfs'] = {}
    # for df in trial['dfs']:
    #     dfs_sel[df] = trial['dfs'][df].query(trial['selection'])
    #     trial['sel_bkg_dfs'][df] = dfs_sel[df]
    # for df in trial['dfs']:
    #     trial['dfs'][df] = trial['dfs'][df].copy()
    #     trial['dfs'][df] = trial['dfs'][df].query(trial['selection'])

    # data_handlers = load_data_handlers(trial)
    fitters = load_mc_fitters(trial, data_handlers, cfg)
    # First, we fit the mc signal to get means and sigmas
    fitters['mc_prompt'].mass_zfit()
    fitters['mc_prompt'].mass_zfit()
    mean = fitters['mc_prompt'].get_mass(0)[0]
    sigma = fitters['mc_prompt'].get_sigma(0)[0]
    n_sigma = cfg['infiles']['background']['n_sigma']
    limits = [
        [trial['min_mass'], mean - n_sigma * sigma],
        [mean + n_sigma * sigma, trial['max_mass']]
    ]

    fitters['data'] = F2MassFitter(
        data_handlers['data'], name_signal_pdf=['nosignal'],
        name_background_pdf=[cfg['infiles']['background']['fit_func']],
        name=f"fit_data_pt_{trial['pt_min'] * 10:.0f}_{trial['pt_max'] * 10:.0f}_{trial['selection']}",
        limits=limits,
        verbosity=1, tol=1.e-1
    )
    if cfg['infiles']['background']['fit_func'] == "expo":
        fitters['data'].set_background_initpar(0, "lam", -4)
    elif cfg['infiles']['background']['fit_func'] == "chebpol2":
        fitters['data'].set_background_initpar(0, "c0", 0.6)
        fitters['data'].set_background_initpar(0, "c1", -0.2)
        fitters['data'].set_background_initpar(0, "c2", 0.01)

    fitters['data'].mass_zfit()
    #fig, ax = fitters['data'].plot_mass_fit(figsize=(8, 8))
    #fig.savefig(f"/home/fchinu/Run3/Ds_pp_13TeV/Optimization/fits/mass_fit_pt_{trial['pt_min'] * 10:.0f}_{trial['pt_max'] * 10:.0f}_{trial['selection']}.pdf")
    #plt.close(fig)
    scale_factor = cfg['n_expected_events'] / cfg['infiles']['background']['n_events'] / trial['fraction_to_keep']

    bkg = fitters['data'].get_background(
        min=mean - n_sigma * sigma,
        max=mean + n_sigma * sigma
    )[0] * scale_factor

    bkg = {
        'prompt': bkg,
        'fd': bkg
    }

    return bkg

def get_selections(cut_vars, i_pt):
    selections = []
    var_selections = []
    for var in cut_vars:
        var_selections.append(np.linspace(cut_vars[var]['min'][i_pt], cut_vars[var]['max'][i_pt], cut_vars[var]['steps'][i_pt]).tolist())
    combined_selections = itertools.product(*var_selections)
    for selection in combined_selections:
        sel = ""
        for i, var in enumerate(cut_vars):
            if cut_vars[var]['upper_lower_cut'] == 'upper':
                sign = "<"
            else:
                sign = ">"
            sel += f'{var} {sign} {selection[i]} and '
        selections.append(sel[:-4])
    return selections

def get_trial(dfs, predictions, selection, cfg, i_pt):
    return {
        'dfs': dfs,
        'predictions': predictions,
        'selection': selection,
        'min_mass': cfg['min_mass'][i_pt],
        'max_mass': cfg['max_mass'][i_pt],
        'fraction_to_keep': cfg['infiles']['background']['fraction_to_keep'][i_pt],
        'i_pt': i_pt,
        'pt_min': cfg['pt_mins'][i_pt],
        'pt_max': cfg['pt_maxs'][i_pt]
    }


def load_data_handlers(dfs, min_mass, max_mass, fraction_to_keep):
    data_handlers = {}
    for df in dfs:
        if df == 'data':
            data_handlers[df] = DataHandler(
                dfs[df].sample(frac=fraction_to_keep, random_state=42),
                var_name='fM', limits=[min_mass, max_mass]
            )
        else:
            data_handlers[df] = DataHandler(
                dfs[df], var_name='fM',
                limits=[min_mass, max_mass]
            )
    return data_handlers

def __get_signif(sgn, bkg):
    """
    Calculate the significance of a signal over background.

    Parameters:
    - sgn (float): The number of signal events.
    - bkg (float): The number of background events.

    Returns:
    - float: The significance of the signal.
    """
    return sgn / np.sqrt(sgn + bkg)

def get_expected_significances(expected_signals, expected_bkg):
    """
    Calculate the expected significances for different signal and background combinations.

    Parameters:
    - expected_signals (dict): A dictionary containing the expected signal counts.
    - expected_bkg (dict): A dictionary containing the expected background counts.

    Returns:
    - dict: A dictionary with the calculated significances for each signal type.
    """

    # No difference between expected_bkg for prompt and feeddown
    return {
        'prompt': __get_signif(expected_signals['prompt'], expected_bkg['prompt']),
        'fd': __get_signif(expected_signals['fd'], expected_bkg['fd'])
    }

def get_fracs(expected_signals, trial):
    """
    Calculate the prompt and non-prompt fractions.

    Parameters:
    - expected_signals (dict): A dictionary containing the expected signal counts.
    - trial (dict): The considered trial.

    Returns:
    - tuple: A tuple containing the prompt and non-prompt fractions.
    """

    fracs = {}

    fracs['prompt'] = expected_signals['prompt'] / (expected_signals['prompt'] + expected_signals['fd'])
    fracs['fd'] = expected_signals['fd'] / (expected_signals['prompt'] + expected_signals['fd'])

    return fracs

def run_selection(trial, cfg):

    efficiencies = get_efficiencies(trial)
    print(f"\nEfficiencies: {efficiencies}")
    expected_signals = get_expected_signals(trial, cfg, efficiencies)
    print(f"\nExpected signals: {expected_signals}")
    expected_bkgs = get_expected_bkgs(trial, cfg)
    print(f"\nExpected bkgs: {expected_bkgs}")
    expected_significances = get_expected_significances(expected_signals, expected_bkgs)
    print(f"\nExpected significances: {expected_significances}")
    fracs = get_fracs(expected_signals, trial)
    print(f"\nFractions: {fracs}")

    return {
        'efficiencies': efficiencies,
        'expected_signals': expected_signals,
        'expected_bkgs': expected_bkgs,
        'expected_significances': expected_significances,
        'fracs': fracs
    }


def run_selection_scan(config_file_name, draw=False):
    with open(config_file_name, 'r', encoding='utf8') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['output_dir'], exist_ok=True)

    pt_mins = cfg['pt_mins']
    pt_maxs = cfg['pt_maxs']

    dfs = load_data(cfg)
    # Apply preselections
    for cut in cfg['presel_cuts']:
        for df in dfs:
            dfs[df] = dfs[df].query(f"{cfg['presel_cuts'][cut]['min']} < {cut} < {cfg['presel_cuts'][cut]['max']}")
    predictions = load_predictions(cfg['predictions'])

    for i_pt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
        dfs_pt = {}
        if not draw:
            for df in dfs:
                dfs_pt[df] = dfs[df].query(f'{pt_min} < fPt < {pt_max}')
            predictions_pt = get_pt_predictions(predictions, pt_min, pt_max, cfg)
            selections = get_selections(cfg['cut_vars'], i_pt)

            results = []
            with ThreadPoolExecutor(max_workers=cfg['n_workers']) as executor:
                for selection in selections:
                    dfs_trial = {k: v.copy(deep=True) for k, v in dfs_pt.items()}
                    trial = get_trial(dfs_trial, predictions_pt, selection, cfg, i_pt)
                    results.append((executor.submit(run_selection, trial, cfg), selection))
            
            # Check results for errors
            for future, selection in results:
                if future.exception() is not None:
                    logger(f"Worker generated an exception: {future.exception()}", "ERROR")
                    raise future.exception()

            out_df = dump_results(results, cfg, pt_min, pt_max)
        else:
            infile = os.path.join(
                cfg['output_dir'],
                f"selection_scan_pt_{pt_min * 10:.0f}_{pt_max * 10:.0f}.parquet"
            )
            out_df = pd.read_parquet(infile)
        draw_selection_scan(out_df, cfg, i_pt)


if __name__=="__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Optimise model parameters")
    parser.add_argument("config_file", type=str, help="Path to the config file")
    parser.add_argument('--draw', action='store_true', help='Just draw the selection scan')
    args = parser.parse_args()

    run_selection_scan(args.config_file, args.draw)
