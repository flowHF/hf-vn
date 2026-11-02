import os
import yaml
import argparse
import ROOT
from ROOT import kRed, kBlue, kGreen, kOrange, kViolet, kCyan, kMagenta, kBlack, gStyle, TFile, TMath, TCanvas, TPad, TLegend, TLine, SetOwnership
import sys
sys.path.append('../utils/')
from StyleFormatter import SetGlobalStyle, SetObjectStyle, GetROOTColor, GetROOTMarker #pylint: disable=wrong-import-position,import-error

CLASSES = ['Bkg', 'Prompt', 'NonPrompt']
XAXISLABELS = ['Bkg Score (<)', 'Prompt Score (>)', 'NonPrompt Score (<)']


gStyle.SetOptStat(0)

def load_hists(models, pt_min, pt_max, cfg):
    """Load the 9 score histograms for all models in one pT bin."""
    result = {}

    for model_name, model_cfg in models.items():
        root_file = (
            f"{model_cfg['dir']}/pt_{pt_min}_{pt_max}/"
            f"apply_test_set/TestSetScores_pT_{pt_min}_{pt_max}.root"
        )

        if not os.path.isfile(root_file):
            print(f'WARNING: missing {root_file}')
            continue

        f = TFile.Open(root_file)
        if not f or f.IsZombie():
            print(f'WARNING: cannot open {root_file}')
            continue

        result[model_name] = {}

        for score in CLASSES:
            result[model_name][score] = {}

            for true in CLASSES:
                name = f'h{score}Score_{true}'
                hist = f.Get(name)

                if not hist:
                    print(f'WARNING: {name} not found in {root_file}')
                    continue

                hist = hist.Clone(f'{name}_{model_name}_{pt_min}_{pt_max}')
                hist.SetDirectory(0)
                result[model_name][score][true] = hist

        f.Close()

    return result


def draw_page(hists, pt_min, pt_max, score, true, cfg):
    """Draw one score/true-class comparison page with a ratio panel."""

    canvas = TCanvas(
        f'c_{score}_{true}_{pt_min}_{pt_max}',
        '',
        cfg.get('width', 850),
        cfg.get('height', 750)
    )

    SetOwnership(canvas, False)

    # Pads
    pad_main = TPad(
        f'pad_main_{score}_{true}_{pt_min}_{pt_max}',
        '',
        0.0, 0.30, 1.0, 1.0
    )

    pad_ratio = TPad(
        f'pad_ratio_{score}_{true}_{pt_min}_{pt_max}',
        '',
        0.0, 0.0, 1.0, 0.30
    )

    SetOwnership(pad_main, False)
    SetOwnership(pad_ratio, False)

    pad_main.SetLeftMargin(0.13)
    pad_main.SetRightMargin(0.04)
    pad_main.SetBottomMargin(0.01)
    pad_main.SetTopMargin(0.07)

    pad_ratio.SetLeftMargin(0.13)
    pad_ratio.SetRightMargin(0.04)
    pad_ratio.SetBottomMargin(0.32)
    pad_ratio.SetTopMargin(0.01)

    # Thicker frame/ticks
    pad_main.SetFrameLineWidth(2)
    pad_main.SetTickx(1)
    pad_main.SetTicky(1)

    pad_ratio.SetFrameLineWidth(2)
    pad_ratio.SetTickx(1)
    pad_ratio.SetTicky(1)

    if cfg.get('log_y', True):
        pad_main.SetLogy()

    canvas.cd()
    pad_main.Draw()
    pad_ratio.Draw()

    # Main panel
    pad_main.cd()

    ymax = max(
        hists[model][score][true].GetMaximum()
        for model in hists
        if score in hists[model]
        and true in hists[model][score]
    )

    first = True
    first_model = None
    first_hist = None

    for model, model_cfg in hists.items():

        if score not in model_cfg or true not in model_cfg[score]:
            continue

        hist = model_cfg[score][true]

        model_color = kBlack

        for model_cfg_yaml in cfg['models']:
            if model_cfg_yaml['name'] == model:
                model_color = GetROOTColor(model_cfg_yaml['color'])
                break

        hist.SetLineColor(model_color)
        hist.SetLineWidth(cfg.get('line_width', 2))
        hist.SetFillStyle(0)

        if first:
            first_model = model
            first_hist = hist

            hist.SetMinimum(cfg.get('ymin', 1e-5))
            hist.SetMaximum(
                ymax * cfg.get('ymax_factor', 1.3)
            )

            hist.SetTitle(
                f';{XAXISLABELS[CLASSES.index(true)]} for {score};'
                f'Surviving Fraction'
            )

            # Larger axis fonts
            hist.GetXaxis().SetTitleSize(0.060)
            hist.GetYaxis().SetTitleSize(0.060)
            hist.GetXaxis().SetLabelSize(0.050)
            hist.GetYaxis().SetLabelSize(0.050)

            hist.GetXaxis().SetTitleOffset(1.00)
            hist.GetYaxis().SetTitleOffset(1.05)

            # Thicker axes
            hist.GetXaxis().SetAxisColor(kBlack)
            hist.GetYaxis().SetAxisColor(kBlack)

            hist.Draw('HIST')
            first = False

        else:
            hist.Draw('HIST SAME')

    # Legend
    leg = TLegend(
        cfg['legend_x1'], cfg['legend_y1'],
        cfg['legend_x2'], cfg['legend_y2']
    )

    SetOwnership(leg, False)

    leg.SetHeader(
        f'{pt_min} < #it{{p}}_{{T}} < {pt_max} GeV/#it{{c}}'
    )

    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)

    # Larger legend font
    leg.SetTextSize(0.045)

    # Slightly thicker legend line
    leg.SetLineWidth(2)

    for model, model_cfg in hists.items():

        if score in model_cfg and true in model_cfg[score]:
            leg.AddEntry(
                model_cfg[score][true],
                model,
                'l'
            )

    leg.Draw()

    # Ratio panel
    pad_ratio.cd()

    ratio_hists = []

    ratio_min = float('inf')
    ratio_max = float('-inf')

    reference = first_hist

    if reference is None:
        return canvas

    # Construct ratio histograms
    for model, model_cfg in hists.items():

        if score not in model_cfg or true not in model_cfg[score]:
            continue

        hist = model_cfg[score][true]

        ratio = hist.Clone(
            f'ratio_{model}_{score}_{true}_{pt_min}_{pt_max}'
        )

        SetOwnership(ratio, False)

        ratio.Reset('ICES')

        for ibin in range(1, hist.GetNbinsX() + 1):

            numerator = hist.GetBinContent(ibin)
            denominator = reference.GetBinContent(ibin)

            if denominator > 0:

                value = numerator / denominator

                ratio.SetBinContent(ibin, value)

                num_error = hist.GetBinError(ibin)
                den_error = reference.GetBinError(ibin)

                if numerator > 0:
                    error = value * TMath.Sqrt(
                        (num_error / numerator) ** 2
                        + (den_error / denominator) ** 2
                    )

                    ratio.SetBinError(ibin, error)

            else:
                ratio.SetBinContent(ibin, 0.0)
                ratio.SetBinError(ibin, 0.0)

        # Style
        model_color = kBlack

        for model_cfg_yaml in cfg['models']:
            if model_cfg_yaml['name'] == model:
                model_color = GetROOTColor(model_cfg_yaml['color'])
                break

        ratio.SetLineColor(model_color)
        ratio.SetLineWidth(cfg.get('line_width', 2))
        ratio.SetMarkerColor(model_color)
        ratio.SetMarkerStyle(20)
        ratio.SetMarkerSize(0.45)

        ratio_hists.append(ratio)

        # Determine range
        for ibin in range(1, ratio.GetNbinsX() + 1):

            value = ratio.GetBinContent(ibin)

            if reference.GetBinContent(ibin) > 0:
                ratio_min = min(ratio_min, value)
                ratio_max = max(ratio_max, value)

    # Ratio range
    if ratio_min == float('inf'):
        ratio_min = 0.8
        ratio_max = 1.2

    if ratio_max == ratio_min:
        margin = 0.2 * abs(ratio_min)

        if margin == 0:
            margin = 0.2
    else:
        margin = cfg.get('ratio_margin', 0.15) * (
            ratio_max - ratio_min
        )

    ratio_ymin = max(0.0, ratio_min - margin)
    ratio_ymax = ratio_max + margin

    ratio_ymin = min(ratio_ymin, 1.0)
    ratio_ymax = max(ratio_ymax, 1.0)

    # Draw ratio
    first_ratio = True

    for ratio in ratio_hists:

        if first_ratio:

            ratio.SetMinimum(ratio_ymin)
            ratio.SetMaximum(ratio_ymax)

            ratio.SetTitle(
                f';'
                f'{XAXISLABELS[CLASSES.index(true)]} for {score};'
                'Ratio to first model'
            )

            # Larger ratio fonts
            ratio.GetXaxis().SetTitleSize(0.13)
            ratio.GetXaxis().SetLabelSize(0.105)

            ratio.GetYaxis().SetTitleSize(0.11)
            ratio.GetYaxis().SetLabelSize(0.095)

            ratio.GetXaxis().SetTitleOffset(1.00)
            ratio.GetYaxis().SetTitleOffset(0.55)

            ratio.GetXaxis().SetAxisColor(kBlack)
            ratio.GetYaxis().SetAxisColor(kBlack)

            ratio.GetYaxis().SetNdivisions(505)

            ratio.Draw('HIST')
            first_ratio = False

        else:
            ratio.Draw('HIST SAME')

    # Ratio = 1 reference line
    ratio_line = TLine(
        first_hist.GetXaxis().GetXmin(),
        1.0,
        first_hist.GetXaxis().GetXmax(),
        1.0
    )

    SetOwnership(ratio_line, False)

    ratio_line.SetLineStyle(2)
    ratio_line.SetLineWidth(2)
    ratio_line.Draw()

    # Keep objects alive
    canvas._pad_main = pad_main
    canvas._pad_ratio = pad_ratio
    canvas._legend = leg
    canvas._ratio_hists = ratio_hists
    canvas._ratio_line = ratio_line

    # Update
    pad_main.Modified()
    pad_ratio.Modified()

    canvas.Modified()
    canvas.Update()

    return canvas



# def draw_page(hists, pt_min, pt_max, score, true, cfg):
#     """Draw one score/true-class comparison page with a ratio panel."""

#     canvas = TCanvas(
#         f'c_{score}_{true}_{pt_min}_{pt_max}',
#         '',
#         cfg.get('width', 850),
#         cfg.get('height', 750)
#     )

#     # Keep ROOT objects owned by C++/canvas rather than Python
#     SetOwnership(canvas, False)

#     # Pads
#     pad_main = TPad(
#         f'pad_main_{score}_{true}_{pt_min}_{pt_max}',
#         '',
#         0.0, 0.30, 1.0, 1.0
#     )

#     pad_ratio = TPad(
#         f'pad_ratio_{score}_{true}_{pt_min}_{pt_max}',
#         '',
#         0.0, 0.0, 1.0, 0.30
#     )

#     SetOwnership(pad_main, False)
#     SetOwnership(pad_ratio, False)

#     pad_main.SetLeftMargin(0.13)
#     pad_main.SetRightMargin(0.04)
#     pad_main.SetBottomMargin(0.01)
#     pad_main.SetTopMargin(0.07)

#     pad_ratio.SetLeftMargin(0.13)
#     pad_ratio.SetRightMargin(0.04)
#     pad_ratio.SetBottomMargin(0.32)
#     pad_ratio.SetTopMargin(0.01)

#     if cfg.get('log_y', True):
#         pad_main.SetLogy()

#     # Draw pads onto canvas
#     canvas.cd()
#     pad_main.Draw()
#     pad_ratio.Draw()

#     # Main panel
#     pad_main.cd()

#     ymax = max(
#         hists[model][score][true].GetMaximum()
#         for model in hists
#         if score in hists[model]
#         and true in hists[model][score]
#     )

#     first = True
#     first_model = None
#     first_hist = None

#     for model, model_cfg in hists.items():

#         if score not in model_cfg or true not in model_cfg[score]:
#             continue

#         hist = model_cfg[score][true]

#         # Model colour
#         model_color = kBlack

#         for model_cfg_yaml in cfg['models']:
#             if model_cfg_yaml['name'] == model:
#                 model_color = GetROOTColor(
#                     model_cfg_yaml['color']
#                 )
#                 break

#         hist.SetLineColor(model_color)
#         hist.SetLineWidth(cfg.get('line_width', 2))
#         hist.SetFillStyle(0)

#         if first:
#             first_model = model
#             first_hist = hist

#             hist.SetMinimum(cfg.get('ymin', 1e-5))
#             hist.SetMaximum(
#                 ymax * cfg.get('ymax_factor', 1.3)
#             )

#             hist.SetTitle(
#                 f';{XAXISLABELS[CLASSES.index(true)]} for {score};'
#                 f'Surviving Fraction'
#             )

#             hist.GetXaxis().SetTitleSize(0.052)
#             hist.GetYaxis().SetTitleSize(0.052)
#             hist.GetXaxis().SetLabelSize(0.045)
#             hist.GetYaxis().SetLabelSize(0.045)
#             hist.GetYaxis().SetTitleOffset(1.15)

#             hist.Draw('HIST')
#             first = False

#         else:
#             hist.Draw('HIST SAME')

#     # Legend
#     leg = TLegend(
#         cfg['legend_x1'], cfg['legend_y1'],
#         cfg['legend_x2'], cfg['legend_y2']
#     )

#     SetOwnership(leg, False)

#     leg.SetHeader(
#         f'{pt_min} < #it{{p}}_{{T}} < {pt_max} GeV/#it{{c}}'
#     )
#     leg.SetBorderSize(0)
#     leg.SetFillStyle(0)
#     leg.SetTextFont(42)
#     leg.SetTextSize(0.038)

#     for model, model_cfg in hists.items():

#         if score in model_cfg and true in model_cfg[score]:
#             leg.AddEntry(
#                 model_cfg[score][true],
#                 model,
#                 'l'
#             )

#     leg.Draw()

#     # Ratio panel
#     pad_ratio.cd()

#     ratio_hists = []

#     ratio_min = float('inf')
#     ratio_max = float('-inf')

#     reference = first_hist

#     if reference is None:
#         return canvas

#     # ------------------------------------------------------------
#     # Construct ratio histograms
#     # ------------------------------------------------------------
#     for model, model_cfg in hists.items():

#         if score not in model_cfg or true not in model_cfg[score]:
#             continue

#         hist = model_cfg[score][true]

#         ratio = hist.Clone(
#             f'ratio_{model}_{score}_{true}_{pt_min}_{pt_max}'
#         )

#         # Prevent Python ownership/destruction
#         SetOwnership(ratio, False)

#         ratio.Reset('ICES')

#         for ibin in range(1, hist.GetNbinsX() + 1):

#             numerator = hist.GetBinContent(ibin)
#             denominator = reference.GetBinContent(ibin)

#             if denominator > 0:

#                 value = numerator / denominator

#                 ratio.SetBinContent(ibin, value)

#                 num_error = hist.GetBinError(ibin)
#                 den_error = reference.GetBinError(ibin)

#                 if numerator > 0:

#                     error = value * TMath.Sqrt(
#                         (num_error / numerator) ** 2
#                         + (den_error / denominator) ** 2
#                     )

#                     ratio.SetBinError(ibin, error)

#                 if TMath.Finite(value):
#                     ratio_min = min(ratio_min, value)
#                     ratio_max = max(ratio_max, value)

#             else:
#                 ratio.SetBinContent(ibin, 0.0)
#                 ratio.SetBinError(ibin, 0.0)

#         # Style
#         model_color = kBlack

#         for model_cfg_yaml in cfg['models']:
#             if model_cfg_yaml['name'] == model:
#                 model_color = GetROOTColor(
#                     model_cfg_yaml['color']
#                 )
#                 break

#         ratio.SetLineColor(model_color)
#         ratio.SetLineWidth(cfg.get('line_width', 2))
#         ratio.SetMarkerColor(model_color)
#         ratio.SetMarkerStyle(20)
#         ratio.SetMarkerSize(0.45)

#         ratio_hists.append(ratio)

#     # Ratio range
#     if ratio_min == float('inf'):
#         ratio_min = 0.8
#         ratio_max = 1.2

#     if ratio_max == ratio_min:
#         margin = 0.2 * abs(ratio_min)

#         if margin == 0:
#             margin = 0.2
#     else:
#         margin = cfg.get('ratio_margin', 0.15) * (
#             ratio_max - ratio_min
#         )

#     ratio_ymin = ratio_min - margin
#     ratio_ymax = ratio_max + margin

#     # Keep ratio=1 visible
#     ratio_ymin = min(ratio_ymin, 1.0)
#     ratio_ymax = max(ratio_ymax, 1.0)

#     # Ratio cannot be negative
#     ratio_ymin = max(0.0, ratio_ymin)

#     # Draw ratio
#     first_ratio = True

#     for ratio in ratio_hists:

#         if first_ratio:

#             ratio.SetMinimum(ratio_ymin)
#             ratio.SetMaximum(ratio_ymax)

#             ratio.SetTitle(
#                 f';'
#                 f'{XAXISLABELS[CLASSES.index(true)]} for {score};'
#                 'Ratio to first model'
#             )

#             ratio.GetXaxis().SetTitleSize(0.12)
#             ratio.GetXaxis().SetLabelSize(0.10)

#             ratio.GetYaxis().SetTitleSize(0.10)
#             ratio.GetYaxis().SetLabelSize(0.085)
#             ratio.GetYaxis().SetTitleOffset(0.55)
#             ratio.GetYaxis().SetNdivisions(505)

#             ratio.Draw('HIST')
#             first_ratio = False

#         else:
#             ratio.Draw('HIST SAME')

#     # Ratio = 1 reference line
#     ratio_line = TLine(
#         first_hist.GetXaxis().GetXmin(),
#         1.0,
#         first_hist.GetXaxis().GetXmax(),
#         1.0
#     )

#     SetOwnership(ratio_line, False)

#     ratio_line.SetLineStyle(2)
#     ratio_line.SetLineWidth(1)
#     ratio_line.Draw()

#     # Keep objects attached to canvas
#     canvas._pad_main = pad_main
#     canvas._pad_ratio = pad_ratio
#     canvas._legend = leg
#     canvas._ratio_hists = ratio_hists
#     canvas._ratio_line = ratio_line

#     # Force ROOT to render both pads
#     pad_main.Modified()
#     pad_ratio.Modified()

#     canvas.Modified()
#     canvas.Update()

#     return canvas


# def draw_page(hists, pt_min, pt_max, score, true, cfg):
#     """Draw one score/true-class comparison page."""
#     canvas = TCanvas(
#         f'c_{score}_{true}_{pt_min}_{pt_max}', '',
#         cfg.get('width', 850), cfg.get('height', 650)
#     )

#     canvas.SetLeftMargin(0.13)
#     canvas.SetRightMargin(0.04)
#     canvas.SetBottomMargin(0.14)
#     canvas.SetTopMargin(0.07)

#     if cfg.get('log_y', True):
#         canvas.SetLogy()

#     ymax = max(
#         hists[model][score][true].GetMaximum()
#         for model in hists
#         if score in hists[model] and true in hists[model][score]
#     )

#     first = True

#     for model, model_cfg in hists.items():

#         if score not in model_cfg or true not in model_cfg[score]:
#             continue

#         hist = model_cfg[score][true]

#         for model_cfg in cfg['models']:
#             if model_cfg['name'] == model:
#                 model_color = model_cfg['color']
#                 break

#         hist.SetLineColor(GetROOTColor(model_color))
#         hist.SetLineWidth(cfg.get('line_width', 2))
#         hist.SetFillStyle(0)

#         if first:
#             hist.SetMinimum(cfg.get('ymin', 1e-5))
#             hist.SetMaximum(
#                 ymax * cfg.get('ymax_factor', 1.3)
#             )

#             hist.SetTitle(
#                 f';{XAXISLABELS[CLASSES.index(true)]} for {score};'
#                 f'Surviving Fraction'
#             )

#             hist.GetXaxis().SetTitleSize(0.052)
#             hist.GetYaxis().SetTitleSize(0.052)
#             hist.GetXaxis().SetLabelSize(0.045)
#             hist.GetYaxis().SetLabelSize(0.045)
#             hist.GetYaxis().SetTitleOffset(1.15)

#             hist.Draw('HIST')
#             first = False

#         else:
#             hist.Draw('HIST SAME')

#     # Legend
#     print(f'  Drawing legend for {score} {true}')

#     leg = TLegend(
#         cfg['legend_x1'], cfg['legend_y1'],
#         cfg['legend_x2'], cfg['legend_y2']
#     )

#     leg.SetHeader(
#         f'{pt_min} < #it{{p}}_{{T}} < {pt_max} GeV/#it{{c}}'
#     )
#     leg.SetBorderSize(0)
#     leg.SetFillStyle(0)
#     leg.SetTextFont(42)
#     leg.SetTextSize(0.038)

#     for model, model_cfg in hists.items():
#         print(f'  Adding legend entry for {model} {score} {true}')

#         if score in model_cfg and true in model_cfg[score]:
#             leg.AddEntry(
#                 model_cfg[score][true],
#                 model,
#                 'l'
#             )

#     # Draw legend after all histograms
#     leg.Draw()

#     # Keep legend alive and force rendering before returning
#     canvas.Modified()
#     canvas.Update()

#     return canvas


def main():
    parser = argparse.ArgumentParser(description='Compare ML score distributions between models.')
    parser.add_argument('config', help='YAML configuration file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models = {
        model['name']: model
        for model in cfg['models']
    }

    pt_bins = cfg['pt_bins']
    out_dir = cfg['out_file']

    os.makedirs(out_dir, exist_ok=True)

    for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):

        print(f'\nProcessing pT bin: {pt_min} - {pt_max} GeV/c')
        hists = load_hists(models, pt_min, pt_max, cfg)
        if not hists:
            print('  No input files found, skipping.')
            continue

        pdf = os.path.join(out_dir, f'MLScoreComparison_pT_{pt_min}_{pt_max}.pdf')
        first_page = True
        i_page = 0

        # 9 pages:
        # each page compares the same (score, true class)
        # between all models
        for score in CLASSES:
            for true in CLASSES:
                i_page += 1
                # Check that at least one model has this histogram
                available = any(
                    score in hists[model]
                    and true in hists[model][score]
                    for model in hists
                )

                if not available:
                    continue

                canvas = draw_page(hists, pt_min, pt_max, score, true, cfg)

                if first_page:
                    canvas.Print(pdf + '(')
                    first_page = False
                elif i_page == 9:
                    canvas.Print(pdf + ')')
                else:
                    canvas.Print(pdf)

                canvas.Close()

        print(f'  Written: {pdf}')


if __name__ == '__main__':
    main()
