import sys
import os
import numpy as np
import argparse
import yaml
import array
import ROOT
from ROOT import TFile, TCanvas, kFullSquare, kBlack, kOrange, kAzure, kGray, kRed, TLegend, kCyan, kSpring, kGreen, kBlue, kMagenta, kFullCircle, TGraphErrors, TH1F, TMultiGraph
script_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.join(script_dir, '../..', 'utils'))
from utils import logger
from StyleFormatter import SetGlobalStyle, SetObjectStyle

SetGlobalStyle(titleoffsety=1.4, maxdigits=3, topmargin=0.1, bottommargin=0.4, leftmargin=1, rightmargin=0.15,
               labelsizey=0.04, setoptstat=0, setopttitle=0, setdecimals=True,titleoffsetx=0.94)
cols = [ROOT.TColor.GetColorTransparent(c, 0.8) for c in  [kBlack, kRed+1, kOrange+7, kOrange+1, kSpring+2, kGreen+2, kCyan+2, kAzure+4, kBlue-4, kMagenta+2,]]
labels = ['default', 'random', 'even', 'odd', 'step1', 'step2', 'minus3low', 'minus3high', 'minus3']

def get_hdeltav(href, hsyst):
    hdelta = hsyst.Clone('')
    hdelta.Reset()
    hdelta.SetDirectory(0)
    for ibin in range(1, href.GetNbinsX()+1):
        hdelta.SetBinContent(ibin, hsyst.GetBinContent(ibin)-href.GetBinContent(ibin))
        hdelta.SetBinError(ibin, 1.e-9)

    return hdelta

def get_rms_shift(histsdeltav2):
    gsyst = TGraphErrors(-1)
    SetObjectStyle(gsyst, color=cols[1], fillcolor=cols[1], linewidth=0, linestyle=8, fillalpha=0.1)
    for ibin in range(1, histsdeltav2[0].GetNbinsX()+1):
        hsyst = TH1F('', '', 20000, -0.1, 0.1)
        for hist in histsdeltav2[1:]: # skipping default
            hsyst.Fill(hist.GetBinContent(ibin))
        gsyst.SetPoint(ibin, histsdeltav2[0].GetBinCenter(ibin), 0)
        gsyst.SetPointError(ibin, histsdeltav2[0].GetBinWidth(ibin)/2, np.sqrt(hsyst.GetRMS()**2 + hsyst.GetMean()**2))

    return gsyst

def get_division_panels(nplots):
    if nplots <= 6:
        return 3, 2
    if nplots <= 8:
        return 4, 2
    if nplots == 9:
        return 3, 3
    if nplots == 10:
        return 5, 2
    if nplots == 14:
        return 4, 4
    else:
        logger(f'Number of plots {nplots} too high, defaulting to 4x4 grid', level='FATAL')


def compute_syst(infiles, outputdir, suffix):
    #______________________________________________________________________________________
    # Collect all .root files
    hv2_prompt = []
    hv2_fd = []
    hdeltav2_prompt = []
    hdeltav2_fd = []
    gv2vsfrac = {}
    tfiles = []
    for ifile, file in enumerate(infiles):
        logger(f'Processing file {file}')
        tfile = TFile().Open(file)
        # check if the file has the right keys
        if not tfile.GetListOfKeys().Contains('hV2VsPtFD') or not tfile.GetListOfKeys().Contains('hV2VsPtPrompt'):
            logger(f'File {file} does not contain the required histograms, skipping it', level='WARNING')
            continue
        tfiles.append(tfile)
        hv2prompt = tfiles[-1].Get('hV2VsPtPrompt')
        hv2fd = tfiles[-1].Get('hV2VsPtFD')
        if hv2prompt is None or hv2fd is None:
            hv2prompt.SetDirectory(0)
            hv2fd.SetDirectory(0)
        hv2_prompt.append(hv2prompt)
        hv2_fd.append(hv2fd)

        gv2vsfrac[labels[ifile]] = []
        for ibin in range(1, hv2_prompt[0].GetNbinsX()+1):
            ptmin = hv2_prompt[0].GetXaxis().GetBinLowEdge(ibin)
            ptmax = hv2_prompt[0].GetXaxis().GetBinLowEdge(ibin) + hv2_prompt[0].GetXaxis().GetBinWidth(ibin)
            gv2_vs_frc = tfiles[-1].Get(f'pt_{ptmin*10:.0f}_{ptmax*10:.0f}/gV2VsFrac').Clone(f'gv2vsfrac_pt{ptmin:.1f}_{ptmax:.1f}_{labels[ifile]}')
            gv2_vs_frc.SetTitle(f'{(ptmin*10):.0f} < #it{{p}}_{{T}} < {(ptmax*10):.0f} GeV/#it{{c}}')
            gv2_vs_frc.GetXaxis().SetTitle('#it{f}_{FD}')
            gv2_vs_frc.GetYaxis().SetTitle(f'#it{{v}}_{{2}} ({ptmin:.1f} < #it{{p}}_{{T}} < {ptmax:.1f} GeV/#it{{c}})')
            gv2_vs_frc.GetYaxis().SetRangeUser(-0.3, 0.3)
            gv2_vs_frc.GetYaxis().SetDecimals()
            gv2_vs_frc.GetXaxis().SetRangeUser(0, 1)
            gv2vsfrac[labels[ifile]].append(gv2_vs_frc)
            SetObjectStyle(gv2vsfrac[labels[ifile]][-1], color=cols[ifile], fillcolor=cols[ifile], markerstyle=kFullCircle, markersize=1.4, linewidth=3, fillalpha=0.2)
        SetObjectStyle(hv2_prompt[-1], color=cols[ifile], fillcolor=cols[ifile], markerstyle=kFullCircle, markersize=1.4, linewidth=3)
        SetObjectStyle(hv2_fd[-1], color=cols[ifile], fillcolor=cols[ifile], markerstyle=kFullSquare, markersize=1.4, linewidth=3)

        if ifile == 0:
            continue  # skip the default for the delta v2
        hdeltav2_prompt.append(get_hdeltav(hv2_prompt[0], hv2_prompt[ifile]))
        hdeltav2_fd.append(get_hdeltav(hv2_fd[0], hv2_fd[ifile]))
    gsyst_prompt = get_rms_shift(hdeltav2_prompt)
    gsyst_fd = get_rms_shift(hdeltav2_fd)

    canv_fFD = TCanvas('canv', 'canv', 1400, 800)
    n_panels_x, n_panels_y = get_division_panels(len(gv2vsfrac[labels[0]]))
    canv_fFD.Divide(n_panels_x, n_panels_y)

    legv2 = TLegend(0.6, 0.6, 0.9, 0.9)
    legv2.SetBorderSize(0)
    legv2.SetFillStyle(0)
    legv2.SetTextSize(0.03)
    legv2.SetTextFont(42)
    for igs, gfracs in enumerate(gv2vsfrac):
        for ig, gfrac in enumerate(gv2vsfrac[gfracs]):
            canv_fFD.cd(ig+1)
            canv_fFD.cd(ig+1).SetBottomMargin(0.16)
            canv_fFD.cd(ig+1).SetLeftMargin(0.16)
            if igs == 0:
                # Set log scale on x axis
                gfrac.GetXaxis().SetMoreLogLabels()
                gfrac.GetXaxis().SetNoExponent()
                gfrac.Draw('peaz')
                if ig == len(gv2vsfrac[gfracs]) - 1:
                    legv2.Draw()
            else:
                # Set log scale on x axis
                gfrac.GetXaxis().SetMoreLogLabels()
                gfrac.GetXaxis().SetNoExponent()
                gfrac.Draw('same pez')
            if ig == 0:
                legv2.AddEntry(hv2_prompt[igs], labels[igs], 'ep')
    canv_fFD.Update()
    canv_fFD.SaveAs(f'{outputdir}/gV2vsFrac_{suffix}.pdf')

    #______________________________________________________________________________________
    # Plot the systematics
    canv = TCanvas('canv', 'canv', 1600, 1600)
    canv.Divide(2, 2)
    
    legsyst = TLegend(0.6, 0.79, 0.9, 0.89)
    legsyst.SetBorderSize(0)
    legsyst.SetFillStyle(0)
    legsyst.SetTextSize(0.03)
    legsyst.SetTextFont(42)
    
    canv.cd(1).SetLeftMargin(0.16)
    canv.cd(1)
    for ihist, hist in enumerate(hv2_prompt):
        hist.SetStats(0)
        hist.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
        hist.GetYaxis().SetTitle('prompt #it{v}_{2}')
        hist.GetXaxis().SetRangeUser(0, 25)
        hist.GetYaxis().SetRangeUser(-0.1, 0.35)
        hist.Draw('same')
    legv2.Draw()
    canv.cd(2).SetLeftMargin(0.16)
    for _, hist in enumerate(hv2_fd):
        hist.SetStats(0)
        hist.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
        hist.GetYaxis().SetTitle('non-prompt #it{v}_{2}')
        hist.GetXaxis().SetRangeUser(0, 25)
        hist.GetYaxis().SetRangeUser(-0.2, 0.35)
        hist.Draw('same')
    canv.cd(3).SetGridy()
    canv.cd(3).SetLeftMargin(0.16)
    for _, hist in enumerate(hdeltav2_prompt):
        hist.SetStats(0)
        hist.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
        hist.GetYaxis().SetTitle('prompt #it{v}_{2}^{syst.} #minus #it{v}_{2}^{ref.}')
        hist.GetXaxis().SetRangeUser(0, 25)
        hist.GetYaxis().SetRangeUser(-0.003, 0.003)
        hist.Draw('samepe')
    gsyst_prompt.Draw('same5')
    legsyst.AddEntry(gsyst_prompt, f'#sqrt{{shift^{{2}} + rms^{{2}}}}', 'f')
    legsyst.Draw()
    canv.cd(4).SetGridy()
    canv.cd(4).SetLeftMargin(0.16)
    for _, hist in enumerate(hdeltav2_fd):
        hist.SetStats(0)
        hist.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
        hist.GetYaxis().SetTitle('non-prompt #it{v}_{2}^{syst.} #minus #it{v}_{2}^{ref.}')
        hist.GetXaxis().SetRangeUser(0, 25)
        hist.GetYaxis().SetRangeUser(-0.006, 0.004)
        hist.Draw('samepe')
    gsyst_fd.Draw('same5')

    canv.Update()

    canvSystPrompt = TCanvas('canvSystPrompt', 'canvSystPrompt', 800, 800)
    canvSystPrompt.SetLeftMargin(0.16)
    canvSystPrompt.SetLogy()
    canvSystPrompt.SetGridy()
    gsyst_prompt.SetStats(0)
    gsyst_prompt.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
    gsyst_prompt.GetYaxis().SetTitle('Systematic uncertainty on prompt #it{v}_{2}')
    gsyst_prompt.GetXaxis().SetRangeUser(0, 25)
    gsyst_prompt.GetYaxis().SetRangeUser(1.e-5, 0.05)
    gsyst_prompt.Draw('a5')
    canvSystPrompt.Update()

    canvSystFD = TCanvas('canvSystFD', 'canvSystFD', 800, 800)
    canvSystFD.SetLeftMargin(0.16)
    canvSystFD.SetLogy()
    canvSystFD.SetGridy()
    gsyst_fd.SetStats(0)
    gsyst_fd.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
    gsyst_fd.GetYaxis().SetTitle('Systematic uncertainty on non-prompt #it{v}_{2}')
    gsyst_fd.GetXaxis().SetRangeUser(0, 25)
    gsyst_fd.GetYaxis().SetRangeUser(1.e-5, 0.05)
    gsyst_fd.Draw('a5')
    canvSystFD.Update()

    #______________________________________________________________________________________
    # Save output
    outputfile = os.path.join(outputdir, f'syst_fFD_{suffix}.root')
    logger(f'Saving output to {outputfile}')
    for outformat in ['root', 'pdf', 'png']:
        canv.SaveAs(f'{outputdir}/SystfFD_{suffix}.{outformat}')
        canvSystPrompt.SaveAs(f'{outputdir}/SystOnlyfPrompt_{suffix}.{outformat}')
        canvSystFD.SaveAs(f'{outputdir}/SystOnlyfFD_{suffix}.{outformat}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('infiles', metavar='text', nargs='*', default='path/to/syst/files')
    parser.add_argument("--suffix", "-s", metavar="text",
                        default="", help="suffix for output files")
    parser.add_argument("--outputdir", "-o", metavar="text",
                        default=".", help="output directory")
    args = parser.parse_args()

    compute_syst(args.infiles,
                 args.outputdir,
                 args.suffix)
