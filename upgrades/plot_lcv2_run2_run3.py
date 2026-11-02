import ROOT
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import sys
from functools import wraps
import numpy as np
from ROOT import TFile, TLegend, TCanvas, TH1F, TF1,  gROOT, TGaxis, gStyle
import sys
sys.path.append('./')

colors = [ROOT.TColor.GetColorTransparent(c, 0.8) for c in [ROOT.kRed+1, ROOT.kAzure+4, ROOT.kCyan+2, ROOT.kOrange-3, ROOT.kGray+1
                                                            , ROOT.kOrange+7, ROOT.kBlue+2, ROOT.kBlack, ROOT.kGreen+2]]

latex = ROOT.TLatex()
latex.SetTextFont(42)
latex.SetTextSize(0.05)
latexmedium = ROOT.TLatex()
latexmedium.SetTextFont(42)
latexmedium.SetTextSize(0.058)
latexdetail =ROOT.TLatex()
latexdetail.SetTextFont(42)
latexdetail.SetTextSize(0.04)
latexsmall =ROOT.TLatex()
latexsmall.SetTextFont(42)
latexsmall.SetTextSize(0.03)
latexdetail2 =ROOT.TLatex()
latexdetail2.SetTextFont(42)
latexdetail2.SetTextSize(0.045)
latexlarge = ROOT.TLatex()
latexlarge.SetTextFont(42)
latexlarge.SetTextSize(0.07)

def DrawLineAtX(x, ymin, ymax, color=ROOT.kGray+2, style=2, width=1):
    line = ROOT.TLine(x, ymin, x, ymax)
    line.SetLineColor(color)
    line.SetLineStyle(style)
    line.SetLineWidth(width)
    line.Draw("same")
    return line

def compute_central_graph(graph_upper, graph_lower):
    """
    Given two TGraphAsymmErrors (upper and lower limits), computes the central values.
    Ensures all unique x-values from both graphs are included.

    Parameters:
        graph_upper (ROOT.TGraphAsymmErrors): Graph representing the upper limit.
        graph_lower (ROOT.TGraphAsymmErrors): Graph representing the lower limit.

    Returns:
        ROOT.TGraphAsymmErrors: Graph with central values and computed uncertainties.
    """
    # Extract x and y values from both graphs
    x_upper = np.array([graph_upper.GetPointX(i) for i in range(graph_upper.GetN())])
    print(x_upper)
    y_upper = np.array([graph_upper.GetPointY(i) for i in range(graph_upper.GetN())])

    x_lower = np.array([graph_lower.GetPointX(i) for i in range(graph_lower.GetN())])
    print(x_lower)
    y_lower = np.array([graph_lower.GetPointY(i) for i in range(graph_lower.GetN())])

    # Get all unique x-values from both graphs
    x_common = np.sort(np.unique(np.concatenate((x_upper, x_lower))))

    # Interpolate/extrapolate the y-values for both graphs at these x-values
    y_upper_interp = np.interp(x_common, x_upper, y_upper, left=y_upper[0], right=y_upper[-1])
    y_lower_interp = np.interp(x_common, x_lower, y_lower, left=y_lower[0], right=y_lower[-1])

    # Compute central values
    y_central = 0.5 * (y_upper_interp + y_lower_interp)

    # Compute uncertainties as differences from central values
    y_err_low = np.abs(y_central - y_lower_interp)
    y_err_high = np.abs(y_upper_interp - y_central)
    print(x_common)

    # Create a new TGraphAsymmErrors
    graph_central = ROOT.TGraphAsymmErrors(len(x_common))
    for i, x in enumerate(x_common):
        graph_central.SetPoint(i, x, y_central[i])
        graph_central.SetPointError(i, 
                                    0, 0,  # No x-errors assumed
                                    y_err_low[i], y_err_high[i])

    return graph_central

def SystX(graph, syst, percentage=0.2):
    for i in range(graph.GetN()):
        stat = 2*graph.GetErrorXlow(i)
        syst.SetPointEXlow(i, stat*percentage)
        syst.SetPointEXhigh(i, stat*percentage)
    return syst

def SetObjectStyle(obj, **kwargs):
    '''
    Method to set root object style.

    Parameters
    ----------

    - obj: object to set style

    - linecolor (int) default 1 (black)
    - linealpha (float) default 1
    - linewidth (int) default 2
    - linestyle (int) default 1

    - markercolor (int) default 1 (black)
    - markeralpha (float) default 1
    - markerstyle (int) default 20 (full circle)
    - markersize (int) default 20 (full circle)

    - fillcolor (int) default no filling
    - fillalpha (float) default 1
    - fillstyle (int) default 0 (no style)

    - color (int) sets same color for line, marker and fill
    - alpha (float) sets same alpha for line, marker and fill
    '''

    # alpha parameters
    lalpha = kwargs.get('linealpha', 1)
    malpha = kwargs.get('markeralpha', 1)
    falpha = kwargs.get('fillalpha', 1)
    if 'alpha' in kwargs:
        lalpha = kwargs['alpha']
        malpha = kwargs['alpha']
        falpha = kwargs['alpha']
    if 'linealpha' in kwargs:
        lalpha = kwargs['linealpha']
    if 'markeralpha' in kwargs:
        malpha = kwargs['markeralpha']
    if 'fillalpha' in kwargs:
        falpha = kwargs['fillalpha']

    # line styles
    if 'linecolor' in kwargs:
        if lalpha < 1:
            obj.SetLineColorAlpha(kwargs['linecolor'], lalpha)
        else:
            obj.SetLineColor(kwargs['linecolor'])
    else:
        if lalpha < 1:
            obj.SetLineColorAlpha(1, lalpha)
        else:
            obj.SetLineColor(1)

    if 'linewidth' in kwargs:
        obj.SetLineWidth(kwargs['linewidth'])
    else:
        obj.SetLineWidth(2)

    if 'linestyle' in kwargs:
        obj.SetLineStyle(kwargs['linestyle'])
    else:
        obj.SetLineStyle(1)

    # marker styles
    if 'markercolor' in kwargs:
        if malpha < 1:
            obj.SetMarkerColorAlpha(kwargs['markercolor'], malpha)
        else:
            obj.SetMarkerColor(kwargs['markercolor'])
    else:
        if malpha < 1:
            obj.SetMarkerColorAlpha(1, malpha)
        else:
            obj.SetMarkerColor(1)

    if 'markersize' in kwargs:
        obj.SetMarkerSize(kwargs['markersize'])
    else:
        obj.SetMarkerSize(1)

    if 'markerstyle' in kwargs:
        obj.SetMarkerStyle(kwargs['markerstyle'])
    else:
        obj.SetMarkerStyle(20)

    # fill styles
    if 'fillcolor' in kwargs:
        if falpha < 1:
            obj.SetFillColorAlpha(kwargs['fillcolor'], falpha)
        else:
            obj.SetFillColor(kwargs['fillcolor'])

    if 'fillstyle' in kwargs:
        obj.SetFillStyle(kwargs['fillstyle'])

    #global color
    if 'color' in kwargs:
        if lalpha < 1:
            obj.SetLineColorAlpha(kwargs['color'], lalpha)
        else:
            obj.SetLineColor(kwargs['color'])
        if malpha < 1:
            obj.SetMarkerColorAlpha(kwargs['color'], malpha)
        else:
            obj.SetMarkerColor(kwargs['color'])
        if falpha < 1:
            obj.SetFillColorAlpha(kwargs['color'], falpha)
        else:
            obj.SetFillColor(kwargs['color'])

def SetGlobalStyle(**kwargs):
    '''
    Method to set global style.

    Parameters
    ----------

    - padrightmargin (float), default = 0.035
    - padleftmargin (float), default = 0.12
    - padtopmargin (float), default = 0.035
    - padbottommargin (float), default = 0.1

    - titlesize (float), default = 0.050
    - titlesizex (float), default = 0.050
    - titlesizey (float), default = 0.050
    - titlesizez (float), default = 0.050

    - labelsize (float), default = 0.045
    - labelsizex (float), default = 0.045
    - labelsizey (float), default = 0.045
    - labelsizez (float), default = 0.045

    - titleoffset (float), default = 1.2
    - titleoffsetx (float), default = 1.2
    - titleoffsey (float), default = 1.2
    - titleoffsetz (float), default = 1.2

    - opttitle (int), default = 0
    - optstat (int), default = 0

    - padtickx (int), default = 1
    - padticky (int), default = 1

    - maxdigits (int), default no max value

    - palette (int), default kBird
    '''

    # pad margins
    if 'padrightmargin' in kwargs:
        gStyle.SetPadRightMargin(kwargs['padrightmargin'])
    else:
        gStyle.SetPadRightMargin(0.035)

    if 'padleftmargin' in kwargs:
        gStyle.SetPadLeftMargin(kwargs['padleftmargin'])
    else:
        gStyle.SetPadLeftMargin(0.12)

    if 'padtopmargin' in kwargs:
        gStyle.SetPadTopMargin(kwargs['padtopmargin'])
    else:
        gStyle.SetPadTopMargin(0.035)

    if 'padbottommargin' in kwargs:
        gStyle.SetPadBottomMargin(kwargs['padbottommargin'])
    else:
        gStyle.SetPadBottomMargin(0.1)

    # title sizes
    if 'titlesize' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesize'], 'xyz')
    else:
        gStyle.SetTitleSize(0.050, 'xyz')

    if 'titlesizex' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesizex'], 'x')
    if 'titlesizey' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesizex'], 'y')
    if 'titlesizez' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesizex'], 'z')

    # label sizes
    if 'labelsize' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsize'], 'xyz')
    else:
        gStyle.SetLabelSize(0.045, 'xyz')

    if 'labelsizex' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsizex'], 'x')
    if 'labelsizey' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsizey'], 'y')
    if 'labelsizez' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsizez'], 'z')

    # title offsets
    if 'titleoffset' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffset'], 'xyz')
    else:
        gStyle.SetTitleOffset(1.2, 'xyz')

    if 'titleoffsetx' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffsetx'], 'x')
    if 'titleoffsety' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffsety'], 'y')
    if 'titleoffsetz' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffsetz'], 'z')

    # other options
    if 'opttitle' in kwargs:
        gStyle.SetOptTitle(kwargs['opttitle'])
    else:
        gStyle.SetOptTitle(0)

    if 'optstat' in kwargs:
        gStyle.SetOptStat(kwargs['optstat'])
    else:
        gStyle.SetOptStat(0)

    if 'padtickx' in kwargs:
        gStyle.SetPadTickX(kwargs['padtickx'])
    else:
        gStyle.SetPadTickX(1)

    if 'padticky' in kwargs:
        gStyle.SetPadTickY(kwargs['padticky'])
    else:
        gStyle.SetPadTickY(1)

    gStyle.SetLegendBorderSize(0)

    if 'maxdigits' in kwargs:
        TGaxis.SetMaxDigits(kwargs['maxdigits'])

    if 'palette' in kwargs:
        gStyle.SetPalette(kwargs['palette'])

    gROOT.ForceStyle()

def GetPrediction(file, xmin, xmax, scale, isUnc, model='tamu'):
    if model == 'tamu':
        splinev2, _, _, _ = ReadTAMUv2(file)
        gpred = ROOT.TGraphAsymmErrors(1)
        for ipt in range(xmin, xmax):
            index = ipt
            ipt *= scale

            gpred.AddPoint(ipt, splinev2['yCent'](ipt))
            if isUnc:
                gpred.SetPointError(index, 0., 0.,
                                    float(splinev2['yCent'](ipt) - splinev2['yMin'](ipt)),
                                    float(splinev2['yMax'](ipt) - splinev2['yCent'](ipt)))
    elif model == 'lido':
        splinev2, _, _, _ = ReadLIDOV2(file)
        gpred = ROOT.TGraphAsymmErrors(1)
        for ipt in range(xmin, xmax):
            ipt *= scale

            gpred.AddPoint(ipt, splinev2['yCent'](ipt))
            if isUnc:
                gpred.SetPointError(ipt, 0., 0.,
                                    float(splinev2['yCent'](ipt) - splinev2['yMin'](ipt)),
                                    float(splinev2['yMax'](ipt) - splinev2['yCent'](ipt)))
    elif model == 'lgr':
        splinev2, _, _, _ = ReadLGRV2(file)
        gpred = ROOT.TGraphAsymmErrors(0)
        index=0
        for ipt in range(xmin, xmax):
            ipt *= scale

            gpred.AddPoint(ipt, splinev2['yCent'](ipt))
            if isUnc:
                index+=1
                errlow=splinev2['yMax'](ipt) - splinev2['yCent'](ipt)
                print(errlow)
                gpred.SetPointError(index, 0., 0.,
                                    float(splinev2['yCent'](ipt) - splinev2['yMin'](ipt)),
                                    float(splinev2['yMax'](ipt) - splinev2['yCent'](ipt)))
    elif model == 'phsd':
        splinePHSD3040, splinePHSD4050, splinePHSD6080, _, _, _ = ReadPHSDV2(file)
        gpred3040 = ROOT.TGraphAsymmErrors(0)
        index=0
        for ipt in range(xmin, xmax):
            ipt *= scale

            gpred3040.AddPoint(ipt, splinePHSD3040['yCent'](ipt))
            if isUnc:
                index+=1
                errlow=splinePHSD3040['yMax'](ipt) - splinePHSD3040['yCent'](ipt)
                print(errlow)
                gpred3040.SetPointError(index, 0., 0.,
                                    float(splinePHSD3040['yCent'](ipt) - splinePHSD3040['yMin'](ipt)),
                                    float(splinePHSD3040['yMax'](ipt) - splinePHSD3040['yCent'](ipt)))
        gpred3040.RemovePoint(0)
        gpred3040.RemovePoint(0)

        gpred4050 = ROOT.TGraphAsymmErrors(0)
        index=0
        for ipt in range(xmin, xmax):
            ipt *= scale

            gpred4050.AddPoint(ipt, splinePHSD4050['yCent'](ipt))
            if isUnc:
                index+=1
                errlow=splinePHSD4050['yMax'](ipt) - splinePHSD4050['yCent'](ipt)
                print(errlow)
                gpred4050.SetPointError(index, 0., 0.,
                                    float(splinePHSD4050['yCent'](ipt) - splinePHSD4050['yMin'](ipt)),
                                    float(splinePHSD4050['yMax'](ipt) - splinePHSD4050['yCent'](ipt)))
        gpred4050.RemovePoint(0)
        gpred4050.RemovePoint(0)

        gpred6080 = ROOT.TGraphAsymmErrors(0)
        index=0
        for ipt in range(xmin, xmax):
            ipt *= scale

            gpred6080.AddPoint(ipt, splinePHSD6080['yCent'](ipt))
            if isUnc:
                index+=1
                errlow=splinePHSD6080['yMax'](ipt) - splinePHSD6080['yCent'](ipt)
                print(errlow)
                gpred6080.SetPointError(index, 0., 0.,
                                    float(splinePHSD6080['yCent'](ipt) - splinePHSD6080['yMin'](ipt)),
                                    float(splinePHSD6080['yMax'](ipt) - splinePHSD6080['yCent'](ipt)))
        gpred6080.RemovePoint(0)
        gpred6080.RemovePoint(0)
        
        return gpred3040, gpred4050, gpred6080
    return gpred

def ReadLIDOV2(fileName):
    '''
    Method to read LIDO Raa files

    Inputs
    ----------
    - fileName: file name

    Returns
    ----------
    splineLIDO: dictionary of splines with LIDO predictions {yCent, yMin, yMax}
    dfLIDO: pandas dataframe with original values
    ptMin: minimum pt for which the model is valid
    ptMax: maximum pt for which the model is valid
    '''

    dfLIDO = pd.read_csv(fileName, sep=' ', comment='#')
    dfLIDO['v2_min'] = dfLIDO['v2'] - dfLIDO['v2-error']
    dfLIDO['v2_max'] = dfLIDO['v2'] + dfLIDO['v2-error']

    splineLIDO, ptMin, ptMax = InterpolateModel(dfLIDO['pT'], dfLIDO['v2'], dfLIDO['v2_max'], dfLIDO['v2_min'])

    return splineLIDO, dfLIDO, ptMin, ptMax

def ReadLGRV2(fileName):
    '''
    Method to read LGR Raa files

    Inputs
    ----------
    - fileName: file name

    Returns
    ----------
    splineLGR: dictionary of splines with LGR predictions {yCent, yMin, yMax}
    dfLGR: pandas dataframe with original values
    ptMin: minimum pt for which the model is valid
    ptMax: maximum pt for which the model is valid
    '''

    dfLGR = pd.read_csv(fileName, sep=' ', comment='#')
    splineLGR, ptMin, ptMax = InterpolateModel(dfLGR['pT'], dfLGR['v2'], dfLGR['v2_min'], dfLGR['v2_max'])

    return splineLGR, dfLGR, ptMin, ptMax

def InterpolateModel(ptCent, yCent, yMin=None, yMax=None):
    '''
    Helper function to interpolate model predictions.
    The returned splines will raise an error if applied out of the data boundary.

    Parameters
    -----------
    ptCent: list of pT centres to interpolate
    yCent: list of central values to interpolate
    yMin: list of min values to interpolate
    yMax: list of max values to interpolate

    Returns:
    -----------
    splinesAll: dictionary with splines {yCent, yMin, yMax}
    ptMin: minimum pt for which the interpolation is valid
    ptMax: maximum pt for which the interpolation is valid
    '''

    splinesAll = {}
    splinesAll['yCent'] = InterpolatedUnivariateSpline(ptCent, yCent, ext='raise', check_finite=True)

    if yMin is not None and yMin.any():
        splinesAll['yMin'] = InterpolatedUnivariateSpline(ptCent, yMin, ext='raise', check_finite=True)
    if yMax is not None and yMax.any():
        splinesAll['yMax'] = InterpolatedUnivariateSpline(ptCent, yMax, ext='raise', check_finite=True)

    return splinesAll, min(ptCent), max(ptCent)

def ReadPHSDV2(fileName):
    '''
    Method to read PHSD v2 files

    Inputs
    ----------
    - fileName: file name

    Returns
    ----------
    splinePHSD: dictionary of splines with PHSD predictions {yCent, yMin, yMax}
    dfPHSD: pandas dataframe with original values
    ptMin: minimum pt for which the model is valid
    ptMax: maximum pt for which the model is valid
    '''

    dfPHSD = pd.read_csv(fileName, sep=' ', comment='#')
    dfPHSD['3040_min'] = dfPHSD['3040'] - dfPHSD['3040_unc']
    dfPHSD['3040_max'] = dfPHSD['3040'] + dfPHSD['3040_unc']
    dfPHSD['4050_min'] = dfPHSD['4050'] - dfPHSD['4050_unc']
    dfPHSD['4050_max'] = dfPHSD['4050'] + dfPHSD['4050_unc']
    dfPHSD['6080_min'] = dfPHSD['6080'] - dfPHSD['6080_unc']
    dfPHSD['6080_max'] = dfPHSD['6080'] + dfPHSD['6080_unc']

    splinePHSD3040, ptMin, ptMax = InterpolateModel(dfPHSD['pT'], dfPHSD['3040'], dfPHSD['3040_max'], dfPHSD['3040_min'])
    splinePHSD4050, _, _ = InterpolateModel(dfPHSD['pT'], dfPHSD['4050'], dfPHSD['4050_max'], dfPHSD['4050_min'])
    splinePHSD6080, _, _ = InterpolateModel(dfPHSD['pT'], dfPHSD['6080'], dfPHSD['6080_max'], dfPHSD['6080_min'])

    return splinePHSD3040, splinePHSD4050, splinePHSD6080, dfPHSD, ptMin, ptMax

def ReadTAMUv2(fileNameTAMUv2):
    '''
    Helper function to read TAMU v2 txt files

    Parameters
    -----------
    fileNameTAMU: TAMU file name

    Returns:
    -----------
    splineTAMU: dictionary with splines {yCent, yMin, yMax}
    dfTAMU: pandas dataframe with original values
    ptMin: minimum pt for which the model is valid
    ptMax: maximum pt for which the model is valid
    '''
    dfTAMU = pd.read_csv(fileNameTAMUv2, sep=' ', comment='#')
    if 'v2min' in dfTAMU and 'v2max' in dfTAMU:
        dfTAMU['v2'] = (dfTAMU['v2min'] + dfTAMU['v2max']) / 2 #central value taken as average of min and max
        splineTAMU, ptMin, ptMax = InterpolateModel(dfTAMU['pT'], dfTAMU['v2'],
                                                    dfTAMU['v2min'], dfTAMU['v2max'])
    else:
        splineTAMU, ptMin, ptMax = InterpolateModel(dfTAMU['pT'], dfTAMU['v2'])

    return splineTAMU, dfTAMU, ptMin, ptMax 

def rebin_tgraph(graph_to_rebin, reference_graph):
    """
    Rebin a Asymmetric TGraph to match the binning of a reference Asymmetric TGraph.
    Parameters
    ----------
    graph_to_rebin : ROOT.TGraphAsymmErrors
        The graph to be rebinned.
    reference_graph : ROOT.TGraphAsymmErrors
        The reference graph with the desired binning.
    Returns
    -------
    ROOT.TGraphAsymmErrors
        The rebinned graph.
    """
    n_ref = reference_graph.GetN()
    rebinned_graph = ROOT.TGraphAsymmErrors(n_ref)
    rebinned_graph.SetName(graph_to_rebin.GetName() + "_rebinned")
    xbins = [2, 3, 4, 5, 6, 8, 12]

    for i, (xmin, xmax) in enumerate(zip(xbins[:-1], xbins[1:])):
        x_ref = reference_graph.GetX()[i]
        exl_ref = reference_graph.GetErrorXlow(i)
        exh_ref = reference_graph.GetErrorXhigh(i)

        # Find the corresponding point in the original graph
        nmatches = 0
        yrebin = 0
        eyl_rebin = 0
        weights = 0
        # weigthed average
        for j in range(graph_to_rebin.GetN()):
            x = graph_to_rebin.GetX()[j]
            if xmin <= x < xmax:
                y = graph_to_rebin.GetY()[j]
                eyl = graph_to_rebin.GetErrorYlow(j)
                eyh = graph_to_rebin.GetErrorYhigh(j)
                weight = 1.0 / ((eyl) / 2) ** 2 if (eyl + eyh) > 0 else 0
                yrebin += y * weight
                eyl_rebin += (eyl ** 2) * weight
                weights += weight
                nmatches += 1
        if nmatches > 0 and weights > 0:
            yrebin /= weights
            eyl_rebin = np.sqrt(eyl_rebin) / weights
        else:
            yrebin = 0
            eyl_rebin = 0
        rebinned_graph.SetPoint(i, x_ref, yrebin)
        rebinned_graph.SetPointError(i, exl_ref, exh_ref, eyl_rebin, eyl_rebin)

    return rebinned_graph

def make_tgraph_from_file2(filename, name="gData", title=";x;y"):
    """
    Read a two-column text file and create a ROOT TGraph.

    Parameters
    ----------
    filename : str
        Path to input file (two columns: x y).
    name : str
        Name of the TGraph.
    title : str
        Title of the TGraph (use ROOT format: 'title;x-axis;y-axis').

    Returns
    -------
    ROOT.TGraph
    """

    x_vals = []
    y_vals = []
    yunc_vals = []

    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            x, y, yunc = map(float, line.split())
            x_vals.append(x)
            y_vals.append(y)
            yunc_vals.append((yunc-y) * np.sqrt(2))


    graph = ROOT.TGraphErrors(len(x_vals))
    graph.SetName(name)
    graph.SetTitle(title)

    for i, (x, y, yunc) in enumerate(zip(x_vals, y_vals, yunc_vals)):
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, 0, yunc)

    return graph

def clone_and_scale_uncertainty(graph, scale_factor, new_name=None, xshift=0.0):

    # Clone correctly
    if isinstance(graph, ROOT.TGraphAsymmErrors):
        g = ROOT.TGraphAsymmErrors(graph)
    elif isinstance(graph, ROOT.TGraphErrors):
        g = ROOT.TGraphErrors(graph)
    elif isinstance(graph, ROOT.TGraph):
        g = ROOT.TGraph(graph)
    else:
        raise TypeError("Unsupported graph type")

    if new_name:
        g.SetName(new_name)

    n = g.GetN()

    for i in range(n):
        # Read values directly
        x = g.GetX()[i] + xshift
        y = g.GetY()[i]

        if isinstance(g, ROOT.TGraphAsymmErrors):
            exl = g.GetErrorXlow(i)
            exh = g.GetErrorXhigh(i)
            eyl = g.GetErrorYlow(i) / scale_factor
            eyh = g.GetErrorYhigh(i) / scale_factor

            g.SetPoint(i, x, y)
            g.SetPointError(i, exl, exh, eyl, eyh)

        elif isinstance(g, ROOT.TGraphErrors):
            ex = g.GetErrorX(i)
            ey = g.GetErrorY(i) / scale_factor

            g.SetPoint(i, x, y)
            g.SetPointError(i, ex, ey)

        else:
            g.SetPoint(i, x, y)

    return g

def clone_and_scale_uncertainty_th1(th1, scale_factor, new_name=None, xshift=0.0):

    # Convert TH1 to TGraphErrors
    n = th1.GetNbinsX()
    graph = ROOT.TGraphErrors(n)
    for i in range(1, n + 1):
        x = th1.GetBinCenter(i) + xshift
        y = th1.GetBinContent(i) * scale_factor
        ey = th1.GetBinError(i)
        binwidth = th1.GetBinWidth(i)

        graph.SetPoint(i - 1, x, y)
        graph.SetPointError(i - 1, binwidth / 2, 0)
    if new_name:
        graph.SetName(new_name)
    return graph

def graph_from_hist_with_inv_uncertainty(
    filename,
    histname,
    graph,
    xshift=0.0
):
    """
    Load a TH1 from file and convert it to a TGraphErrors
    with uncertainty = 1 / bin content.

    Parameters
    ----------
    filename : str
        ROOT file path
    histname : str
        Name of the TH1 inside the file
    graph_name : str
        Name of the output TGraphErrors
    graph_title : str
        Title of the graph

    Returns
    -------
    ROOT.TGraphErrors
    """

    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        print(f"Cannot open file: {filename}")

    h = f.Get(histname)
    if not h:
        print(f"Histogram '{histname}' not found in file")

    n = h.GetNbinsX()

    for i in range(1, n + 1):
        y = h.GetBinContent(i)

        if y > 0:
            err = 1.0 / y
        else:
            err = 0.0

        graph.SetPointError(i - 1, 0.0, err)
        graph.SetPoint(i - 1, graph.GetX()[i - 1] + xshift, graph.GetY()[i - 1])

    return graph

def make_tgraph_from_file(filename, name="gData", title=";x;y"):
    """
    Read a two-column text file and create a ROOT TGraph.

    Parameters
    ----------
    filename : str
        Path to input file (two columns: x y).
    name : str
        Name of the TGraph.
    title : str
        Title of the TGraph (use ROOT format: 'title;x-axis;y-axis').

    Returns
    -------
    ROOT.TGraph
    """

    x_vals = []
    y_vals = []

    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            x, y = map(float, line.split())
            x_vals.append(x)
            y_vals.append(y)

    graph = ROOT.TGraphErrors(len(x_vals))
    graph.SetName(name)
    graph.SetTitle(title)

    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, 0, 0)  # No errors provided

    return graph

def LoadGraphAndSyst(path_to_file, graph_name, syst_name, syst_fd, color, marker, markersize=2):
    """
    Loads graph and systematic uncertainties from a ROOT file.
    
    Args:
        path_to_file (str): Path to ROOT file.
        graph_name (str): Name of the graph.
        syst_name (str): Name of systematic uncertainty graph.
        syst_fd (str): Additional systematic uncertainty graph.
        color (int): Color of the graph.
        marker (int): Marker style.
        markersize (float): Marker size.
    
    Returns:
        tuple: (graph, syst, syst_fd) if both systematic uncertainties are provided,
               (graph, syst) if only one is provided, or (graph) otherwise.
    """
    infile = ROOT.TFile.Open(path_to_file)
    graph = infile.Get(graph_name)
    SetObjectStyle(graph, markerstyle=marker, markercolor=color, markersize=markersize, linecolor=color, linewidth=2)

    # set x unc to 0
    n = graph.GetN()
    for i in range(n):
        eyl = graph.GetErrorYlow(i)
        eyh = graph.GetErrorYhigh(i)
        graph.SetPointError(i, 0, 0, eyl, eyh)

    if syst_name:
        gsyst = infile.Get(syst_name)
        SetObjectStyle(gsyst, markerstyle=marker, markercolor=color, markersize=markersize, linecolor=color, linewidth=2, fillalpha=0, fillstyle=0)
        gsyst = SystX(graph, gsyst)

        if syst_fd:
            gsyst_fd = infile.Get(syst_fd)
            SetObjectStyle(gsyst_fd, markerstyle=marker, markercolor=color, markersize=markersize, linecolor=color, linewidth=1, fillcolor=color, fillalpha=0.4, fillstyle=1000)
            gsyst_fd = SystX(graph, gsyst_fd, 0.1)
            return graph, gsyst, gsyst_fd

        return graph, gsyst
    return graph


SetGlobalStyle(padleftmargin=0.16, padbottommargin=0.14, padtopmargin=0.08,
               opttitle=1, titleoffsety=1.6, labelsize=0.05, titlesize=0.05,
               labeloffset=0.01, titleoffset=1.2, labelfont=42, titlefont=42, palette=ROOT.kRainBow)

def main():
    # Plot settings
    plot23=False
    plot24=False
    plot25=False
    plot26=True
    plotIts2Old=False
    plotIts2=False
    plotIts3=False
    plotIts3RealLumi=True
    plotIts3Deadzone=True
    plotD0=False

    # Read histograms
    lcfile = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/input/v2_Lc_d0.root')
    hlcv2 = lcfile.Get('h1')
    hlcv2.SetDirectory(0)
    d0file = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/input/v2VsFracD0_OO_010.root')
    hd0v2 = d0file.Get('hV2VsPtPrompt')
    hd0v2.SetDirectory(0)
    d0filecms = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/input/v2_OO_pt_cms_pbpb_6080.root')
    gd0v2cmspPb = d0filecms.Get('g0')
    dpfile = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/input/v2VsFrac_010_Dplus_combined.root')
    hdplusv2 = dpfile.Get('hV2VsPtPrompt')
    hdplusv2.SetDirectory(0)
    dsfile = TFile.Open('/home/spolitan/alice/ds_v2_oo/output/111225/v2/v2VsFrac.root')
    hdsv2 = dsfile.Get('hV2VsPtPrompt')
    hdsv2.SetDirectory(0)
    k0sfile = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/input/v2ofK0s_020.root')
    hk0sv2 = k0sfile.Get('v2datapoints')
    hk0sv2.SetDirectory(0)
    lambdafile = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/input/v2ofLambda_020.root')
    hlambdav2 = lambdafile.Get('v2datapoints')
    hlambdav2.SetDirectory(0)
    gd0cmspp = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/input/HEPData-ins1817310-v1-Table_1.root')
    d0ppv2 = gd0cmspp.Get('Table 1/Graph1D_y1')
    gist_lc_pass4_3050, gist_lc_pass4_syst_3050, gist_lc_pass4_fd_3050 = LoadGraphAndSyst('/home/spolitan/alice/hcv2-prl-figures/input/lc-prompt-allpt-wTotsyst.root',
                                                       'gV2PromptStat',
                                                       'gSystTotPrompt',
                                                       'gSystTotPrompt',
                                                       ROOT.kGreen+2,
                                                       ROOT.kFullCrossX)
    gist_d0_pass4_3050, gist_d0_pass4_syst_3050, gist_d0_pass4_fd_3050 = LoadGraphAndSyst('/home/spolitan/alice/hcv2-prl-figures/ste/finalresults/Dzero/v2_wsyst/v2_prompt_wsyst_D0_3050_finer.root',
                                                       'gvn_prompt_stat',
                                                       'tot_syst',
                                                       'fd_syst',
                                                        ROOT.kRed-4,
                                                        ROOT.kFullCircle,
                                                        markersize=1.2)
    gist_d0_pass4_3050 = rebin_tgraph(gist_d0_pass4_3050, gist_lc_pass4_3050)
    gist_lc_2023_2024 = clone_and_scale_uncertainty(gist_lc_pass4_3050, np.sqrt(2), new_name="gist_lc_2023_2024", xshift=0.1)
    gist_lc_2023_2024_2025 = clone_and_scale_uncertainty(gist_lc_pass4_3050, np.sqrt(5/1.5), new_name="gist_lc_2023_2024_2025", xshift=0.2)
    gist_lc_2023_2024_2025_2026 = clone_and_scale_uncertainty(gist_lc_pass4_3050, np.sqrt(7/1.5), new_name="gist_lc_2023_2024_2025_2026", xshift=0.)
    gist_d0_2023_2024 = clone_and_scale_uncertainty(gist_d0_pass4_3050, np.sqrt(2), new_name="gist_d0_2023_2024", xshift=0.1)
    gist_d0_2023_2024_2025 = clone_and_scale_uncertainty(gist_d0_pass4_3050, np.sqrt(5/1.5), new_name="gist_d0_2023_2024_2025", xshift=0.2)
    gist_d0_2023_2024_2025_2026 = clone_and_scale_uncertainty(gist_d0_pass4_3050, np.sqrt(7/1.5), new_name="gist_d0_2023_2024_2025_2026", xshift=0.3)

    infile_gsignif_lc_2023 = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/input/raw_yields_2023-uncr-pt4-12-pass4mc-weight_02.root')
    gsignif_lc_2023 = infile_gsignif_lc_2023.Get('hRawYieldsSignificanceSimFit')
    gsignif_lc_2023.SetDirectory(0)
    infile_gsignif_lc_2024 = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/input/raw_yields_2024-pass3-pass4mc-uncr_02.root')
    gsignif_lc_2024 = infile_gsignif_lc_2024.Get('hRawYieldsSignificanceSimFit')
    gsignif_lc_2024.SetDirectory(0)
    gsignif_lc_2023_2024 = clone_and_scale_uncertainty_th1(gsignif_lc_2023, np.sqrt(2), new_name="gsignif_lc_2023_2024")
    gsignif_lc_2023_2024_2025 = clone_and_scale_uncertainty_th1(gsignif_lc_2023, np.sqrt(5/1.5), new_name="gsignif_lc_2023_2024_2025")
    gsignif_lc_2023_2024_2025_2026 = clone_and_scale_uncertainty_th1(gsignif_lc_2023, np.sqrt(7/1.5), new_name="gsignif_lc_2023_2024_2025_2026")
    gsignif_lc_2023 = clone_and_scale_uncertainty_th1(gsignif_lc_2023, 1, new_name="gsignif_lc_2023_2024")
    gsignif_lc_2024 = clone_and_scale_uncertainty_th1(gsignif_lc_2024, 1, new_name="gsignif_lc_2024")

    glcits2 = make_tgraph_from_file('/home/spolitan/alice/hcv2-prl-figures/code/input_v2lc_run2.txt', name="glc_run2", title=";p_{T} (GeV/c);v_{2}")
    glcits3 = make_tgraph_from_file('/home/spolitan/alice/hcv2-prl-figures/code/input_v2lc_run2.txt', name="glc_run3", title=";p_{T} (GeV/c);v_{2}")
    glcits2old = make_tgraph_from_file2('/home/spolitan/alice/hcv2-prl-figures/code/input_v2lc_run2_old.txt', name="glc_run2_old", title=";p_{T} (GeV/c);v_{2}")
    graph_from_hist_with_inv_uncertainty('/home/spolitan/alice/hcv2-prl-figures/input/hSignifLcIts2.root', 'UP cut7 signif', glcits2, xshift=0.1)
    graph_from_hist_with_inv_uncertainty('/home/spolitan/alice/hcv2-prl-figures/input/hSignifLcIts3.root', 'SUP cut7 signif', glcits3, xshift=0.2)

    gtamuv2D0 = GetPrediction('/home/spolitan/alice/hcv2-prl-figures/ste/models/tamu/PromptD_TAMU_v2_5TeV_3050.txt',
                            1, 12, 1, True)
    gtamuv2D0.SetLineColor(ROOT.kAzure+4)
    gtamuv2D0.SetLineStyle(1)
    gtamuv2D0.SetLineWidth(0)
    gtamuv2D0.SetFillColorAlpha(ROOT.kAzure+4, 0.2)

    gtamuv2lcup = GetPrediction('/home/spolitan/alice/hcv2-prl-figures/ste/lc-up.dat',
                            11, 110, 0.1, False)
    gtamuv2lclow = GetPrediction('/home/spolitan/alice/hcv2-prl-figures/ste/lc-low.dat',
                            12, 110, 0.1, False)
    gtamuv2lc = compute_central_graph(gtamuv2lcup, gtamuv2lclow)
    gtamuv2lcup.SetLineColor(colors[0])
    gtamuv2lcup.SetLineStyle(9)
    gtamuv2lcup.SetLineWidth(3)
    gtamuv2lcup.SetFillColorAlpha(colors[4], 0.3)
    gtamuv2lclow.SetLineColor(colors[4])
    gtamuv2lclow.SetLineStyle(9)
    gtamuv2lclow.SetLineWidth(3)
    gtamuv2lc.SetFillColorAlpha(colors[4], 0.3)
    gtamuv2lc.SetLineColor(colors[4])
    gtamuv2lc.SetLineStyle(9)
    gtamuv2lc.SetLineWidth(0)
    gtamuv2lc.SetFillColorAlpha(colors[4], 0.3)
    
    SetObjectStyle(gist_lc_pass4_3050, markercolor=colors[7], linecolor=colors[7], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gist_lc_2023_2024, markercolor=colors[3], linecolor=colors[3], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gist_lc_2023_2024_2025, markercolor=colors[5], linecolor=colors[5], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gist_lc_2023_2024_2025_2026, markercolor=colors[0], linecolor=colors[0], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gist_d0_pass4_3050, markercolor=colors[4], linecolor=colors[4], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gist_d0_2023_2024, markercolor=colors[2], linecolor=colors[2], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gist_d0_2023_2024_2025, markercolor=colors[1], linecolor=colors[1], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gist_d0_2023_2024_2025_2026, markercolor=colors[6], linecolor=colors[6], markersize=1.2, markerstyle=20, linewidth=3)

    SetObjectStyle(gsignif_lc_2023, markercolor=colors[7], linecolor=colors[7], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gsignif_lc_2024, markercolor=colors[1], linecolor=colors[1], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gsignif_lc_2023_2024, markercolor=colors[3], linecolor=colors[3], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gsignif_lc_2023_2024_2025, markercolor=colors[5], linecolor=colors[5], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(gsignif_lc_2023_2024_2025_2026, markercolor=colors[0], linecolor=colors[0], markersize=1.2, markerstyle=20, linewidth=3)
    
    SetObjectStyle(glcits2, markercolor=colors[1], linecolor=colors[1], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(glcits2old, markercolor=colors[2], linecolor=colors[2], markersize=1.2, markerstyle=20, linewidth=3)
    SetObjectStyle(glcits3, markercolor=colors[6], linecolor=colors[6], markersize=1.2, markerstyle=20, linewidth=3)

    npoints = glcits2.GetN()
    for i in range(npoints):
        glcits2.SetPoint(i, gist_lc_pass4_3050.GetX()[i]+0.1, gist_lc_pass4_3050.GetY()[i])
        glcits3.SetPoint(i, gist_lc_pass4_3050.GetX()[i]+0.2, gist_lc_pass4_3050.GetY()[i])


    glcits3_reallumi = glcits3.Clone("glcits3_reallumi")
    npoints = glcits3_reallumi.GetN()
    for i in range(npoints):
        eyl = glcits3_reallumi.GetErrorYlow(i) * np.sqrt(10/3)
        glcits3_reallumi.SetPoint(i, glcits3_reallumi.GetX()[i]+0.15, glcits3_reallumi.GetY()[i])
        glcits3_reallumi.SetPointError(i,
                                       glcits3_reallumi.GetErrorX(i),
                                       eyl)
    SetObjectStyle(glcits3_reallumi, markercolor=colors[8], linecolor=colors[8], markersize=1.2, markerstyle=20, linewidth=3)
    glcits3_deadzone = glcits3_reallumi.Clone("glcits3_deadzone")
    # scale unc of 10%
    npoints = glcits3_deadzone.GetN()
    for i in range(npoints):
        eyl = glcits3_deadzone.GetErrorYlow(i) * 1.10
        glcits3_deadzone.SetPoint(i, glcits3_deadzone.GetX()[i]-0.18, glcits3_deadzone.GetY()[i])
        glcits3_deadzone.SetPointError(i,
                                       glcits3_deadzone.GetErrorX(i),
                                       eyl)
    SetObjectStyle(glcits3_deadzone, markercolor=colors[3], linecolor=colors[3], markersize=1.2, markerstyle=20, linewidth=3, fillcolor=colors[6], fillalpha=0.4, fillstyle=1000)

    canv = TCanvas("canv_hf_lf_v2", "canv_hf_lf_v2", 800, 800)
    leghflf = TLegend(0.2, 0.63, 0.45, 0.82)
    leghflf.SetTextFont(42)
    leghflf.SetTextSize(0.035)
    leghflf.SetBorderSize(0) 
    leghflf.SetFillStyle(0)

    if plotD0: 
        leghflf.SetHeader('#Lambda_{c}', "C")
        legD0 = TLegend(0.45, 0.63, 0.70, 0.78)
        legD0.SetTextFont(42)
        legD0.SetTextSize(0.035)
        legD0.SetBorderSize(0)
        legD0.SetFillStyle(0)
        legD0.SetHeader('D^{0}', "C")

        legNsigma = TLegend(0.70, 0.63, 0.80, 0.78)
        legNsigma.SetTextFont(42)
        legNsigma.SetTextSize(0.035)
        legNsigma.SetBorderSize(0)
        legNsigma.SetFillStyle(0)
        legNsigma.SetHeader('N#sigma', "C")


        legD0.AddEntry(gist_d0_pass4_3050, '2023', 'PEZ')
        if plot24:
            legD0.AddEntry(gist_d0_2023_2024, '2023-2024', 'PEZ')
        if plot25:
            legD0.AddEntry(gist_d0_2023_2024_2025, '2023-2025', 'PEZ')
        if plot26:
            legD0.AddEntry(gist_d0_2023_2024_2025_2026, '2023-2026', 'PEZ')
    
    if plot23:
        leghflf.AddEntry(gist_lc_pass4_3050, '2023, #it{L}_{int} #approx 1.5 nb^{-1}', 'PEZ') if not plotD0 else leghflf.AddEntry(gist_lc_pass4_3050, '2023', 'PEZ')
    if plot24:
        leghflf.AddEntry(gist_lc_2023_2024, '2023-2024, #it{L}_{int} #approx 3.0 nb^{-1}', 'PEZ') if not plotD0 else leghflf.AddEntry(gist_lc_2023_2024, '2023-2024', 'PEZ')
    else:
        if (not plotIts2 and not plotIts3) and (not plotIts2Old):
            leghflf.AddEntry(gist_lc_2023_2024, ' ', '')
    if plot25:
        leghflf.AddEntry(gist_lc_2023_2024_2025, '2023-2025, #it{L}_{int} #approx 5.0 nb^{-1}', 'PEZ') if not plotD0 else leghflf.AddEntry(gist_lc_2023_2024_2025, '2023-2025', 'PEZ')
    else:
        if (not plotIts2 and not plotIts3) and (not plotIts2Old):
            leghflf.AddEntry(gist_lc_2023_2024_2025, ' ', '')
    if plot26:
        leghflf.AddEntry(gist_lc_2023_2024_2025_2026, '2023-2026, #it{L}_{int} #approx 7.0 nb^{-1}', 'PEZ') if not plotD0 else leghflf.AddEntry(gist_lc_2023_2024_2025_2026, '2023-2026', 'PEZ')
    else:
        if not plotIts2 and not plotIts3:
            leghflf.AddEntry(gist_lc_2023_2024_2025_2026, ' ', '')
    if plotIts2Old:
        leghflf.AddEntry(glcits2old, 'ITS2 performance (TDR ITS2), #it{L}_{int} #approx 10 nb^{-1}', 'PEZ')
    if plotIts2:
        leghflf.AddEntry(glcits2, 'ITS2 performance (TDR ITS3), #it{L}_{int} #approx 10 nb^{-1}', 'PEZ')
    else:
        if plotIts2Old:
            leghflf.AddEntry(glcits2, ' ', '')
    if plotIts3:
        leghflf.AddEntry(glcits3, 
                         'ITS3 w/o deadzones, #it{L}_{int} #approx 3.0 nb^{-1}', 'PEZ')
                         #'ITS3 performance (TDR ITS3), #it{L}_{int} #approx 10 nb^{-1}', 'PEZ')
    else:
        if plotIts2 or plotIts2Old:
            leghflf.AddEntry(glcits3, ' ', '')
    if plotIts3RealLumi:
        leghflf.AddEntry(glcits3_reallumi, 
                         'ITS3 w/o deadzones, #it{L}_{int} #approx 3.0 nb^{-1}', 'PEZ')
    
    gist_lc_pass4_3050.GetYaxis().SetDecimals()
    #gist_lc_pass4_3050.GetYaxis().SetNdivisions(505)
    gist_lc_pass4_3050.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
    gist_lc_pass4_3050.GetYaxis().SetTitle('#it{v}_{2} (#Lambda_{c})') if not plotD0 else gist_lc_pass4_3050.GetYaxis().SetTitle('#it{v}_{2}')
    gist_lc_pass4_3050.GetYaxis().CenterTitle(False)
    gist_lc_pass4_3050.GetYaxis().SetRangeUser(-0.08, 0.45)
    gist_lc_pass4_3050.GetXaxis().SetRangeUser(1, 12)
    gist_lc_pass4_3050.SetTitle('')
    gist_lc_2023_2024_2025_2026.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
    gist_lc_2023_2024_2025_2026.GetYaxis().SetTitle('#it{v}_{2} (#Lambda_{c})') if not plotD0 else gist_lc_2023_2024_2025_2026.GetYaxis().SetTitle('#it{v}_{2}')
    gist_lc_2023_2024_2025_2026.GetYaxis().CenterTitle(False)
    gist_lc_2023_2024_2025_2026.GetYaxis().SetRangeUser(-0.08, 0.45)
    gist_lc_2023_2024_2025_2026.GetXaxis().SetRangeUser(1, 12)
    gist_lc_2023_2024_2025_2026.SetTitle('')
    gsignif_lc_2023.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
    gsignif_lc_2023.GetYaxis().SetTitle('Significance (#Lambda_{c})')
    gsignif_lc_2023.GetYaxis().CenterTitle(False)
    gsignif_lc_2023.GetYaxis().SetRangeUser(1, 200)
    gsignif_lc_2023.GetXaxis().SetRangeUser(4, 12)
    gsignif_lc_2023.SetTitle('')
    gtamuv2lc.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
    gtamuv2lc.GetYaxis().SetTitle('#it{v}_{2} (#Lambda_{c})') if not plotD0 else gtamuv2lc.GetYaxis().SetTitle('#it{v}_{2}')
    gtamuv2lc.GetYaxis().CenterTitle(False)
    gtamuv2lc.GetYaxis().SetRangeUser(-0.08, 0.45)
    gtamuv2lc.GetXaxis().SetRangeUser(1, 12)
    gtamuv2lc.SetTitle('')      

    legTamu = TLegend(0.22, 0.19, 0.45, 0.25)
    if plotD0: legTamu.SetNColumns(2)
    legTamu.SetTextFont(42)
    legTamu.SetTextSize(0.035)
    legTamu.SetBorderSize(0) 
    legTamu.SetFillStyle(0)
    if plotD0:
        legTamu.AddEntry(gtamuv2lc, '#Lambda_{c}', 'LF')
        legTamu.AddEntry(gtamuv2D0, 'D^{0}', 'LF')
        legTamu.SetHeader('TAMU')
    else:
        legTamu.AddEntry(gtamuv2lc, 'TAMU', 'LF')

    line2 = DrawLineAtX(0, 1.03, 12-0.3)
    
    '''
    gsignif_lc_2023.Draw('APEZ')
    gsignif_lc_2023_2024.Draw('PEZ same')
    gsignif_lc_2023_2024_2025.Draw('PEZ same')
    gsignif_lc_2023_2024_2025_2026.Draw('PEZ same')
    gsignif_lc_2024.Draw('PEZ same')

    leghflf.AddEntry(gsignif_lc_2023, '2023, #it{L}_{int} #approx 1.5 nb^{-1}', 'PEZ')
    leghflf.AddEntry(gsignif_lc_2024, '2024, #it{L}_{int} #approx 1.5 nb^{-1}', 'PEZ')
    leghflf.AddEntry(gsignif_lc_2023_2024, '2023-2024, #it{L}_{int} #approx 3.0 nb^{-1}', 'PEZ')
    leghflf.AddEntry(gsignif_lc_2023_2024_2025, '2023-2025, #it{L}_{int} #approx 5.0 nb^{-1}', 'PEZ')
    leghflf.AddEntry(gsignif_lc_2023_2024_2025_2026, '2023-2026, #it{L}_{int} #approx 7.0 nb^{-1}', 'PEZ')
    '''

    gtamuv2lc.Draw('A3c')
    if plotD0:
        gtamuv2D0.Draw('3c same')
    if plot23:
        gist_lc_pass4_3050.Draw('PZ same')
    if plot24:
        gist_lc_2023_2024.Draw('PZ same')
    if plot25:
        gist_lc_2023_2024_2025.Draw('PZ same')
    if plot26:
        gist_lc_2023_2024_2025_2026.Draw('PZ same')
    if plotIts2:
        glcits2.Draw('PZ same')
    if plotIts2Old:
        glcits2old.Draw('PZ same')
    if plotIts3:
        glcits3.Draw('PZ same')
    if plotIts3Deadzone:
        glcits3_deadzone.Draw('PEZ same')
        leghflf.AddEntry(glcits3_deadzone, 'ITS3 w/ deadzones, #it{L}_{int} #approx 3.0 nb^{-1}', 'PEZ')
    if plotIts3RealLumi:
        glcits3_reallumi.Draw('PZ same')
    if plotD0:
        z23, z24, z25, z26 = [], [], [], []
        chi2_23, chi2_24, chi2_25, chi2_26 = 0, 0, 0, 0
        pvalues = []
        nsigma = []
        npoints = gist_lc_pass4_3050.GetN()

        for i in range(npoints):
            if gist_lc_pass4_3050.GetX()[i] < 4 or gist_lc_pass4_3050.GetX()[i] > 10:
                print("Skipping pT bin", gist_lc_pass4_3050.GetX()[i])
                continue
            yD023 = gist_d0_pass4_3050.GetY()[i]
            yD024 = gist_d0_2023_2024.GetY()[i]
            yD025 = gist_d0_2023_2024_2025.GetY()[i]
            yD026 = gist_d0_2023_2024_2025_2026.GetY()[i]
            yD023Unc = gist_d0_pass4_3050.GetErrorYhigh(i)
            yD024Unc = gist_d0_2023_2024.GetErrorYhigh(i)
            yD025Unc = gist_d0_2023_2024_2025.GetErrorYhigh(i)
            yD026Unc = gist_d0_2023_2024_2025_2026.GetErrorYhigh(i)
            yLc23 = gist_lc_pass4_3050.GetY()[i]
            yLc24 = gist_lc_2023_2024.GetY()[i]
            yLc25 = gist_lc_2023_2024_2025.GetY()[i]
            yLc26 = gist_lc_2023_2024_2025_2026.GetY()[i]
            yLc23Unc = gist_lc_pass4_3050.GetErrorYhigh(i)
            yLc24Unc = gist_lc_2023_2024.GetErrorYhigh(i)
            yLc25Unc = gist_lc_2023_2024_2025.GetErrorYhigh(i)
            yLc26Unc = gist_lc_2023_2024_2025_2026.GetErrorYhigh(i)

            z23.append(np.abs(yLc23 - yD023) / np.sqrt(yLc23Unc**2 + yD023Unc**2))
            z24.append(np.abs(yLc24 - yD024) / np.sqrt(yLc24Unc**2 + yD024Unc**2))
            z25.append(np.abs(yLc25 - yD025) / np.sqrt(yLc25Unc**2 + yD025Unc**2))
            z26.append(np.abs(yLc26 - yD026) / np.sqrt(yLc26Unc**2 + yD026Unc**2))

        print("Z 2026:", z26)
        for i in range(len(z23)):
            chi2_23 += z23[i]**2
            chi2_24 += z24[i]**2
            chi2_25 += z25[i]**2
            print("Chi2 2025 intermediate:", chi2_25)
            chi2_26 += z26[i]**2
        print("Chi2 2023:", chi2_23)
        print("Chi2 2024:", chi2_24)
        print("Chi2 2025:", chi2_25)
        print("Chi2 2026:", chi2_26)
        
        pvalues.append(ROOT.Math.chisquared_cdf_c(chi2_23, len(z23)))
        pvalues.append(ROOT.Math.chisquared_cdf_c(chi2_24, len(z24)))
        pvalues.append(ROOT.Math.chisquared_cdf_c(chi2_25, len(z25)))
        pvalues.append(ROOT.Math.chisquared_cdf_c(chi2_26, len(z26)))
        print("P-value 2023:", pvalues[0])
        print("P-value 2024:", pvalues[1])
        print("P-value 2025:", pvalues[2])
        print("P-value 2026:", pvalues[3])

        for pvalue in pvalues:
            nsigma.append(ROOT.Math.normal_quantile_c(pvalue, 1.0))  

        if plot23:
            legNsigma.AddEntry(gist_d0_pass4_3050, f'{nsigma[0]:.2f}', '')
        if plot24:
            legNsigma.AddEntry(gist_d0_2023_2024, f'{nsigma[1]:.2f}', '')
        if plot25:
            legNsigma.AddEntry(gist_d0_2023_2024_2025, f'{nsigma[2]:.2f}', '')
        if plot26:
            legNsigma.AddEntry(gist_d0_2023_2024_2025_2026, f'{nsigma[3]:.2f}', '')
        legNsigma.Draw()

        gist_d0_pass4_3050.Draw('PZ same')
        if plot24:
            gist_d0_2023_2024.Draw('PZ same')
        if plot25:
            gist_d0_2023_2024_2025.Draw('PZ same')
        if plot26:
            gist_d0_2023_2024_2025_2026.Draw('PZ same')
    
    latex.DrawLatexNDC(0.22, 0.85, 'ALICE, performance study')
    latexdetail.DrawLatexNDC(0.22, 0.80, '30#minus50% Pb#minusPb, #sqrt{#it{s}_{NN}} = 5.36 TeV')
    #latexdetail.DrawLatexNDC(0.8, 0.86, '|#it{y}| < 0.8')
    gsignif_lc_2023_2024.Draw('P same')
    gsignif_lc_2023_2024_2025.Draw('P same')
    gsignif_lc_2023_2024_2025_2026.Draw('P same')
    leghflf.Draw()
    if plotD0:
        legD0.Draw()

    legTamu.Draw()
    line2.Draw()
    canv.SaveAs("./lc_v2_run2_run3.pdf")
    canv.SaveAs("./lc_v2_run2_run3.root")
    print("Saved lc_v2_run2_run3.png")


    # 2023 vs 2025 significance and significance/sqrt(N) plots (inputs and values from Xufei)
    infile_23 = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/code/raw_yields_cr-pt2-24_00_2023pass4.root')
    infile_25 = TFile.Open('/home/spolitan/alice/hcv2-prl-figures/code/raw_yields_cr-pt2-24_00_2025pass1.root')
    hsignif_lc_2023 = infile_23.Get('hRawYieldsSignificanceSimFit')
    hsignif_lc_2023.SetDirectory(0)
    hsignif_lc_2025 = infile_25.Get('hRawYieldsSignificanceSimFit')
    hsignif_lc_2025.SetDirectory(0)
    infile_23.Close()
    infile_25.Close()
    nev_23 =  1.64e9
    nev_25 =  2.3e9

    hsignif_lc_2023.SetLineColor(colors[1])
    hsignif_lc_2025.SetLineColor(colors[0])
    hsignif_lc_2023.SetMarkerColor(colors[1])
    hsignif_lc_2025.SetMarkerColor(colors[0])
    hsignif_lc_2023.SetMarkerSize(1.2)
    hsignif_lc_2025.SetMarkerSize(1.2)
    hsignif_lc_2023.SetLineWidth(2)
    hsignif_lc_2025.SetLineWidth(2)
    hsignif_lc_2023.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
    hsignif_lc_2023.SetMarkerStyle(ROOT.kFullCircle)
    hsignif_lc_2025.GetXaxis().SetTitle('#it{p}_{T} (GeV/#it{c})')
    hsignif_lc_2025.SetMarkerStyle(ROOT.kFullCircle)

    # significance/sqrt(N) plots
    hsignif_lc_2023_sqrtN = hsignif_lc_2023.Clone("hsignif_lc_2023_sqrtN")
    hsignif_lc_2025_sqrtN = hsignif_lc_2025.Clone("hsignif_lc_2025_sqrtN")
    hsignif_lc_2023_sqrtN.Scale(1/np.sqrt(nev_23))
    hsignif_lc_2025_sqrtN.Scale(1/np.sqrt(nev_25))
    hsignif_lc_2023_sqrtN.GetYaxis().SetTitle('#Lambda_{c}^{+} significance/#sqrt{#it{N}_{ev}}')
    hsignif_lc_2023_sqrtN.GetYaxis().SetRangeUser(1.e-4, 0.002)
    hsignif_lc_2023_sqrtN.GetYaxis().SetDecimals()
    hsignif_lc_2023_sqrtN.GetYaxis().SetMaxDigits(2)
    hsignif_lc_2023_sqrtN.GetXaxis().SetRangeUser(2, 24)

    canv_signif_nev = TCanvas("canv_signif", "canv_signif", 800, 800)
    leg_signif = TLegend(0.2, 0.65, 0.3, 0.78)
    leg_signif.SetTextFont(42)
    leg_signif.SetTextSize(0.035)
    leg_signif.SetBorderSize(0)
    leg_signif.SetFillStyle(0)
    leg_signif.AddEntry(hsignif_lc_2023, f'2023 (#it{{N}}_{{ev}} = {nev_23:.2e})', 'PE')
    leg_signif.AddEntry(hsignif_lc_2025, f'2025 (#it{{N}}_{{ev}} = {nev_25:.2e})', 'PE')

    hsignif_lc_2023_sqrtN.Draw('P')
    hsignif_lc_2025_sqrtN.Draw('P same')
    leg_signif.Draw()

    latex.DrawLatexNDC(0.2, 0.85, 'ALICE, performance study')
    latexdetail.DrawLatexNDC(0.2, 0.80, '30#minus50% Pb#minusPb, #sqrt{#it{s}_{NN}} = 5.36 TeV')

    canv_signif_nev.SaveAs("./lc_v2_run2_run3_significance_over_nev.pdf")
    canv_signif_nev.SaveAs("./lc_v2_run2_run3_significance_over_nev.root")
    print("Saved lc_v2_run2_run3_significance.png")

    # significance/sqrt(N) plots with ratio 2025/2023
    hratio_23_25_nev = hsignif_lc_2025_sqrtN.Clone("hratio_23_25_nev")
    hratio_23_25_nev.Divide(hsignif_lc_2023_sqrtN)
    hratio_23_25_nev.GetYaxis().SetTitle('#Lambda_{c}^{+} significance/#sqrt{#it{N}_{ev}} ratio 2025/2023')
    hratio_23_25_nev.GetYaxis().SetRangeUser(0, 2)
    hratio_23_25_nev.GetYaxis().SetDecimals()
    hratio_23_25_nev.GetYaxis().SetMaxDigits(2)
    hratio_23_25_nev.GetXaxis().SetRangeUser(2, 24)
    canv_signif_nev_wratio = TCanvas("canv_signif", "canv_signif", 1600, 800)
    canv_signif_nev_wratio.Divide(2, 1)
    
    canv_signif_nev_wratio.cd(1)
    leg_signif = TLegend(0.2, 0.65, 0.3, 0.78)
    leg_signif.SetTextFont(42)
    leg_signif.SetTextSize(0.035)
    leg_signif.SetBorderSize(0)
    leg_signif.SetFillStyle(0)
    leg_signif.AddEntry(hsignif_lc_2023, f'2023 (#it{{N}}_{{ev}} = {nev_23:.2e})', 'PE')
    leg_signif.AddEntry(hsignif_lc_2025, f'2025 (#it{{N}}_{{ev}} = {nev_25:.2e})', 'PE')
    hsignif_lc_2023_sqrtN.Draw('P')
    hsignif_lc_2025_sqrtN.Draw('P same')
    latex.DrawLatexNDC(0.2, 0.85, 'ALICE, performance study')
    latexdetail.DrawLatexNDC(0.2, 0.80, '30#minus50% Pb#minusPb, #sqrt{#it{s}_{NN}} = 5.36 TeV')
    leg_signif.Draw()
    
    canv_signif_nev_wratio.cd(2)
    hratio_23_25_nev.Draw('P')
    line_at_1 = ROOT.TLine(2, 1, 24, 1)
    line_at_1.SetLineColor(ROOT.kGray+2)
    line_at_1.SetLineStyle(2)
    line_at_1.Draw()
    canv_signif_nev_wratio.cd(1)

    canv_signif_nev_wratio.SaveAs("./lc_v2_run2_run3_significance_over_nev_ratio.pdf")
    canv_signif_nev_wratio.SaveAs("./lc_v2_run2_run3_significance_over_nev_ratio.root")
    print("Saved lc_v2_run2_run3_significance_over_nev_ratio.png")


    # significance plots
    hsignif_lc_2023.GetYaxis().SetTitle('#Lambda_{c}^{+} significance')
    hsignif_lc_2023.GetYaxis().SetRangeUser(0, 100)
    hsignif_lc_2023.GetYaxis().SetDecimals()
    hsignif_lc_2023.GetYaxis().SetMaxDigits(4)
    hsignif_lc_2023.GetXaxis().SetRangeUser(2, 24)

    canv_signif = TCanvas("canv_signif", "canv_signif", 800, 800)
    leg_signif = TLegend(0.2, 0.65, 0.3, 0.78)
    leg_signif.SetTextFont(42)
    leg_signif.SetTextSize(0.035)
    leg_signif.SetBorderSize(0)
    leg_signif.SetFillStyle(0)
    leg_signif.AddEntry(hsignif_lc_2023, f'2023', 'PE')
    leg_signif.AddEntry(hsignif_lc_2025, f'2025', 'PE')

    hsignif_lc_2023.Draw('P')
    hsignif_lc_2025.Draw('P same')
    leg_signif.Draw()

    latex.DrawLatexNDC(0.2, 0.85, 'ALICE, performance study')
    latexdetail.DrawLatexNDC(0.2, 0.80, '30#minus50% Pb#minusPb, #sqrt{#it{s}_{NN}} = 5.36 TeV')

    canv_signif.SaveAs("./lc_v2_run2_run3_significance.pdf")
    canv_signif.SaveAs("./lc_v2_run2_run3_significance.root")
    print("Saved lc_v2_run2_run3_significance.png")


main()