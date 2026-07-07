// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DhCorrelationFitter.cxx
/// \brief class for for performing the fit of azimuthal correlations
/// \author Samuele Cattaruzzi <samuele.cattaruzzi@cern.ch>
/// \author Swapnesh Santosh Khade <swapnesh.santosh.khade@cern.ch>

#include "DhCorrelationFitter.h"

#include <TError.h>
#include <TF1.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TMath.h>
#include <TMathBase.h>
#include <TMatrixD.h> // IWYU pragma: keep (do not replace with TMatrixDfwd.h)
#include <TMatrixDfwd.h>
#include <TMinuit.h>
#include <TPaveText.h>
#include <TString.h>
#include <TText.h>
#include <TVirtualFitter.h>
#include <TVirtualPad.h>
#include <TLegend.h>

#include <Rtypes.h>
#include <RtypesCore.h>

#include <cstdio>
#include <iostream>

DhCorrelationFitter::DhCorrelationFitter() : // default constructor
                                             fVerbose(true),
                                             fIsReflected(kFALSE),
                                             fTypeOfFitFunc(kConstwoGaus),
                                             fFixBase(0),
                                             fFixMean(0),
                                             fMinCandPt(0.),
                                             fMaxCandPt(99.),
                                             fMinAssoPt(0.),
                                             fMaxAssoPt(99.),
                                             fNpars(0),
                                             fExtParsVals(0x0),
                                             fExtParsLowBounds(0x0),
                                             fExtParsUppBounds(0x0),
                                             fUseExternalPars(kFALSE),
                                             fShiftBaselineUp(kFALSE),
                                             fShiftBaselineDown(kFALSE),
                                             fIsTotal(kTRUE),
                                             fWithPedLM(kFALSE),
                                             fFixLMFactor(kFALSE),
                                             fNbasleinePoints(0),
                                             fBinsBaseline(0x0),
                                             fHist(0x0),
                                             fTempHisto(0x0),
                                             fGraph(0x0),
                                             fSpline(0x0),
                                             fLMOutput(0x0),
                                             fTemplateFunc(0x0),
                                             fTempFunc(0),
                                             fPairsPerMassBin(0.0),
                                             fMinCorr(0),
                                             fMaxCorr(0),
                                             fBaseline(0.),
                                             fErrBaseline(0.),
                                             fFit(0x0),
                                             fGausNS(0x0),
                                             fGausAS(0x0),
                                             fPed(0x0),
                                             fLM(0x0),
                                             fFlow(0x0),
                                             fv1(0x0),
                                             fv2(0x0),
                                             fv3(0x0),
                                             fv4(0x0),
                                             fv5(0x0),
                                             fv1_lm(0x0),
                                             fv2_lm(0x0),
                                             fv3_lm(0x0),
                                             fv4_lm(0x0),
                                             fv5_lm(0x0),
                                             fv2AssocPart(0.),
                                             fv2Dmeson(0.),
                                             fv2Delta(0.),
                                             fRyTrigger(1.),
                                             fRyTriggerErr(0.),
                                             fLMRyTrigger(1.),
                                             fLMRyTriggerErr(0.),
                                             fNSyieldBinCount(0.),
                                             fErrNSyieldBinCount(0.),
                                             fASyieldBinCount(0.),
                                             fErrASyieldBinCount(0.)
{
}

DhCorrelationFitter::DhCorrelationFitter(TH1F* histoToFit, Double_t min, Double_t max) : // standard constructor
                                                                                         fVerbose(true),
                                                                                         fIsReflected(kFALSE),
                                                                                         fTypeOfFitFunc(kConstwoGaus),
                                                                                         fFixBase(0),
                                                                                         fFixMean(0),
                                                                                         fMinCandPt(0.),
                                                                                         fMaxCandPt(99.),
                                                                                         fMinAssoPt(0.),
                                                                                         fMaxAssoPt(99.),
                                                                                         fNpars(0),
                                                                                         fExtParsVals(0x0),
                                                                                         fExtParsLowBounds(0x0),
                                                                                         fExtParsUppBounds(0x0),
                                                                                         fUseExternalPars(kFALSE),
                                                                                         fShiftBaselineUp(kFALSE),
                                                                                         fShiftBaselineDown(kFALSE),
                                                                                         fIsTotal(kTRUE),
                                                                                         fWithPedLM(kFALSE),
                                                                                         fFixLMFactor(kFALSE),
                                                                                         fNbasleinePoints(0),
                                                                                         fBinsBaseline(0x0),
                                                                                         fHist(0x0),
                                                                                         fTempHisto(0x0),
                                                                                         fGraph(0x0),
                                                                                         fSpline(0x0),
                                                                                         fLMOutput(0x0),
                                                                                         fTemplateFunc(0x0),
                                                                                         fTempFunc(0),
                                                                                         fPairsPerMassBin(0.0),
                                                                                         fMinCorr(0.),
                                                                                         fMaxCorr(0.),
                                                                                         fBaseline(0.),
                                                                                         fErrBaseline(0.),
                                                                                         fFit(0x0),
                                                                                         fGausNS(0x0),
                                                                                         fGausAS(0x0),
                                                                                         fPed(0x0),
                                                                                         fLM(0x0),
                                                                                         fFlow(0x0),
                                                                                         fv1(0x0),
                                                                                         fv2(0x0),
                                                                                         fv3(0x0),
                                                                                         fv4(0x0),
                                                                                         fv5(0x0),
                                                                                         fv1_lm(0x0),
                                                                                         fv2_lm(0x0),
                                                                                         fv3_lm(0x0),
                                                                                         fv4_lm(0x0),
                                                                                         fv5_lm(0x0),
                                                                                         fBaseTransvReg(0x0),
                                                                                         fv2AssocPart(0.),
                                                                                         fv2Dmeson(0.),
                                                                                         fv2Delta(0.),
                                                                                         fRyTrigger(1.),
                                                                                         fRyTriggerErr(0.),
                                                                                         fLMRyTrigger(1.),
                                                                                         fLMRyTriggerErr(0.),
                                                                                         fNSyieldBinCount(0.),
                                                                                         fErrNSyieldBinCount(0.),
                                                                                         fASyieldBinCount(0.),
                                                                                         fErrASyieldBinCount(0.)
{
  fHist = histoToFit;
  fMinCorr = min;
  fMaxCorr = max;
}

DhCorrelationFitter::DhCorrelationFitter(const DhCorrelationFitter& source) : // copy constructor
                                                                              fVerbose(true),
                                                                              fIsReflected(source.fIsReflected),
                                                                              fTypeOfFitFunc(source.fTypeOfFitFunc),
                                                                              fFixBase(source.fFixBase),
                                                                              fFixMean(source.fFixMean),
                                                                              fMinCandPt(source.fMinCandPt),
                                                                              fMaxCandPt(source.fMaxCandPt),
                                                                              fMinAssoPt(source.fMinAssoPt),
                                                                              fMaxAssoPt(source.fMaxAssoPt),
                                                                              fNpars(source.fNpars),
                                                                              fExtParsVals(source.fExtParsVals),
                                                                              fExtParsLowBounds(source.fExtParsLowBounds),
                                                                              fExtParsUppBounds(source.fExtParsUppBounds),
                                                                              fUseExternalPars(source.fUseExternalPars),
                                                                              fShiftBaselineUp(source.fShiftBaselineUp),
                                                                              fShiftBaselineDown(source.fShiftBaselineDown),
                                                                              fIsTotal(source.fIsTotal),
                                                                              fWithPedLM(source.fWithPedLM),
                                                                              fFixLMFactor(source.fFixLMFactor),
                                                                              fNbasleinePoints(source.fNbasleinePoints),
                                                                              fBinsBaseline(source.fBinsBaseline),
                                                                              fHist(source.fHist),
                                                                              fMinCorr(source.fMinCorr),
                                                                              fMaxCorr(source.fMaxCorr),
                                                                              fBaseline(source.fBaseline),
                                                                              fErrBaseline(source.fErrBaseline),
                                                                              fFit(source.fFit),
                                                                              fGausNS(source.fGausNS),
                                                                              fGausAS(source.fGausAS),
                                                                              fPed(source.fPed),
                                                                              fLM(source.fLM),
                                                                              fFlow(source.fFlow),
                                                                              fv1(source.fv1),
                                                                              fv2(source.fv2),
                                                                              fv3(source.fv3),
                                                                              fv4(source.fv4),
                                                                              fv5(source.fv5),
                                                                              fv1_lm(source.fv1_lm),
                                                                              fv2_lm(source.fv2_lm),
                                                                              fv3_lm(source.fv3_lm),
                                                                              fv4_lm(source.fv4_lm),
                                                                              fv5_lm(source.fv5_lm),
                                                                              fBaseTransvReg(source.fBaseTransvReg),
                                                                              fv2AssocPart(source.fv2AssocPart),
                                                                              fv2Dmeson(source.fv2Dmeson),
                                                                              fv2Delta(source.fv2Delta),
                                                                              fNSyieldBinCount(source.fNSyieldBinCount),
                                                                              fErrNSyieldBinCount(source.fErrNSyieldBinCount),
                                                                              fASyieldBinCount(source.fASyieldBinCount),
                                                                              fErrASyieldBinCount(source.fErrASyieldBinCount)
{
}

DhCorrelationFitter::~DhCorrelationFitter()
// destructor
{
  Info("DhCorrelationFitter.cxx", "Destructor is calling");

  if (fHist) {
    delete fHist;
    fHist = 0;
  }
  if (fTempHisto) {
    delete fTempHisto;
    fTempHisto = 0;
  }
  if (fLMOutput) {
    delete fLMOutput;
    fLMOutput = 0;
  }
  if (fTemplateFunc) {
    delete fTemplateFunc;
    fTemplateFunc = 0;
  }
  if (fFit) {
    delete fFit;
    fFit = 0;
  }
  if (fGausNS) {
    delete fGausNS;
    fGausNS = 0;
  }
  // if (fGausNS2) {delete fGausNS2; fGausNS2 = 0;}
  if (fPed) {
    delete fPed;
    fPed = 0;
  }
  if (fLM) {
    delete fLM;
    fLM = 0;
  }
  if (fFlow) {
    delete fFlow;
    fFlow = 0;
  }
}

DhCorrelationFitter& DhCorrelationFitter::operator=(const DhCorrelationFitter& cfit)
// assignment operator
{
  if (&cfit == this)
    return *this;

  fIsReflected = cfit.fIsReflected;
  fTypeOfFitFunc = cfit.fTypeOfFitFunc;
  fFixBase = cfit.fFixBase;
  fFixMean = cfit.fFixMean;
  fMinCandPt = cfit.fMinCandPt;
  fMaxCandPt = cfit.fMaxCandPt;
  fMinAssoPt = cfit.fMinAssoPt;
  fMaxAssoPt = cfit.fMaxAssoPt;
  fNpars = cfit.fNpars;
  fExtParsVals = cfit.fExtParsVals;
  fExtParsLowBounds = cfit.fExtParsLowBounds;
  fExtParsUppBounds = cfit.fExtParsUppBounds;
  fUseExternalPars = cfit.fUseExternalPars;
  fWithPedLM = cfit.fWithPedLM;
  fFixLMFactor = cfit.fFixLMFactor;
  fNbasleinePoints = cfit.fNbasleinePoints;
  fBinsBaseline = cfit.fBinsBaseline;
  fHist = cfit.fHist;
  fMinCorr = cfit.fMinCorr;
  fMaxCorr = cfit.fMaxCorr;
  fBaseline = cfit.fBaseline;
  fErrBaseline = cfit.fErrBaseline;
  fFit = cfit.fFit;
  fGausNS = cfit.fGausNS;
  fGausAS = cfit.fGausAS;
  fPed = cfit.fPed;
  fNSyieldBinCount = cfit.fNSyieldBinCount;
  fErrNSyieldBinCount = cfit.fErrNSyieldBinCount;
  fASyieldBinCount = cfit.fASyieldBinCount;
  fErrASyieldBinCount = cfit.fErrASyieldBinCount;
  fLM = cfit.fLM;
  fFlow = cfit.fFlow;

  return *this;
}

void DhCorrelationFitter::SetExternalValsAndBounds(Int_t nPars, Double_t* vals, Double_t* lowBounds, Double_t* uppBounds)
{

  fNpars = nPars;

  fExtParsVals = new Double_t[fNpars];
  fExtParsLowBounds = new Double_t[fNpars];
  fExtParsUppBounds = new Double_t[fNpars];

  for (int i = 0; i < fNpars; i++) {
    fExtParsVals[i] = vals[i];
    fExtParsLowBounds[i] = lowBounds[i];
    fExtParsUppBounds[i] = uppBounds[i];
  }

  return;
}

void DhCorrelationFitter::Fitting(Bool_t drawSplitTerm, Bool_t useExternalPars)
{
  if (fHist == nullptr) {
    Error("DhCorrelationFitter::Fitting", "No histogram to fit!");
    return;
  }
  // -> fFixBase = 0 : baseline free
  //             = 1 : fix the baseline to the minimum of the histogram
  //             < 0 : fix the baseline to the weighted average of the abs(fFixBaseline) lower points
  //             = 2 : zyam at pi/2. Fix the baseline averaging the 2 points around +-pi/2 value
  //             = 3 : fix the baseline to the weighted average of the points passed through the function SetPointsForBaseline()
  //             = 4 : fix the baseline to the weighted average of the points in the transverse region in default configuration (i.e. for 32 bins the first 2, the last 2 and the 4 middle ones)
  //
  // -> fFixMean = 0 : NS & AS mean free
  //             = 1 : NS mean fixed to 0, AS mean free
  //             = 2 : AS mean fixed to pi, NS mean free
  //             = 3 : NS mean fixed to 0, AS mean to pi

  if (useExternalPars)
    fUseExternalPars = kTRUE;

  /*if (fFixBase != 0 && fFixBase != 6) {
    Printf("[INFO] DhCorrelationFitter::Fitting, Finding baseline");
    FindBaseline();
  }*/
  /*if (fFixBase == 0) {
    // set initial value of the fBaseline
    fBaseline = CalculateBaseline(fHist, fIsTotal);
  }*/
  if (fVerbose) { Printf("[INFO] DhCorrelationFitter::Fitting, Setting Function"); }
  if (fTypeOfFitFunc == 7) { // case for v2 modulation
    FitBaselineWv2();        // to contrain the B parameter in the fit function for the pedestal
    Printf("[INFO] B parameter for v2 fit: %.3f", fBaseline);
  }
  SetFitFunction();

  /*if (fFixBase != 0 && fTypeOfFitFunc != 8 && fTypeOfFitFunc != 9) {
    fFit->FixParameter(0, fBaseline);
  }
  if ((fFixMean == 1 || fFixMean == 3) && fTypeOfFitFunc != 8 && fTypeOfFitFunc != 9) {
    fFit->FixParameter(2, 0.);
  }
  if (fFixMean == 2 || fFixMean == 3) {
    if (fTypeOfFitFunc != 0 && fTypeOfFitFunc != 3 && fTypeOfFitFunc != 8 && fTypeOfFitFunc != 9)
      fFit->FixParameter(5, TMath::Pi());
    if (fTypeOfFitFunc == 3 || fTypeOfFitFunc == 6)
      fFit->FixParameter(2, TMath::Pi());
  }*/
  if (fVerbose) {
    printf("[INFO] DhCorrelationFitter::Fitting, Starting fit with function type: %d \n", fTypeOfFitFunc);
    printf("[INFO] DhCorrelationFitter::Fitting, Fit range: %.3f to %.3f \n", fMinCorr, fMaxCorr);
    printf("[INFO] DhCorrelationFitter::Fitting, bins in fit range: %d to %d \n", fHist->GetXaxis()->FindBin(fMinCorr), fHist->GetXaxis()->FindBin(fMaxCorr));
    printf("[INFO] DhCorrelationFitter::Fitting, Number of points in histogram: %d \n", fHist->GetNbinsX());
    printf("[INFO] DhCorrelationFitter::Fitting, Name of histogram: %s \n", fHist->GetName());

    Printf("[INFO] DhCorrelationFitter::Fitting, Fitting");
  }
  TVirtualFitter::SetMaxIterations(50000);
  TFitResultPtr fitptr = fHist->Fit(fFit, "RIMESQ", "", fMinCorr, fMaxCorr);
  if (fitptr.Get() == nullptr || fitptr->Status() != 0) {
    std::cout << "[ERROR] Fit failed or invalid TFitResult!" << std::endl;
    //return;
  }
  TMatrixD cor = fitptr->GetCorrelationMatrix();
  TMatrixD cov = fitptr->GetCovarianceMatrix();
  if (fVerbose) {
    printf("[INFO] Correlation Matrix - The final one! \n");
    cor.Print();
    gMinuit->mnmatu(1);
    printf("[INFO] Covariance Matrix - The final one! \n");
    cov.Print();
  }
  if (fFixBase == 0) {
    fBaseline = fFit->GetParameter(0);
    fErrBaseline = fFit->GetParError(0);
  }
  if (fVerbose) { Printf("[INFO] DhCorrelationFitter::Fitting, Calculating yields with BC"); }
  CalculateYieldsAboveBaseline();
  fHist->SetTitle(";#Delta#varphi (rad); #frac{1}{N_{D}}#frac{dN^{assoc}}{d#Delta#varphi} (rad^{-1})");
  if (fVerbose) { Printf("[INFO] DhCorrelationFitter::Fitting, Now drawing, if requested"); }
  SetSingleTermsForDrawing(drawSplitTerm);

  /*// NS yield from bin counting
  double fNSyield = 0.;
  double fNSyieldErr = 0.;
  double baselinBinCount = 0;
  for (int iBin = 1; iBin <= 6; iBin++) {                                                // first six bins
    fNSyield += 2 * (fHist->GetBinContent(iBin) - fBaseline) * fHist->GetBinWidth(iBin); // x2 due to the fatct the histogram is reflected
    fNSyieldErr += 4 * (TMath::Power(fHist->GetBinError(iBin), 2) + TMath::Power(fErrBaseline, 2)) * (fHist->GetBinWidth(iBin) * fHist->GetBinWidth(iBin));
    baselinBinCount += fBaseline * fHist->GetBinWidth(iBin);
  }
  fNSyieldErr = TMath::Sqrt(fNSyieldErr);

  // AS yield from bin counting
  double fASyield = 0.;
  double fASyieldErr = 0.;
  for (int iBin = 11; iBin <= 16; iBin++) { // last six bins
    fASyield += 2 * (fHist->GetBinContent(iBin) - fBaseline) * fHist->GetBinWidth(iBin);
    fASyieldErr += 4 * (TMath::Power(fHist->GetBinError(iBin), 2) + TMath::Power(fErrBaseline, 2)) * (fHist->GetBinWidth(iBin) * fHist->GetBinWidth(iBin));
  }
  fASyieldErr = TMath::Sqrt(fASyieldErr);

  printf("[RESULT MINE] Bin counting results: NS Yield = %.3f +- %.3f \n[RESULT MINE] Bin counting results: AS Yield: %.3f +- %.3f \n[RESULT MINE] baseline = %.3f \n", fNSyield, fNSyieldErr, fASyield, fASyieldErr, baselinBinCount);*/
}

void DhCorrelationFitter::SetFitFunction()
{
  // -> fitFunc = 1: const + G NS + G AS (w/o periodicity)
  //            = 2: const + G NS + G AS  (w/ periodicity)
  //            = 3: const + G AS
  //            = 4: const + GenG NS + G AS  (w/ periodicity)
  //            = 5: const + VonMises NS + VonMises AS (w/periodicity)
  //            = 6: const + VonMises AS
  //            = 7: baseline w v2 modulation + G NS + G AS  (w/ periodicity)
  //            = 8: only baseline w v2 modulation

  if (fFit) {
    delete fFit;
    delete fGausNS;
    // delete fGausNS2;
    delete fGausAS;
    // delete fGausAS2;
    delete fPed;
  }
  switch (fTypeOfFitFunc) {
    case 1:
      fFit = new TF1("ConstwoGaus", "[0]+[1]/TMath::Sqrt(2.*TMath::Pi())/[3]*TMath::Exp(-(x-[2])*(x-[2])/2./([3]*[3]))+[4]/TMath::Sqrt(2.*TMath::Pi())/[6]*TMath::Exp(-(x-[5])*(x-[5])/2./([6]*[6]))", fMinCorr, fMaxCorr);
      fGausNS = new TF1("fGausNS", "[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-[1])*(x-[1])/2./([2]*[2]))", fMinCorr, fMaxCorr);
      fGausAS = new TF1("fGausAS", "[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-[1])*(x-[1])/2./([2]*[2]))", fMinCorr, fMaxCorr);
      fPed = new TF1("fPed", "[0]", fMinCorr, fMaxCorr);

      if (!fUseExternalPars) {
        fFit->SetParLimits(0, 0, 9999.);
        fFit->SetParLimits(1, 0, 999.);
        fFit->SetParLimits(2, -1, 1);
        fFit->SetParLimits(3, 0, 3.14 / 3.);
        fFit->SetParLimits(4, 0, 999.);
        fFit->SetParLimits(5, 2., 4);
        fFit->SetParLimits(6, 0, 3.14 / 2.);

        fFit->SetParameter(0, 3);
        fFit->SetParameter(1, 2);
        fFit->SetParameter(2, 0.);
        fFit->SetParameter(3, 0.3);
        fFit->SetParameter(4, 2);
        fFit->SetParameter(5, 3.14);
        fFit->SetParameter(6, 0.3);

      } else {
        for (int i = 0; i < fNpars; i++) {
          fFit->SetParameter(i, fExtParsVals[i]);
          fFit->SetParLimits(i, fExtParsLowBounds[i], fExtParsUppBounds[i]);
        }
      }

      fFit->SetParName(0, "ped");
      fFit->SetParName(1, "NS Y");
      fFit->SetParName(2, "NS mean");
      fFit->SetParName(3, "NS #sigma");
      fFit->SetParName(4, "AS Y");
      fFit->SetParName(5, "AS mean");
      fFit->SetParName(6, "AS #sigma");

      break;

    case 2:
      fFit = new TF1("TwoGausPeriodicity", "[0]+[1]/TMath::Sqrt(2.*TMath::Pi())/[3]*TMath::Exp(-(x-[2])*(x-[2])/2./([3]*[3]))+[4]/TMath::Sqrt(2.*TMath::Pi())/[6]*TMath::Exp(-(x-[5])*(x-[5])/2./([6]*[6]))+[1]/TMath::Sqrt(2.*TMath::Pi())/[3]*TMath::Exp(-(x-2.*TMath::Pi()-[2])*(x-2.*TMath::Pi()-[2])/2./([3]*[3]))+[1]/TMath::Sqrt(2.*TMath::Pi())/[3]*TMath::Exp(-(x+2.*TMath::Pi()-[2])*(x+2.*TMath::Pi()-[2])/2./([3]*[3]))+[4]/TMath::Sqrt(2.*TMath::Pi())/[6]*TMath::Exp(-(x+2.*TMath::Pi()-[5])*(x+2.*TMath::Pi()-[5])/2./([6]*[6]))+[4]/TMath::Sqrt(2.*TMath::Pi())/[6]*TMath::Exp(-(x-2.*TMath::Pi()-[5])*(x-2.*TMath::Pi()-[5])/2./([6]*[6]))", fMinCorr, fMaxCorr);
      fGausNS = new TF1("fGausNSper", "[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-[1])*(x-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-2.*TMath::Pi()-[1])*(x-2.*TMath::Pi()-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x+2.*TMath::Pi()-[1])*(x+2.*TMath::Pi()-[1])/2./([2]*[2]))", fMinCorr, fMaxCorr);
      fGausAS = new TF1("fGausASper", "[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-[1])*(x-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-2.*TMath::Pi()-[1])*(x-2.*TMath::Pi()-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x+2.*TMath::Pi()-[1])*(x+2.*TMath::Pi()-[1])/2./([2]*[2]))", fMinCorr, fMaxCorr);
      fPed = new TF1("fPed", "[0]", fMinCorr, fMaxCorr);

      if (!fUseExternalPars) {
        fFit->SetParLimits(0, 0., 999.);
        fFit->SetParLimits(1, 0, 999.);
        fFit->SetParLimits(2, -0.55, 0.55);
        fFit->SetParLimits(3, 0, 3.14 / 3.);
        fFit->SetParLimits(4, 0, 999.);
        fFit->SetParLimits(5, 2.85, 3.55);
        fFit->SetParLimits(6, 0, 3.14 / 2.);

        fFit->SetParameter(0, 0.6);
        fFit->SetParameter(1, 3);
        fFit->SetParameter(2, 0.);
        fFit->SetParameter(3, 0.3);
        fFit->SetParameter(4, 2);
        fFit->SetParameter(5, TMath::Pi());
        fFit->SetParameter(6, 0.3);

      } else {
        for (int i = 0; i < fNpars; i++) {
          fFit->SetParameter(i, fExtParsVals[i]);
          fFit->SetParLimits(i, fExtParsLowBounds[i], fExtParsUppBounds[i]);
        }
      }

      fFit->SetParName(0, "ped");
      fFit->SetParName(1, "NS Y");
      fFit->SetParName(2, "NS mean");
      fFit->SetParName(3, "NS #sigma");
      fFit->SetParName(4, "AS Y");
      fFit->SetParName(5, "AS mean");
      fFit->SetParName(6, "AS #sigma");

      break;

    case 3:
      fFit = new TF1("OneGausPeriodicity", "[0]+[1]/TMath::Sqrt(2.*TMath::Pi())/[3]*TMath::Exp(-(x-[2])*(x-[2])/2./([3]*[3]))+[1]/TMath::Sqrt(2.*TMath::Pi())/[3]*TMath::Exp(-(x+2.*TMath::Pi()-[2])*(x+2.*TMath::Pi()-[2])/2./([3]*[3]))+[1]/TMath::Sqrt(2.*TMath::Pi())/[3]*TMath::Exp(-(x-2.*TMath::Pi()-[2])*(x-2.*TMath::Pi()-[2])/2./([3]*[3]))", fMinCorr, fMaxCorr);
      fGausAS = new TF1("fGausASper", "[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-[1])*(x-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-2.*TMath::Pi()-[1])*(x-2.*TMath::Pi()-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x+2.*TMath::Pi()-[1])*(x+2.*TMath::Pi()-[1])/2./([2]*[2]))", fMinCorr, fMaxCorr);
      fPed = new TF1("fPed", "[0]", fMinCorr, fMaxCorr);

      // TODO: add possibility to use external parameters
      // if(!fUseExternalPars) {
      fFit->SetParLimits(0, 0, 9999.);
      fFit->SetParLimits(1, 0, 999.);
      fFit->SetParLimits(2, 2., 4.);
      fFit->SetParLimits(3, 0, 3.14 / 2.);

      fFit->SetParameter(0, 3);
      fFit->SetParameter(1, 2);
      fFit->SetParameter(2, 3.14);
      fFit->SetParameter(3, 0.3);

      /*} else {
        for(int i=0; i<fNpars; i++) {
          fFit -> SetParameter(i, fExtParsVals[i]);
          fFit -> SetParLimits(i, fExtParsLowBounds[i], fExtParsUppBounds[i]);
        }
      }  */

      fFit->SetParName(0, "ped");
      fFit->SetParName(1, "AS Y");
      fFit->SetParName(2, "AS mean");
      fFit->SetParName(3, "AS #sigma");

      break;

    case 4:
      fFit = new TF1("kModifNSGausPeriodicity", "[0]+[1]*([7]*TMath::Sqrt(TMath::Gamma(3./[7]))/(2.*[3]*TMath::Power(TMath::Gamma(1./[7]),3./2.))*TMath::Exp(-TMath::Power(TMath::Abs(x-[2])*TMath::Sqrt(TMath::Gamma(3./[7]))/([3]*TMath::Sqrt(TMath::Gamma(1./[7]))),[7])))+[4]/TMath::Sqrt(2.*TMath::Pi())/[6]*TMath::Exp(-(x-[5])*(x-[5])/2./([6]*[6]))+[1]*([7]*TMath::Sqrt(TMath::Gamma(3./[7]))/(2.*[3]*TMath::Power(TMath::Gamma(1./[7]),3./2.))*TMath::Exp(-TMath::Power(TMath::Abs(x-2*TMath::Pi()-[2])*TMath::Sqrt(TMath::Gamma(3./[7]))/([3]*TMath::Sqrt(TMath::Gamma(1./[7]))),[7])))+[1]*([7]*TMath::Sqrt(TMath::Gamma(3./[7]))/(2.*[3]*TMath::Power(TMath::Gamma(1./[7]),3./2.))*TMath::Exp(-TMath::Power(TMath::Abs(x+2*TMath::Pi()-[2])*TMath::Sqrt(TMath::Gamma(3./[7]))/([3]*TMath::Sqrt(TMath::Gamma(1./[7]))),[7])))+[4]/TMath::Sqrt(2.*TMath::Pi())/[6]*TMath::Exp(-(x-2.*TMath::Pi()-[5])*(x-2.*TMath::Pi()-[5])/2./([6]*[6]))+[4]/TMath::Sqrt(2.*TMath::Pi())/[6]*TMath::Exp(-(x+2.*TMath::Pi()-[5])*(x+2.*TMath::Pi()-[5])/2./([6]*[6]))", fMinCorr, fMaxCorr);
      fGausNS = new TF1("fModGausNSper", "[0]*([3]*TMath::Sqrt(TMath::Gamma(3./[3]))/(2.*[2]*TMath::Power(TMath::Gamma(1./[3]),3./2.))*TMath::Exp(-TMath::Power(TMath::Abs(x-[1])*TMath::Sqrt(TMath::Gamma(3./[3]))/([2]*TMath::Sqrt(TMath::Gamma(1./[3]))),[3])))+[0]*([3]*TMath::Sqrt(TMath::Gamma(3./[3]))/(2.*[2]*TMath::Power(TMath::Gamma(1./[3]),3./2.))*TMath::Exp(-TMath::Power(TMath::Abs(x-2*TMath::Pi()-[1])*TMath::Sqrt(TMath::Gamma(3./[3]))/([2]*TMath::Sqrt(TMath::Gamma(1./[3]))),[3])))+[0]*([3]*TMath::Sqrt(TMath::Gamma(3./[3]))/(2.*[2]*TMath::Power(TMath::Gamma(1./[3]),3./2.))*TMath::Exp(-TMath::Power(TMath::Abs(x+2*TMath::Pi()-[1])*TMath::Sqrt(TMath::Gamma(3./[3]))/([2]*TMath::Sqrt(TMath::Gamma(1./[3]))),[3])))", fMinCorr, fMaxCorr);
      fGausAS = new TF1("fGausASper", "[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-[1])*(x-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-2.*TMath::Pi()-[1])*(x-2.*TMath::Pi()-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x+2.*TMath::Pi()-[1])*(x+2.*TMath::Pi()-[1])/2./([2]*[2]))", fMinCorr, fMaxCorr);
      fPed = new TF1("fPed", "[0]", fMinCorr, fMaxCorr);

      fFit->SetParLimits(0, 0., 999.);
      fFit->SetParLimits(1, 0.005, 25.);
      fFit->SetParLimits(2, -0.55, 0.55);
      fFit->SetParLimits(3, 0, 0.8);
      fFit->SetParLimits(4, 0.005, 25.);
      fFit->SetParLimits(5, 2.85, 3.55);
      fFit->SetParLimits(6, 0.05, 3.14 / 2.);
      fFit->SetParLimits(7, 0.5, 3.5);

      // default starting pars
      fFit->SetParameter(0, 1.);
      fFit->SetParameter(1, 1.);
      fFit->SetParameter(2, 0.);
      fFit->SetParameter(3, 0.3);
      fFit->SetParameter(4, 0.25);
      fFit->SetParameter(5, TMath::Pi());
      fFit->SetParameter(6, 0.3);
      fFit->SetParameter(7, 2);

      fFit->SetParName(0, "ped");
      fFit->SetParName(1, "NS Y");
      fFit->SetParName(2, "NS mean");
      fFit->SetParName(3, "NS #sigma");
      fFit->SetParName(4, "AS Y");
      fFit->SetParName(5, "AS mean");
      fFit->SetParName(6, "AS #sigma");
      fFit->SetParName(7, "NS shape par"); // beta of the gen. gaussian
      break;

    case 5:
      fFit = new TF1("kVonMises", "[0] +[1]/(2*TMath::Pi()*TMath::BesselI0([3]))*TMath::Exp([3]*TMath::Cos(x- 2*TMath::Pi() - [2])) + [4]/(2*TMath::Pi()*TMath::BesselI0([6]))*TMath::Exp([6]*TMath::Cos(x- 2*TMath::Pi()-[5]))", fMinCorr, fMaxCorr);
      fGausNS = new TF1("fVonMisesNS", "[0]/(2*TMath::Pi()*TMath::BesselI0([2]))*TMath::Exp([2]*TMath::Cos(x- 2*TMath::Pi() - [1]))", fMinCorr, fMaxCorr);
      fGausAS = new TF1("fVonMisesAS", "[0]/(2*TMath::Pi()*TMath::BesselI0([2]))*TMath::Exp([2]*TMath::Cos(x- 2*TMath::Pi()-[1]))", fMinCorr, fMaxCorr);
      fPed = new TF1("fPed", "[0]", fMinCorr, fMaxCorr);

      fFit->SetParLimits(0, 0., 999.);
      fFit->SetParLimits(1, 0.005, 25.);
      fFit->SetParLimits(2, -0.55, 0.55);
      fFit->SetParLimits(3, 0, 15.);
      fFit->SetParLimits(4, 0.005, 25.);
      fFit->SetParLimits(5, 2.85, 3.55);
      fFit->SetParLimits(6, 0., 15.);

      // default starting pars
      fFit->SetParameter(0, 3.5);
      fFit->SetParameter(1, 0.8);
      fFit->SetParameter(2, 0.);
      fFit->SetParameter(3, 1.);
      fFit->SetParameter(4, 1.5);
      fFit->SetParameter(5, TMath::Pi());
      fFit->SetParameter(6, 1.);

      fFit->SetParName(0, "ped");
      fFit->SetParName(1, "NS Y");
      fFit->SetParName(2, "NS mean");
      fFit->SetParName(3, "NS #sigma");
      fFit->SetParName(4, "AS Y");
      fFit->SetParName(5, "AS mean");
      fFit->SetParName(6, "AS #sigma");

      break;

    case 6:
      fFit = new TF1("kSingleVonMises", "[0] +[1]/(2*TMath::Pi()*TMath::BesselI0([3]))*TMath::Exp([3]*TMath::Cos(x- 2*TMath::Pi() - [2]))", fMinCorr, fMaxCorr);
      fGausAS = new TF1("fVonMisesAS", "[0]/(2*TMath::Pi()*TMath::BesselI0([2]))*TMath::Exp([2]*TMath::Cos(x- 2*TMath::Pi()-[1]))", fMinCorr, fMaxCorr);
      fPed = new TF1("fPed", "[0]", fMinCorr, fMaxCorr);

      fFit->SetParLimits(0, 0., 999.);
      fFit->SetParLimits(1, 0.005, 25.);
      fFit->SetParLimits(2, 2.85, 3.55);
      fFit->SetParLimits(3, 0., 15.);

      // default starting pars
      fFit->SetParameter(0, 3.5);
      fFit->SetParameter(1, 1.5);
      fFit->SetParameter(2, TMath::Pi());
      fFit->SetParameter(3, 1.);

      fFit->SetParName(0, "ped");
      fFit->SetParName(1, "AS Y");
      fFit->SetParName(2, "AS mean");
      fFit->SetParName(3, "AS #sigma");

      break;

    case 7: // case 2 Gaus w periodicity + v2 modulation

      fFit = new TF1("kTwoGausPeriodicityPlusV2modulation", "[1]/TMath::Sqrt(2.*TMath::Pi())/[3]*TMath::Exp(-(x-[2])*(x-[2])/2./([3]*[3]))+[4]/TMath::Sqrt(2.*TMath::Pi())/[6]*TMath::Exp(-(x-[5])*(x-[5])/2./([6]*[6]))+[1]/TMath::Sqrt(2.*TMath::Pi())/[3]*TMath::Exp(-(x-2.*TMath::Pi()-[2])*(x-2.*TMath::Pi()-[2])/2./([3]*[3]))+[1]/TMath::Sqrt(2.*TMath::Pi())/[3]*TMath::Exp(-(x+2.*TMath::Pi()-[2])*(x+2.*TMath::Pi()-[2])/2./([3]*[3]))+[4]/TMath::Sqrt(2.*TMath::Pi())/[6]*TMath::Exp(-(x+2.*TMath::Pi()-[5])*(x+2.*TMath::Pi()-[5])/2./([6]*[6]))+[4]/TMath::Sqrt(2.*TMath::Pi())/[6]*TMath::Exp(-(x-2.*TMath::Pi()-[5])*(x-2.*TMath::Pi()-[5])/2./([6]*[6]))+[0]*(1+2*[7]*[8]*TMath::Cos(2*x))", fMinCorr, fMaxCorr);
      fGausNS = new TF1("fGausNSper", "[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-[1])*(x-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-2.*TMath::Pi()-[1])*(x-2.*TMath::Pi()-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x+2.*TMath::Pi()-[1])*(x+2.*TMath::Pi()-[1])/2./([2]*[2]))", fMinCorr, fMaxCorr);
      fGausAS = new TF1("fGausASper", "[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-[1])*(x-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x-2.*TMath::Pi()-[1])*(x-2.*TMath::Pi()-[1])/2./([2]*[2]))+[0]/TMath::Sqrt(2.*TMath::Pi())/[2]*TMath::Exp(-(x+2.*TMath::Pi()-[1])*(x+2.*TMath::Pi()-[1])/2./([2]*[2]))", fMinCorr, fMaxCorr);
      fPed = new TF1("fPedv2Mod", "[0]*(1+2*[1]*[2]*TMath::Cos(2*x))", fMinCorr, fMaxCorr);

      fFit->SetParLimits(0, 0., 999.);
      fFit->SetParLimits(1, 0, 999.);
      fFit->SetParLimits(2, -0.55, 0.55);
      fFit->SetParLimits(3, 0, 3.14 / 3.);
      fFit->SetParLimits(4, 0, 999.);
      fFit->SetParLimits(5, 2.85, 3.55);
      fFit->SetParLimits(6, 0, 3.14 / 2.);
      fFit->SetParLimits(7, -1, 1);
      fFit->SetParLimits(8, -1, 1);

      fFit->FixParameter(0, fBaseline);
      fFit->SetParameter(1, 3);
      fFit->SetParameter(2, 0.);
      fFit->SetParameter(3, 0.3);
      fFit->SetParameter(4, 2);
      fFit->SetParameter(5, TMath::Pi());
      fFit->SetParameter(6, 0.3);
      fFit->SetParameter(7, 0);
      fFit->SetParameter(8, 0);

      if (fUseExternalPars) { // overwrites previous configuration
        for (int i = 0; i < fNpars; i++) {
          fFit->SetParameter(i, fExtParsVals[i]);
          fFit->SetParLimits(i, fExtParsLowBounds[i], fExtParsUppBounds[i]);
        }
      }

      fFit->FixParameter(7, fv2AssocPart);
      fFit->FixParameter(8, fv2Dmeson);

      fPed->FixParameter(0, fBaseline);
      fPed->FixParameter(1, fv2AssocPart);
      fPed->FixParameter(2, fv2Dmeson);

      fFit->SetParName(0, "ped");
      fFit->SetParName(1, "NS Y");
      fFit->SetParName(2, "NS mean");
      fFit->SetParName(3, "NS #sigma");
      fFit->SetParName(4, "AS Y");
      fFit->SetParName(5, "AS mean");
      fFit->SetParName(6, "AS #sigma");
      fFit->SetParName(7, "v_{2} hadron");
      fFit->SetParName(8, "v_{2} D meson");
      break;

      case 8: // case 2 Gaus w periodicity + v2 modulation (delta) low mult

      fFit = new TF1("fTotFit", "[0] + [1]* (1 + 2*[2]*TMath::Cos(x) + 2*[3]*TMath::Cos(2*x) + 2*[4]*TMath::Cos(3*x))", fMinCorr, fMaxCorr);
      fv1 = new TF1("fv1", "[0] + 2*[0]*[1]*TMath::Cos(x)", fMinCorr, fMaxCorr);
      fv2 = new TF1("fv2", "[0] + 2*[0]*[1]*TMath::Cos(2*x)", fMinCorr, fMaxCorr);
      fv3 = new TF1("fv3", "[0] + 2*[0]*[1]*TMath::Cos(3*x)", fMinCorr, fMaxCorr);
      fPed = new TF1("fPed", "[0]", fMinCorr, fMaxCorr);
      // fFit = new TF1("fTotFit", "[0] + [1] / 2 / TMath::Pi() * (1 + 2*[2]*TMath::Cos(x) + 2*[3]*TMath::Cos(2*x) + 2*[4]*TMath::Cos(3*x))", fMinCorr, fMaxCorr);
      // fv1 = new TF1("fv1", "[1]  / 2 / TMath::Pi() * 2*[2]*TMath::Cos(x)", fMinCorr, fMaxCorr);
      // fv2 = new TF1("fv2", "[1]  / 2 / TMath::Pi() * 2*[3]*TMath::Cos(2*x)", fMinCorr, fMaxCorr);
      // fv3 = new TF1("fv3", "[1]  / 2 / TMath::Pi() * 2*[4]*TMath::Cos(3*x)", fMinCorr, fMaxCorr);
      // fv4 = new TF1("fv4", "2*[0]*[1]/2/TMath::Pi()*TMath::Cos(4*x)", fMinCorr, fMaxCorr);
      // fv5 = new TF1("fv5", "2*[0]*[1]*TMath::Cos(5*x)", fMinCorr, fMaxCorr);
      /*fv4 = new TF1("fv4", "2*[0]*[1]*TMath::Cos(4*x)", fMinCorr, fMaxCorr);
      fv5 = new TF1("fv5", "2*[0]*[1]*TMath::Cos(5*x)", fMinCorr, fMaxCorr);*/

      if (fUseExternalPars) { // overwrites previous configuration
        for (int i = 0; i < fNpars; i++) {
          fFit->SetParameter(i, fExtParsVals[i]);
          fFit->SetParLimits(i, fExtParsLowBounds[i], fExtParsUppBounds[i]);
        }
      }

      fFit->FixParameter(0, 0.0);
      fFit->SetParameter(1, 0.5);
      // Double_t ave = 0.;
      // for (int i = 1; i < fHist->GetNbinsX(); i++) {
      //   ave += fHist->GetBinContent(i);
      // }
      // ave /= fHist->GetNbinsX();
      // std::cout << "Average bin content: " << ave << std::endl;
      // fFit->FixParameter(1, ave);

      fFit->SetParName(0, "czyam");
      fFit->SetParName(1, "ped");
      fFit->SetParName(2, "v_{1}");
      fFit->SetParName(3, "v_{2} delta");
      fFit->SetParName(4, "v_{3}");
      /*fFit->SetParName(5, "v_{4}");
      fFit->SetParName(6, "v_{5}");*/

      break;

      case 9: // case 2 Gaus w periodicity + v2 modulation (delta) high mult

      fFit = new TF1("fTotFit", "[0]+[1]*(1 + 2*[2]*TMath::Cos(x) + 2*[3] *TMath::Cos(2*x) + 2*[4] *TMath::Cos(3*x)) + [5]*([6]+[7]*(1 + 2*[8]*TMath::Cos(x) + 2*[9]*TMath::Cos(2*x) + 2*[10]*TMath::Cos(3*x)))", fMinCorr, fMaxCorr);

      if (fUseExternalPars) { // overwrites previous configuration
        for (int i = 0; i < fNpars; i++) {
          fFit->SetParameter(i, fExtParsVals[i]);
          fFit->SetParLimits(i, fExtParsLowBounds[i], fExtParsUppBounds[i]);
        }
      }

      fFit->FixParameter(0, 0.0);
      fFit->SetParameter(1, fBaseline);

      fFit->SetParName(0, "czyam");
      fFit->SetParName(1, "ped");
      fFit->SetParName(2, "v_{1}");
      fFit->SetParName(3, "v_{2} delta");
      fFit->SetParName(4, "v_{3}");
      /*fFit->SetParName(5, "v_{4}");
      fFit->SetParName(6, "v_{5}");*/
      fFit->SetParName(5, "scaling");
      fFit->SetParName(6, "czyam_lm");
      fFit->SetParName(7, "ped_lm");
      fFit->SetParName(8, "v_{1}_lm");
      fFit->SetParName(9, "v_{2}_lm delta");
      fFit->SetParName(10, "v_{3}_lm");
      /*fFit->SetParName(13, "v_{4}_lm");
      fFit->SetParName(14, "v_{5}_lm");*/

      case 10: // template fit
      // Build LM template based on tempFunc (delegated to BuildLMOutput)
      BuildLMOutput();
      fFit = new TF1("fTotFit", this, &DhCorrelationFitter::TotalTemplateFitFunction, fMinCorr, fMaxCorr, 4);
      
      fFit->SetParName(0, "F (LM Scale)");
      fFit->SetParName(1, "G (Flow Scale)");
      fFit->SetParName(2, "v_{2} delta");
      fFit->SetParName(3, "v_{3} delta");
      // fFit->SetParName(4, "v_{1} delta");
      // Set initial F (LM Scale) from raw-yield ratio
      // if (fRyTrigger > 0 && fLMRyTrigger > 0) {
      //   fFit->SetParameter(0, fRyTrigger / fLMRyTrigger);
      // } else {
      //   fFit->SetParameter(0, 1.1);
      // }
      // Limits from actual raw-yield uncertainties:
      //   max = (fRyTrigger + fRyTriggerErr) / (fLMRyTrigger - fLMRyTriggerErr)
      //   min = (fRyTrigger - fRyTriggerErr) / (fLMRyTrigger + fLMRyTriggerErr)
      if (fRyTrigger > 0 && fLMRyTrigger > 0) {
        Double_t ryErr = fRyTriggerErr;
        Double_t lmErr = fLMRyTriggerErr;
        Double_t tempAvgPairs = fTempHisto->Integral() / fTempHisto->GetNbinsX(); // average bin content of template

        Double_t flowAvgPairs = fHist->Integral() / fHist->GetNbinsX(); // average bin content of data
        printf("[INFO] DhCorrelationFitter: total pairs: data=%.3f  temp=%.3f\n", fHist->Integral(), fTempHisto->Integral());
        printf("[INFO] DhCorrelationFitter: bins: data=%d  temp=%d\n", fHist->GetNbinsX(), fTempHisto->GetNbinsX());
        printf("[INFO] DhCorrelationFitter: average pairs per bin: data=%.3f  temp=%.3f\n", flowAvgPairs, tempAvgPairs);
        Double_t normalizedMean = flowAvgPairs / fRyTrigger;
        Double_t normalizedTempMean = tempAvgPairs / fLMRyTrigger;
        printf("[INFO] DhCorrelationFitter: normalized average pairs per trigger: data=%.3f  temp=%.3f  (data/template = %.3f)\n",
              normalizedMean, normalizedTempMean, normalizedMean / normalizedTempMean);

        // fFit->SetParameter(0, normalizedMean / normalizedTempMean);
        fFit->SetParameter(0, fRyTrigger / fLMRyTrigger);
        fFit->SetParLimits(0, 0., 999.);
        if (fFixLMFactor) {
          fFit->FixParameter(0, fFit->GetParameter(0));
          printf("[INFO] DhCorrelationFitter: FixLMFactor is true -> fixing LM scale factor (par 0) to %.6f\n",
                fFit->GetParameter(0));
        }
        if (ryErr > 0 && lmErr > 0) {
          Double_t numHi = fRyTrigger + ryErr;
          Double_t numLo = fRyTrigger - ryErr;
          Double_t denHi = fLMRyTrigger + lmErr;
          Double_t denLo = fLMRyTrigger - lmErr;
          Double_t fLo = numLo / denHi;  // conservative lower bound
          Double_t fHi = numHi / denLo;  // conservative upper bound
          if (fLo < 0.01) fLo = 0.01;
          printf("[INFO][NOT DONE] DhCorrelationFitter: F scale init=%.3f  limits=[%.3f, %.3f]  (ryRatio, errors from fit)\n",
                normalizedMean / normalizedTempMean, fLo, fHi);
          // fFit->SetParLimits(0, fLo, fHi);
        }
      } else {
        std::cout << "[WARNING] DhCorrelationFitter: Cannot set initial F (LM Scale) from raw-yield ratio due to non-positive values (fRyTrigger=" << fRyTrigger << ", fLMRyTrigger=" << fLMRyTrigger << "). Setting default initial value to 1.1." << std::endl;
        fFit->SetParameter(0, 1.1);
      }

      fFit->SetParameter(1, fHist->GetMinimum());
      fFit->SetParameter(2, 0.05);
      fFit->SetParameter(3, 0.01);
      // fFit->SetParameter(4, 0.01);

      fLM = new TF1("fLM", this, &DhCorrelationFitter::TemplateFitFunctionWPed, fMinCorr, fMaxCorr, 2); // LM
      fFlow = new TF1("fFlow", "[0] * (1 + 2*[1]*TMath::Cos(2*x) + 2*[2]*TMath::Cos(3*x))", fMinCorr, fMaxCorr); // HM
      fv2 = new TF1("fv2", "[0] + 2*[0]*[1]*TMath::Cos(2*x)", fMinCorr, fMaxCorr); // v2 component
      fv3 = new TF1("fv3", "[0] + 2*[0]*[1]*TMath::Cos(3*x)", fMinCorr, fMaxCorr); // v3 component
      fPed = new TF1("fPed", "[0]", fMinCorr, fMaxCorr); // Pedestal

      break;
  }
}

void DhCorrelationFitter::intepolateTemp()
{
  if (!fTempHisto) {
    Printf("[ERROR] No template histogram set for template fit! Please set it with SetTemplateHisto(TH1* histo) method");
    return;
  }
  
  if (fGraph) { delete fGraph; fGraph = nullptr; }
  if (fSpline) { delete fSpline; fSpline = nullptr; }

  fTempHisto->Smooth(1);
  fGraph = new TGraphErrors(fTempHisto->GetNbinsX());
  for (int i = 1; i <= fTempHisto->GetNbinsX(); i++) {
    Double_t val = fTempHisto->GetBinContent(i);
    Double_t valErr = fTempHisto->GetBinError(i);
    Double_t xErr = fTempHisto->GetBinWidth(i) / 2.0;
    fGraph->SetPoint(i - 1, fTempHisto->GetBinCenter(i), val);
    fGraph->SetPointError(i - 1, xErr, valErr);
  }

  fSpline = new TSpline3("tempSpline", fGraph);
}

void DhCorrelationFitter::BuildLMOutput()
{
  if (!fTempHisto) {
    Printf("[ERROR] BuildLMOutput: No template histogram set");
    return;
  }
  // ---- Clean up previous outputs ----
  if (fLMOutput) { delete fLMOutput; fLMOutput = 0; }
  if (fTemplateFunc) { delete fTemplateFunc; fTemplateFunc = 0; }
  if (fSpline) { delete fSpline; fSpline = 0; }
  if (fGraph) { delete fGraph; fGraph = 0; }

  printf("[INFO] total pairs in template histogram: %.3f\n", fTempHisto->Integral());
  printf("[INFO] Scaled total pairs in template histogram (scaled by raw-yield ratio): %.3f\n", fTempHisto->Integral() * fPairsPerMassBin);
  printf("[INFO] total pairs in data histogram: %.3f\n", fHist->Integral());

  // ---- Build template based on tempFunc ----
  if (fTempFunc == 0) {
    // --- Raw histogram template: no smooth, no spline, direct bin lookup ---
    if (fFixBase != 0 && fTempHisto) {
      TH1F* savedHist = fHist;
      TH1F* tmpHist = new TH1F("tmpLMBaseline", "", fTempHisto->GetNbinsX(),
                                fTempHisto->GetXaxis()->GetXmin(),
                                fTempHisto->GetXaxis()->GetXmax());
      for (int i = 1; i <= fTempHisto->GetNbinsX(); i++) {
        tmpHist->SetBinContent(i, fTempHisto->GetBinContent(i));
        tmpHist->SetBinError(i, fTempHisto->GetBinError(i));
      }
      fHist = tmpHist;
      FindBaseline();
      printf("[INFO] DhCorrelationFitter: LM template baseline = %.6f +/- %.6f (subtracted from template)\n",
              fBaseline, fErrBaseline);
      fHist = savedHist;
      delete tmpHist;
    }
    printf("[INFO] BuildLMOutput: Using raw histogram as LM template (tempFunc=0)\n");

  } else if (fTempFunc == 1) {
    // --- Raw + spline: no smooth, build spline for interpolation ---------
    if (fFixBase != 0 && fTempHisto) {
      TH1F* savedHist = fHist;
      TH1F* tmpHist = new TH1F("tmpLMBaseline", "", fTempHisto->GetNbinsX(),
                                fTempHisto->GetXaxis()->GetXmin(),
                                fTempHisto->GetXaxis()->GetXmax());
      for (int i = 1; i <= fTempHisto->GetNbinsX(); i++) {
        tmpHist->SetBinContent(i, fTempHisto->GetBinContent(i));
        tmpHist->SetBinError(i, fTempHisto->GetBinError(i));
      }
      fHist = tmpHist;
      FindBaseline();
      printf("[INFO] DhCorrelationFitter: LM template baseline = %.6f +/- %.6f (subtracted from template)\n",
              fBaseline, fErrBaseline);
      fHist = savedHist;
      delete tmpHist;
    }
    // Build graph + spline from raw histogram (no Smooth)
    fGraph = new TGraphErrors(fTempHisto->GetNbinsX());
    for (int i = 1; i <= fTempHisto->GetNbinsX(); i++) {
      fGraph->SetPoint(i - 1, fTempHisto->GetBinCenter(i), fTempHisto->GetBinContent(i));
      fGraph->SetPointError(i - 1, fTempHisto->GetBinWidth(i) / 2.0, fTempHisto->GetBinError(i));
    }
    fSpline = new TSpline3("tempSpline", fGraph);
    printf("[INFO] BuildLMOutput: Using raw+spline LM template (tempFunc=1)\n");

  } else if (fTempFunc == 2) {
    // --- Smooth + spline --------------------------------------------------
    if (fFixBase != 0 && fTempHisto) {
      TH1F* savedHist = fHist;
      TH1F* tmpHist = new TH1F("tmpLMBaseline", "", fTempHisto->GetNbinsX(),
                                fTempHisto->GetXaxis()->GetXmin(),
                                fTempHisto->GetXaxis()->GetXmax());
      for (int i = 1; i <= fTempHisto->GetNbinsX(); i++) {
        tmpHist->SetBinContent(i, fTempHisto->GetBinContent(i));
        tmpHist->SetBinError(i, fTempHisto->GetBinError(i));
      }
      fHist = tmpHist;
      FindBaseline();
      printf("[INFO] DhCorrelationFitter: LM template baseline = %.6f +/- %.6f (subtracted from template)\n",
              fBaseline, fErrBaseline);
      fHist = savedHist;
      delete tmpHist;
    }
    // Smooth then build spline
    fTempHisto->Smooth(1);
    fGraph = new TGraphErrors(fTempHisto->GetNbinsX());
    for (int i = 1; i <= fTempHisto->GetNbinsX(); i++) {
      fGraph->SetPoint(i - 1, fTempHisto->GetBinCenter(i), fTempHisto->GetBinContent(i));
      fGraph->SetPointError(i - 1, fTempHisto->GetBinWidth(i) / 2.0, fTempHisto->GetBinError(i));
    }
    fSpline = new TSpline3("tempSpline", fGraph);
    printf("[INFO] BuildLMOutput: Using smooth+spline LM template (tempFunc=2)\n");

  } else if (fTempFunc == 3) {
    // --- Fit Gaus to LM template ------------------------------------------
    TVirtualFitter::SetMaxIterations(50000);
    Double_t minVal = fTempHisto->GetMinimum();
    Double_t maxVal = fTempHisto->GetMaximum();
    TF1* fGausFit = new TF1("fGausFit",
      "[0]+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-TMath::Power(x,2)/(2*TMath::Power([2],2)))+[3]/(TMath::Sqrt(2*TMath::Pi())*[4])*TMath::Exp(-TMath::Power(x-TMath::Pi(),2)/(2*TMath::Power([4],2)))",
      fMinCorr, fMaxCorr);
    fGausFit->SetParameter(0, minVal);
    fGausFit->SetParLimits(0, 0, maxVal*2);
    fGausFit->SetParameters(1, 0.5*(maxVal-minVal));
    fGausFit->SetParLimits(1, 0.01, maxVal-minVal);
    fGausFit->SetParameter(2, TMath::Pi()/8);
    fGausFit->SetParLimits(2, TMath::Pi()/16, TMath::Pi());
    fGausFit->SetParameter(3, 0.5*(maxVal-minVal));
    fGausFit->SetParLimits(3, 0.01, maxVal-minVal);
    fGausFit->SetParameter(4, TMath::Pi()/8);
    fGausFit->SetParLimits(4, TMath::Pi()/16, TMath::Pi());
    fTempHisto->Fit(fGausFit, "RIS", "", fMinCorr, fMaxCorr);

    fTemplateFunc = static_cast<TF1*>(fTempHisto->GetFunction("fGausFit")->Clone("fLMTemplateFunc"));
    fTemplateFunc->SetRange(fMinCorr, fMaxCorr);
    fBaseline = fTemplateFunc->GetMinimum(fMinCorr, fMaxCorr);
    fErrBaseline = 0.0;
    printf("[INFO] BuildLMOutput: Fitted Gaus LM template (tempFunc=3) baseline=%.6f\n", fBaseline);
    delete fGausFit;

  } else if (fTempFunc == 4) {
    // --- Fit GausPeriodic to LM template ----------------------------------
    TVirtualFitter::SetMaxIterations(50000);
    Double_t minVal = fTempHisto->GetMinimum();
    Double_t maxVal = fTempHisto->GetMaximum();
    TF1* fGausPerFit = new TF1("fGausPerFit",
      "[0]+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-x*x/(2*[2]*[2]))+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-(x-2*TMath::Pi())*(x-2*TMath::Pi())/(2*[2]*[2]))+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-(x+2*TMath::Pi())*(x+2*TMath::Pi())/(2*[2]*[2]))+[3]/(TMath::Sqrt(2*TMath::Pi())*[4])*TMath::Exp(-(x-TMath::Pi())*(x-TMath::Pi())/(2*[4]*[4]))+[3]/(TMath::Sqrt(2*TMath::Pi())*[4])*TMath::Exp(-(x-3*TMath::Pi())*(x-3*TMath::Pi())/(2*[4]*[4]))+[3]/(TMath::Sqrt(2*TMath::Pi())*[4])*TMath::Exp(-(x+TMath::Pi())*(x+TMath::Pi())/(2*[4]*[4]))",
      fMinCorr, fMaxCorr);
    fGausPerFit->SetParameter(0, minVal);
    fGausPerFit->SetParLimits(0, 0, maxVal*2);
    fGausPerFit->SetParameters(1, 0.5*(maxVal-minVal));
    fGausPerFit->SetParLimits(1, 0.01, maxVal-minVal);
    fGausPerFit->SetParameter(2, TMath::Pi()/8);
    fGausPerFit->SetParLimits(2, TMath::Pi()/16, TMath::Pi());
    fGausPerFit->SetParameter(3, 0.5*(maxVal-minVal));
    fGausPerFit->SetParLimits(3, 0.01, maxVal-minVal);
    fGausPerFit->SetParameter(4, TMath::Pi()/8);
    fGausPerFit->SetParLimits(4, TMath::Pi()/16, TMath::Pi());
    fTempHisto->Fit(fGausPerFit, "RIS", "", fMinCorr, fMaxCorr);

    fTemplateFunc = static_cast<TF1*>(fTempHisto->GetFunction("fGausPerFit")->Clone("fLMTemplateFunc"));
    fTemplateFunc->SetRange(fMinCorr, fMaxCorr);
    fBaseline = fTemplateFunc->GetMinimum(fMinCorr, fMaxCorr);
    fErrBaseline = 0.0;
    printf("[INFO] BuildLMOutput: Fitted GausPeriodic LM template (tempFunc=4) baseline=%.6f\n", fBaseline);
    delete fGausPerFit;
  }

  // ---- Build fLMOutput (clone of original template data for drawing) ----
  fLMOutput = (TH1D*)fTempHisto->Clone("fLMOutput");
  fLMOutput->SetDirectory(0);
  fLMOutput->SetStats(0);
  fLMOutput->GetListOfFunctions()->Clear(); // remove fit function from output histogram

  // // ---- Scale template to match data yield (based on fPairsPerMassBin) ----
  // if (fPairsPerMassBin > 0) {
  //   Double_t scale = fPairsPerMassBin;
  //   printf("[INFO] BuildLMOutput: Scaling template by fPairsPerMassBin=%.6f\n", scale);

  //   // Scale the fLMOutput histogram (clone of fTempHisto) — show scaled data in plot
  //   fLMOutput->Scale(scale);

  //   if (fTempFunc == 1 || fTempFunc == 2) {
  //     // Rebuild graph and spline from scaled histogram contents
  //     if (fGraph) { delete fGraph; fGraph = nullptr; }
  //     if (fSpline) { delete fSpline; fSpline = nullptr; }
  //     fGraph = new TGraphErrors(fTempHisto->GetNbinsX());
  //     for (int i = 1; i <= fTempHisto->GetNbinsX(); i++) {
  //       fGraph->SetPoint(i - 1, fTempHisto->GetBinCenter(i), fTempHisto->GetBinContent(i) * scale);
  //       fGraph->SetPointError(i - 1, fTempHisto->GetBinWidth(i) / 2.0, fTempHisto->GetBinError(i) * scale);
  //     }
  //     fSpline = new TSpline3("tempSpline", fGraph);
  //     fBaseline *= scale;
  //     fErrBaseline *= scale;
  //   } else if (fTempFunc == 3 || fTempFunc == 4) {
  //     // Scale ALL parameters so f(x) = scale * original_f(x)
  //     fTemplateFunc->SetParameter(0, fTemplateFunc->GetParameter(0) * scale);
  //     fTemplateFunc->SetParameter(1, fTemplateFunc->GetParameter(1) * scale);
  //     fTemplateFunc->SetParameter(3, fTemplateFunc->GetParameter(3) * scale);
  //     fBaseline *= scale;
  //     fErrBaseline *= scale;
  //   }
  // }

  // ---- Add sub-component functions (near-side, away-side) for drawing ----
  // Created AFTER scaling so they inherit scaled parameter values
  if (fTempFunc == 1 || fTempFunc == 2) {
    if (fSpline) fLMOutput->GetListOfFunctions()->Add(fSpline);
  }
  if (fTempFunc == 3 && fTemplateFunc) {
    fLMOutput->GetListOfFunctions()->Add(fTemplateFunc);
    // Gaus: near = BL + A_N*Gaus(0,sig_N), away = BL + A_A*Gaus(pi,sig_A)
    TF1* fNear = new TF1("fNearGaus",
      "[0]+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-x*x/(2*[2]*[2]))",
      fMinCorr, fMaxCorr);
    fNear->SetParameters(fTemplateFunc->GetParameter(0),
                         fTemplateFunc->GetParameter(1),
                         fTemplateFunc->GetParameter(2));
    fNear->SetLineColor(kAzure+2);
    fNear->SetLineWidth(2);
    fNear->SetLineStyle(9);
    fLMOutput->GetListOfFunctions()->Add(fNear);

    TF1* fAway = new TF1("fAwayGaus",
      "[0]+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-(x-TMath::Pi())*(x-TMath::Pi())/(2*[2]*[2]))",
      fMinCorr, fMaxCorr);
    fAway->SetParameters(fTemplateFunc->GetParameter(0),
                         fTemplateFunc->GetParameter(3),
                         fTemplateFunc->GetParameter(4));
    fAway->SetLineColor(kGreen+2);
    fAway->SetLineWidth(2);
    fAway->SetLineStyle(9);
    fLMOutput->GetListOfFunctions()->Add(fAway);
  } else if (fTempFunc == 4 && fTemplateFunc) {
    // GausPeriodic: near = BL + A_N*[Gaus(0)+Gaus(2pi)+Gaus(-2pi)]
    TF1* fNear = new TF1("fNearGausPer",
      "[0]+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-x*x/(2*[2]*[2]))+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-(x-2*TMath::Pi())*(x-2*TMath::Pi())/(2*[2]*[2]))+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-(x+2*TMath::Pi())*(x+2*TMath::Pi())/(2*[2]*[2]))",
      fMinCorr, fMaxCorr);
    fNear->SetParameters(fTemplateFunc->GetParameter(0),
                         fTemplateFunc->GetParameter(1),
                         fTemplateFunc->GetParameter(2));
    fNear->SetLineColor(kAzure+2);
    fNear->SetLineWidth(2);
    fNear->SetLineStyle(9);
    fLMOutput->GetListOfFunctions()->Add(fNear);

    TF1* fAway = new TF1("fAwayGausPer",
      "[0]+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-(x-TMath::Pi())*(x-TMath::Pi())/(2*[2]*[2]))+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-(x-3*TMath::Pi())*(x-3*TMath::Pi())/(2*[2]*[2]))+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])*TMath::Exp(-(x+TMath::Pi())*(x+TMath::Pi())/(2*[2]*[2]))",
      fMinCorr, fMaxCorr);
    fAway->SetParameters(fTemplateFunc->GetParameter(0),
                         fTemplateFunc->GetParameter(3),
                         fTemplateFunc->GetParameter(4));
    fAway->SetLineColor(kGreen+2);
    fAway->SetLineWidth(2);
    fAway->SetLineStyle(9);
    fLMOutput->GetListOfFunctions()->Add(fAway);
  }
  printf("[INFO] DhCorrelationFitter: BuildLMOutput completed for tempFunc=%d\n", fTempFunc);
}

Double_t DhCorrelationFitter::TemplateFitFunction(Double_t* x, Double_t* par)
{
  Double_t templateVal = 0.0;
  if (fTempFunc == 0) {
    // Raw histogram: direct bin lookup
    if (!fTempHisto) return 0.0;
    templateVal = fTempHisto->GetBinContent(fTempHisto->FindBin(x[0]));
  } else if (fTempFunc == 1 || fTempFunc == 2) {
    // Spline interpolation
    if (!fSpline) return 0.0;
    templateVal = fSpline->Eval(x[0]);
  } else if (fTempFunc == 3 || fTempFunc == 4) {
    // Gaus/GausPeriodic function evaluation
    if (!fTemplateFunc) return 0.0;
    templateVal = fTemplateFunc->Eval(x[0]);
  }

  if (fWithPedLM) {
    return par[0] * templateVal;
  } else {
    if (templateVal < fBaseline) return 0.0;
    return par[0] * (templateVal - fBaseline);
  }
}

Double_t DhCorrelationFitter::TemplateFitFunctionWPed(Double_t* x, Double_t* par)
{
  Double_t templateVal = 0.0;
  if (fTempFunc == 0) {
    if (!fTempHisto) return 0.0;
    templateVal = fTempHisto->GetBinContent(fTempHisto->FindBin(x[0]));
  } else if (fTempFunc == 1 || fTempFunc == 2) {
    if (!fSpline) return 0.0;
    templateVal = fSpline->Eval(x[0]);
  } else if (fTempFunc == 3 || fTempFunc == 4) {
    if (!fTemplateFunc) return 0.0;
    templateVal = fTemplateFunc->Eval(x[0]);
  }

  if (fWithPedLM) {
    return par[0] * templateVal + par[1];
  } else {
    return par[0] * (templateVal - fBaseline) + par[1];
  }
}

Double_t DhCorrelationFitter::TotalTemplateFitFunction(Double_t* x, Double_t* par)
{
    // par[0] = F (template scaling factor)
    // par[1] = G (flow scaling factor)
    // par[2] = V_{2\Delta}
    // par[3] = V_{3\Delta}

    Double_t dphi = x[0];

    // par[0] * fSpline->Eval(x[0])
    Double_t template_term = TemplateFitFunction(x, par); 

    Double_t flow_term = par[1] * (1.0 + 2.0 * par[2] * TMath::Cos(2.0 * dphi) 
                                       + 2.0 * par[3] * TMath::Cos(3.0 * dphi)
                                      //  + 2.0 * par[4] * TMath::Cos(1.0 * dphi)
                                       );
    return template_term + flow_term;
}

void DhCorrelationFitter::SetPointsForBaseline(Int_t nBaselinePoints, Int_t* binsBaseline)
{

  fNbasleinePoints = nBaselinePoints;

  fBinsBaseline = new Int_t[fNbasleinePoints];

  for (int i = 0; i < fNbasleinePoints; i++) {
    fBinsBaseline[i] = binsBaseline[i];
  }

  return;
}

Double_t DhCorrelationFitter::FindBaseline()
{

  // baseline free
  if (fFixBase == 0) {
    Printf("[INFO] DhCorrelationFitter::FindBasline(). The baseline option is set to free baseline: now the full fit will be done. Beware!");
    Fitting(); // TODO: not sure
    return fBaseline;
  }

  // fix the baseline to the minimum of the histogram
  if (fFixBase == 1) {
    Double_t min = 1.e10;
    Int_t iBin = -1;
    for (Int_t j = 1; j <= fHist->GetNbinsX(); j++) {
      if (fHist->GetBinContent(j) < min) {
        min = fHist->GetBinContent(j);
        iBin = j;
      }
    }
    fBaseline = min;
    fErrBaseline = fHist->GetBinError(iBin);

    if (fShiftBaselineUp) {
      fBaseline += fErrBaseline;
      printf("[INFO] Shift baseline up of its statistical uncertainty");
    }

    if (fShiftBaselineDown) {
      fBaseline -= fErrBaseline;
      printf("[INFO] Shift baseline down of its statistical uncertainty");
    }

    return fBaseline;
  }

  // fix the baseline to the weighted average of the abs(fFixBaseline) lower points
  if (fFixBase < 0) {
    Int_t npointsAv = TMath::Abs(fFixBase);
    Int_t* ind = new Int_t[fHist->GetNbinsX()];
    Float_t* hval = new Float_t[fHist->GetNbinsX()];
    for (Int_t k = 1; k <= fHist->GetNbinsX(); k++) {
      hval[k - 1] = fHist->GetBinContent(k);
    }
    Double_t errAv = 0., Av = 0.;
    TMath::Sort(fHist->GetNbinsX(), hval, ind, kFALSE); //  KFALSE -> increasing order
    // Average of abs(fFixBase) lower points
    for (Int_t k = 0; k < npointsAv; k++) {
      if (fHist->GetBinError(ind[k] + 1) == 0.) // in case of null entries which induce a crash. Could bias the basline in upward direction!
      {
        printf("[WARNING] Null entries found in histogram to be fit! These points are been excluded from baseline evaluation...\n");
        npointsAv++;
        continue;
      }
      Av += fHist->GetBinContent(ind[k] + 1) / (fHist->GetBinError(ind[k] + 1) * fHist->GetBinError(ind[k] + 1));
      errAv += 1. / (fHist->GetBinError(ind[k] + 1) * fHist->GetBinError(ind[k] + 1));
    }
    Av /= errAv;
    errAv = TMath::Sqrt(1. / errAv);
    if (fVerbose) { printf("[RESULT] Average fBaseline: %.3f +- %.3f", Av, errAv); }
    fBaseline = Av;
    fErrBaseline = errAv;

    if (fShiftBaselineUp) {
      fBaseline += fErrBaseline;
      if (fVerbose) { printf("[INFO] Shift baseline up of its statistical uncertainty"); }
    }

    if (fShiftBaselineDown) {
      fBaseline -= fErrBaseline;
      if (fVerbose) { printf("[INFO] Shift baseline down of its statistical uncertainty"); }
    }

    return fBaseline;
  }

  // zyam at pi/2. Fix the baseline averaging the 2 points around +-pi/2 value
  if (fFixBase == 2) {
    Double_t errAv = 0., Av = 0.;
    Int_t binPhi = fHist->FindBin(TMath::Pi() / 2.);
    Av += fHist->GetBinContent(binPhi) / (fHist->GetBinError(binPhi) * fHist->GetBinError(binPhi));
    errAv += 1. / (fHist->GetBinError(binPhi) * fHist->GetBinError(binPhi));
    if (!fIsReflected) {
      binPhi = fHist->FindBin(-TMath::Pi() / 2.);
      if (binPhi < 1)
        binPhi = 1;
      Av += fHist->GetBinContent(binPhi) / (fHist->GetBinError(binPhi) * fHist->GetBinError(binPhi));
      errAv += 1. / (fHist->GetBinError(binPhi) * fHist->GetBinError(binPhi));
    } else {
      printf("[INFO] Reflected histo: only the point at +pi/2 used to evaluate baseline");
    }
    Av /= errAv;
    errAv = TMath::Sqrt(1. / errAv);
    printf("[RESULT] Average fBaseline: %.3f +- %.3f \n", Av, errAv);
    fBaseline = Av;
    fErrBaseline = errAv;

    if (fShiftBaselineUp) {
      fBaseline += fErrBaseline;
      printf("[INFO] Shift baseline up of its statistical uncertainty");
    }

    if (fShiftBaselineDown) {
      fBaseline -= fErrBaseline;
      printf("[INFO] Shift baseline down of its statistical uncertainty");
    }

    return fBaseline;
  }

  // fix the baseline to the weighted average of the points passed through the function SetPointsForBaseline()
  if (fFixBase == 3) {
    if (fNbasleinePoints == 0) {
      printf("[ERROR] No baseline points set for the baseline evaluation, SetPointsForBaseline(Int_t nBaselinePoints, Double_t* valsBaseline). Returning -1");
      return -1;
    }
    Double_t errAv = 0., Av = 0.;
    for (int i = 0; i < fNbasleinePoints; i++) {
      Av += fHist->GetBinContent(fBinsBaseline[i]) / (fHist->GetBinError(fBinsBaseline[i]) * fHist->GetBinError(fBinsBaseline[i]));
      errAv += 1. / (fHist->GetBinError(fBinsBaseline[i]) * fHist->GetBinError(fBinsBaseline[i]));
    }
    Av /= errAv;
    errAv = TMath::Sqrt(1. / errAv);
    printf("[RESULT] Average fBaseline: %.3f +- %.3f \n", Av, errAv);
    fBaseline = Av;
    fErrBaseline = errAv;

    if (fShiftBaselineUp) {
      fBaseline += fErrBaseline;
      printf("[INFO] Shift baseline up of its statistical uncertainty \n");
    }

    if (fShiftBaselineDown) {
      fBaseline -= fErrBaseline;
      printf("[INFO] Shift baseline down of its statistical uncertainty \n");
    }

    return fBaseline;
  }

  if (fFixBase == 4) {
    fBaseline = CalculateBaseline(fHist, fIsTotal); // TODO: add the option for total range/ reflected range to pass in input
    fErrBaseline = CalculateBaselineError(fHist, fIsTotal);

    if (fShiftBaselineUp) {
      fBaseline += fErrBaseline;
      printf("[INFO] Shift baseline up of its statistical uncertainty \n");
    }

    if (fShiftBaselineDown) {
      fBaseline -= fErrBaseline;
      printf("[INFO] Shift baseline down of its statistical uncertainty \n");
    }

    return fBaseline;
  }

  Printf("[ERROR] DhCorrelationFitter::FindBaseline - WRONG BASELINE OPTION SET. Returning -1");
  return -1.;
}

void DhCorrelationFitter::FitBaselineWv2()
{

  fBaseTransvReg = new TF1("fBaseTransvReg", [](double* x, double* p) {
    double xx = x[0]; // x value
    if ((xx >= -TMath::Pi()/2 && xx <= -3*TMath::Pi()/8) || (xx >= 3*TMath::Pi()/8 && xx <= 5*TMath::Pi()/8) || (xx >= 11*TMath::Pi()/8 && xx <= 3*TMath::Pi()/2)) {
        // Gaussian example: p[0] = amplitude, p[1] = mean, p[2] = sigma
        return p[0]*(1+2*p[1]*p[2]*TMath::Cos(2*xx));
    }
    return 0.; }, -TMath::Pi() / 2, 3 * TMath::Pi() / 2, 3); // Function valid for [0,10], with 3 parameters

  fBaseTransvReg->FixParameter(1, fv2AssocPart);
  fBaseTransvReg->FixParameter(2, fv2Dmeson);

  TFitResultPtr rFit = fHist->Fit(fBaseTransvReg, "RIMES", "", fMinCorr, fMaxCorr);
  fBaseline = fBaseTransvReg->GetParameter(0);

  return;
}

void DhCorrelationFitter::CalculateYieldsAboveBaseline()
{

  fNSyieldBinCount = 0;
  fErrNSyieldBinCount = 0;
  fASyieldBinCount = 0;
  fErrASyieldBinCount = 0;
  if (fVerbose) { cout << "[RESULT] Baseline: " << fBaseline << " +- " << fErrBaseline << endl; }
  Int_t binMinNS = fHist->FindBin(-1.5); // slightly more than -pi/2
  if (binMinNS < 1)
    binMinNS = 1;      // with this, it is ok even in the case of a reflected fHist (range 0 - pi)
  Int_t binMaxNS = 6;  // fHist -> FindBin(1.5); // slightly less than +pi/2
  Int_t binMinAS = 11; // fHist -> FindBin(1.6); // slightly more than +pi/2
  Int_t binMaxAS = 16; // fHist -> FindBin(3.14+1.5); // slightly less than +3pi/2
  if (binMaxAS > fHist->GetNbinsX())
    binMaxAS = fHist->GetNbinsX(); // with this, it is ok even in the case of a reflected fHist (range 0 - pi)
  // Near Side Yield from bin counting
  for (Int_t bmNS = binMinNS; bmNS <= binMaxNS; bmNS++) {
    fNSyieldBinCount += 2 * (fHist->GetBinContent(bmNS) - fBaseline) * fHist->GetBinWidth(bmNS);
    fErrNSyieldBinCount += 4 * (fHist->GetBinError(bmNS) * fHist->GetBinError(bmNS)) * fHist->GetBinWidth(bmNS) * fHist->GetBinWidth(bmNS);
  }
  fErrNSyieldBinCount = TMath::Sqrt(fErrNSyieldBinCount);

  // Away Side Yield from bin counting
  for (Int_t bmAS = binMinAS; bmAS <= binMaxAS; bmAS++) {
    fASyieldBinCount += 2 * (fHist->GetBinContent(bmAS) - fBaseline) * fHist->GetBinWidth(bmAS);
    fErrASyieldBinCount += 4 * (fHist->GetBinError(bmAS) * fHist->GetBinError(bmAS)) * fHist->GetBinWidth(bmAS) * fHist->GetBinWidth(bmAS);
  }
  fErrASyieldBinCount = TMath::Sqrt(fErrASyieldBinCount);

  if (fVerbose) { printf("[RESULT] Bin counting results: NS Yield = %.3f +- %.3f \n[RESULT] Bin counting results: AS Yield: %.3f +- %.3f \n", fNSyieldBinCount, fErrNSyieldBinCount, fASyieldBinCount, fErrASyieldBinCount); }

  return;
}

Double_t DhCorrelationFitter::CalculateBaseline(TH1F*& histo, Bool_t totalRange)
{

  if (histo == nullptr) {
    printf("[ERROR] DhCorrelationFitter::CalculateBaseline. Null pointer for input histogram \n");
    return -1.;
  }
  // total range = 2*Pi
  // half range = Pi , for histogram reflected under symmetric assumption

  Double_t baseline, errBaseline;
  Int_t nBinsPhi = histo->GetNbinsX();
  Int_t binPhiHalf = nBinsPhi / 2;
  Int_t binPhiHalfMinus1 = nBinsPhi / 2 - 1;
  Int_t binPhiHalfPlus1 = nBinsPhi / 2 + 1;
  Int_t binPhiHalfPlus2 = nBinsPhi / 2 + 1;

  if (totalRange) {
    printf("[INFO] Using total deltaPhi range \n");
    // baseline evaluated considering: the two first points, the last two points and four points in the middle (corresponding to the outer points)
    if (nBinsPhi >= 32) {
      printf("[INFO] nBinsPhi >= 32 \n");
      baseline =
        ((histo->GetBinContent(1)) * (1. / TMath::Power(histo->GetBinError(1), 2)) +
         (histo->GetBinContent(2)) * (1. / TMath::Power(histo->GetBinError(2), 2)) +
         (histo->GetBinContent(binPhiHalfMinus1)) * (1. / TMath::Power(histo->GetBinError(binPhiHalfMinus1), 2)) +
         (histo->GetBinContent(binPhiHalf)) * (1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
         (histo->GetBinContent(binPhiHalfPlus1)) * (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)) +
         (histo->GetBinContent(binPhiHalfPlus2)) * (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus2), 2)) +
         (histo->GetBinContent(nBinsPhi - 1)) * (1. / TMath::Power(histo->GetBinError(nBinsPhi - 1), 2)) +
         (histo->GetBinContent(nBinsPhi)) * (1. / TMath::Power(histo->GetBinError(nBinsPhi), 2))) /
        ((1. / TMath::Power(histo->GetBinError(1), 2)) +
         (1. / TMath::Power(histo->GetBinError(2), 2)) +
         (1. / TMath::Power(histo->GetBinError(binPhiHalfMinus1), 2)) +
         (1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
         (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)) +
         (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus2), 2)) +
         (1. / TMath::Power(histo->GetBinError(nBinsPhi - 1), 2)) +
         (1. / TMath::Power(histo->GetBinError(nBinsPhi), 2)));
    } else {
      printf("[INFO] nBinsPhi < 32 \n");
      baseline =
        ((histo->GetBinContent(1)) * (1. / TMath::Power(histo->GetBinError(1), 2)) +
         (histo->GetBinContent(binPhiHalf)) * (1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
         (histo->GetBinContent(binPhiHalfPlus1)) * (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)) +
         (histo->GetBinContent(nBinsPhi)) * (1. / TMath::Power(histo->GetBinError(nBinsPhi), 2))) /
        ((1. / TMath::Power(histo->GetBinError(1), 2)) +
         (1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
         (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)) +
         (1. / TMath::Power(histo->GetBinError(nBinsPhi), 2)));
    }
  } else {
    printf("[INFO] Using reflected deltaPhi range \n");
    // baseline evaluated using the 4 middle points in the transverese region
    if (nBinsPhi >= 16) {
      printf("[INFO] 4 central points in the transverse region for baseline \n");
      baseline =
        ((histo->GetBinContent(binPhiHalfMinus1)) * (1. / TMath::Power(histo->GetBinError(binPhiHalfMinus1), 2)) +
         (histo->GetBinContent(binPhiHalf)) * (1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
         (histo->GetBinContent(binPhiHalfPlus1)) * (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)) +
         (histo->GetBinContent(binPhiHalfPlus2)) * (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus2), 2))) /
        ((1. / TMath::Power(histo->GetBinError(binPhiHalfMinus1), 2)) +
         (1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
         (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)) +
         (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus2), 2)));
    } else {
      baseline =
        ((histo->GetBinContent(binPhiHalf)) * (1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
         (histo->GetBinContent(binPhiHalfPlus1)) * (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2))) /
        ((1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
         (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)));
    }
  }

  return baseline;
}

Double_t DhCorrelationFitter::CalculateBaselineError(TH1F*& histo, Bool_t totalRange)
{
  if (histo == nullptr) {
    printf("[ERROR] DhCorrelationFitter::CalculateBaselineError. Null pointer for input histogram \n");
    return -1.;
  }
  // total range = 2*Pi
  // half range = Pi , for histogram reflected under symmetric assumption

  Double_t errBaseline;
  Int_t nBinsPhi = histo->GetNbinsX();
  Int_t binPhiHalf = nBinsPhi / 2;
  Int_t binPhiHalfMinus1 = nBinsPhi / 2 - 1;
  Int_t binPhiHalfPlus1 = nBinsPhi / 2 + 1;
  Int_t binPhiHalfPlus2 = nBinsPhi / 2 + 1;

  if (totalRange) {
    // baseline evaluated considering: the two first points, the last two points and four points in the middle (corresponding to the outer points)
    if (nBinsPhi >= 32) {
      errBaseline = 1. /
                    TMath::Sqrt((1. / TMath::Power(histo->GetBinError(1), 2)) +
                                (1. / TMath::Power(histo->GetBinError(2), 2)) +
                                (1. / TMath::Power(histo->GetBinError(binPhiHalfMinus1), 2)) +
                                (1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
                                (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)) +
                                (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus2), 2)) +
                                (1. / TMath::Power(histo->GetBinError(nBinsPhi - 1), 2)) +
                                (1. / TMath::Power(histo->GetBinError(nBinsPhi), 2)));
    } else { // fon nBinsPhi = 16 (rebin 4)
      errBaseline = 1. /
                    TMath::Sqrt((1. / TMath::Power(histo->GetBinError(1), 2)) +
                                (1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
                                (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)) +
                                (1. / TMath::Power(histo->GetBinError(nBinsPhi), 2)));
    }
  } else {
    // baseline evaluated using the 4 middle points in the transverese region
    if (nBinsPhi >= 32) {
      errBaseline = 1. /
                    TMath::Sqrt((1. / TMath::Power(histo->GetBinError(binPhiHalfMinus1), 2)) +
                                (1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
                                (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)) +
                                (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus2), 2)));
    } else {
      errBaseline = 1. /
                    TMath::Sqrt((1. / TMath::Power(histo->GetBinError(binPhiHalf), 2)) +
                                (1. / TMath::Power(histo->GetBinError(binPhiHalfPlus1), 2)));
    }
  }

  return errBaseline;
}

void DhCorrelationFitter::SetSingleTermsForDrawing(Bool_t draw)
{
  Double_t* par = 0;
  if (fTypeOfFitFunc == 1 || fTypeOfFitFunc == 2 || fTypeOfFitFunc == 5) {
    par = new Double_t[7];
  } else if (fTypeOfFitFunc == 3 || fTypeOfFitFunc == 6) {
    par = new Double_t[4];
  } else if (fTypeOfFitFunc == 4) {
    par = new Double_t[8];
  } else if (fTypeOfFitFunc == 7) {
    par = new Double_t[9];
  } else if (fTypeOfFitFunc == 8) {
    par = new Double_t[5];
  } else if (fTypeOfFitFunc == 9) {
    par = new Double_t[11];
  } else if (fTypeOfFitFunc == 10) {
    par = new Double_t[4];
  } else {
    Printf("[ERROR] DhCorrelationFitter::SetSingleTermsForDrawing, wrong type of function");
    return;
  }

  if (fTypeOfFitFunc == 3 || fTypeOfFitFunc == 6) {
    fFit->GetParameters(par);
    fFit->SetLineWidth(4);

    fPed->SetParameter(0, par[0]);
    fPed->SetLineColor(6); // pink
    fPed->SetLineStyle(9);
    fPed->SetLineWidth(4);

    fGausAS->SetParameter(0, par[1]);
    fGausAS->SetParameter(1, par[2]);
    fGausAS->SetParameter(2, par[3]);

    fGausAS->SetLineStyle(9);
    fGausAS->SetLineColor(kGreen);
    fGausAS->SetLineWidth(4);

    TPaveText* pvStatTests1 = new TPaveText(0.51, 0.58, 0.85, 0.90, "NDC");
    pvStatTests1->SetFillStyle(0);
    pvStatTests1->SetTextSize(0.045);
    pvStatTests1->SetBorderSize(0);
    TText *t0, *t1, *t2, *t3;
    t0 = pvStatTests1->AddText(0., 1.00, Form("#chi^{2}/ndf = %.1f/%d ", fFit->GetChisquare(), fFit->GetNDF()));
    t1 = pvStatTests1->AddText(0., 0.80, Form("Ped = %.3f#pm%.3f ", fBaseline, fErrBaseline));
    t2 = pvStatTests1->AddText(0., 0.65, Form("AS Y = %.3f#pm%.3f ", fFit->GetParameter("AS Y"), fFit->GetParError(fFit->GetParNumber("AS Y"))));
    t3 = pvStatTests1->AddText(0., 0.50, Form("AS #sigma = %.3f#pm%.3f ", fFit->GetParameter("AS #sigma"), fFit->GetParError(fFit->GetParNumber("AS #sigma"))));

    if (draw) {
      fFit->Draw("same");
      fPed->Draw("same");
      fGausAS->Draw("same");
      pvStatTests1->Draw("same");
    }
  } else if (fTypeOfFitFunc == 4) {
    fFit->GetParameters(par);
    fFit->SetLineWidth(4);

    fPed->SetParameter(0, par[0]);
    fPed->SetLineColor(6); // pink
    fPed->SetLineStyle(9);
    fPed->SetLineWidth(4);

    fGausNS->SetParameter(0, par[1]);
    fGausNS->SetParameter(1, par[2]);
    fGausNS->SetParameter(2, par[3]);
    fGausNS->SetParameter(3, par[7]);
    fGausAS->SetParameter(0, par[4]);
    fGausAS->SetParameter(1, par[5]);
    fGausAS->SetParameter(2, par[6]);

    fGausNS->SetLineStyle(9);
    fGausNS->SetLineColor(kBlue);
    fGausAS->SetLineStyle(9);
    fGausAS->SetLineColor(kGreen);
    fGausNS->SetLineWidth(4);
    fGausAS->SetLineWidth(4);

    TPaveText* pvStatTests1 = new TPaveText(0.51, 0.58, 0.85, 0.90, "NDC");
    pvStatTests1->SetFillStyle(0);
    pvStatTests1->SetTextSize(0.045);
    pvStatTests1->SetBorderSize(0);
    TText *t0, *t1, *t2, *t3, *t4, *t5, *t6;
    t0 = pvStatTests1->AddText(0., 1.00, Form("#chi^{2}/ndf = %.1f/%d ", fFit->GetChisquare(), fFit->GetNDF()));
    t1 = pvStatTests1->AddText(0., 0.80, Form("Ped = %.3f#pm%.3f ", fBaseline, fErrBaseline));
    t2 = pvStatTests1->AddText(0., 0.65, Form("NS Y = %.3f#pm%.3f ", fFit->GetParameter("NS Y"), fFit->GetParError(fFit->GetParNumber("NS Y"))));
    t3 = pvStatTests1->AddText(0., 0.50, Form("NS #sigma = %.3f#pm%.3f ", fFit->GetParameter("NS #sigma"), fFit->GetParError(fFit->GetParNumber("NS #sigma"))));
    t4 = pvStatTests1->AddText(0., 0.35, Form("AS Y = %.3f#pm%.3f ", fFit->GetParameter("AS Y"), fFit->GetParError(fFit->GetParNumber("AS Y"))));
    t5 = pvStatTests1->AddText(0., 0.20, Form("AS #sigma = %.3f#pm%.3f ", fFit->GetParameter("AS #sigma"), fFit->GetParError(fFit->GetParNumber("AS #sigma"))));
    t6 = pvStatTests1->AddText(0., 0.05, Form("#beta = %.3f#pm%.3f ", fFit->GetParameter("NS shape par"), fFit->GetParError(fFit->GetParNumber("NS shape par"))));

    if (draw) {
      fFit->Draw("same");
      fPed->Draw("same");
      fGausAS->Draw("same");
      fGausNS->Draw("same");
      pvStatTests1->Draw("same");
    }
  } else if (fTypeOfFitFunc == 7) {
    fFit->GetParameters(par);
    fFit->SetLineWidth(4);

    fBaseTransvReg->SetLineColor(15);
    fBaseTransvReg->SetLineStyle(9);
    fBaseTransvReg->SetLineWidth(4);

    fPed->SetParameter(0, par[0]);
    fPed->SetParameter(1, par[7]);
    fPed->SetParameter(2, par[8]);
    fPed->SetLineColor(6); // pink
    fPed->SetLineStyle(9);
    fPed->SetLineWidth(4);

    fGausNS->SetParameter(0, par[1]);
    fGausNS->SetParameter(1, par[2]);
    fGausNS->SetParameter(2, par[3]);
    // fGausNS -> SetParameter(3, par[7]);
    fGausAS->SetParameter(0, par[4]);
    fGausAS->SetParameter(1, par[5]);
    fGausAS->SetParameter(2, par[6]);

    fGausNS->SetLineStyle(9);
    fGausNS->SetLineColor(kBlue);
    fGausAS->SetLineStyle(9);
    fGausAS->SetLineColor(kGreen);
    fGausNS->SetLineWidth(4);
    fGausAS->SetLineWidth(4);

    TPaveText* pvStatTests1 = new TPaveText(0.51, 0.58, 0.85, 0.90, "NDC");
    pvStatTests1->SetFillStyle(0);
    pvStatTests1->SetTextSize(0.045);
    pvStatTests1->SetBorderSize(0);
    TText *t0, *t1, *t2, *t3, *t4, *t5, *t6, *t7, *t8;
    t0 = pvStatTests1->AddText(0., 1.00, Form("#chi^{2}/ndf = %.1f/%d ", fFit->GetChisquare(), fFit->GetNDF()));
    t2 = pvStatTests1->AddText(0., 0.80, Form("NS Y = %.3f#pm%.3f ", fFit->GetParameter("NS Y"), fFit->GetParError(fFit->GetParNumber("NS Y"))));
    t3 = pvStatTests1->AddText(0., 0.65, Form("NS #sigma = %.3f#pm%.3f ", fFit->GetParameter("NS #sigma"), fFit->GetParError(fFit->GetParNumber("NS #sigma"))));
    t4 = pvStatTests1->AddText(0., 0.50, Form("AS Y = %.3f#pm%.3f ", fFit->GetParameter("AS Y"), fFit->GetParError(fFit->GetParNumber("AS Y"))));
    t5 = pvStatTests1->AddText(0., 0.35, Form("AS #sigma = %.3f#pm%.3f ", fFit->GetParameter("AS #sigma"), fFit->GetParError(fFit->GetParNumber("AS #sigma"))));
    // t6 = pvStatTests1 -> AddText(0., 0.20, Form("#beta = %.3f#pm%.3f ", fFit -> GetParameter("NS shape par"), fFit -> GetParError(fFit->GetParNumber("NS shape par"))));

    TPaveText* pvStatTests2 = new TPaveText(0.51, 0.28, 0.85, 0.60, "NDC");
    pvStatTests2->SetFillStyle(0);
    pvStatTests2->SetTextSize(0.045);
    pvStatTests2->SetBorderSize(0);
    t1 = pvStatTests2->AddText(0., 1.00, Form("Ped = %.3f#pm%.3f ", fFit->GetParameter("ped"), fErrBaseline /*fFit -> GetParError(fFit->GetParNumber("ped")*/));
    t7 = pvStatTests2->AddText(0., 0.65, Form("v_{2}^{hadron} = %.3f#pm%.3f ", fFit->GetParameter("v_{2} hadron"), fFit->GetParError(fFit->GetParNumber("v_{2} hadron"))));
    t8 = pvStatTests2->AddText(0., 0.35, Form("v_{2}^{D} = %.3f#pm%.3f ", fFit->GetParameter("v_{2} D meson"), fFit->GetParError(fFit->GetParNumber("v_{2} D meson"))));

    if (draw) {
      fFit->Draw("same");
      fPed->Draw("same");
      fBaseTransvReg->Draw("same");
      fGausAS->Draw("same");
      fGausNS->Draw("same");
      pvStatTests1->Draw("same");
      pvStatTests2->Draw("same");
    }
  } else if (fTypeOfFitFunc == 8 || fTypeOfFitFunc == 9) {
    fFit->GetParameters(par);
    fFit->SetLineWidth(4);

    // Pedestal line
    fPed->SetParameter(0, par[1]);
    fPed->SetLineColor(6);
    fPed->SetLineStyle(9);
    fPed->SetLineWidth(4);

    // Components drawn as: ped + modulation_term (stacked on pedestal)
    fv1->SetParameter(0, par[0]);
    fv1->SetParameter(1, par[1]);
    fv1->SetLineColor(6);
    fv1->SetLineStyle(9);
    fv1->SetLineWidth(4);

    fv2->SetParameter(0, par[1]);
    fv2->SetParameter(1, par[3]);
    fv2->SetLineColor(kOrange+7);
    fv2->SetLineStyle(8);
    fv2->SetLineWidth(4);

    fv3->SetParameter(0, par[1]);
    fv3->SetParameter(1, par[4]);
    fv3->SetLineColor(kGreen+2);
    fv3->SetLineStyle(8);
    fv3->SetLineWidth(4);

    /*fv4->SetParameter(0, par[1]);
    fv4->SetParameter(1, par[5]);
    fv4->SetLineColor(kRed+2);
    fv4->SetLineStyle(7);
    fv4->SetLineWidth(4);

    fv5->SetParameter(0, par[1]);
    fv5->SetParameter(1, par[6]);
    fv5->SetLineColor(kAzure+5);
    fv5->SetLineStyle(7);
    fv5->SetLineWidth(4);*/

    TPaveText* pvStatTests1 = new TPaveText(0.603, 0.284, 0.943, 0.753, "NDC");
    pvStatTests1->SetFillStyle(0);
    pvStatTests1->SetTextSize(0.045);
    pvStatTests1->SetBorderSize(0);
    TText *t0, *t1, *t2, *t3, *t4;
    t0 = pvStatTests1->AddText(0., 1.00, Form("#chi^{2}/ndf = %.1f/%d ", fFit->GetChisquare(), fFit->GetNDF()));
    t1 = pvStatTests1->AddText(0., 0.85, Form("Ped = %.4f #pm %.4f ", par[1], fFit->GetParError(fFit->GetParNumber("ped"))));
    t2 = pvStatTests1->AddText(0., 0.65, Form("v1 = %.5f#pm%.5f ", fFit->GetParameter("v_{1}"), fFit->GetParError(fFit->GetParNumber("v_{1}"))));
    t3 = pvStatTests1->AddText(0., 0.50, Form("v2 = %.5f#pm%.5f ", fFit->GetParameter("v_{2} delta"), fFit->GetParError(fFit->GetParNumber("v_{2} delta"))));
    t4 = pvStatTests1->AddText(0., 0.35, Form("v3 = %.5f#pm%.5f ", fFit->GetParameter("v_{3}"), fFit->GetParError(fFit->GetParNumber("v_{3}"))));

    if (draw) {
      fFit->Draw("same");
      fPed->Draw("same");
      fv1->Draw("same");
      fv2->Draw("same");
      fv3->Draw("same");
      pvStatTests1->Draw("same");
    }
  } else if (fTypeOfFitFunc == 10) { 
    fFit->GetParameters(par);
    fFit->SetLineWidth(4);

    // Pedestal line
    fPed->SetParameter(0, par[1]);
    fPed->SetLineColor(kGray+2);
    fPed->SetLineStyle(3);
    fPed->SetLineWidth(4);

    fLM->SetParameter(0, par[0]);
    fLM->SetParameter(1, par[1]);
    fLM->SetLineColor(kAzure+2);
    fLM->SetLineStyle(8); // dashed
    fLM->SetLineWidth(4);

    // Flow (total flow component: G * [1 + 2*v2*cos(2x) + 2*v3*cos(3x)])
    fFlow->SetParameter(0, par[1]);  // g
    fFlow->SetParameter(1, par[2]); // v2
    fFlow->SetParameter(2, par[3]); // v3
    fFlow->SetLineColor(kGreen+2);
    fFlow->SetLineStyle(9); // dotted
    fFlow->SetLineWidth(4);

    fv2->SetParameter(0, par[1]);
    fv2->SetParameter(1, par[2]);
    fv2->SetLineColor(46);
    fv2->SetLineStyle(10);
    fv2->SetLineWidth(4);

    fv3->SetParameter(0, par[1]);
    fv3->SetParameter(1, par[3]);
    fv3->SetLineColor(41);
    fv3->SetLineStyle(10); // dashed + dotted
    fv3->SetLineWidth(4);

    TPaveText* pvStatTests1 = new TPaveText(0.18, 0.55, 0.45, 0.88, "NDC");
    pvStatTests1->SetFillStyle(0);
    pvStatTests1->SetTextSize(0.040);
    pvStatTests1->SetBorderSize(0);
    pvStatTests1->AddText(0., 1.00, Form("#chi^{2}/ndf = %.1f/%d ",
                                        fFit->GetChisquare(), fFit->GetNDF()));
    // pvStatTests1->AddText(0., 0.82, Form("Ped = %.4f #pm %.4f ",
    //                                     fBaseline, fErrBaseline));
    pvStatTests1->AddText(0., 0.65, Form("LM Factor = %.3f #pm %.3f ",
                                        fFit->GetParameter(0), fFit->GetParError(0)));
    pvStatTests1->AddText(0., 0.52, Form("Flow Factor = %.3f #pm %.3f ",
                                        fFit->GetParameter(1), fFit->GetParError(1)));
    pvStatTests1->AddText(0., 0.42, Form("v_{2#Delta} = %.5f #pm %.5f ",
                                        fFit->GetParameter(2), fFit->GetParError(2)));
    pvStatTests1->AddText(0., 0.32, Form("v_{3#Delta} = %.5f #pm %.5f ",
                                        fFit->GetParameter(3), fFit->GetParError(3)));
    TLegend* leg = new TLegend(0.65, 0.65, 0.95, 0.88);   // NDC coordinates
    leg->SetFillStyle(0);                                 // transparent
    leg->SetBorderSize(0);
    leg->SetTextSize(0.04);

    leg->AddEntry(fFit,  "Global Fit",      "l");
    leg->AddEntry(fPed,  "Pedestal",        "l");
    leg->AddEntry(fLM,   "LM",       "l");
    leg->AddEntry(fFlow, "HM", "l");
    leg->AddEntry(fv2,   "v_{2} component", "l");
    leg->AddEntry(fv3,   "v_{3} component", "l");

    if (draw) {
      fPed->Draw("same");
      fv3->Draw("same");
      fv2->Draw("same");
      fFlow->Draw("same");
      fLM->Draw("same");
      fFit->Draw("same");
      pvStatTests1->Draw("same");
      leg->Draw("same");
    }
  }  else {
    fFit->GetParameters(par);
    fFit->SetLineWidth(4);

    fPed->SetParameter(0, par[0]);
    fPed->SetLineColor(6); // pink
    fPed->SetLineStyle(9);
    fPed->SetLineWidth(4);

    fGausNS->SetParameter(0, par[1]);
    fGausNS->SetParameter(1, par[2]);
    fGausNS->SetParameter(2, par[3]);
    fGausAS->SetParameter(0, par[4]);
    fGausAS->SetParameter(1, par[5]);
    fGausAS->SetParameter(2, par[6]);

    fGausNS->SetLineStyle(9);
    fGausNS->SetLineColor(kBlue);
    fGausAS->SetLineStyle(9);
    fGausAS->SetLineColor(kGreen);
    fGausNS->SetLineWidth(4);
    fGausAS->SetLineWidth(4);

    TPaveText* pvStatTests1 = new TPaveText(0.51, 0.58, 0.85, 0.90, "NDC");
    pvStatTests1->SetFillStyle(0);
    pvStatTests1->SetTextSize(0.045);
    pvStatTests1->SetBorderSize(0);
    TText *t0, *t1, *t2, *t3, *t4, *t5, *t6;
    t0 = pvStatTests1->AddText(0., 1.00, Form("#chi^{2}/ndf = %.1f/%d ", fFit->GetChisquare(), fFit->GetNDF()));
    t1 = pvStatTests1->AddText(0., 0.80, Form("Ped = %.3f#pm%.3f ", fBaseline, fErrBaseline));
    t2 = pvStatTests1->AddText(0., 0.65, Form("NS Y = %.3f#pm%.3f ", fFit->GetParameter("NS Y"), fFit->GetParError(fFit->GetParNumber("NS Y"))));
    t3 = pvStatTests1->AddText(0., 0.50, Form("NS #sigma = %.3f#pm%.3f ", fFit->GetParameter("NS #sigma"), fFit->GetParError(fFit->GetParNumber("NS #sigma"))));
    t4 = pvStatTests1->AddText(0., 0.35, Form("AS Y = %.3f#pm%.3f ", fFit->GetParameter("AS Y"), fFit->GetParError(fFit->GetParNumber("AS Y"))));
    t5 = pvStatTests1->AddText(0., 0.20, Form("AS #sigma = %.3f#pm%.3f ", fFit->GetParameter("AS #sigma"), fFit->GetParError(fFit->GetParNumber("AS #sigma"))));

    if (draw) {
      fFit->Draw("same");
      fPed->Draw("same");
      fGausAS->Draw("same");
      fGausNS->Draw("same");
      pvStatTests1->Draw("same");
    }
  }
}
