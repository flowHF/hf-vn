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

/// \file DhCorrelationFitter.h
/// \brief Class to perform the azimuthal correlation fit
/// \author Samuele Cattaruzzi <samuele.cattaruzzi@cern.ch>
/// \author Swapnesh Santosh Khade <swapnesh.santosh.khade@cern.ch>

#ifndef PWGHF_HFC_MACROS_DHCORRELATIONFITTER_H_
#define PWGHF_HFC_MACROS_DHCORRELATIONFITTER_H_

#include <TF1.h>
#include <TH1.h>
#include <TGraphErrors.h>
#include <TSpline.h>
#include <RtypesCore.h>

#include <cstdio>

class DhCorrelationFitter
{

 public:
  enum FunctionType { kConstwoGaus = 1,
                      kTwoGausPeriodicity = 2,
                      kSingleGaus = 3,
                      kGenGaus = 4,
                      kVonMises = 5,
                      kSingleVonMises = 6,
                      kTwoGausPeriodicityPlusV2modulation = 7,
                      kV2DeltaModulationLowMult = 8,
                      kV2DeltaModulationHighMult = 9,
                      kTemplateFit = 10};

  /// Constructors
  DhCorrelationFitter();
  DhCorrelationFitter(TH1F* histoToFit, Double_t min, Double_t max);
  virtual ~DhCorrelationFitter();
  DhCorrelationFitter(const DhCorrelationFitter& source);
  DhCorrelationFitter& operator=(const DhCorrelationFitter& cfit);

  /// Setters
  void SetVerbosity(Int_t isVerbose) { fVerbose = isVerbose; }
  void SetHistoIsReflected(Bool_t isrefl) { fIsReflected = isrefl; }
  void SetFuncType(FunctionType fitType) { fTypeOfFitFunc = fitType; }
  void SetFixBaseline(Int_t fixBase) { fFixBase = fixBase; }
  void SetFixMean(Int_t fixMean) { fFixMean = fixMean; }
  void SetPtRanges(Double_t PtCandMin, Double_t PtCandMax, Double_t PtAssocMin, Double_t PtAssocMax)
  {
    fMinCandPt = PtCandMin;
    fMaxCandPt = PtCandMax;
    fMinAssoPt = PtAssocMin;
    fMaxAssoPt = PtAssocMax;
  }
  void SetExternalValsAndBounds(Int_t nPars, Double_t* vals, Double_t* lowBounds, Double_t* uppBounds);
  void SetPointsForBaseline(Int_t nBaselinePoints, Int_t* binsBaseline);
  void SetReflectedCorrHisto(Bool_t isReflected) { fIsTotal = isReflected; }
  void SetBaselineUpOrDown(Bool_t baseUp, Bool_t baseDown)
  {
    fShiftBaselineUp = baseUp;
    fShiftBaselineDown = baseDown;
  }
  void Setv2(Double_t v2AssocPart, Double_t v2Dmeson)
  {
    fv2AssocPart = v2AssocPart;
    fv2Dmeson = v2Dmeson;
  }
  void SetLMTemplate(TH1D* tempHisto) { fTempHisto = tempHisto; }
  void SetTempFunc(Int_t tempFunc) { fTempFunc = tempFunc; }
  void SetLMPairs(Double_t pairs) { fPairsPerMassBin = pairs; }
  void SetRyTrigger(Double_t val, Double_t err) { fRyTrigger = val; fRyTriggerErr = err; }
  void SetLMRyTrigger(Double_t val, Double_t err) { fLMRyTrigger = val; fLMRyTriggerErr = err; }
  void SetWithPedLM(Bool_t withPed) { fWithPedLM = withPed; }
  void SetFixLMFactor(Bool_t fixFactor) { fFixLMFactor = fixFactor; }

  /// Functions for fitting
  void Fitting(Bool_t drawSplitTerm = kTRUE, Bool_t useExternalPars = kFALSE);
  void SetFitFunction();
  Double_t TotalTemplateFitFunction(Double_t* x, Double_t* par);
  Double_t TemplateFitFunctionWPed(Double_t* x, Double_t* par);
  Double_t TemplateFitFunction(Double_t* x, Double_t* par);
  void CalculateYieldsAboveBaseline();
  void FitBaselineWv2();
  Double_t CalculateBaseline(TH1F*& histo, Bool_t totalRange = kTRUE);
  Double_t CalculateBaselineError(TH1F*& histo, Bool_t totalRange = kTRUE);
  void SetSingleTermsForDrawing(Bool_t draw);
  Double_t FindBaseline();
  void intepolateTemp();
  void BuildLMOutput();

  /// Getters
  Double_t GetNSSigma() { return fFit->GetParameter("NS #sigma"); } // TODO: case kConstThreeGausPeriodicity
  Double_t GetASSigma() { return fFit->GetParameter("AS #sigma"); } // TODO: case kConstThreeGausPeriodicity
  Double_t GetNSYield() { return fFit->GetParameter("NS Y"); }
  Double_t GetASYield() { return fFit->GetParameter("AS Y"); }
  Double_t GetBeta() { return fFit->GetParameter(7); }
  Double_t GetPedestal() { return fBaseline; }
  Double_t Getv2hadron() { return fFit->GetParameter("v_{2} hadron"); }
  Double_t Getv2Dmeson() { return fFit->GetParameter("v_{2} D meson"); }
  Double_t Getv2Delta() { return fFit->GetParameter("v_{2} delta"); }
  Double_t Getv2Delta_lm() { return fFit->GetParameter("v_{2}_lm delta"); }
  Double_t GetLMFactor() { return fFit->GetParameter("F (LM Scale)"); }
  Double_t GetLMFactorError() { return fFit->GetParError(fFit->GetParNumber("F (LM Scale)")); }
  Double_t GetNSSigmaError() { return fFit->GetParError(fFit->GetParNumber("NS #sigma")); } // TODO: case kConstThreeGausPeriodicity
  Double_t GetASSigmaError() { return fFit->GetParError(fFit->GetParNumber("AS #sigma")); } // TODO: case kConstThreeGausPeriodicityAS
  Double_t GetNSYieldError() { return fFit->GetParError(fFit->GetParNumber("NS Y")); }
  Double_t GetASYieldError() { return fFit->GetParError(fFit->GetParNumber("AS Y")); }
  Double_t GetBetaError() { return fFit->GetParError(7); }
  Double_t GetPedestalError() { return fErrBaseline; }
  Double_t Getv2hadronError() { return fFit->GetParError(fFit->GetParNumber("v_{2} hadron")); }
  Double_t Getv2DmesonError() { return fFit->GetParError(fFit->GetParNumber("v_{2} D meson")); }
  Double_t Getv2DeltaError() { return fFit->GetParError(fFit->GetParNumber("v_{2} delta")); }
  Double_t Getv2DeltaError_lm() { return fFit->GetParError(fFit->GetParNumber("v_{2}_lm delta")); }
  Double_t GetBinCountingNSYield() { return fNSyieldBinCount; }
  Double_t GetBinCountingASYield() { return fASyieldBinCount; }
  Double_t GetBinCountingNSYieldErr() { return fErrNSyieldBinCount; }
  Double_t GetBinCountingASYieldErr() { return fErrASyieldBinCount; }
  TH1D* GetLMOutput() { return fLMOutput; }
  TF1* GetLMTemplateFunc() { return fTemplateFunc; }
  Int_t GetTempFunc() { return fTempFunc; }
  TF1* GetFitFunction()
  {
    if (!fFit) {
      printf("[ERROR] DhCorrelationFitter::GetFitFunction, No fit function");
      return NULL;
    }
    return fFit;
  }

 private:
  bool fVerbose = true;
  TH1F* fHist; // 1D azimuthal correlation histogram
  TH1D* fTempHisto; // 1D histogram for the template fit function
  TH1D* fLMOutput; // LM template histogram clone (for drawing hLM_template)
  TF1* fTemplateFunc; // LM template function (from Gaus/GausPeriodic fit)
  TGraphErrors* fGraph; // Graph created from the template histogram for interpolation in the fit function
  TSpline3* fSpline;

  TF1* fFit;           // Total fit function
  TF1* fGausNS;        // Near-Side (NS) Gaussian
  TF1* fGausAS;        // Away-Side (AS) Gaussian
  TF1* fPed;           // Baseline function
  TF1* fBaseTransvReg; // Baseline function with v2
  TF1* fLM;            // Template function for the low-mult. v2 modulation
  TF1* fFlow;          // Flow modulation function for the low-mult. v2 modulation
  TF1* fv1;            // v1 flow coefficient
  TF1* fv2;            // v2 flow coefficient
  TF1* fv3;            // v3 flow coefficient
  TF1* fv4;            // v4 flow coefficient
  TF1* fv5;            // v5 flow coefficient
  TF1* fv1_lm;         // v1_lm flow coefficient
  TF1* fv2_lm;         // v2_lm flow coefficient
  TF1* fv3_lm;         // v3_lm flow coefficient
  TF1* fv4_lm;         // v4_lm flow coefficient
  TF1* fv5_lm;         // v5_lm flow coefficient

  Bool_t fIsReflected;       // To use if reflected azimuthal correlation are given as input
  Bool_t fUseExternalPars;   // To use external fit parameters initial values and bounds
  Bool_t fShiftBaselineUp;   // To shift the baseline up of its statistical uncertainty
  Bool_t fShiftBaselineDown; // To shift baseline down of its statistical uncertainty
  Bool_t fIsTotal;           // Total range of 2*pi in the azimuthal correlation distribution
  Bool_t fWithPedLM;         // If true, LM template function includes pedestal (TemplateFitFunctionWPed); if false, no pedestal (TemplateFitFunction)
  Bool_t fFixLMFactor;       // If true, fix the LM scale factor (parameter 0); if false, let it float

  FunctionType fTypeOfFitFunc; // Type of fit function

  Int_t fFixBase;         // Fix baseline or keep it as free parameter (see Fitting())
  Int_t fFixMean;         // Fix means or keep them as free parameters (see Fitting())
  Int_t fNpars;           // Number of parameters of the fit function
  Int_t fNbasleinePoints; // Number of points passed as inputs for the baseline estimation
  Int_t fTempFunc;        // Template function type (0=raw, 1=raw+spline, 2=smooth+spline, 3=Gaus, 4=GausPeriodic)

  Int_t* fBinsBaseline; // Bin values for calculating the baseline

  Double_t fMinCorr;            // Min value of correlation histogram
  Double_t fMaxCorr;            // Max value of correlation histogram
  Double_t fMinCandPt;          // Min pt value of candidate
  Double_t fMaxCandPt;          // Max pt value of candidate
  Double_t fMinAssoPt;          // Min pt value of assoc. particles
  Double_t fMaxAssoPt;          // Max pt value of assoc. particles
  Double_t fBaseline;           // Baseline value
  Double_t fErrBaseline;        // Baseline error
  Double_t fNSyieldBinCount;    // NS Yield from bin counting
  Double_t fErrNSyieldBinCount; // NS Yield error from bin counting
  Double_t fASyieldBinCount;    // AS Yield from bin counting
  Double_t fErrASyieldBinCount; // AS Yield error from bin counting
  Double_t fv2AssocPart;        // v2 associated particles
  Double_t fv2Dmeson;           // v2 of D mesons
  Double_t fv2Delta;            // v2 of D mesons * v2 assoc part
  Double_t fRyTrigger;          // raw yield of trigger (D meson) from mass fit
  Double_t fRyTriggerErr;       // uncertainty of fRyTrigger
  Double_t fLMRyTrigger;        // raw yield of trigger (D meson) from mass fit (LM template)
  Double_t fLMRyTriggerErr;     // uncertainty of fLMRyTrigger
  Double_t fPairsPerMassBin;    // pairs per mass bin for scaling LM template

  Double_t* fExtParsVals;      // Fit parameters initial values
  Double_t* fExtParsLowBounds; // Fit parameters lower bounds
  Double_t* fExtParsUppBounds; // Fit parameters upper bounds
};

#endif // PWGHF_HFC_MACROS_DHCORRELATIONFITTER_H_
