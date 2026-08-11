#include "CCDB/CcdbApi.h"
#include "TDirectoryFile.h"
#include "TFile.h"
#include "TH2D.h"
#include "TH3F.h"
#include "TMath.h"
#include "TSystem.h"

enum Detectors {
  kFT0C = 0,
  kFT0A,
  kFT0M,
  kFV0A,
  kTPCPOS,
  kTPCNEG,
  kTPCALL,
  kNDetectors
};

struct DetectorConfig {
    std::string name;
    std::string dirsuffix;
    std::string ref;
};

using DetectorMap = std::array<DetectorConfig, kNDetectors>;

void Recenter(TH2* h, std::vector<double>& corr){
    corr.push_back(h->GetMean(1));
    corr.push_back(h->GetMean(2));
}

double CalcB(double rho, double sigmax, double sigmay){
    return rho * sigmax * sigmay * TMath::Sqrt(2.0 * (sigmax * sigmax + sigmay * sigmay - 2.0 * sigmax * sigmay * TMath::Sqrt(1.0 - rho * rho)) / ((sigmax * sigmax - sigmay * sigmay) * (sigmax * sigmax - sigmay * sigmay) + 4.0 * (sigmax * sigmay * rho) * (sigmax * sigmay * rho)));
}

void Twist(TH2* h, std::vector<double>& corr){
    double aPlus, aMinus;
    double lambdaPlus, lambdaMinus;
    double b = CalcB(h->GetCorrelationFactor(), h->GetStdDev(1), h->GetStdDev(2));

    aPlus = TMath::Sqrt(2. * TMath::Power(h->GetStdDev(1), 2.) - TMath::Power(b, 2.));
    aMinus = TMath::Sqrt(2. * TMath::Power(h->GetStdDev(2), 2.) - TMath::Power(b, 2.));

    lambdaPlus = b / aPlus;
    lambdaMinus = b / aMinus;

    corr.push_back(lambdaPlus);
    corr.push_back(lambdaMinus);
}

void Rescale(TH2* h, std::vector<double>& corr){
    double aPlus, aMinus;
    double b = CalcB(h->GetCorrelationFactor(), h->GetStdDev(1), h->GetStdDev(2));
    aPlus = TMath::Sqrt(2. * TMath::Power(h->GetStdDev(1), 2.) - TMath::Power(b, 2.));
    aMinus = TMath::Sqrt(2. * TMath::Power(h->GetStdDev(2), 2.) - TMath::Power(b, 2.));

    corr.push_back(aPlus);
    corr.push_back(aMinus);
}


std::vector<double> fillCorrections(string outdir, TFile* infile, string detector, string dirname, string ref, int nmode){

    TH3F* hQxQyCentUncor = (TH3F*)infile->Get(Form("%s/histQvec%sUncorV%d",dirname.c_str(),ref.c_str(),nmode));

    TFile* c = new TFile(Form("%s/debugQvecs_%s.root", outdir.c_str(), detector.c_str()), "recreate");
    const int nCentBins = 100;      // 1% centrality differential
    TH2F* hQvecUncor[nCentBins];
    std::vector<double> CorUncor;
    for(int i=0;i<nCentBins;i++){
        hQxQyCentUncor->GetZaxis()->SetRange(i+1, i+1);
        hQvecUncor[i] = (TH2F*)(hQxQyCentUncor->Project3D("yx"));
        int centMin = static_cast<int>(hQxQyCentUncor->GetZaxis()->GetBinLowEdge(i+1));
        int centMax = static_cast<int>(hQxQyCentUncor->GetZaxis()->GetBinUpEdge(i+1));
        hQvecUncor[i]->SetTitle(Form("hQxQyCent_%i_%i_Diff;Q_{x};Q_{y}", centMin, centMax));
        hQvecUncor[i]->Write(Form("hQxQyCent_%i_%i_Diff", centMin, centMax));
        Recenter(hQvecUncor[i], CorUncor);
        Twist(hQvecUncor[i], CorUncor);
        Rescale(hQvecUncor[i], CorUncor);
        // std::cout << "[CentBin " << i << "] ref = \"" << ref << "\", nmode = " << nmode << "] Recenter " << CorUncor[i*6 + 0] << ", " << CorUncor[i*6 + 1] << std::endl;
        // std::cout << "[CentBin " << i << "] ref = \"" << ref << "\", nmode = " << nmode << "] Twist " << CorUncor[i*6 + 2] << ", " << CorUncor[i*6 + 3] << std::endl;
        // std::cout << "[CentBin " << i << "] ref = \"" << ref << "\", nmode = " << nmode << "] Rescale " << CorUncor[i*6 + 4] << ", " << CorUncor[i*6 + 5] << std::endl;
        // std::cout << std::endl;
    }
    c->Close();
    return CorUncor;
}


void getCorrections(TFile* inFile,
                    std::string outDir,
                    const DetectorMap& detMap,
                    int run,
                    int vn,
                    TFile* corrFile,
                    bool isEse) {

    int nCentBins = 100;
    int nDetectorBins = 10;
    int nCorrectionParams = 6;


    TH3F* hCCDB = new TH3F("ccdb", "",
                           nCentBins, 0, nCentBins,                          // cent
                           nCorrectionParams, 0-0.5, nCorrectionParams-0.5,  // const
                           nDetectorBins, 0-0.5, nDetectorBins-0.5);         // det

    std::string histBaseDir = isEse ? "q-vectors-correction" : "q-vectors-correction_ScalarProd";
    std::vector<double> QvecCorFT0C   = fillCorrections(outDir, inFile, detMap[kFT0C].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFT0C].dirsuffix.c_str()),   detMap[kFT0C].ref,   vn);
    std::vector<double> QvecCorFT0A   = fillCorrections(outDir, inFile, detMap[kFT0A].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFT0A].dirsuffix.c_str()),   detMap[kFT0A].ref,   vn);
    std::vector<double> QvecCorFT0M   = fillCorrections(outDir, inFile, detMap[kFT0M].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFT0M].dirsuffix.c_str()),   detMap[kFT0M].ref,   vn);
    std::vector<double> QvecCorFV0A   = fillCorrections(outDir, inFile, detMap[kFV0A].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFV0A].dirsuffix.c_str()),   detMap[kFV0A].ref,   vn);
    std::vector<double> QvecCorTPCPOS = fillCorrections(outDir, inFile, detMap[kTPCPOS].name, Form("%s%s", histBaseDir.c_str(), detMap[kTPCPOS].dirsuffix.c_str()), detMap[kTPCPOS].ref, vn);
    std::vector<double> QvecCorTPCNEG = fillCorrections(outDir, inFile, detMap[kTPCNEG].name, Form("%s%s", histBaseDir.c_str(), detMap[kTPCNEG].dirsuffix.c_str()), detMap[kTPCNEG].ref, vn);
    std::vector<double> QvecCorTPCall = fillCorrections(outDir, inFile, detMap[kTPCALL].name, Form("%s%s", histBaseDir.c_str(), detMap[kTPCALL].dirsuffix.c_str()), detMap[kTPCALL].ref, vn);

    // Map your detector IDs to their corresponding data vector reference and enum index
    struct DetConfig {
        int id;
        const std::vector<double>& data;
    };
    std::vector<DetConfig> detectors = {
        {kFT0C,   QvecCorFT0C},   {kFT0A,   QvecCorFT0A},   {kFT0M,   QvecCorFT0M},
        {kFV0A,   QvecCorFV0A},   {kTPCPOS, QvecCorTPCPOS}, {kTPCNEG, QvecCorTPCNEG},
        {kTPCALL, QvecCorTPCall}
    };

    // Names of correction parameters matching your indexing order (0 to 5)
    std::vector<std::string> paramNames = {"RecenterX", "RecenterY", "TwistX", "TwistY", "RescaleX", "RescaleY"};

    // Setup 1D QA histograms programmatically 
    std::map<int, std::vector<TH1F>> hParams;
    for (const auto& det : detectors) {
        for (const auto& name : paramNames) {
            std::string hName = Form("hPar%s_%s", name.c_str(), detMap[det.id].name.c_str());
            std::string hTitle = Form("%s - %s", name.c_str(), detMap[det.id].name.c_str());
            hParams[det.id].emplace_back(hName.c_str(), hTitle.c_str(), nCentBins, 0, nCentBins);
        }
    }

    // 2. Fill Histograms using streamlined loops
    for (int i = 0; i < nCentBins; i++) {
        for (int j = 0; j < nCorrectionParams; j++) {
            int linearIndex = j + i * nCorrectionParams;

            for (const auto& det : detectors) {
                double val = det.data.at(linearIndex);
                hCCDB->SetBinContent(i + 1, j + 1, det.id + 1, val);
                hParams[det.id][j].SetBinContent(i + 1, val);
            }

            // Dummy detector bins (8, 9, 10) using TPCNEG data
            for (int dummyDet = 8; dummyDet <= 10; dummyDet++) {
                hCCDB->SetBinContent(i + 1, j + 1, dummyDet, QvecCorTPCNEG.at(linearIndex));
            }
        }
    }

    // 3. Create Directories and Write Histograms cleanly
    corrFile->cd();
    hCCDB->Write();

    for (const auto& det : detectors) {
        corrFile->mkdir(detMap[det.id].name.c_str());
        corrFile->cd(detMap[det.id].name.c_str());

        for (int j = 0; j < nCorrectionParams; j++) {
            hParams[det.id][j].Write();
        }
    }
}

void calibQVecs(std::string outDir,
                std::string inFileDir,
                int run,
                std::string wagonSuffixFT0A,
                std::string whichDetFT0A,
                std::string wagonSuffixFT0C,
                std::string whichDetFT0C,
                std::string wagonSuffixFT0M,
                std::string whichDetFT0M,
                std::string wagonSuffixFV0A,
                std::string whichDetFV0A,
                std::string wagonSuffixTPCPOS,
                std::string whichDetTPCPOS,
                std::string wagonSuffixTPCNEG,
                std::string whichDetTPCNEG,
                std::string wagonSuffixTPCALL,
                std::string whichDetTPCALL,
                bool isEse) {

    gErrorIgnoreLevel = kError;

    std::string qVectType = isEse ? "EsE" : "SP";
    gSystem->mkdir(Form("%s/runs/%d/%s/qvecCorrs/qvec_corrections.root", outDir.c_str(), run, qVectType.c_str()), true);

    DetectorMap detMap = {{
        {"FT0C",   wagonSuffixFT0C,   whichDetFT0C},
        {"FT0A",   wagonSuffixFT0A,   whichDetFT0A},
        {"FT0M",   wagonSuffixFT0M,   whichDetFT0M},
        {"FV0A",   wagonSuffixFV0A,   whichDetFV0A},
        {"TPCPOS", wagonSuffixTPCPOS, whichDetTPCPOS},
        {"TPCNEG", wagonSuffixTPCNEG, whichDetTPCNEG},
        {"TPCALL", wagonSuffixTPCALL, whichDetTPCALL}
    }};

    TFile* fileCorrs = new TFile(Form("%s/runs/%d/%s/qvecCorrs/qvec_corrections.root", outDir.c_str(), run, qVectType.c_str()), "recreate");
    TFile* inputFile = new TFile(Form("%s/AnalysisResults_%d.root", inFileDir.c_str(), run), "read");

    std::array<int, 1> harmonics = {2}; // ,3,4};
    for (int vn : harmonics) {
        getCorrections(inputFile,
                       Form("%s/runs/%d/%s/qvecCorrs/", outDir.c_str(), run, qVectType.c_str()),
                       detMap,
                       run,
                       vn,
                       fileCorrs,
                       isEse);
    }

    fileCorrs->Close();

}
