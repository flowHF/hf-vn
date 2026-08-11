#include "TFile.h"
#include "TH2D.h"
#include "TH3F.h"
#include "TError.h"
#include "TSystem.h"
#include <iostream>

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


void projQVecs(std::vector<std::pair<int, int>> ranges, TFile* outfile, TFile* infile, string detector, string dirname, string ref, string corrtype, string corrfolder, int nmode){

    TH3F* hQxQyVsCent = (TH3F*)infile->Get(Form("%s/histQvec%s%sV%d",dirname.c_str(),ref.c_str(),corrfolder.c_str(),nmode));

    int nCentRanges = ranges.size();
    TH2F* hQvecCorrSteps[nCentRanges];
    for(int i=0;i<nCentRanges;i++){

        int firstBin = hQxQyVsCent->GetZaxis()->FindBin(ranges[i].first); // Use the lower edge of the centrality range for binning
        int lastBin = hQxQyVsCent->GetZaxis()->FindBin(ranges[i].second - 1); // Use the upper edge of the centrality range for binning
        int centMin = static_cast<int>(hQxQyVsCent->GetZaxis()->GetBinLowEdge(firstBin));
        int centMax = static_cast<int>(hQxQyVsCent->GetZaxis()->GetBinUpEdge(lastBin));

        const auto dirName = Form("cent_%i_%i/%s", centMin, centMax, detector.c_str());
        TDirectory* dir = outfile->GetDirectory(dirName);
        if (!dir) {
            dir = outfile->mkdir(dirName);
        }
        outfile->cd(dirName);

        hQxQyVsCent->GetZaxis()->SetRange(firstBin, lastBin);
        hQvecCorrSteps[i] = (TH2F*)hQxQyVsCent->Project3D("yx");
        std::string histoProjName = Form("hQvec%s%sV%d_cent_%i_%i_%s",detector.c_str(),corrtype.c_str(),nmode, centMin, centMax, corrtype.c_str());
        hQvecCorrSteps[i]->SetTitle(Form("%s;Q_{x};Q_{y}", histoProjName.c_str()));
        hQvecCorrSteps[i]->Write(histoProjName.c_str(), TObject::kOverwrite);
    }

    return;
}


void projEvtPlane(std::vector<std::pair<int, int>> ranges, TFile* outfile, TFile* infile, string detector, string dirname, string ref, string corrtype, string corrfolder, int nmode){

    TH2F* hEvtPlaneVsCent = (TH2F*)infile->Get(Form("%s/histEvtPl%s%sV%d",dirname.c_str(),ref.c_str(),corrfolder.c_str(),nmode));

    int nCentRanges = ranges.size();
    TH1D* hPsiN[nCentRanges];
    for(int i=0;i<nCentRanges;i++){

        int firstBin = hEvtPlaneVsCent->GetYaxis()->FindBin(ranges[i].first); // Use the lower edge of the centrality range for binning
        int lastBin = hEvtPlaneVsCent->GetYaxis()->FindBin(ranges[i].second - 1); // Use the upper edge of the centrality range for binning
        int centMin = static_cast<int>(hEvtPlaneVsCent->GetYaxis()->GetBinLowEdge(firstBin));
        int centMax = static_cast<int>(hEvtPlaneVsCent->GetYaxis()->GetBinUpEdge(lastBin));

        const auto dirName = Form("cent_%i_%i/%s", centMin, centMax, detector.c_str());
        TDirectory* dir = outfile->GetDirectory(dirName);
        if (!dir) {
            dir = outfile->mkdir(dirName);
        }
        outfile->cd(dirName);

        std::string histoProjName = Form("hEvtPlane%s%sV%d_cent_%i_%i_%s",detector.c_str(),corrtype.c_str(),nmode, centMin, centMax, corrtype.c_str());
        hPsiN[i] = static_cast<TH1D*>(hEvtPlaneVsCent->ProjectionX(histoProjName.c_str(), firstBin, lastBin));
        hPsiN[i]->SetTitle(Form("%s;#Psi_{%i};Counts", histoProjName.c_str(), nmode));
        hPsiN[i]->Write(histoProjName.c_str(), TObject::kOverwrite);
    }

    return;
}


void checkCalibrations(std::string outDir, std::string inFileDir, int run,
                       std::string wagonSuffixFT0A,   std::string whichDetFT0A,
                       std::string wagonSuffixFT0C,   std::string whichDetFT0C,
                       std::string wagonSuffixFT0M,   std::string whichDetFT0M,
                       std::string wagonSuffixFV0A,   std::string whichDetFV0A,
                       std::string wagonSuffixTPCPOS, std::string whichDetTPCPOS,
                       std::string wagonSuffixTPCNEG, std::string whichDetTPCNEG,
                       std::string wagonSuffixTPCALL, std::string whichDetTPCALL,
                       bool isEse) {

    gErrorIgnoreLevel = kError;

    std::string qVectType = isEse ? "EsE" : "SP";
    std::string runOutDir = Form("%s/CheckCalibrations/%s/", outDir.c_str(), qVectType.c_str());
    gSystem->mkdir(runOutDir.c_str(), true);

    DetectorMap detMap = {{
        {"FT0C",   wagonSuffixFT0C,   whichDetFT0C},
        {"FT0A",   wagonSuffixFT0A,   whichDetFT0A},
        {"FT0M",   wagonSuffixFT0M,   whichDetFT0M},
        {"FV0A",   wagonSuffixFV0A,   whichDetFV0A},
        {"TPCPOS", wagonSuffixTPCPOS, whichDetTPCPOS},
        {"TPCNEG", wagonSuffixTPCNEG, whichDetTPCNEG},
        {"TPCALL", wagonSuffixTPCALL, whichDetTPCALL}
    }};
    std::map<std::string, std::string> correctionSteps = {
        {"Uncorrected", "Uncor"},
        {"Recenter",    "Rectr"},
        {"Twist",       "Twist"},
        {"Rescale",     "Final"},
        {"Final",       "Final"}
    };

    std::vector<std::pair<int, int>> centRanges;
    centRanges.emplace_back(0, 80);
    for (int i = 0; i < 100; i += 10) {
        centRanges.emplace_back(i, i + 10);
    }
    for (int i = 0; i < 100; ++i) {
        centRanges.emplace_back(i, i + 1);
    }

    TFile* inFile = new TFile(Form("%s/AnalysisResults_%d.root", inFileDir.c_str(), run), "read");
    // Gracefully exit if the input file cannot be opened
    if (!inFile || inFile->IsZombie()) {
        std::cerr << "Error: Could not open input file: " << Form("%s/AnalysisResults_%d.root", inFileDir.c_str(), run) << std::endl;
        delete inFile;
        return;
    }

    std::array<int, 1> harmonics = {2}; // ,3,4};
    TFile* outFileQVecs = new TFile(Form("%s/CheckCalibrations/%s/checkQVecs_%d.root", outDir.c_str(), qVectType.c_str(), run), "recreate");
    for (int vn : harmonics) {
        for (const auto& [corrType, corrFolderName] : correctionSteps) {
            std::string histBaseDir = isEse ? "q-vectors-correction" : "q-vectors-correction_ScalarProd";
            projQVecs(centRanges, outFileQVecs, inFile, detMap[kFT0C].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFT0C].dirsuffix.c_str()),   detMap[kFT0C].ref,   corrType, corrFolderName, vn);
            projQVecs(centRanges, outFileQVecs, inFile, detMap[kFT0A].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFT0A].dirsuffix.c_str()),   detMap[kFT0A].ref,   corrType, corrFolderName, vn);
            projQVecs(centRanges, outFileQVecs, inFile, detMap[kFT0M].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFT0M].dirsuffix.c_str()),   detMap[kFT0M].ref,   corrType, corrFolderName, vn);
            projQVecs(centRanges, outFileQVecs, inFile, detMap[kFV0A].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFV0A].dirsuffix.c_str()),   detMap[kFV0A].ref,   corrType, corrFolderName, vn);
            projQVecs(centRanges, outFileQVecs, inFile, detMap[kTPCPOS].name, Form("%s%s", histBaseDir.c_str(), detMap[kTPCPOS].dirsuffix.c_str()), detMap[kTPCPOS].ref, corrType, corrFolderName, vn);
            projQVecs(centRanges, outFileQVecs, inFile, detMap[kTPCNEG].name, Form("%s%s", histBaseDir.c_str(), detMap[kTPCNEG].dirsuffix.c_str()), detMap[kTPCNEG].ref, corrType, corrFolderName, vn);
            projQVecs(centRanges, outFileQVecs, inFile, detMap[kTPCALL].name, Form("%s%s", histBaseDir.c_str(), detMap[kTPCALL].dirsuffix.c_str()), detMap[kTPCALL].ref, corrType, corrFolderName, vn);
        }
    }
    outFileQVecs->Close();

    TFile* outFileEvtPlane = new TFile(Form("%s/CheckCalibrations/%s/checkEvtPlane_%d.root", outDir.c_str(), qVectType.c_str(), run), "recreate");
    for (int vn : harmonics) {
        for (const auto& [corrType, corrFolderName] : correctionSteps) {

            std::string histBaseDir = isEse ? "q-vectors-correction" : "q-vectors-correction_ScalarProd";
            projEvtPlane(centRanges, outFileEvtPlane, inFile, detMap[kFT0C].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFT0C].dirsuffix.c_str()),   detMap[kFT0C].ref,   corrType, corrFolderName, vn);
            projEvtPlane(centRanges, outFileEvtPlane, inFile, detMap[kFT0A].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFT0A].dirsuffix.c_str()),   detMap[kFT0A].ref,   corrType, corrFolderName, vn);
            projEvtPlane(centRanges, outFileEvtPlane, inFile, detMap[kFT0M].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFT0M].dirsuffix.c_str()),   detMap[kFT0M].ref,   corrType, corrFolderName, vn);
            projEvtPlane(centRanges, outFileEvtPlane, inFile, detMap[kFV0A].name,   Form("%s%s", histBaseDir.c_str(), detMap[kFV0A].dirsuffix.c_str()),   detMap[kFV0A].ref,   corrType, corrFolderName, vn);
            projEvtPlane(centRanges, outFileEvtPlane, inFile, detMap[kTPCPOS].name, Form("%s%s", histBaseDir.c_str(), detMap[kTPCPOS].dirsuffix.c_str()), detMap[kTPCPOS].ref, corrType, corrFolderName, vn);
            projEvtPlane(centRanges, outFileEvtPlane, inFile, detMap[kTPCNEG].name, Form("%s%s", histBaseDir.c_str(), detMap[kTPCNEG].dirsuffix.c_str()), detMap[kTPCNEG].ref, corrType, corrFolderName, vn);
            projEvtPlane(centRanges, outFileEvtPlane, inFile, detMap[kTPCALL].name, Form("%s%s", histBaseDir.c_str(), detMap[kTPCALL].dirsuffix.c_str()), detMap[kTPCALL].ref, corrType, corrFolderName, vn);
        }
    }
    outFileEvtPlane->Close();
}
