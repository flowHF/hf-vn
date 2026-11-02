#include "CCDB/CcdbApi.h"
#include "TFile.h"
#include "TH3F.h"
#include "TROOT.h"

int loadCorrectionsToCCDB(std::string correctionsPath,
                          int run,
                          long long sor,
                          long long eor,
                          std::string ccdbDir,
                          bool loadEse,
                          bool loadSp,
                          bool loadGainEq,
                          std::string apass,
                          int vn = 2) {

    gROOT->SetBatch(kTRUE);

    std::cout << "Loading corrections to CCDB for run " << run << " from file " << correctionsPath << ", SOR: " << sor << ", EOR: " << eor << std::endl;

    const string ccdbPath = "http://alice-ccdb.cern.ch";
    o2::ccdb::CcdbApi ccdb;
    map<string, string> metadata;//, metadataRCT, header; // NOTE: Re-enable the other two if timing information is needed.
    ccdb.init(Form("%s", ccdbPath.data()));

    if (loadSp) {
        std::cout << "Loading SP corrections from file " << Form("%s/runs/%d/SP/qvecCorrs/qvec_corrections.root", correctionsPath.data(), run) << std::endl;
        TFile* inFile = new TFile(Form("%s/runs/%d/SP/qvecCorrs/qvec_corrections.root", correctionsPath.data(), run), "read");
        TH3F* hCCDB = (TH3F*)inFile->Get("ccdb");
        if(vn==2){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "v2";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/v2", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/v2", ccdbDir.data()), metadata, sor, eor);
        }

        if(vn==3){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "v3";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/v3", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/v3", ccdbDir.data()), metadata, sor, eor);
        }

        if(vn==4){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "v4";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/v4", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/v4", ccdbDir.data()), metadata, sor, eor);
        }
        metadata.clear();
        inFile->Close();
    }

    if (loadEse) {
        TFile* inFile = new TFile(Form("%s/runs/%d/EsE/qvecCorrs/qvec_corrections.root", correctionsPath.data(), run), "read");
        TH3F* hCCDB = (TH3F*)inFile->Get("ccdb");
        if(vn == 2){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "eseq2";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/eseq2", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/eseq2", ccdbDir.data()), metadata, sor, eor);
        }

        if(vn == 3){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "eseq3";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/eseq3", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/eseq3", ccdbDir.data()), metadata, sor, eor);
        }

        if(vn == 4){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "eseq4";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/eseq4", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/eseq4", ccdbDir.data()), metadata, sor, eor);
        }
        metadata.clear();
        inFile->Close();
    }

    if (loadGainEq) {

        TFile* inFile = new TFile(Form("%s/runs/%d/gainCor/summary.root", correctionsPath.data(), run), "read");
        inFile->ls();

        TH1F* hRelGainsFT0A = (TH1F*)inFile->Get("histRelGainsFT0A");
        TH1F* hRelGainsFT0C = (TH1F*)inFile->Get("histRelGainsFT0C");
        TH1F* hRelGainsFV0 = (TH1F*)inFile->Get("histRelGainsFV0");

        // Unique vector for FT0A and FT0C, but FT0A is read
        // before FT0C in qVectorsTable, so we follow this order
        std::vector<float> relGainsFT0, relGainsFV0;

        std::cout << "\n\nLoading FT0A relative gains to CCDB for run " << run << std::endl;
        for (int iCh=0; iCh<hRelGainsFT0A->GetNbinsX(); iCh++) {
            std::cout << "FT0A channel " << iCh << " relative gain: " << hRelGainsFT0A->GetBinContent(iCh + 1) << std::endl;
            relGainsFT0.push_back(hRelGainsFT0A->GetBinContent(iCh + 1));
        }
        std::cout << "\n\nLoading FT0C relative gains to CCDB for run " << run << std::endl;
        for (int iCh=0; iCh<hRelGainsFT0C->GetNbinsX(); iCh++) {
            std::cout << "FT0C channel " << iCh << " relative gain: " << hRelGainsFT0C->GetBinContent(iCh + 1) << std::endl;
            relGainsFT0.push_back(hRelGainsFT0C->GetBinContent(iCh + 1));
        }
        std::cout << "\n\nLoading FV0 relative gains to CCDB for run " << run << std::endl;
        for (int iCh=0; iCh<hRelGainsFV0->GetNbinsX(); iCh++) {
            std::cout << "FV0 channel " << iCh << " relative gain: " << hRelGainsFV0->GetBinContent(iCh + 1) << std::endl;
            relGainsFV0.push_back(hRelGainsFV0->GetBinContent(iCh + 1));
        }

        metadata["runnum"] = std::to_string(run);
        metadata["detector"] = "FT0";
        metadata["apass"] = apass;
        ccdb.storeAsTFileAny(&relGainsFT0, Form("%s/GainEq/%s", ccdbDir.data(), "FT0"), metadata, sor, eor);
        ccdb.storeAsTFileAny(&relGainsFV0, Form("%s/GainEq/%s", ccdbDir.data(), "FV0"), metadata, sor, eor);
        metadata.clear();
    }

    return 0;
}
