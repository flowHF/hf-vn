#include "CCDB/CcdbApi.h"
#include "TFile.h"
#include "TH3F.h"
#include "TFile.h"
#include "TROOT.h"

int LoadResolutionToCCDB(std::string resoPath, std::string ccdbFolder, std::string timestampFile, std::string apass, int year, bool load) {

    gROOT->SetBatch(kTRUE);

    std::cout << "Loading resolution from " << resoPath << " to ccdbFolder " 
              << ccdbFolder << ", timestamps taken from " << timestampFile 
              << ", will load: " << (load ? "true" : "false") << std::endl;

    // Determine lowest SOR and highest EOR from the timestamp file
    long long firstTimestamp = std::numeric_limits<long long>::max();
    long long lastTimestamp = std::numeric_limits<long long>::min();

    std::ifstream tsFile(timestampFile);
    if (!tsFile.is_open()) {
        std::cerr << "Error: could not open timestamp file: " << timestampFile << std::endl;
        return -1;
    }

    int run;
    long long sor, eor;

    std::string header;
    std::getline(tsFile, header); // Skip RUN SOR EOR header

    while (tsFile >> run >> sor >> eor) {
        firstTimestamp = std::min(firstTimestamp, sor);
        lastTimestamp = std::max(lastTimestamp, eor);
    }
    if (firstTimestamp == std::numeric_limits<long long>::max() ||
        lastTimestamp == std::numeric_limits<long long>::min()) {
        std::cerr << "Error: no valid timestamps found in " << timestampFile << std::endl;
        return -1;
    }
    std::cout << "CCDB validity interval: " << firstTimestamp << " -> " << lastTimestamp << std::endl;

    const string ccdbPath = "http://alice-ccdb.cern.ch";
    o2::ccdb::CcdbApi ccdb;
    map<string, string> metadata;//, metadataRCT, header; // NOTE: Re-enable the other two if timing information is needed.
    ccdb.init(Form("%s", ccdbPath.data()));

    TFile* resoFile = TFile::Open(resoPath.data(), "READ");
    std::vector<std::string> tripletDirs = {
        "FT0c_FT0a_FV0a",
        "FT0c_FT0a_TPCpos",
        "FT0c_FT0a_TPCneg",
        "FT0c_FT0a_TPCtot",
        "FT0c_FV0a_TPCpos",
        "FT0c_FV0a_TPCneg",
        "FT0c_FV0a_TPCtot",
        "FT0c_TPCpos_TPCneg",
        "FT0a_FV0a_TPCpos",
        "FT0a_FV0a_TPCneg",
        "FT0a_FV0a_TPCtot",
        "FT0a_TPCpos_TPCneg",
        "FT0m_FV0a_TPCpos",
        "FT0m_FV0a_TPCneg",
        "FT0m_FV0a_TPCtot",
        "FT0m_TPCpos_TPCneg",
        "FV0a_TPCpos_TPCneg"
    };

    for (const auto& triplet : tripletDirs) {

        std::array<float, 100> resoValsVsCent{};

        std::string histoName = Form("%s/histo_reso", triplet.data());
        TH1F* hReso = (TH1F*)resoFile->Get(histoName.data());
        if (!hReso) {
            std::cerr << "Error: Histogram " << histoName << " not found in file " << resoPath << std::endl;
            continue;
        }

        std::cout << std::endl;
        std::cout << "Processing triplet: " << triplet << std::endl;
        for (int i = 0; i < hReso->GetNbinsX(); ++i) {
            std::cout << "Pushing back reso value " << hReso->GetBinContent(i + 1) << " for centrality " << i + 0.5f << std::endl;
            resoValsVsCent[i] = hReso->GetBinContent(i + 1);
        }
        std::cout << std::endl;

        std::cout << "Will load resolution values for triplet " << triplet 
                  << " to ccdb path " << Form("%s/%s", ccdbFolder.data(), triplet.data()) 
                  << " with metadata: " << apass << ", " << year 
                  << " and timestamps: " << firstTimestamp << " -> " << lastTimestamp
                  << std::endl;
        if (load) {
            std::cout << "Loading to CCDB..." << std::endl;
            metadata["triplet"] = triplet;
            metadata["year"] = std::to_string(year);
            metadata["RecoPassName"] = apass;
            ccdb.storeAsTFileAny(&resoValsVsCent, Form("%s/%s", ccdbFolder.data(), triplet.data()), metadata, firstTimestamp, lastTimestamp);
            metadata.clear();
        }
    }

    return 0;
}
