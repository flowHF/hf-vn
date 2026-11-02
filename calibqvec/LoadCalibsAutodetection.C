#include "CCDB/CcdbApi.h"
#include "TH3F.h"

std::vector<int> Runs2023Pass4 = {
    545367, 545345, 545332, 545312, 545311, 545296, 545295, 545294, 545291,
    545289, 545262, 545249, 545246, 545223, 545222, 545210, 545185, 545184,
    545171, 545117, 545103, 545086, 545066, 545064, 545063, 545062, 545060,
    545047, 545044, 545042, 545041, 545009, 545008, 545004, 544992, 544991,
    544968, 544964, 544963, 544961, 544947, 544931, 544917, 544914, 544913,
    544911, 544896, 544887, 544886, 544868, 544813, 544797, 544795, 544794,
    544767, 544754, 544742, 544739, 544696, 544694, 544693, 544692, 544674,
    544672, 544653, 544652, 544640, 544614, 544585, 544583, 544582, 544580,
    544568, 544567, 544565, 544564, 544551, 544550, 544549, 544548, 544518,
    544515, 544514, 544512, 544511, 544510, 544508, 544492, 544491, 544490,
    544477, 544476, 544475, 544474, 544454, 544451, 544392, 544391, 544390,
    544389, 544185, 544184, 544124, 544123, 544122, 544121, 544116, 544098,
    544095, 544091, 544032, 544028, 544013
};

std::vector<int> Runs2024Pass3 = {
    559781, 559802, 559803, 559827, 559843, 559856, 559917, 559933, 559966,
    559968, 559969, 559970, 559987, 560012, 560031, 560033, 560049, 560066,
    560067, 560070, 560089, 560090, 560105, 560106, 560123, 560127, 560141,
    560142, 560184, 559575, 559596, 559828, 559830, 559901, 559919, 560164,
    559574, 559595, 560162, 560163, 559544, 559545, 559561, 559613, 559614,
    559615, 559616, 559617, 559631, 559632, 559679, 559680, 559681, 559713,
    559731, 559749, 559762, 559782, 559783, 559784, 559804, 559902, 559903,
    // 559920, 560034, 560168, 560169
};

std::vector<int> Runs2025Pass1 = {
    568721, 568709, 568695, 568694, 568678, 568664, 568663, 568648, 568647,
    568646, 568631, 568630, 568617, 568616, 568602, 568601, 568600, 568587,
    568586, 568568, 568539, 568538, 568526, 568525, 568524, 568522, 568521,
    568500, 568486, 568484, 568468, 568467, 568452, 568451, 568422, 568421,
    568409, 568393, 568380, 568367, 568365, 568322, 568321, 568294, 568282,
    568269, 568256, 568242, 568229, 568228, 568227, 568211, 568210, 568209,
    568191, 568146, 568117, 568080, 568067, 568041, 568040, 568039, 568026,
    568012, 567999, 567986, 567967, 567966, 567939, 567927, 567925, 567912,
    567911, 567905,
};

std::vector<int> Runs2026Pass1 = {
    572715, 572704, 572693, 572669, 572658, 572647, 572646, 572635, 572622,
    572609, 572593, 572581, 572570, 572559, 572557, 572556, 572543, 572542,
    572541, 572527, 572515, 572504, 572502, 572501, 572489, 572488, 572487,
    572469, 572467, 572449, 572438, 572422, 572408, 572397, 572384, 572369,
    572355, 572354, 572353, 572340, 572339, 572338, 572337, 572324, 572323,
    572311, 572307, 572306, 572304, 572303, 572292, 572280, 572279, 572268,
    572266, 572246, 572245, 572244, 572227, 572216, 572196, 572184, 572183,
    572150, 572106,
};

struct QVecCalibInfo {
    std::string uploadPath;
    std::string pathQVecCalib;
    std::string pathGainEq;
    std::string apassMetadata;
    bool requireApassWhenLoading;
    std::string timeStartName;
    std::string timeEndName;
};

QVecCalibInfo Info2023Pass4 = {
    "Users/m/mdicosta/QVecsAutodetect/CalibV1/LHC23_apass4",
    "Analysis/EventPlane/QVecCorrections",
    "Analysis/EventPlane/GainEq",
    "apass4",
    true,
    "SOR",
    "EOR"
};

QVecCalibInfo Info2024Pass3 = {
    "Users/m/mdicosta/QVecsAutodetect/CalibV1/LHC24_apass3",
    "Users/m/mavirta/Qvector/LHC24_PbPb_pass3/QvecCalib",
    "Users/m/mavirta/Qvector/LHC24_PbPb_pass3/GainEq",
    "apass3",
    false,
    "SOR",
    "EOR"
};

QVecCalibInfo Info2025Pass1 = {
    "Users/m/mdicosta/QVecsAutodetect/CalibV1/LHC25_apass1",
    "Users/j/junlee/Qvector/Pass1/QvecCalib",
    "Users/j/junlee/Qvector/Pass1/GainEq",
    "apass1",
    false,
    "SOR",
    "EOR"
};

QVecCalibInfo Info2026Pass1 = {
    "Users/m/mdicosta/QVecsAutodetect/CalibV1/LHC26_apass1",
    "Users/j/junlee/Qvector/Pass1/QvecCalib",
    "Users/j/junlee/Qvector/Pass1/GainEq",
    "apass1",
    false,
    "STF",
    "ETF"
};

std::vector<std::tuple<std::vector<int>, QVecCalibInfo>> folderPeriodMap = {
    // {Runs2023Pass4, Info2023Pass4},
    // {Runs2024Pass3, Info2024Pass3},
    // {Runs2025Pass1, Info2025Pass1},
    {Runs2026Pass1, Info2026Pass1},
};

std::vector<std::string> harmonics = {"2"}; // {"2", "3", "4"};
std::vector<std::string> gainEqDets = {"FT0", "FV0"};

int LoadCalibsAutodetection()
{
    o2::ccdb::CcdbApi ccdb;
    ccdb.init(o2::base::NameConf::getCCDBServer());

    for (const auto &[runs, qVecCalibInfo] : folderPeriodMap) {
        std::string outFileName = qVecCalibInfo.uploadPath.substr(qVecCalibInfo.uploadPath.find_last_of("/") + 1);
        TFile *outFile = new TFile(Form("%s.root", outFileName.c_str()), "RECREATE");

        for (const auto &run : runs) {
            const auto headers = ccdb.retrieveHeaders("RCT/Info/RunInformation", {}, run);
            const auto itSOR = headers.find(qVecCalibInfo.timeStartName);
            const auto itEOR = headers.find(qVecCalibInfo.timeEndName);

            if (itSOR == headers.end() || itEOR == headers.end()) {
                std::cerr << "[SKIP] Run " << run << ": missing SOR/EOR" << std::endl;
                continue;
            }

            const uint64_t sorTimestamp = std::stoull(itSOR->second);
            const uint64_t eorTimestamp = std::stoull(itEOR->second);

            std::cout << "\n---> Run: " << run
                      << ", " << qVecCalibInfo.timeStartName << ": " << sorTimestamp
                      << ", " << qVecCalibInfo.timeEndName << ": " << eorTimestamp << std::endl;

            std::map<std::string, std::string> mapMetadata;
            if (qVecCalibInfo.requireApassWhenLoading) {
                mapMetadata["RecoPassName"] = qVecCalibInfo.apassMetadata;
            }

            std::map<std::string, std::string> mapHeader;

            // Objects retrieved for this run.
            std::vector<TH3F*> qVecCalibs;
            std::map<std::string, std::vector<float>> gainEqCalibs;
            bool skipRun = false;

            // Retrieve all Q-vector calibrations
            for (const auto &harmonic : harmonics) {
                try {
                    TH3F *qVec = ccdb.retrieveFromTFileAny<TH3F>(
                        Form("%s/v%s", qVecCalibInfo.pathQVecCalib.c_str(), harmonic.c_str()),
                        mapMetadata, sorTimestamp, &mapHeader);

                    if (!qVec) {
                        throw std::runtime_error("retrieved object is nullptr");
                    }

                    qVec->SetName(Form("QVecCalib_Run%d_v%s", run, harmonic.c_str()));
                    qVecCalibs.push_back(qVec);
                } catch (const std::exception &e) {
                    std::cerr << "[SKIP] Run " << run
                              << ": missing QVec harmonic " << harmonic
                              << ": " << e.what() << std::endl;
                    skipRun = true;
                    break;
                }
            }

            if (skipRun) {
                std::cout << "[SKIP] Run " << run << ": skipping due to missing QVec calibrations." << std::endl;
                continue;
            }

            // Retrieve all GainEq calibrations
            for (const auto &gainEqDet : gainEqDets) {
                try {
                    auto gainEq = ccdb.retrieveFromTFileAny<std::vector<float>>(
                        Form("%s/%s", qVecCalibInfo.pathGainEq.c_str(), gainEqDet.c_str()),
                        mapMetadata, sorTimestamp, &mapHeader);

                    if (!gainEq)
                        throw std::runtime_error("retrieved object is nullptr");

                    gainEqCalibs[gainEqDet] = *gainEq;
                } catch (const std::exception &e) {
                    std::cerr << "[SKIP] Run " << run
                              << ": missing GainEq " << gainEqDet
                              << ": " << e.what() << std::endl;
                    skipRun = true;
                    break;
                }
            }

            if (skipRun) {
                std::cout << "[SKIP] Run " << run << ": skipping due to missing GainEq calibrations." << std::endl;
                continue;
            }

            // All requested calibrations exist.
            std::cout << "[OK] Run " << run << ": all calibrations retrieved." << std::endl;

            // Load to CCDB based on qVecCalibInfo.uploadPath
            std::map<string, string> metadata; //, metadataRCT, header;
            for (size_t i = 0; i < qVecCalibs.size(); ++i) {
                TH3F *hCCDB = qVecCalibs[i];
                const std::string &harmonic = harmonics[i];

                metadata["runnum"] = std::to_string(run);
                metadata["RecoPassName"] = qVecCalibInfo.apassMetadata;
                metadata["harmonics"] = harmonic;
                std::cout << "\nWill load histogram to ccdb path " << Form("%s/v%s", qVecCalibInfo.uploadPath.data(), harmonic.c_str()) << ", sorTimestamp: " << sorTimestamp << ", eorTimestamp: " << eorTimestamp << std::endl;
                ccdb.storeAsTFileAny(hCCDB, Form("%s/v%s", qVecCalibInfo.uploadPath.data(), harmonic.c_str()), metadata, sorTimestamp, eorTimestamp);
                // Print metadata for verification
                std::cout << "Metadata for harmonic " << harmonic << ": ";
                for (const auto &[key, value] : metadata) {
                    std::cout << key << "=" << value << " ";
                }
                std::cout << std::endl;
                metadata.clear();

            }

            metadata["runnum"] = std::to_string(run);
            metadata["RecoPassName"] = qVecCalibInfo.apassMetadata;
            metadata["detector"] = "FT0";
            for (const auto &[key, value] : metadata) {
                std::cout << key << "=" << value << " ";
            }
            std::cout << std::endl;
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/GainEq/%s", qVecCalibInfo.uploadPath.data(), "FT0") << ", sorTimestamp: " << sorTimestamp << ", eorTimestamp: " << eorTimestamp << std::endl;
            ccdb.storeAsTFileAny(&gainEqCalibs.at("FT0"), Form("%s/GainEq/%s", qVecCalibInfo.uploadPath.data(), "FT0"), metadata, sorTimestamp, eorTimestamp);
            metadata["detector"] = "FV0";
            for (const auto &[key, value] : metadata) {
                std::cout << key << "=" << value << " ";
            }
            std::cout << std::endl;
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/GainEq/%s", qVecCalibInfo.uploadPath.data(), "FV0") << ", sorTimestamp: " << sorTimestamp << ", eorTimestamp: " << eorTimestamp << std::endl;
            ccdb.storeAsTFileAny(&gainEqCalibs.at("FV0"), Form("%s/GainEq/%s", qVecCalibInfo.uploadPath.data(), "FV0"), metadata, sorTimestamp, eorTimestamp);
            metadata.clear();

            // Write the objects after all retrievals succeeded.
            std::string runDir = Form("Run%d_SOR_%llu_EOR_%llu", run,
                                      static_cast<unsigned long long>(sorTimestamp),
                                      static_cast<unsigned long long>(eorTimestamp));

            for (size_t i = 0; i < qVecCalibs.size(); ++i) {

                TH3F *qVec = qVecCalibs[i];
                const std::string &harmonic = harmonics[i];
                std::string dir = Form("%s/QVecCalibs/v%s", runDir.c_str(), harmonic.c_str());

                outFile->mkdir(dir.c_str());
                outFile->cd(dir.c_str());

                qVec->Write();

                TH1 *projXY = qVec->Project3D("xy");
                projXY->SetName(Form("QVecCalibProjXY_Run%d", run));
                projXY->Write();
                TH1 *projXZ = qVec->Project3D("xz");
                projXZ->SetName(Form("QVecCalibProjXZ_Run%d", run));
                projXZ->Write();
                TH1 *projYZ = qVec->Project3D("yz");
                projYZ->SetName(Form("QVecCalibProjYZ_Run%d", run));
                projYZ->Write();
            }

            for (const auto &[det, values] : gainEqCalibs) {

                std::string dir = Form("%s/GainEqCalibs/%s", runDir.c_str(), det.c_str());
                outFile->mkdir(dir.c_str());
                outFile->cd(dir.c_str());

                std::string histoName = Form("GainEqCalib_Run%d_Det%s", run, det.c_str());
                TH1F histoGainEq(histoName.c_str(), histoName.c_str(),
                                 values.size(), 0, values.size());

                for (size_t i = 0; i < values.size(); ++i)
                    histoGainEq.SetBinContent(i + 1, values[i]);

                histoGainEq.Write();
            }
        }

        outFile->Close();
        delete outFile;
    }

    return 0;
}
