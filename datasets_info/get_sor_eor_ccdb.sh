#!/bin/bash

echo "================================================="
echo "Getting SOR and EOR information from CCDB for 2026 Pb-Pb runs"
echo "================================================="

OUTFILE="run_sor_eor_pbpb_2026.txt"
: > "$OUTFILE"

RUNSPBPB2026=(
572715 572704 572693 572669 572658 572647 572646 572635 572622
572609 572593 572581 572570 572559 572557 572556 572543 572542
572541 572527 572515 572504 572502 572501 572489 572488 572487
572469 572467 572449 572438 572422 572408 572397 572384 572369
572355 572354 572353 572340 572339 572338 572337 572324 572323
572311 572307 572306 572304 572303 572292 572280 572279 572268
572266 572246 572245 572244 572227 572216 572196 572184 572183
572150 572106
)

echo "RUN          SOR                 EOR" >> "$OUTFILE"


for RUN in "${RUNSPBPB2026[@]}"; do

    STF=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}" \
        | awk '/^STF:/ {print $2}' \
        | tr -d '\r\n')

    ETF=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}" \
        | awk '/^ETF:/ {print $2}' \
        | tr -d '\r\n')

    # Display all the information in curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}"
    display_info=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}")
    echo "Information for RUN ${RUN}:"
    echo "$display_info"

    echo "${RUN}       ${STF}       ${ETF}"
    echo "${RUN}       ${STF}       ${ETF}" >> "$OUTFILE"

done

echo "Saved to ${OUTFILE}"


# echo "================================================="
# echo "Getting SOR and EOR information from CCDB for 2025 Pb-Pb runs"
# echo "================================================="

# OUTFILE="run_sor_eor_pbpb_2025.txt"
# : > "$OUTFILE"

# RUNSPBPB2025=(
# 568721 568709 568695 568694 568678 568664 568663 568648 568647 568646
# 568631 568630 568617 568616 568602 568601 568600 568587 568586 568568
# 568539 568538 568526 568525 568524 568522 568521 568500 568486 568484
# 568468 568467 568452 568451 568422 568421 568409 568393 568380 568367
# 568365 568322 568321 568294 568282 568269 568256 568242 568229 568228
# 568227 568211 568210 568209 568191 568146 568117 568080 568067 568041
# 568040 568039 568026 568012 567999 567986 567967 567966 567939 567927
# 567925 567912 567911 567905
# )

# echo "RUN          SOR                 EOR" >> "$OUTFILE"


# for RUN in "${RUNSPBPB2025[@]}"; do

#     SOR=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}" \
#         | awk '/^SOR:/ {print $2}' \
#         | tr -d '\r\n')

#     EOR=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}" \
#         | awk '/^EOR:/ {print $2}' \
#         | tr -d '\r\n')

#     echo "${RUN}       ${SOR}       ${EOR}" >> "$OUTFILE"

# done

# echo "Saved to ${OUTFILE}"


# echo "================================================="
# echo "Getting SOR and EOR information from CCDB for 2024 Pb-Pb runs"
# echo "================================================="

# OUTFILE="run_sor_eor_pbpb_2024.txt"
# : > "$OUTFILE"

# RUNSPBPB2024=(
# 560401 560399 560371 560355 560352 560331 560330
# 560402 560397 560229 560334 560314 560313 560310
# 560279 560244 560243 560223 560169 560168 560034
# 559920 559903 559902 559804 559784 559783 559782
# 559762 559749 559731 559713 559681 559680 559679
# 559632 559631 559617 559616 559615 559614 559613
# 559561 559545 559544 560163 560162 559595 559574
# 560164 559919 559901 559830 559828 559596 559575
# 560184 560142 560141 560127 560123 560106 560105
# 560090 560089 560070 560067 560066 560049 560033
# 560031 560012 559987 559970 559969 559968 559966
# 559933 559917 559856 559843 559827 559803 559802
# 559781
# )

# echo "RUN          SOR                 EOR" >> "$OUTFILE"


# for RUN in "${RUNSPBPB2024[@]}"; do

#     SOR=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}" \
#         | awk '/^SOR:/ {print $2}' \
#         | tr -d '\r\n')

#     EOR=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}" \
#         | awk '/^EOR:/ {print $2}' \
#         | tr -d '\r\n')

#     echo "${RUN}       ${SOR}       ${EOR}" >> "$OUTFILE"

# done

# echo "Saved to ${OUTFILE}"


# echo "================================================="
# echo "Getting SOR and EOR information from CCDB for 2023 Pb-Pb runs"
# echo "================================================="

# OUTFILE="run_sor_eor_pbpb_2023.txt"
# : > "$OUTFILE"

# RUNSPBPB2023=(
# 545367 545345 545332 545312 545311 545296 545295 545294
# 545291 545289 545262 545249 545246 545223 545222 545210
# 545185 545184 545171 545117 545103 545086 545066 545064
# 545063 545062 545060 545047 545044 545042 545041 545009
# 545008 545004 544992 544991 544968 544964 544963 544961
# 544947 544931 544917 544914 544913 544911 544896 544887
# 544886 544868 544813 544797 544795 544794 544767 544754
# 544742 544739 544696 544694 544693 544692 544674 544672
# 544653 544652 544640 544614 544585 544583 544582 544580
# 544568 544567 544565 544564 544551 544550 544549 544548
# 544518 544515 544514 544512 544511 544510 544508 544492
# 544491 544490 544477 544476 544475 544474 544454 544451
# 544392 544391 544390 544389 544185 544184 544124 544123
# 544122 544121 544116 544098 544095 544091 544032 544028
# 544013
# )

# echo "RUN          SOR                 EOR" >> "$OUTFILE"


# for RUN in "${RUNSPBPB2023[@]}"; do

#     SOR=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}" \
#         | awk '/^SOR:/ {print $2}' \
#         | tr -d '\r\n')

#     EOR=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}" \
#         | awk '/^EOR:/ {print $2}' \
#         | tr -d '\r\n')

#     echo "${RUN}       ${SOR}       ${EOR}" >> "$OUTFILE"

# done

# echo "Saved to ${OUTFILE}"


# echo "================================================="
# echo "Getting SOR and EOR information from CCDB for 2025 OO runs"
# echo "================================================="

# OUTFILE="run_sor_eor_oo_2025.txt"
# : > "$OUTFILE"

# RUNSOO2025=(
# 564445 564430 564414 564400 564387 
# 564374 564373 564359 564356
# )

# echo "RUN          SOR                 EOR" >> "$OUTFILE"


# for RUN in "${RUNSOO2025[@]}"; do

#     SOR=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}" \
#         | awk '/^SOR:/ {print $2}' \
#         | tr -d '\r\n')

#     EOR=$(curl -sI "http://alice-ccdb.cern.ch/RCT/Info/RunInformation/${RUN}" \
#         | awk '/^EOR:/ {print $2}' \
#         | tr -d '\r\n')

#     echo "${RUN}       ${SOR}       ${EOR}" >> "$OUTFILE"

# done

# echo "Saved to ${OUTFILE}"
