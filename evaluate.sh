#!/usr/bin/env bash

set -eu
python ./la_evaluate.py ./Scores/LA/Bmamba5_LA_WCE_1e-06_ES144_NE12.txt ./keys/LA eval

python ./df_evaluate.py ./Scores/DF/Bmamba3_LA_WCE_1e-06_ES144_NE12.txt ./ASVspoof2021DF_eval/ eval

python ./in_wild_evaluate.py ./Scores/In-the-Wild/Bmamba5_In-the-Wild_WCE_1e-06_ES144_NE12.txt ./ eval