#!/bin/bash
#BSUB -J predict
module load miniconda/pytorch-tensorflow-cpu

python 01_promoter_logo.py --data_path "./data"  --predict T  &> predict.txt
