#!/bin/bash
#BSUB -J validation
module load miniconda/pytorch-tensorflow-cpu

python 01_promoter_logo.py --data_path "./data"  --validation T  &> validation.txt
