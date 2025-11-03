#!/bin/bash
#BSUB -J validation
#BSUB -n 32
module load miniconda/pytorch-tensorflow-cpu

python 01_promoter_logo.py --data_path "./data"  --validation T  &> validation.txt
