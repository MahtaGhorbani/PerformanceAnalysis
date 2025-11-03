#!/bin/bash
#BSUB -J validation2
module load miniconda/pytorch-tensorflow-cpu

python 01_promoter_logo.py --data_path "./data"  --validation T  &> validation2.txt
