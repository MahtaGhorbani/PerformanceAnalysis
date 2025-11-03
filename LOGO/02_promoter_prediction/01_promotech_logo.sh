#!/bin/bash
#BSUB -J LOGO_promo
#BSUB -o output.log
#BSUB -e error.log

module load miniconda/pytorch-tensorflow-cpu
python 01_promotech_logo.py\
	 --model_name promotech \
	--bacteria_name 'CJEJUNI'\
	--ngram 5  &> promotech_cjejuni.txt 
