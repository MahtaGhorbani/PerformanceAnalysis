#!/bin/bash
#BSUB -J LOGO_promo_mlp

module load miniconda/pytorch-tensorflow-cpu
python 01_promotech_logo.py\
	 --model_name mlp \
	--bacteria_name 'CJEJUNI'\
	--ngram 5  &> mlp.txt 
