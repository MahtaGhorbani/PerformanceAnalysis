#!/bin/bash
#BSUB -J train-human
#BSUB -n 64
module load miniconda/pytorch-tensorflow-cpu

python 01_promoter_logo.py --data_path "./data" --train T --pretrain_weight_path '../LOGO_5_gram_2_layer_8_heads_256_dim_weights_32-0.885107.hdf5'  &> train.txt
