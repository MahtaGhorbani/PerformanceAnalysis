#!/bin/bash
#CUDA_VISIBLE_DEVICES=0,1,2,3
#BSUB -J finetune 
#BSUB -o output.log
#BSUB -e error.log
module load miniconda/pytorch-tensorflow-gpu
#/admin/cair-sw/anaconda3/ppc64le/bin/conda activate logo_3
CUDA_VISIBLE_DEVICES=4 python 02_train_gene_transformer_lm_hg_bert4keras_tfrecord.py \
  --save ./model \
  --train-data ../data/train/tf \
  --seq-len 1000 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 256 \
  --ngram 5 \
  --stride 5 \
  --model-name genebert \
  --steps-per-epoch 4000 \
  --weight-path LOGO_5_gram_2_layer_8_heads_256_dim_weights_32-0.885107.hdf5 \
  --shuffle-size 4000 &> shout.txt

