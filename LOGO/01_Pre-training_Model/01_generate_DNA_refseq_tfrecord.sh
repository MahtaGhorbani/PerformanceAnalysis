#!/bin/bash
#BSUB -J tfrecord
#BSUB -o output.log
#BSUB -e error.log
#BSUB -n 32
python 01_generate_DNA_refseq_tfrecord.py \
  --data ../data/train \
  --output ../data/train/tf \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 5 \
  --stride 1 \
  --slice-size 100000 \
  --hg-name bacteria\
  --pool-size 1

