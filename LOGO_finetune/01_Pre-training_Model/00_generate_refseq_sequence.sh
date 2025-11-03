#!/bin/bash

python 00_generate_refseq_sequence.py \
  --data ../data/ncbi-genomes-2023-05-30/all_in_one.fna \
  --output ../data/train \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 5 \
  --stride 1 \
  --slice-size 100000 \
  --hg-name bacteria \
  --pool-size 32 &> out.txt

