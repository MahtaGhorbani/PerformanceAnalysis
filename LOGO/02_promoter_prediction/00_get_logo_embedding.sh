#!/bin/bash
#BSUB -J LOGO_embedding
module load miniconda/pytorch-tensorflow-cpu
# Define the directory where your bacteria folders are located
data_dir="../Promotech/datasets/training/40nt-sequences/bacteria-1-10-ratio"

# Loop through each folder in the data directory
for bacteria_folder in "$data_dir"/*; do
    if [ -d "$bacteria_folder" ]; then
        # Get the bacteria name from the folder name (assuming the folder name is the bacteria name)
        bacteria_name=$(basename "$bacteria_folder")

        # Run your Python script with the appropriate arguments for this folder
        python 00_get_logo_embedding.py --ngram 5 \
            --fasta_files_positive "$bacteria_folder/positive.fasta" \
            --fasta_files_negative "$bacteria_folder/negative.fasta" \
            --bacteria_name "$bacteria_name"\
           --pretrain_weight_path ../01_Pre-training_Model/genebert_weights_99-0.875329.tf
    fi
done >& get_embeddings.txt
