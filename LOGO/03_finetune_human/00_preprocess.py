
import os
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
import warnings
warnings.filterwarnings('ignore')
from Bio import SeqIO
import argparse
sys.path.append("../")
from bgi.common.refseq_utils import get_word_dict_for_n_gram_alphabet

embedding_size = 128
model_dim = 256
max_depth = 2
vocab_size = 3138
batch_size = 32

def create_df(file_path, label):
    records = []
    seen = []
    for record in SeqIO.parse(file_path, "fasta"):
        if str(record.seq) not in seen:
            seen.append(str(record.seq))
            records.append(record)
    records = [str(record.seq) for record in records]
    df = pd.DataFrame(records, columns=['seq_record'])
    df['label'] = label
    return df
def process_raw_text(sequences,
                     labels,
                     seq_size=1000,
                     ngram=3,
                     stride=1,
                     filter_txt=None,
                     skip_n: bool = False,
                     word_dict: dict = None,
                     output_path: str = './',
                     task_name: str = 'validation',
                     gene_type_dict: dict = None):
    slice_index = 0
    slice_seq_data = []
    slice_label_data = []

    set_atcg = set(list('ATCG'))

    print("seq_size: ", seq_size)

    for row in range(len(sequences)):
        seq = sequences[row]
        label = labels[row]

        #print("SEQ: ", seq)
        seq = seq.upper()
        seq_number = [
            word_dict.get(seq[i : i + ngram], 0)
            for i in range(0, seq_size, stride)
            if i + ngram <= len(seq) and word_dict is not None
        ]
        # Pad or truncate the sequence to have a uniform length (seq_size)
        if len(seq_number) < seq_size:
            seq_number = seq_number + [0] * (seq_size - len(seq_number))
        elif len(seq_number) > seq_size:
            seq_number = seq_number[:seq_size]
        slice_seq_data.append(seq_number)
        slice_label_data.append(label)

        slice_index += 1

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    if slice_seq_data and slice_label_data:
        save_dict = {
            'sequence': slice_seq_data,
            'label': slice_label_data
        }
        save_path = os.path.join(output_path, f'{task_name}_{str(ngram)}_gram_{bacteria_name}.npz')
        np.savez_compressed(save_path, **save_dict)
        

if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='A data preprocessing of the Transformer language model in Genomics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--ngram',  
        type=int,
        default=5,
        help='ngram')
    _argparser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='stride')
    _argparser.add_argument(
        '--fasta_files_positive',
        type=str,
        default='./positive.fna',
        help='fasta_files_positive')
    _argparser.add_argument(
        '--fasta_files_negative',
        type=str,
        default='./negative.fna',
        help='fasta_files_negative')
    _argparser.add_argument(
        '--embedding_size',
        type=int,
        default=128,
        help='embedding_size')
    _argparser.add_argument(
        '--model_dim',
        type=int,
        default=256,
        help='model_dim')
    _argparser.add_argument(
        '--bacteria_name',
        type=str,
        default='test',
        help='bacteria_name')
    _argparser.add_argument(
        '--task_name',
        type=str,
        default='validation')
    
    _args = _argparser.parse_args()
    ngram = _args.ngram
    stride = _args.stride
    fasta_files_positive = _args.fasta_files_positive
    fasta_files_negative = _args.fasta_files_negative
    embedding_size = _args.embedding_size
    model_dim = _args.model_dim
    bacteria_name = _args.bacteria_name
    task_name = _args.task_name
    word_dict = get_word_dict_for_n_gram_alphabet(n_gram=ngram)
    df_positive = create_df(fasta_files_positive, 1)
    df_negative = create_df(fasta_files_negative, 0)
    df = pd.concat([df_positive, df_negative])
    df = df.sample(frac=1).reset_index(drop=True)

    sequences = df['seq_record'].values
    labels = df['label'].values
    process_raw_text(
        sequences=sequences,
        labels=labels,
        ngram=ngram,
        word_dict=word_dict,
        task_name=task_name,
    ) 



