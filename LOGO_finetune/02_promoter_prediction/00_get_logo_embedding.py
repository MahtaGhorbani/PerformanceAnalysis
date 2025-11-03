import os
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from Bio import SeqIO
import tensorflow as tf
import argparse
sys.path.append("../")
from bgi.common.refseq_utils import get_word_dict_for_n_gram_alphabet
from bgi.bert4keras.models import build_transformer_model


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
                     task_name: str = 'train',
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
        save_path = os.path.join(output_path, f'{task_name}_{str(ngram)}_gram_{str(bacteria_name)}.npz')
        np.savez_compressed(save_path, **save_dict)
        
def model(pretrain_weight_path):
    config = {
            "attention_probs_dropout_prob": 0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0,
            "embedding_size": embedding_size,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": model_dim * 4,
            "max_position_embeddings": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": max_depth,
            "num_hidden_groups": 1,
            "net_structure_type": 0,
            "gap_size": 0,
            "num_memory_blocks": 0,
            "inner_group_num": 1,
            "down_scale_factor": 1,
            "type_vocab_size": 0,
            "vocab_size": vocab_size,
            "custom_masked_sequence": False,
        }

    bert = build_transformer_model(
            configs=config,
            model='bert',
            with_mlm='linear',
            application='lm',
            return_keras_model=False,
            weights=pretrain_weight_path
        )
    albert = bert.model
    albert.summary()
    albert.compile(optimizer='adam', loss=[tf.keras.losses.SparseCategoricalCrossentropy()], metrics=['accuracy'])
    
    return albert

def load_npz_data_for_classification(file_name, ngram=3, only_one_slice=True, ngram_index=None, masked=True):
    """
    Import npz data
    :param file_name:
    :param ngram:
    :param only_one_slice:
    :param ngram_index:
    :return:
    """
    x_data_all = []
    y_data_all = []
    if str(file_name).endswith('.npz') is False or os.path.exists(file_name) is False:
        return x_data_all, None, y_data_all

    loaded = np.load(file_name)
    x_data = loaded['sequence']
    y_data = loaded['label']

    print("Load: ", file_name)
    print("X: ", x_data.shape)
    print("Y: ", y_data.shape)
    if only_one_slice is True:
        for ii in range(ngram):
            if ngram_index is not None and ii != ngram_index:
                continue
            kk = ii
            slice_indexes = []
            max_slice_seq_len = x_data.shape[1] // ngram * ngram
            for gg in range(kk, max_slice_seq_len, ngram):
                slice_indexes.append(gg)
            x_data_slice = x_data[:, slice_indexes]
            x_data_all.append(x_data_slice)
            y_data_all.append(y_data)
    else:
        x_data_all.append(x_data)
        y_data_all.append(y_data)

    return x_data_all, y_data_all
def load_all_data(record_names: list, ngram=3, only_one_slice=True, ngram_index=None, masked=False):
    x_data_all = []
    y_data_all = []
    for file_name in record_names:
        print(file_name)
        data = load_npz_data_for_classification(file_name,
                                                        ngram,
                                                        only_one_slice,
                                                        ngram_index,
                                                        masked=masked)
        print(len(data))                                                
        if len(data) == 2:
            x_data , y_data = data
        else:
            print("Error: ", file_name)
            
        x_data_all.extend(x_data)
        y_data_all.extend(y_data)

    x_data_all = np.concatenate(x_data_all)
    y_data_all = np.concatenate(y_data_all)
    return x_data_all, y_data_all

def create_segment_ids_and_input_mask(data, max_length):
    segment_ids_list = []
    input_mask_list = []

    for seq in data:
        # Check if the sequence is longer than max_length after tokenization
        if len(seq) > max_length:
            seq = seq[:max_length]  # Truncate the sequence if it's longer

        # Create segment IDs and input mask
        segment_ids = [0] * len(seq) + [0] * (max_length - len(seq))
        input_mask = [1] * len(seq) + [0] * (max_length - len(seq))

        segment_ids_list.append(segment_ids)
        input_mask_list.append(input_mask)

    return np.array(segment_ids_list), np.array(input_mask_list)

def kfold_splitting(X,Y,ngram,my_model):
    kf = KFold(n_splits=5, shuffle=True)
    for k_fold, (train, test) in enumerate(kf.split(X, Y)):

        x_train_data = X[train]
        y_train_data = Y[train]
        x_test_data = X[test]
        y_test_data = Y[test]
        
        max_length = 200
        x_train_segment_ids, x_train_input_mask = create_segment_ids_and_input_mask(x_train_data, max_length)
        x_test_segment_ids, x_test_input_mask = create_segment_ids_and_input_mask(x_test_data, max_length)
        x = my_model.predict([x_train_data, x_train_segment_ids, x_train_input_mask])
        x_test = my_model.predict([x_test_data, x_test_segment_ids, x_test_input_mask])
        # save data as a npz file
        np.savez(
            f'./data/kfold_{str(k_fold)}_train_{str(ngram)}_gram_{bacteria_name}.npz',
            x = x,
            label = y_train_data,)
        np.savez(
            f'./data/kfold_{str(k_fold)}_test_{str(ngram)}_gram_{bacteria_name}.npz',
            x = x_test,
            label = y_test_data,)
            
        

if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='A data preprocessing of the Transformer language model in Genomics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--ngram',  
        type=int,
        default=3,
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
        '--pretrain_weight_path',
        type=str,
        default='./',
        help='pretrain_weight_path')
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
        default='C_JEJUNI',
        help='bacteria_name')
    
    _args = _argparser.parse_args()
    ngram = _args.ngram
    stride = _args.stride
    fasta_files_positive = _args.fasta_files_positive
    fasta_files_negative = _args.fasta_files_negative
    pretrain_weight_path = _args.pretrain_weight_path
    embedding_size = _args.embedding_size
    model_dim = _args.model_dim
    bacteria_name = _args.bacteria_name
    
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
        
    )

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    vocab_size = len(word_dict) + 10


    train_data_file = f'train_{str(ngram)}_gram_{str(bacteria_name)}.npz'
    #data_path = './data/' + '{}_gram'.format(ngram)

    ## load data: sequence, label
    #train_promoter_files = [os.path.join(data_path, train_data_file)]
    train_files = [train_data_file]

    only_one_slice = True
    print(train_files)
    X, Y = load_all_data(train_files, ngram=ngram, only_one_slice=only_one_slice,ngram_index=None, )
    my_model = model(pretrain_weight_path)
    kfold_splitting(X,Y,ngram,my_model)
