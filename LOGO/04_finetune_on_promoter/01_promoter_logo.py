# -*- coding:utf-8 -*-
import os
import sys

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Reshape, Permute, concatenate
from tensorflow.keras.layers import Lambda, Dense, Layer
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
import numpy
from sklearn import metrics
import argparse
sys.path.append("../")
from bgi.bert4keras.models import build_transformer_model
from bgi.common.callbacks import LRSchedulerPerStep
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number
import warnings
warnings.filterwarnings("ignore")



def load_npz_data_for_classification(file_name, ngram=5, only_one_slice=True, ngram_index=None, masked=True):
    """
    Import npz data
    :param file_name:
    :param ngram:
    :param only_one_slice:
    :param ngram_index:
    :return:
    """
    x_data_all = []
    anno_data_all = []
    y_data_all = []
    if str(file_name).endswith('.npz') is False or os.path.exists(file_name) is False:
        print("File not exists: ", file_name)
        return x_data_all, None, y_data_all

    loaded = np.load(file_name)
    x_data = loaded['sequence']
    y_data = loaded['label']

    # if masked:
    #     positive_samples = np.sum(y_data)
    #     RANDOM_STATE = 42
    #     print("positive_samples: ", positive_samples)
    #     x_data, y_data = make_imbalance(x_data, y_data,
    #                                     sampling_strategy={0: positive_samples, 1: positive_samples},
    #                                     random_state=RANDOM_STATE)

    print("Load: ", file_name)
    print("X: ", x_data.shape)
    print("Y: ", y_data.shape)
    # if only_one_slice is True:
    #     print("Only one slice")
    #     for ii in range(ngram):
    #         if ngram_index is not None and ii != ngram_index:
    #             continue
    #         kk = ii
    #         slice_indexes = []
    #         max_slice_seq_len = x_data.shape[1] // ngram * ngram
    #         print("max_slice_seq_len: ", max_slice_seq_len)
    #         for gg in range(kk, max_slice_seq_len, ngram):
    #             slice_indexes.append(gg)
                
    #         print("slice_indexes: ", slice_indexes)
    #         x_data_slice = x_data[:, slice_indexes]
    #         print("x_data_slice: ", x_data_slice.shape)
    #         x_data_all.append(x_data_slice)
    #         print("x_data_all: ", len(x_data_all))
            
    #         y_data_all.append(y_data)
    #         print("y_data_all: ", len(y_data_all))
    # else:
    x_data_all.append(x_data)
    y_data_all.append(y_data)

    return x_data_all, anno_data_all, y_data_all


def load_all_data(record_names: list, ngram=5, only_one_slice=True, ngram_index=None, masked=False):
    x_data_all = []
    y_data_all = []
    
    for file_name in record_names:
        try:
            x_data, anno_data, y_data = load_npz_data_for_classification(file_name, ngram, only_one_slice, ngram_index, masked=masked)
            if x_data is not None and y_data is not None:
                print("x_data: ", len(x_data))
                print("y_data: ", len(y_data))
                x_data_all.extend(x_data)
                y_data_all.extend(y_data)
                print("x_data_all: ", len(x_data_all))
                print("y_data_all: ", len(y_data_all))
        except Exception as e:
            print(f"Error loading data from {file_name}: {str(e)}")

    if not x_data_all:
        print("No data loaded for " + str(record_names) )
        return None, None

    print("x_data_all: ", len(x_data_all))
    print("y_data_all: ", len(y_data_all))
    x_data_all = np.concatenate(x_data_all)
    y_data_all = np.concatenate(y_data_all)
    print("x_data_all: ", x_data_all.shape)
    print("y_data_all: ", y_data_all.shape)    
    return x_data_all, y_data_all



# @tf.function
def load_npz_dataset_for_classification(x_promoter_data_all,
                                        y_data_all,
                                        promoter_seq_len,
                                        ngram=5,
                                        only_one_slice=True,
                                        ngram_index=None,
                                        shuffle=False,
                                        seq_len=200,
                                        num_classes=1,
                                        masked=True,
                                        ):
    """
    Read sequence data from NPZ file and generate tf.data.DataSet
    :param record_names:
    :param batch_size:
    :param ngram:
    :param only_one_slice: Slice by ngram
    :param ngram_index:
    :param shuffle:
    :param seq_len:
    :param num_classes:
    :param num_parallel_calls:
    :return:
    """

    # if not isinstance(enhacer_record_names, list):
    #     enhacer_record_names = [enhacer_record_names]
    #
    # if not isinstance(promoter_record_names, list):
    #     promoter_record_names = [promoter_record_names]

    if num_classes == 1:
        y_data_all = np.reshape(y_data_all, (y_data_all.shape[0], 1))

    # Data Generator
    def data_generator():
        total_size = len(x_promoter_data_all)
        #print('total_size: ', total_size)
        #print('x_promoter_data_all: ', x_promoter_data_all.shape)
        indexes = np.arange(total_size)
        if shuffle is True:
            np.random.shuffle(indexes)

        ii = 0
        while True:
            if ii < total_size:
                index = indexes[ii]
            else:
                print("Shuffle ..............................................")
                total_size = len(x_promoter_data_all)
                indexes = np.arange(total_size)
                if shuffle is True:
                    np.random.shuffle(indexes)

                ii = 0
                index = indexes[ii]

            x_promoter = x_promoter_data_all[index]
            #print("x_promoter: ", x_promoter.shape)
            segment_promoter = np.zeros_like(x_promoter)
            #print("segment_promoter: ", segment_promoter.shape)
            y = y_data_all[index]
            ii += 1
            yield x_promoter, segment_promoter, y

    classes_shape = tf.TensorShape([num_classes])
    if num_classes == 1:
        classes_shape = tf.TensorShape([1])

    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.int16, tf.int16, tf.int16),
                                             output_shapes=(
                                                 tf.TensorShape([promoter_seq_len]),
                                                 tf.TensorShape([promoter_seq_len]),
                                                 classes_shape
                                             ))
    return dataset


def parse_function(x_promoter, segment_x, y):
    x = {
        'Input-Token': x_promoter,
        'Input-Segment': segment_x,
    }

    y = {
        'CLS-Activation': y
    }
    return x, y


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    TP = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1), 'float'))
    FP = K.sum(tf.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1), 'float'))
    FN = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0), 'float'))
    TN = K.sum(tf.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 0), 'float'))
    P = TP / (TP + FP + K.epsilon())
    R = TP / (TP + FN + K.epsilon())
    F1 = 2 * P * R / (P + R + K.epsilon())

    return F1


def auprc_score(y_true, y_pred):
    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    auprc = tf.py_function(metrics.average_precision_score, (y_true, y_pred), tf.float64)
    return auprc


def average_precision(y_true, y_pred):
    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def model_def(embedding_size=128,
              hidden_size=128,
              num_heads=8,
              num_hidden_layers=1,
              vocab_size=10000,
              drop_rate=0.25,
              weight=None):
    config = {
        "attention_probs_dropout_prob": 0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0,
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "max_position_embeddings": 1024 * 2,
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_hidden_layers,
        "num_hidden_groups": 1,
        "net_structure_type": 0,
        "gap_size": 0,
        "num_memory_blocks": 0,
        "inner_group_num": 1,
        "down_scale_factor": 1,
        "type_vocab_size": 0,
        "vocab_size": vocab_size,
        "custom_masked_sequence": False,
        "custom_conv_layer": False,
        "use_segment_ids": True,
        "use_position_ids": True,
        "multi_inputs": []
    }
    bert = build_transformer_model(
        configs=config,
        model='bert',
        return_keras_model=False,
       weights=weight
    )

    promoter_output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = BatchNormalization()(promoter_output)
    output = Dropout(drop_rate)(output)
    output = Dense(1, activation='sigmoid', name='CLS-Activation')(output)

    model = tf.keras.models.Model(inputs=bert.model.input, outputs=[output])
    model.summary()

    return model


def validation(validation_data_file, data_path, batch_size=256, 
               ngram=5, vocab_size=10000, PROMOTER_RESIZED_LEN=600, task_name='Promoter', filename=None):
    # Distributed Training
    num_gpu = 1
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync

    GLOBAL_BATCH_SIZE = batch_size * num_gpu
    num_parallel_calls = 16

    # Load validation data
    only_one_slice = True  # You can adjust this as needed
    validation_promoter_files = [os.path.join(data_path, file) for file in validation_data_file]
    print("Validation promoter files: ", validation_promoter_files)
    with strategy.scope():
        model = model_def(vocab_size=vocab_size,weight=filename)
        print('compiling...')
        model.compile(loss='binary_crossentropy',
                        optimizer=tf.keras.optimizers.legacy.Adam(0.0001),
                        metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score, auprc_score])
        print("Loading weights for prediction from: ", filename)
        model.load_weights(filename)
        model.summary()
    for file in validation_promoter_files:
        region1_seq, label = load_all_data([file], ngram=ngram, only_one_slice=only_one_slice, ngram_index=None)

        seed = 7
        numpy.random.seed(seed)
        x_validation_data = region1_seq
        y_validation_data = label
        promoter_seq_len = 1000
        valid_total_size = len(y_validation_data)
        valid_dataset = load_npz_dataset_for_classification(x_validation_data, y_validation_data, promoter_seq_len,
                                                            ngram=ngram, only_one_slice=only_one_slice, ngram_index=None,
                                                            shuffle=False, seq_len=0, masked=False)
        valid_dataset = valid_dataset.batch(batch_size)
        valid_dataset = valid_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        valid_steps_per_epoch = (valid_total_size + GLOBAL_BATCH_SIZE ) // GLOBAL_BATCH_SIZE

        #save valid dataset
        eval = model.evaluate(valid_dataset, steps=valid_steps_per_epoch, verbose=2)
        print("Eval on", file, "is:", eval)



def train(train_data_file,
                data_path,
                batch_size=256,
                epochs=1,
                ngram=5,
                n_splits=10,
                vocab_size=10000,
                PROMOTER_RESIZED_LEN=600,
                task_name='Promoter',
                filename=None):
    # Distributed Training
    num_gpu = 1
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync

    GLOBAL_BATCH_SIZE = batch_size * num_gpu
    num_parallel_calls = 16

    train_promoter_files = [os.path.join(data_path, file) for file in train_data_file]
    print("train_promoter_files: ", train_promoter_files)

    only_one_slice = True
    region1_seq, label = load_all_data(train_promoter_files, ngram=ngram, only_one_slice=only_one_slice,
                                       ngram_index=None)

    seed = 7
    numpy.random.seed(seed)
    X = region1_seq
    Y = label

    promoter_seq_len = 1000

    k_fold = 0
    shuffle = True
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3)

    promoter_indexes = np.arange(len(Y))
    if shuffle is True:
        np.random.shuffle(promoter_indexes)

    train_slice = promoter_indexes[:int(len(promoter_indexes) * 0.9)]
    valid_slice = promoter_indexes[int(len(promoter_indexes) * 0.9):]
    x_train_data = X[train_slice]
    y_train_data = Y[train_slice]
    x_valid_data = X[valid_slice]
    y_valid_data = Y[valid_slice]
    

    with strategy.scope():
        model = model_def(vocab_size=vocab_size,weight=pretrain_weight_path)
        print('compiling...')
        model.compile(loss='binary_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(0.0001),
                        metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score,auprc_score])
        # model.summary()

    

    modelCheckpoint = ModelCheckpoint(filename, monitor='val_acc', save_best_only=True, verbose=1)

    print('fitting...')

    train_total_size = len(y_train_data)
    train_dataset = load_npz_dataset_for_classification(x_train_data,
                                                        y_train_data,
                                                        promoter_seq_len,
                                                        ngram=ngram,
                                                        only_one_slice=only_one_slice,
                                                        ngram_index=None,
                                                        shuffle=True,
                                                        seq_len=0,
                                                        masked=False,
                                                        )

    train_dataset = train_dataset.shuffle(train_total_size, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    valid_total_size = len(y_valid_data)
    valid_dataset = load_npz_dataset_for_classification(x_valid_data,
                                                        y_valid_data,
                                                        promoter_seq_len,
                                                        ngram=ngram,
                                                        only_one_slice=only_one_slice,
                                                        ngram_index=None,
                                                        shuffle=False,
                                                        seq_len=0,
                                                        num_classes=1,
                                                        masked=False,
                                                        )
    valid_dataset = valid_dataset.batch(batch_size)
    valid_dataset = valid_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
    valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    
    train_steps_per_epoch = train_total_size // GLOBAL_BATCH_SIZE
    valid_steps_per_epoch = valid_total_size // GLOBAL_BATCH_SIZE

    print("Training")
    print("batch size: ", GLOBAL_BATCH_SIZE)

    model_train_history = model.fit(train_dataset,
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=epochs,
                                    validation_data=valid_dataset,
                                    validation_steps=valid_steps_per_epoch,
                                    callbacks=[modelCheckpoint, early_stopping],
                                    verbose=2)

    print(model_train_history)
    


def predict(validation_data_file, data_path, batch_size=256, 
            ngram=5, vocab_size=10000, PROMOTER_RESIZED_LEN=600, task_name='Promoter', filename=None):
    num_gpu = 1
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync

    GLOBAL_BATCH_SIZE = batch_size * num_gpu
    num_parallel_calls = 16
    with strategy.scope():
        model = model_def(vocab_size=vocab_size,weight=filename)
        print('compiling...')
        model.compile(loss='binary_crossentropy',
                        optimizer=tf.keras.optimizers.legacy.Adam(0.0001),
                        metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score, auprc_score])

            # Load the trained weights
        print("Loading weights for prediction from: ", filename)
        model.load_weights(filename)
        model.summary()
    only_one_slice = True  # You can adjust this as needed
    validation_promoter_files = [os.path.join(data_path, file) for file in validation_data_file]
    print("Validation promoter files: ", validation_promoter_files)
    
    for file in validation_promoter_files:
        region1_seq, label = load_all_data([file], ngram=ngram, only_one_slice=only_one_slice, ngram_index=None)

        seed = 7
        numpy.random.seed(seed)
        x_validation_data = region1_seq
        y_validation_data = label
        promoter_seq_len = 1000
        valid_total_size = len(y_validation_data)
        
        valid_dataset = load_npz_dataset_for_classification(x_validation_data, y_validation_data, promoter_seq_len,
                                                            ngram=ngram, only_one_slice=only_one_slice, ngram_index=None,
                                                            shuffle=False, seq_len=0, masked=False)
        valid_dataset = valid_dataset.batch(batch_size)
        valid_dataset = valid_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        print("valid_total_size: ", valid_total_size)
        print("batch_size: ", batch_size)
        print("GLOBAL_BATCH_SIZE: ", GLOBAL_BATCH_SIZE)
        # Calculate the number of steps, including the remaining samples
        valid_steps_per_epoch = valid_total_size  // GLOBAL_BATCH_SIZE
        print("valid_steps_per_epoch: ", valid_steps_per_epoch)
        # Adjust the steps parameter in model.predict to include remaining samples
        y_pred = model.predict(valid_dataset, steps=valid_steps_per_epoch+1, verbose=2)

        # Save y_pred and y_true as a csv file
        print("y_pred: ", y_pred.shape)
        print("y_validation_data: ", y_validation_data.shape)
        y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))
        y_validation_data = np.reshape(y_validation_data, (y_validation_data.shape[0], 1))
        file_path = file.split('/')[-1].split('.')[0]
        csv_path_pred = f'./data/y_pred_{file_path}.csv'
        csv_path_true = f'./data/y_true_{file_path}.csv'
        np.savetxt(csv_path_pred, y_pred, delimiter=',')
        np.savetxt(csv_path_true, y_validation_data, delimiter=',')

        
def train_kfold(train_data_file,
                data_path,
                batch_size=256,
                epochs=10,
                ngram=5,
                n_splits=10,
                vocab_size=10000,
                PROMOTER_RESIZED_LEN=600,
                task_name='promoter'):
    # Distributed Training
    num_gpu = 1
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync

    GLOBAL_BATCH_SIZE = batch_size * num_gpu
    num_parallel_calls = 16

    ## load data: sequence, label
    #train_promoter_files = [os.path.join(data_path, train_data_file)]
    train_promoter_files = [os.path.join(data_path, file) for file in train_data_file]
    print("train_promoter_files: ", train_promoter_files)

    only_one_slice = True
    region1_seq, label = load_all_data(train_promoter_files, ngram=ngram, only_one_slice=only_one_slice,
                                       ngram_index=None)

    seed = 7
    numpy.random.seed(seed)
    X = region1_seq
    Y = label

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    promoter_seq_len = PROMOTER_RESIZED_LEN // ngram * ngram
    print(promoter_seq_len)
    print(PROMOTER_RESIZED_LEN)
    print(ngram)
    if only_one_slice:
        promoter_seq_len = promoter_seq_len // ngram
    promoter_seq_len =200
    k_fold = 0
    shuffle = True
    for train, test in kfold.split(X, Y):
        print("Iterating through data")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)

        promoter_indexes = np.arange(len(train))
        if shuffle is True:
            np.random.shuffle(promoter_indexes)

        train_slice = promoter_indexes[:int(len(promoter_indexes) * 0.9)]
        valid_slice = promoter_indexes[int(len(promoter_indexes) * 0.9):]

        x_train_data = X[train[train_slice]]
        y_train_data = Y[train[train_slice]]

        x_valid_data = X[train[valid_slice]]
        y_valid_data = Y[train[valid_slice]]

        x_test_data = X[test]
        y_test_data = Y[test]


        np.savez('./data/kfold_{}_train_and_valid_index_{}_gram_{}.npz'.format(str(k_fold), str(ngram), task_name), train=train,
                 test=test)

        with strategy.scope():
            model = model_def(vocab_size=vocab_size)
            print('compiling...')
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(0.0001),
                          metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score, auprc_score])
            # model.summary()

        filename = './data/promoter_best_model_gene_bert_{}_fold_{}_gram_{}.tf'.format(str(k_fold), str(ngram), task_name)

        modelCheckpoint = ModelCheckpoint(filename, monitor='val_acc', save_best_only=True, verbose=1)

        print('fitting...')

        train_total_size = len(y_train_data)
        train_dataset = load_npz_dataset_for_classification(x_train_data,
                                                            y_train_data,
                                                            promoter_seq_len,
                                                            ngram=ngram,
                                                            only_one_slice=only_one_slice,
                                                            ngram_index=None,
                                                            shuffle=True,
                                                            seq_len=0,
                                                            masked=False,
                                                            )

        train_dataset = train_dataset.shuffle(train_total_size, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        valid_total_size = len(y_valid_data)
        valid_dataset = load_npz_dataset_for_classification(x_valid_data,
                                                            y_valid_data,
                                                            promoter_seq_len,
                                                            ngram=ngram,
                                                            only_one_slice=only_one_slice,
                                                            ngram_index=None,
                                                            shuffle=False,
                                                            seq_len=0,
                                                            num_classes=1,
                                                            masked=False,
                                                            )
        valid_dataset = valid_dataset.batch(batch_size)
        valid_dataset = valid_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        test_total_size = len(y_test_data)
        test_dataset = load_npz_dataset_for_classification(x_test_data,
                                                            y_test_data,
                                                            promoter_seq_len,
                                                            ngram=ngram,
                                                            only_one_slice=only_one_slice,
                                                            ngram_index=None,
                                                            shuffle=False,
                                                            seq_len=0,
                                                            num_classes=1,
                                                            masked=False,
                                                            )
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)


        train_steps_per_epoch = train_total_size // GLOBAL_BATCH_SIZE
        valid_steps_per_epoch = valid_total_size // GLOBAL_BATCH_SIZE
        test_steps_per_epoch = test_total_size // GLOBAL_BATCH_SIZE

        print("Training")
        print("batch size: ", GLOBAL_BATCH_SIZE)

        model_train_history = model.fit(train_dataset,
                                        steps_per_epoch=train_steps_per_epoch,
                                        epochs=epochs,
                                        validation_data=valid_dataset,
                                        validation_steps=valid_steps_per_epoch,
                                        callbacks=[modelCheckpoint, early_stopping],
                                        verbose=2)

        print(model_train_history)

        # Make predictions and reload the optimal weights
        with strategy.scope():
            model = model_def(vocab_size=vocab_size)
            print('compiling...')
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(0.0001),
                          metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score,auprc_score])
            model.load_weights(filename)
        eval = model.evaluate(test_dataset, steps=test_steps_per_epoch, verbose=2)
        print("Eval: ", eval)

        k_fold += 1
    
if __name__ == '__main__':

    _argparser = argparse.ArgumentParser(
        description='Train the model for promoter prediction'
    )
    _argparser.add_argument(
        '--pretrain_weight_path',
        type=str,
        default='../01_Pre-training_Model/genebert_weights_99-0.875329.tf'
    )
    _argparser.add_argument(
        '--data_path',
        type=str,
        default='./data'
    )
    _argparser.add_argument(
        '--task_name',
        type=str,
        default='Promoter'
    )
    _argparser.add_argument(
        '--train_data_file',
        type=str,
        default="train_5_gram_CJEJUNI_2.npz,train_5_gram_C_JEJUNI.npz,train_5_gram_HPYLORI_2.npz,train_5_gram_SONEIDENSIS.npz,train_5_gram_CJEJUNI_3.npz,train_5_gram_CPNEUMONIAE.npz,train_5_gram_HPYLORI.npz,train_5_gram_SPYOGENE.npz,train_5_gram_CJEJUNI_4.npz,train_5_gram_ECOLI_2.npz,train_5_gram_LINTERROGANS.npz,train_5_gram_STYPHIRMURIUM.npz,train_5_gram_CJEJUNI_5.npz,train_5_gram_ECOLI.npz,train_5_gram_SCOELICOLOR.npz"
    )
    _argparser.add_argument(
        '--validation_data_file',
        type=str,
        default='validation_5_gram_CLOSTRIDIUM.npz,validation_5_gram_RHODOBACTER_1.npz,validation_5_gram_BACILLUS.npz,validation_5_gram_MYCOBACTER.npz,validation_5_gram_RHODOBACTER_2.npz'
    )
    _argparser.add_argument(
        '--model_name',
        type=str,
        default='gene_bert'
    )
    _argparser.add_argument(
        '--train',
        type=bool,
        default=False
    )
    _argparser.add_argument(
        '--validation',
        type=bool,
        default=False
    )
    _argparser.add_argument(
        '--predict',
        type=bool,
        default=False
    )
    _argparser.add_argument(
        '--train_kfold',
        type=bool,
        default=False
    )
    _args = _argparser.parse_args()
    pretrain_weight_path = _args.pretrain_weight_path
    train_data_file = _args.train_data_file
    validation_data_file = _args.validation_data_file
    data_path = _args.data_path
    task_name = _args.task_name
    train_data_file = train_data_file.split(',')
    validation_data_file = validation_data_file.split(',') 
    model_name = _args.model_name   
    train_bool = _args.train
    validation_bool = _args.validation
    predict_bool = _args.predict
    train_kfold_bool = _args.train_kfold


    print("train_data_file: ", train_data_file)
    print("validation_data_file: ", validation_data_file)
    # Dynamic allocation of video memory
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    ngram = 5
    stride = 1
    word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)
    vocab_size = len(word_dict) + 10
    filename = './data/promoter_best_model_gene_bert.tf'
    if train_bool :
        train(train_data_file,
                    data_path,
                    batch_size=256,
                    epochs=50,
                    ngram=ngram,
                    n_splits=1,
                    vocab_size=vocab_size,
                    PROMOTER_RESIZED_LEN=600,
                    task_name=task_name,
                    filename=filename)
    if validation_bool:
        validation(validation_data_file, data_path, batch_size=256,
                    ngram=ngram, vocab_size=vocab_size, PROMOTER_RESIZED_LEN=600, task_name=task_name, filename=filename)
        
    if predict_bool:
        predict(validation_data_file, data_path, batch_size=256,
                    ngram=ngram, vocab_size=vocab_size, PROMOTER_RESIZED_LEN=600, task_name=task_name, filename=filename)
        
    if train_kfold_bool:
        train_kfold(train_data_file,
                    data_path,
                    batch_size=256,
                    epochs=10,
                    ngram=ngram,
                    n_splits=10,
                    vocab_size=vocab_size,
                    PROMOTER_RESIZED_LEN=600,
                    task_name=task_name)
    
    

