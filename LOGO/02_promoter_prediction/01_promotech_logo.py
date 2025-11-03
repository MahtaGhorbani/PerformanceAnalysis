import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import  PrecisionRecallDisplay, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from PromotechCNN_Prev import PromotechCNN
import argparse
from keras.models import Model
from keras.metrics import AUC
from keras import layers
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Lambda, Dense, Layer

def read_train_test(trainPath, testPath):
    X_train, y_train = load_data(trainPath)
    X_test, y_test = load_data(testPath)
    return X_train, y_train, X_test, y_test
    
def load_data(path):
    loaded = np.load(path)
    x = loaded['x']
    x = x.reshape(x.shape[0], -1)
    y = loaded['label']
    return x, y

def mlp_model():
    inputs = layers.Input(shape = (None,627000 ))
    x = layers.Dense(32, activation = 'relu')(inputs)
    x = layers.Dense(16, activation = 'relu')(x)
    y = layers.Dense(1, activation = 'sigmoid')(x)
    
    return Model(inputs = inputs, outputs = y)
    
def Mlp():
    model = mlp_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    for i in range(K):
        trainPath = f'./data/kfold_{str(i)}_train_{str(NGRAM)}_gram_{str(bacteria_name)}.npz'
        testPath = f'./data/kfold_{str(i)}_test_{str(NGRAM)}_gram_{str(bacteria_name)}.npz'
        #trainPath = f'./Data/kfold_{str(i)}_train_real_{str(NGRAM)}_gram.npz'
        #testPath = f'./Data/kfold_{str(i)}_test_real_{str(NGRAM)}_gram.npz'
        X_train, y_train, X_test, y_test = read_train_test(trainPath, testPath)
        model.fit(np.array(X_train), np.array(y_train), epochs = 100, batch_size = 16)
        yHat = model.predict(np.array(X_test))
        
        
        precision, recall, _ = precision_recall_curve(np.array(y_test), yHat)
        precRecCurve = PrecisionRecallDisplay(precision, recall)

        AUPRC = AUC(curve = 'PR')
        AUPRC.update_state(np.array(y_test), yHat)
        
    print(f'AUPRC: {AUPRC.result().numpy()}')
    print(f'Precision: {precision_score(np.array(y_test), yHat)}')
    print(f'Recall: {recall_score(np.array(y_test), yHat)}')
def Promotech():
    model = PromotechCNN()
    model.buildNetwork()
    for i in range(K):
        print("In for loop")
        trainPath = f'./data/kfold_{str(i)}_train_{str(NGRAM)}_gram_{str(bacteria_name)}.npz'
        testPath = f'./data/kfold_{str(i)}_test_{str(NGRAM)}_gram_{str(bacteria_name)}.npz'
        X_train, y_train, X_test, y_test = read_train_test(trainPath, testPath)
        #X_train = np.array(X_train).reshape(-1, 40, 4)
        #X_test = np.array(X_test).reshape(-1, 40, 4)
        model.train(X_train, y_train, X_test, y_test)
        print("Compute PCA ...")
#        pca = PCA(n_components=160)
#        X_train_pca = pca.fit_transform(X_train)
#        X_test_pca = pca.transform(X_test)
#        X_train_pca_reshaped = np.reshape(X_train_pca, (-1, 40, 4))
#        X_test_pca_reshaped = np.reshape(X_test_pca, (-1, 40, 4))
        print("Model.train ...")
#        model.train(X_train_pca_reshaped, y_train, X_test_pca_reshaped, y_test)
        results = model.test(X_test_pca_reshaped,y_test)
        print(results)
    
        


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='train a model on a dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--model_name',
        type=str,
        help='name of the model mlp or promotech')
    _argparser.add_argument(
        '--ngram',
        type=int,
        default=5,
        help='ngram size')
    _argparser.add_argument(
        '--kfold',
        type=int,
        default=5,
        help='number of kfold')
    _argparser.add_argument(
        '--bacteria_name',
        type=str,
        default='CJEJUNI',
        help='bacteria name')
    args = _argparser.parse_args()
    NGRAM = args.ngram
    K = args.kfold
    bacteria_name = args.bacteria_name
    if args.model_name == 'mlp':
        Mlp()
    elif args.model_name == 'promotech':
        Promotech()
        
        
        
