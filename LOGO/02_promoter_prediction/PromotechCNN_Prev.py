# Imports
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from typing import List, Tuple
from keras.models import Model
from keras.metrics import AUC
from keras import layers
import tensorflow as tf
import numpy as np

# Hyperparameters
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 10
ALPHA = 5e-2
VERBOSE = 1
PSI = 0.95
STD = 0.5
TAU = 4

# Define Noise Decay Callback Class
class NoiseDecay(tf.keras.callbacks.Callback):
    def __init__(self, psi:float) -> None:
        super().__init__()
        self.psi = psi

    # Decay Standard Deviation of Gaussian Noise by Psi
    def on_epoch_begin(self, epoch:int, logs:dict = None) -> None:
        self.model.layers[1].stddev *= self.psi

# Define 1D Squeeze and Excite Layer Class
class SqueezeAndExcite1D(layers.Layer):
    def __init__(self, filters:int) -> None:
        super().__init__()

        self.pool = layers.GlobalAveragePooling1D(keepdims = True)
        self.squeezeConv = layers.Conv1D(filters  // TAU, kernel_size = 1, activation = 'relu')
        self.exciteConv = layers.Conv1D(filters, kernel_size = 1, activation = 'sigmoid')

    # Define Forward Pass
    def call(self, inputs:tf.Tensor, training:bool = True) -> tf.Tensor:
        x = self.pool(inputs)
        x = self.squeezeConv(x)
        x = self.exciteConv(x)

        return tf.multiply(x, inputs)
    
# Define PromotechCNN Network Class
class PromotechCNN:
    # One Hot Encode Nucleotides
    def oneHot(self, nuc:str) -> tf.Tensor:
        return tf.one_hot(np.array([*map(lambda x:{'A':0, 'C':1, 'G':2, 'T':3}[x] if x in 'ACTG' else 4, nuc)]), depth = 4, dtype = np.uint8)
    
    # Get Data from File
    def getData(self, filename:str, label:int) -> Tuple[List[tf.Tensor], List[int]]:
        X, y = [], []

        with open(filename, 'r') as f:
            for line in f.readlines():
                if line[0] != '>' and not line.isspace() and len(line) > 40:
                    line = line.strip('\n').upper()
                    enc = self.oneHot([*line])

                    if len(line):
                        X += [enc]
                        y += [label]
        
        return X, y
    
    # Build Data from Files
    def buildData(self, negFiles:List[str], posFiles:List[str]) -> Tuple[List[tf.Tensor], List[int]]:
        X, y = [], []

        for file in negFiles:
            if VERBOSE:
                print(f'Getting Data from {file}...')

            XFile, yFile = self.getData(file, 0)
            X += XFile
            y += yFile
        
        for file in posFiles:
            if VERBOSE:
                print(f'Getting Data from {file}...')

            XFile, yFile = self.getData(file, 1)
            X += XFile
            y += yFile

        if VERBOSE:
            print(f'Data Size: {len(X)}')
        
        return X, y
    
    # Initialize Network Architecture
    def buildNetwork(self) -> None:
        inputLayer = layers.Input(shape = (40,4))

        x = layers.GaussianNoise(stddev = STD)(inputLayer)
        x = layers.Conv1D(64, kernel_size = 3, padding = 'same')(x)
        x = layers.LeakyReLU(ALPHA)(x)

        for q in range(5):
            conv = layers.Conv1D(64, kernel_size = 3, padding = 'same', dilation_rate = 2 ** (q + 1))(x)
            conv = layers.LeakyReLU(ALPHA)(conv)
            conv = SqueezeAndExcite1D(64)(conv)

            x = layers.Add()([x, conv])

        x = layers.GlobalAveragePooling1D()(x)
        outputLayer = layers.Dense(1, activation = 'sigmoid')(x)

        self.model = Model(inputLayer, outputLayer)
        self.model.compile(Adam(LEARNING_RATE), loss = BinaryCrossentropy(), metrics = [AUC(curve = 'PR', name = 'AUPRC')])

        if VERBOSE:
            self.model.summary()

    # Train Model
    def train(self, XTrain:List[tf.Tensor], yTrain:List[int], XVal:List[tf.Tensor], yVal:List[int]) -> None:
        earlyStopping = EarlyStopping(monitor = 'val_AUPRC', mode = 'max', verbose = VERBOSE, patience = PATIENCE, restore_best_weights = True)
        noiseDecay = NoiseDecay(PSI)
        print("Fitting data to the model")
        history = self.model.fit(np.array(XTrain), np.array(yTrain), epochs = NUM_EPOCHS, verbose = VERBOSE, validation_data = (np.array(XVal), np.array(yVal)), shuffle = True, callbacks = [earlyStopping, noiseDecay], batch_size = BATCH_SIZE)
        print("Results : ")
        if VERBOSE:
            print("!!!!!!!!!")
            print(history.history)
        
    # Get AUPRC and Precision-Recall Curve
    def test(self, XTest:List[tf.Tensor], yTest:List[int]) -> Tuple[np.float32, PrecisionRecallDisplay]:
        XTest, yTest = np.array(XTest), np.array(yTest)

        yHat = self.model.predict(XTest)

        precision, recall, _ = precision_recall_curve(yTest, yHat)
        precRecCurve = PrecisionRecallDisplay(precision, recall)

        AUPRC = AUC(curve = 'PR')
        AUPRC.update_state(yTest, yHat)

        return AUPRC.result().numpy(), precRecCurve
