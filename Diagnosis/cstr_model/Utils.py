
import numpy as np
from numpy import ma
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker, cm
from matplotlib.pyplot import figure
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
import scipy
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, decomposition, preprocessing
from sklearn.model_selection import KFold

# from model import Autoencoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error

# Hide Warnings
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')



# Prepare the dataset

# This code combines the ci fault and ti fault summary into a master file
# def Data_prep(pattern):
#     path=os.getcwd()
#     # pattern='final'

#     masterfile=pd.DataFrame()
#     files=[f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
#     for filename in files:
#         if filename.startswith(pattern):
#             df=pd.read_excel(filename)
#             masterfile=masterfile.append(df)
#     masterfile=masterfile.iloc[:,2:]
#     return masterfile

def Model_development(n_features,train_data, test_data):
    encoder = keras.Sequential(name='encoder')
    encoder.add(layer=keras.layers.Dense(units=20, activation=keras.activations.relu, input_shape=[n_features]))
    encoder.add(keras.layers.Dropout(0.1))
    encoder.add(layer=keras.layers.Dense(units=10, activation=keras.activations.relu))
    encoder.add(layer=keras.layers.Dense(units=5, activation=keras.activations.relu))

    decoder = keras.Sequential(name='decoder')
    decoder.add(layer=keras.layers.Dense(units=10, activation=keras.activations.relu, input_shape=[5]))
    decoder.add(layer=keras.layers.Dense(units=20, activation=keras.activations.relu))
    decoder.add(keras.layers.Dropout(0.1))
    decoder.add(layer=keras.layers.Dense(units=n_features, activation=keras.activations.sigmoid))

    autoencoder = keras.Sequential([encoder, decoder])

    autoencoder.compile(
        loss=keras.losses.MSE,
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.mean_squared_error,'accuracy'])

    loss = keras.losses.Huber()
    learning_rate = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, restore_best_weights=True)

    history = autoencoder.fit(
        x=train_data, y=train_data,
        batch_size=32,
        epochs=50,
        verbose=0,
        validation_data=(test_data, test_data),
        callbacks=[es])

    # # Plot the learning curves
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

# Save the best model
    best_model = keras.models.clone_model(autoencoder)
    best_model.set_weights(autoencoder.get_weights())
    best_model.save('AE_feature.h5')

    return history




def plot_training_history(history):
    # Plot training history
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()