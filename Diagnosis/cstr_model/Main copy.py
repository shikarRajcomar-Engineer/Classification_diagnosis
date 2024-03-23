
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
import Utils
# Hide Warnings
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')



df = pd.read_excel('Model data.xlsx',engine='openpyxl')
x = df[df.columns[2:9]].to_numpy()
scaler = preprocessing.StandardScaler()
scaled_data = scaler.fit_transform(x)
train_data, test_data = train_test_split(scaled_data, test_size=0.3)
n_features = train_data.shape[1]


Utils.Model_development(n_features,train_data, test_data)


autoencoder=load_model('best_model2.h5')



# I am importing a test file and scaling it then passing the scaled values into the model the model predictions is then inversed to reconstruct the original data(y pred)
# Overall mse is calculated and each signal mse is calculated(MSE vs MAE?)
# All datapointws are being passed but can also only pass Class =1 Anomalies


raw_data=pd.read_excel('Tci amp 20 Fault 1.xlsx',engine='openpyxl')
# raw_data=raw_data.query('Class==1')
raw_data=raw_data.iloc[:,2:9]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(raw_data)
original=scaler.inverse_transform(scaled_data)
predictions = autoencoder.predict(scaled_data)
Predicted_original=scaler.inverse_transform(predictions)
Overall_MSE = keras.losses.mean_squared_error(original, predictions)
Overall_MSE=pd.DataFrame(Overall_MSE,columns=['Overall Error'])


ypred=pd.DataFrame(Predicted_original)
