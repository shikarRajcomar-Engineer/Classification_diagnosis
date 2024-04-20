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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error
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
# Utils.Model_development(n_features,train_data, test_data)
autoencoder=load_model('best_model2.h5')



# I am importing a test file and scaling it then passing the scaled values into the model the model predictions is then inversed to reconstruct the original data(y pred)
# Overall mse is calculated and each signal mse is calculated(MSE vs MAE?)
# All datapointws are being passed but can also only pass Class =1 Anomalies

raw_data=pd.read_excel(r'Fault 1_Bias/Tci.xlsx',engine='openpyxl')
df_class=raw_data
raw_data=raw_data.iloc[:,2:9]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(raw_data)
original=scaler.inverse_transform(scaled_data)
predictions = autoencoder.predict(scaled_data)
Predicted_original=scaler.inverse_transform(predictions)
Overall_MSE = keras.losses.mean_squared_logarithmic_error(original, predictions)
Overall_MSE=pd.DataFrame(Overall_MSE,columns=['Overall Error'])
ypred=pd.DataFrame(Predicted_original)

df1=pd.DataFrame(original)
df2=pd.DataFrame(Predicted_original)

# Calculate the MSLE for each column and store it in a new DataFrame called Error_By_Sensor
Error_By_Sensor = pd.DataFrame()
for col in df1.columns:
    point1 = df1[col].values.reshape(-1, 1)
    point2 = df2[col].values.reshape(-1, 1)

    # Calculate the MSLE between Y and Ypred for each column
    # tf.keras.losses.mean_squared_logarithmic_error
    # tf.keras.losses.mean_squared_error
    # tf.keras.losses.mean_absolute_error


    msle = keras.losses.mean_squared_error(point1, point2).numpy()
    Error_By_Sensor[col] = msle
Error_By_Sensor.columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']

outliers = Utils.detect_outliers_Mahalanobis(Overall_MSE)
Error_By_Sensor['Outlier'] = outliers
Error_By_Sensor['msle'] = df.iloc[:,1:7].mean(axis=1)
Error_By_Sensor['Class']=Error_By_Sensor['Outlier'].astype('int')



# fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8))


# ax1.plot(Error_By_Sensor.index,Error_By_Sensor.Class,label='Anomalies')
# ax1.plot(df_class.index,df_class.Class,label='Class')
# ax1.legend()

# ax2.plot(Error_By_Sensor.index,Error_By_Sensor.msle,label='Reconstruction Error')
# ax2.legend()
# plt.show()

# Error_By_Sensor=Error_By_Sensor[Error_By_Sensor.Class==1]
# # Plotting spread/distribution for each error by sensor
# Error_By_Sensor.boxplot(figsize=(10, 6))
# plt.title('Spread/Distribution of Errors for Each Sensor')
# plt.ylabel('MSE')
# plt.xticks(rotation=45)
# plt.show()

# -------------------
# Anomaly Diagnosis
# # -------------------
# autoencoders = ['AE_feature0.h5','AE_feature1.h5','AE_feature2.h5','AE_feature3.h5','AE_feature4.h5','AE_feature5.h5','AE_feature6.h5']

# # Load test data
# raw_data = pd.read_excel('Ci_SensorBias_.xlsx', engine='openpyxl')
# test_data = raw_data.iloc[:, 2:9]

# # Scale test data
# scaled_test_data = scaler.transform(test_data)
# scaled_test_data=pd.DataFrame(scaled_test_data)
# # Predict and calculate reconstruction error for each column
# reconstruction_errors = []
# predicted_originals = []

# predictions=pd.DataFrame()
# for i, autoencoder in enumerate(autoencoders):
#     model = tf.keras.models.load_model(autoencoder)
#     predicted_data = model.predict(scaled_test_data)
#     predicted_data = pd.DataFrame(predicted_data)
#     predictions = pd.concat([predictions, predicted_data], axis=1)


# predictions=pd.DataFrame(scaler.inverse_transform(predictions))
# print(predictions)

spread_df = Utils.Spread(Error_By_Sensor.drop(columns=['Outlier', 'msle', 'Class']))
widest_spread = pd.DataFrame(Utils.widest_spread_sensor(spread_df),columns=['Diagnosis'])
results=pd.concat([spread_df,widest_spread],axis=1)
print(results)