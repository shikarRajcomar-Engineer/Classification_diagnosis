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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import os
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


df = pd.read_excel(os.getcwd()+'/Fault 1_Bias/C_SensorBias.xlsx',engine='openpyxl')
df['Ci']=df.Ci.apply(np.log)*100
df['C']=df.C.apply(np.log)*100
x = df[df.columns[2:9]].to_numpy()
scaler = preprocessing.MinMaxScaler()
scaled_data = scaler.fit_transform(x)
model=load_model('Test.h5')
recon=model.predict(scaled_data)
recon=pd.DataFrame(recon)






autoencoders = ['AE_model_feature0.h5','AE_model_feature1.h5','AE_model_feature2.h5','AE_model_feature3.h5','AE_model_feature4.h5','AE_model_feature5.h5','AE_model_feature6.h5']

folder_path=os.getcwd()+'/Fault 1_Bias/C_SensorBias.xlsx'
raw_data = pd.read_excel(folder_path, engine='openpyxl')
raw_data['Ci']=raw_data.Ci.apply(np.log)*100
raw_data['C']=raw_data.C.apply(np.log)*100
test_data = raw_data.iloc[:, 2:9]


def create_features(sensor_data):
    rolling_mean = sensor_data.rolling(window=7, min_periods=1).mean()
    rolling_std = sensor_data.rolling(window=7, min_periods=1).std()
    features_df = pd.DataFrame({
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
    })
    features_df.dropna(inplace=True)
    return features_df


dfs = []

for idx, col in enumerate(test_data.columns):
    sensor_data = test_data[col]
    sensor_features = create_features(sensor_data)
    sensor_features=pd.concat([pd.DataFrame(sensor_features),recon.iloc[:,idx]],axis=1)
    dfs.append(pd.concat([sensor_data, sensor_features], axis=1))


# Naming the dataframes
dfs[0].name = 'df1'
dfs[1].name = 'df2'
dfs[2].name = 'df3'
dfs[3].name = 'df4'
dfs[4].name = 'df5'
dfs[5].name = 'df6'
dfs[6].name = 'df7'
dfs[0]['rolling_mean'].fillna(dfs[0]['rolling_mean'].mean(), inplace=True)
dfs[0]['rolling_std'].fillna(dfs[0]['rolling_std'].mean(), inplace=True)
dfs[1]['rolling_mean'].fillna(dfs[1]['rolling_mean'].mean(), inplace=True)
dfs[1]['rolling_std'].fillna(dfs[1]['rolling_std'].mean(), inplace=True)
dfs[2]['rolling_mean'].fillna(dfs[2]['rolling_mean'].mean(), inplace=True)
dfs[2]['rolling_std'].fillna(dfs[2]['rolling_std'].mean(), inplace=True)
dfs[3]['rolling_mean'].fillna(dfs[3]['rolling_mean'].mean(), inplace=True)
dfs[3]['rolling_std'].fillna(dfs[3]['rolling_std'].mean(), inplace=True)
dfs[4]['rolling_mean'].fillna(dfs[4]['rolling_mean'].mean(), inplace=True)
dfs[4]['rolling_std'].fillna(dfs[4]['rolling_std'].mean(), inplace=True)
dfs[5]['rolling_mean'].fillna(dfs[5]['rolling_mean'].mean(), inplace=True)
dfs[5]['rolling_std'].fillna(dfs[5]['rolling_std'].mean(), inplace=True)
dfs[6]['rolling_mean'].fillna(dfs[6]['rolling_mean'].mean(), inplace=True)
dfs[6]['rolling_std'].fillna(dfs[6]['rolling_std'].mean(), inplace=True)


# # Scale test data
scaler=MinMaxScaler()

# Predict and calculate reconstruction error for each column
Error_By_Sensor = pd.DataFrame()

for i, autoencoder in enumerate(autoencoders):
    scaler = MinMaxScaler()  # Initialize the scaler inside the loop
    scaled_test_data = scaler.fit_transform(dfs[i])
    model = tf.keras.models.load_model(autoencoder)
    predicted_data = model.predict(dfs[i])

    # Inverse scaling
    predicted_data = scaler.inverse_transform(predicted_data)
    scaled_test_data = scaler.inverse_transform(scaled_test_data)

    # Calculate MSE for each observation and average across features
    mse_per_observation = np.mean((predicted_data - scaled_test_data)**2, axis=1)

    # Store the MSE in Error_By_Sensor DataFrame
    Error_By_Sensor[f'df{i+1}'] = mse_per_observation

# Rename the columns
Error_By_Sensor.columns = ['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']

# Error_By_Sensor.columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']
dff=pd.concat([Error_By_Sensor,raw_data.Class],axis=1)
dff=dff[dff.Class==1]
dff=dff[['Ci', 'Ti', 'T', 'Qc',  'Tci', 'Tc', 'C']]

# dff.boxplot(figsize=(10, 6))
# plt.title('Boxplot for Each Column')
# plt.ylabel('Values')
# plt.xticks(rotation=45)
# plt.show()


# Plotting spread/distribution for each error by sensor
Error_By_Sensor.boxplot(figsize=(10, 6))
plt.title('Spread/Distribution of Errors for Each Sensor')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.show()

# # # Normalize the error values across sensors
# normalized_errors = (Error_By_Sensor - Error_By_Sensor.mean()) / Error_By_Sensor.std()

# # Plotting spread/distribution for each error by sensor
# normalized_errors.boxplot(figsize=(10, 6))
# plt.title('Spread/Distribution of Normalized Errors for Each Sensor')
# plt.ylabel('Normalized MSE')
# plt.xticks(rotation=45)
# plt.show()