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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


# # Preparing model data-only required if we are retraining a new model
# df = pd.read_excel('Model data.xlsx',engine='openpyxl')
# df['Ci']=df.Ci.apply(np.log)*100
# df['C']=df.C.apply(np.log)*100
# x = df[df.columns[2:9]].to_numpy()

# # Normilize the Data using MinMax scaler
# scaler = preprocessing.MinMaxScaler()
# scaled_data = scaler.fit_transform(x)
# train_data, test_data = train_test_split(scaled_data, test_size=0.3)
# n_features = train_data.shape[1]

# # Function to Build the Main Autoencoder model(using all 7-Sensors)
# def Model_development(n_features,train_data, test_data):
#     encoder = keras.Sequential(name='encoder')
#     encoder.add(layer=keras.layers.Dense(units=20, activation=keras.activations.relu, input_shape=[n_features]))
#     encoder.add(keras.layers.Dropout(0.1))
#     encoder.add(layer=keras.layers.Dense(units=10, activation=keras.activations.relu))
#     encoder.add(layer=keras.layers.Dense(units=5, activation=keras.activations.relu))

#     decoder = keras.Sequential(name='decoder')
#     decoder.add(layer=keras.layers.Dense(units=10, activation=keras.activations.relu, input_shape=[5]))
#     decoder.add(layer=keras.layers.Dense(units=20, activation=keras.activations.relu))
#     decoder.add(keras.layers.Dropout(0.1))
#     decoder.add(layer=keras.layers.Dense(units=n_features, activation=keras.activations.sigmoid))

#     autoencoder = keras.Sequential([encoder, decoder])

#     autoencoder.compile(
#         loss=keras.losses.MSE,
#         optimizer=keras.optimizers.Adam(),
#         metrics=[keras.metrics.mean_squared_error,'accuracy'])

#     loss = keras.losses.Huber()
#     learning_rate = 0.001
#     optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#     es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, restore_best_weights=True)

#     history = autoencoder.fit(
#         x=train_data, y=train_data,
#         batch_size=32,
#         epochs=50,
#         verbose=1,
#         validation_data=(test_data, test_data),
#         callbacks=[es])

#     best_model = keras.models.clone_model(autoencoder)
#     best_model.set_weights(autoencoder.get_weights())
#     best_model.save('MainAE.h5')

#     return history,autoencoder

# ----------------------------------------------------------------------------------------------------------------------
# Train the Main AE model
# ----------------------------------------------------------------------------------------------------------------------

# Training for the Main AE model

autoencoder=load_model('MainAE.h5')



# Load test data which I am using to get reconstruction error used in the individual models as knowledge distillation
raw_data = pd.read_excel('Model Data.xlsx', engine='openpyxl')
raw_data['Ci']=raw_data.Ci.apply(np.log)*100
raw_data['C']=raw_data.C.apply(np.log)*100
test_data = raw_data.iloc[:, 2:9]

# Scale test data
scaler = preprocessing.MinMaxScaler()
scaled_test_data = scaler.fit_transform(test_data)
scaled_test_data=pd.DataFrame(scaled_test_data)

# Predict and calculate reconstruction error for each column
reconstruction_errors = []
predicted_originals = []
predictions=pd.DataFrame()

predicted_data = autoencoder.predict(scaled_test_data)
predicted_data=pd.DataFrame(predicted_data)
predictions=pd.concat([predictions,predicted_data],axis=1)
predictions=pd.DataFrame(scaler.inverse_transform(predictions))

df1=pd.DataFrame(scaler.inverse_transform(scaled_test_data))
df2=predictions


# Calculate the MSLE for each column and store it in a new DataFrame called Error_By_Sensor this will be sent to the individual AE as knowledge distallation
Error_By_Sensor = pd.DataFrame()
for col in df1.columns:
    point1 = df1[col].values.reshape(-1, 1)
    point2 = df2[col].values.reshape(-1, 1)
    msle = tf.keras.losses.mean_squared_logarithmic_error(point1, point2).numpy()
    Error_By_Sensor[col] = msle

Error_By_Sensor.columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']
Error_By_Sensor.to_excel('Reconstruction_Error.xlsx')

# ----------------------------------------------------------------------------------------------------------------------
# Train the indiviudal AE models
# ----------------------------------------------------------------------------------------------------------------------

# Apply Feature engineering to individual models to get more information to help improve performance
def create_features(sensor_data):
    rolling_mean = sensor_data.rolling(window=7, min_periods=1).mean()
    rolling_std = sensor_data.rolling(window=7, min_periods=1).std()
    
    features_df = pd.DataFrame({
        'original': sensor_data, 
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std
    })
    features_df.dropna(inplace=True)
    return features_df

def Individual_autoencoder(n_features, train_data, test_data, model_save_path):
    input_data = keras.Input(shape=(n_features,))  

    encoded = keras.layers.Dense(units=64, activation='relu')(input_data)
    encoded = keras.layers.Dropout(0.2)(encoded)
    encoded = keras.layers.Dense(units=32, activation='relu')(encoded)
    encoded = keras.layers.Dropout(0.6)(encoded)
    encoded = keras.layers.Dense(units=16, activation='relu')(encoded)

    decoded = keras.layers.Dense(units=32, activation='relu')(encoded)
    decoded = keras.layers.Dropout(0.2)(decoded)
    decoded = keras.layers.Dense(units=64, activation='relu')(decoded)
    decoded = keras.layers.Dropout(0.2)(decoded)
    decoded = keras.layers.Dense(units=n_features, activation='linear')(decoded)

    autoencoder = keras.Model(input_data, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mean_squared_error', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

    history = autoencoder.fit(
        x=train_data, y=train_data,
        batch_size=32,
        epochs=100,  
        verbose=0,
        validation_data=(test_data, test_data),
        callbacks=[es])
    return history, autoencoder

# Read data the normal data again -this may be redundant but have it here to understand the code 
df = pd.read_excel('Model Data.xlsx')
x = df[df.columns[2:9]].to_numpy()
df=pd.DataFrame(x,columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C'])
df['Ci']=df.Ci.apply(np.log)*100
df['C']=df.C.apply(np.log)*100

recon=Error_By_Sensor

# Apply feature engineering to each column
dfs = []
for col in df.columns:
    sensor_data = df[col]
    sensor_features = create_features(sensor_data)
    dfs.append(sensor_features)


# Prepare data and train individual models for each feature
for idx, df_feature in enumerate(dfs):
    # Fill NaN values with mean of respective columns
    df_feature=pd.concat([pd.DataFrame(df_feature),recon.iloc[:,idx]],axis=1)
    df_feature.fillna(df_feature.mean(), inplace=True)

    # Prepare data for training
    x = df_feature.to_numpy()
    n_features = x.shape[1]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(x)
    train_data, test_data = train_test_split(scaled_data, test_size=0.3)

    # Define path for saving model
    model_save_path = f'AE_Individual_Model{idx}.h5'

    # Train model and save
    Individual_autoencoder(n_features, train_data, test_data, model_save_path)


# ----------------------------------------------------------------------------------------------------------------------
# Anomaly Detection and diagnosis of sensor
# ----------------------------------------------------------------------------------------------------------------------

# After the main model has been built and the indiviudal models are built now we test the process 

# Preprocess data and pass data through to mainAE to get reconstruction data which is then passed to the individual AE
df = pd.read_excel('C_SensorBias.xlsx',engine='openpyxl')
df['Ci']=df.Ci.apply(np.log)*100
df['C']=df.C.apply(np.log)*100
x = df[df.columns[2:9]].to_numpy()
scaler = preprocessing.MinMaxScaler()
scaled_data = scaler.fit_transform(x)
model=load_model('MainAE.h5')
recon=model.predict(scaled_data) 
recon=pd.DataFrame(recon)



# Load individual AE models  and import fault dataset example :C_SensorBias.xlsx
autoencoders = ['AE_Individual_Model0.h5','AE_Individual_Model1.h5','AE_Individual_Model2.h5','AE_Individual_Model3.h5','AE_Individual_Model4.h5','AE_Individual_Model5.h5','AE_Individual_Model6.h5']

raw_data = pd.read_excel('C_SensorBias.xlsx', engine='openpyxl')
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
# take the fault data pass it thorugh the function to create more features ,in this case its the std and rolling mean then append the reconstruction error from the main AE for the respective sensor to the DFS

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


# Even though im passing the recon error from the main model which is already scaled,do i need to rescale it again ?
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

dff.boxplot(figsize=(10, 6))
plt.title('Boxplot for Each Column')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()



