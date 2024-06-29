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


# Preparing model data-only required if we are retraining a new model
df = pd.read_excel('Model data.xlsx',engine='openpyxl')
df['Ci']=df.Ci.apply(np.log)*100
df['C']=df.C.apply(np.log)*100
x = df[df.columns[2:9]].to_numpy()


scaler = preprocessing.MinMaxScaler()
scaled_data = scaler.fit_transform(x)
train_data, test_data = train_test_split(scaled_data, test_size=0.3)
n_features = train_data.shape[1]

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


    best_model = keras.models.clone_model(autoencoder)
    best_model.set_weights(autoencoder.get_weights())
    best_model.save('Test.h5')

    return history,autoencoder



Model_development(n_features,train_data, test_data)
autoencoder=load_model('Test.h5')


test_data = x

# Scale test data
scaled_test_data = scaler.transform(test_data)
scaled_test_data=pd.DataFrame(scaled_test_data)
reconstruction_errors = []
predicted_originals = []

predictions=pd.DataFrame()

predicted_data = autoencoder.predict(scaled_test_data)
predicted_data=pd.DataFrame(predicted_data)
predictions=pd.concat([predictions,predicted_data],axis=1)

predictions=pd.DataFrame(scaler.inverse_transform(predictions))

df1=pd.DataFrame(scaler.inverse_transform(scaled_test_data))
df2=pd.DataFrame(predictions)

# Calculate the MSLE for each column and store it in a new DataFrame called Error_By_Sensor
Error_By_Sensor = pd.DataFrame()
for col in df1.columns:
    point1 = df1[col].values.reshape(-1, 1)
    point2 = df2[col].values.reshape(-1, 1)

    msle = tf.keras.losses.mean_squared_logarithmic_error(point1, point2).numpy()
    Error_By_Sensor[col] = msle

Error_By_Sensor.columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']

Recon=pd.DataFrame(Error_By_Sensor)

# ------------------------------------------------------------------------
# Individual models
# ------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

def Model_development_individual(n_features, train_data, test_data, model_save_path):
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

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    # plt.show()

    # Save model
    autoencoder.save(model_save_path)

    return history, autoencoder

# Read data
df = pd.read_excel('Model Data.xlsx')
x = df[df.columns[2:9]].to_numpy()
df=pd.DataFrame(x,columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C'])
df['Ci']=df.Ci.apply(np.log)*100
df['C']=df.C.apply(np.log)*100

recon=Recon

# Apply feature engineering to each column
dfs = []
for col in df.columns:
    sensor_data = df[col]
    sensor_features = create_features(sensor_data)
    dfs.append(sensor_features)


# Prepare data and train models for each feature
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
    model_save_path = f'AE_model_feature{idx}.h5'

    # Train model and save
    Model_development_individual(n_features, train_data, test_data, model_save_path)