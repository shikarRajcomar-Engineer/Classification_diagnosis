
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.






df = pd.read_excel('Model data.xlsx',engine='openpyxl')
x = df[df.columns[2:9]].to_numpy()
scaler = preprocessing.StandardScaler()
scaled_data = scaler.fit_transform(x)
train_data, test_data = train_test_split(scaled_data, test_size=0.3)
n_features = train_data.shape[1]  


for i in range(n_features):
    encoder = keras.Sequential(name='encoder')
    encoder.add(keras.layers.Dense(units=64, activation=keras.activations.relu, input_shape=[n_features]))
    encoder.add(keras.layers.BatchNormalization())
    encoder.add(keras.layers.Dropout(0.2))
    encoder.add(keras.layers.Dense(units=32, activation=keras.activations.relu))
    encoder.add(keras.layers.BatchNormalization())
    encoder.add(keras.layers.Dropout(0.2))
    encoder.add(keras.layers.Dense(units=16, activation=keras.activations.relu))
    encoder.add(keras.layers.BatchNormalization())
    encoder.add(keras.layers.Dropout(0.2))

    decoder = keras.Sequential(name='decoder')
    decoder.add(keras.layers.Dense(units=32, activation=keras.activations.relu, input_shape=[16]))
    decoder.add(keras.layers.BatchNormalization())
    decoder.add(keras.layers.Dropout(0.2))
    decoder.add(keras.layers.Dense(units=64, activation=keras.activations.relu))
    decoder.add(keras.layers.BatchNormalization())
    decoder.add(keras.layers.Dropout(0.2))
    decoder.add(keras.layers.Dense(units=n_features, activation=keras.activations.sigmoid))
    decoder.add(keras.layers.BatchNormalization())

    autoencoder = keras.Sequential([encoder, decoder])

    loss = keras.losses.Huber()
    learning_rate = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, restore_best_weights=True)

    autoencoder.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[keras.metrics.MeanSquaredError()])


    history = autoencoder.fit(
        x=train_data, y=train_data,
        batch_size=32,
        epochs=50,
        verbose=0,
        validation_data=(test_data, test_data),
        callbacks=[es])

    # Plot the learning curves
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


best_model = keras.models.clone_model(autoencoder)
best_model.set_weights(autoencoder.get_weights())
best_model.save(f'AE_feature{i}.h5')

# autoencoder=load_model('best_model2.h5')



# Im importing a test file and scaling it then passing the scaled values into the model the model predictions is then inversed to reconstruct the original data(y pred)
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
ypred.to_excel('ypred.xlsx')

# y=pd.DataFrame(original)
# mse_df = pd.DataFrame(index=y.index)

# # Calculate MSE for each column
# for column in y.columns:
#     mse_df[column] = [(y_i - ypred_i) ** 2 for y_i, ypred_i in zip(y[column], ypred[column])]
# Results_df=pd.concat([mse_df,Overall_MSE],axis=1)
# # dfx=df[['Ci','Ti','Tci','Tsp','Qc','Tc','T']]

# # Extract sensor error and plot distribution

# sensor_errors=Results_df.drop(Overall_MSE,axis=1)
# Overall_MSE=Results_df['Overall Error']

# for col in sensor_errors.columns:
#     plt.hist(sensor_errors[col],label=col,alpha=0.7)
# # plt.hist(Overall_MSE,label='Overall Error',alpha=0.7)
# plt.xlabel('Reconstruction Error(MSE)')
# plt.ylabel('Frequency')
# plt.title('Distribution of Reconstruction errror')
# plt.legend()
# # plt.show()


# fig,axis=plt.subplots(nrows=len(sensor_errors.columns),figsize=(10,8))
# for i, (col, ax) in enumerate(zip(sensor_errors.columns, axis[:-1])):
#     ax.hist(sensor_errors[col],label=col,alpha=0.7)
#     ax.set_xlabel('Reconsturction error(MSE)')
#     ax.set_ylabel('Frequency')
#     ax.set_title(f'Reconstruction error for {col}')

# fig.tight_layout()
# # plt.show()


# # Define a function to calculate statistics
# def calculate_stats(series):
#     range = series.max() - series.min()
#     iqr = series.quantile(0.75) - series.quantile(0.25)
#     sd = series.std()
#     cv = series.std() / series.mean()
#     return range, iqr, sd, cv

# # Calculate and print statistics for each sensor
# for col in sensor_errors.columns:
#     range, iqr, sd, cv = calculate_stats(sensor_errors[col])
#     print(f"\nSensor {col}:")
#     print(f"  Range: {range:.4f}")
#     print(f"  IQR: {iqr:.4f}")
#     print(f"  SD: {sd:.4f}")
#     print(f"  CV: {cv:.4f}")