# ----------------------------------------------------------------------------------------------------------
# Pipeline Dependancies
# ----------------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------------
# 1.Importing Data Step in Pipeline
# ----------------------------------------------------------------------------------------------------------

# Import model file-Master data only contains clean records with no Anaomalies
df = pd.read_excel('Model data.xlsx',engine='openpyxl')
column_names=df.columns

name=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']
df1=df.iloc[:,2:9]

# signal_plotter = dataanalysis()

# ----------------------------------------------------------------------------------------------------------
# 2.Data Preprocessing Step in Pipeline
# ----------------------------------------------------------------------------------------------------------

normal_data = df.loc[df["Class"] == 0]
fault_data = df.loc[df["Class"] == 1]

print("normal_data ", normal_data.shape)
print("fault data ", fault_data.shape)
print("Percent fault ", round((len(fault_data)/len(df)), 4),"%")

# Features to be used to describe anomalies
x = df[df.columns[2:9]].to_numpy()

# column with class
y = df[df.columns[1]].to_numpy()

# This portion of code remove anomalies if the dataset contained anomalies
df = pd.concat([pd.DataFrame(x), pd.DataFrame({'anomaly': y})], axis=1)
normal_events = df[df['anomaly'] == 0]
normal_events = normal_events.loc[:, normal_events.columns != 'anomaly']
df_normal_events = df[df['anomaly'] == 0].loc[:, df.columns != 'anomaly']


# Scale data to make mean 0 and std deviation 1
scaler = preprocessing.StandardScaler()
scaler.fit(df.drop('anomaly', 1))
scaled_data = scaler.transform(normal_events)
train_data, test_data = train_test_split(scaled_data, test_size=0.3)
n_features = x.shape[1]

# ----------------------------------------------------------------------------------------------------------
#3. Model Training Step in Pipeline
# ----------------------------------------------------------------------------------------------------------



# # model
# Add LSTM layers in encoder
n_features = train_data.shape[1]  # Assuming train_data is the input data

# # Model
# encoder = keras.Sequential(name='encoder')
# encoder.add(keras.layers.Dense(units=64, activation=keras.activations.relu, input_shape=[n_features]))
# encoder.add(keras.layers.BatchNormalization())
# encoder.add(keras.layers.Dropout(0.2))

# encoder.add(keras.layers.Dense(units=32, activation=keras.activations.relu))
# encoder.add(keras.layers.BatchNormalization())
# encoder.add(keras.layers.Dropout(0.2))

# encoder.add(keras.layers.Dense(units=16, activation=keras.activations.relu))
# encoder.add(keras.layers.BatchNormalization())
# encoder.add(keras.layers.Dropout(0.2))

# decoder = keras.Sequential(name='decoder')
# decoder.add(keras.layers.Dense(units=32, activation=keras.activations.relu, input_shape=[16]))
# decoder.add(keras.layers.BatchNormalization())
# decoder.add(keras.layers.Dropout(0.2))

# decoder.add(keras.layers.Dense(units=64, activation=keras.activations.relu))
# decoder.add(keras.layers.BatchNormalization())
# decoder.add(keras.layers.Dropout(0.2))

# decoder.add(keras.layers.Dense(units=n_features, activation=keras.activations.sigmoid))
# decoder.add(keras.layers.BatchNormalization())

# autoencoder = keras.Sequential([encoder, decoder])

# loss = keras.losses.Huber()

# learning_rate = 0.001

# optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# autoencoder.compile(
#     loss=loss,
#     optimizer=optimizer,
#     metrics=[keras.metrics.MeanSquaredError()])

# # Train the model with a batch size of 8
# es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, restore_best_weights=True)
# history = autoencoder.fit(
#     x=train_data, y=train_data,
#     batch_size=32,
#     epochs=50,
#     verbose=0,
#     validation_data=(test_data, test_data),
#     callbacks=[es])

# # Plot the learning curves
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()


# best_model = keras.models.clone_model(autoencoder)
# best_model.set_weights(autoencoder.get_weights())
# best_model.save('best_model2.h5')

autoencoder=load_model('best_model2.h5')


raw_data=pd.read_excel('Ci Fault 1 0.xlsx',engine='openpyxl')
raw_data=raw_data.iloc[:,2:9]
scaler = StandardScaler()
scaler.fit(df.drop('anomaly', 1))
scaled_data = scaler.transform(raw_data)

# # Make predictions
predictions = autoencoder.predict(scaled_data)



# import itertools

# dimensions = range(7)
# pairs = list(itertools.combinations(dimensions, 2))

# for pair in pairs:
#     plt.scatter(predictions[:, pair[0]], predictions[:, pair[1]], marker='o', label='Anomalies')
#     plt.xlabel(f'Latent Feature {pair[0] + 1}')
#     plt.ylabel(f'Latent Feature {pair[1] + 1}')
#     plt.title(f'Visualization of Anomalies in Latent Space (Pair {pair[0] + 1}, {pair[1] + 1})')
#     plt.legend()
#     plt.show()


# from sklearn.manifold import TSNE

# # Assuming 'predictions' is your encoded representation
# tsne = TSNE(n_components=2)
# reduced_dimensions = tsne.fit_transform(predictions)

# plt.scatter(reduced_dimensions[:, 0], reduced_dimensions[:, 1], c='red', marker='o', label='Anomalies')
# plt.xlabel('T-SNE Dimension 1')
# plt.ylabel('T-SNE Dimension 2')
# plt.title('Visualization of Anomalies in Reduced Latent Space')
# plt.legend()
# plt.show()

import seaborn as sns

# Assuming 'scaled_data' is your input data
encoded_data = autoencoder.layers[0](scaled_data).numpy()  # Using the first layer of the autoencoder as the encoder

# Get the number of latent features
num_latent_features = encoded_data.shape[1]

# Compute the correlation matrix
correlation_matrix = np.corrcoef(encoded_data, rowvar=False)

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=range(1, num_latent_features + 1), yticklabels=range(1, num_latent_features + 1))
plt.title('Correlation Matrix of Latent Features')
plt.show()





# Overall_MSE = keras.losses.mean_squared_error(scaled_data, predictions)
# Overall_MSE=pd.DataFrame(Overall_MSE,columns=['Overall Error'])


# original=scaler.inverse_transform(predictions)
# print(pd.DataFrame(original))
# ypred=pd.DataFrame(predictions)

# y=pd.DataFrame(scaled_data)
# mse_df = pd.DataFrame(index=y.index)

# # Calculate MSE for each column
# for column in y.columns:
#     mse_df[column] = [(y_i - ypred_i) ** 2 for y_i, ypred_i in zip(y[column], ypred[column])]

# Results_df=pd.concat([mse_df,Overall_MSE],axis=1)

# # Display the resulting MSE dataframe
# print(Results_df)
# # Results_df.to_excel('re.xlsx')