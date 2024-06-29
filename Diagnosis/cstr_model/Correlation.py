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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# def create_features(sensor_data):
#     rolling_mean = sensor_data.rolling(window=7, min_periods=1).mean()
#     rolling_std = sensor_data.rolling(window=7, min_periods=1).std()

#     features_df = pd.DataFrame({
#         'original': sensor_data,
#         'rolling_mean': rolling_mean,
#         'rolling_std': rolling_std
#     })
#     features_df.dropna(inplace=True)
#     return features_df

# def build_autoencoder(n_features):
#     input_data = keras.Input(shape=(n_features,))

#     # Encoder
#     x = keras.layers.Dense(units=10, activation='relu')(input_data)
#     x = keras.layers.Dropout(0.1)(x)

#     # Bottleneck
#     encoded = keras.layers.Dense(units=10, activation='relu')(x)

#     # Decoder
#     x = keras.layers.Dense(units=8, activation='relu')(encoded)
#     x = keras.layers.Dropout(0.1)(x)
#     x = keras.layers.Dense(units=4, activation='relu')(x)

#     x = keras.layers.Dropout(0.4)(x)

#     decoded = keras.layers.Dense(units=n_features, activation='relu')(x)

#     autoencoder = keras.Model(input_data, decoded)
#     autoencoder.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001), loss='mean_absolute_error', metrics=['accuracy'])

#     return autoencoder

# def Model_development(n_features, train_data, test_data, model_save_path):
#     autoencoder = build_autoencoder(n_features)

#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

#     history = autoencoder.fit(
#         x=train_data, y=train_data,
#         batch_size=32,
#         epochs=50,
#         verbose=1,
#         validation_data=(test_data, test_data),
#         callbacks=[es])

    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['mean_squared_error'])
    # plt.plot(history.history['val_mean_squared_error'])
    # plt.title('Model Mean Squared Error')
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Squared Error')
    # plt.legend(['Train', 'Validation'], loc='upper left')

    # # plt.subplot(1, 2, 2)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
#     # plt.title('Model Loss')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('Loss')
#     # plt.legend(['Train', 'Validation'], loc='upper left')

#     # plt.tight_layout()
#     # plt.show()

#     # Save model
#     autoencoder.save(model_save_path)

#     return history, autoencoder

# # Read data
# df = pd.read_excel('Model Data.xlsx')
# x = df.Qc.to_numpy()
# df = pd.DataFrame(x, columns=['Qc'])


# recon = pd.read_excel('Recon.xlsx')


# dfs = []
# for col in df.columns:
#     sensor_data = df[col]
#     sensor_features = create_features(sensor_data)
#     dfs.append(sensor_features)


# for idx, df_feature in enumerate(dfs):
#     # df_feature = pd.concat([pd.DataFrame(df_feature), recon.Qc], axis=1)
#     df_feature.fillna(df_feature.mean(), inplace=True)

#     # Prepare data for training
#     x = df_feature.to_numpy()
#     n_features = x.shape[1]
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(x)
#     train_data, test_data = train_test_split(scaled_data, test_size=0.2)

#     # Define path for saving model
#     model_save_path = 'Qc.h5'

#     # Train model and save
#     Model_development(n_features, train_data, test_data, model_save_path)







# autoencoders = ['AE_model_feature0.h5','AE_model_feature1.h5','AE_model_feature2.h5','AE_model_feature3.h5','AE_model_feature4.h5','AE_model_feature5.h5','AE_model_feature6.h5']

# # ['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C'])

# # Preprocess data and pass data through to mainAE to get reconstruction data which is then passed to the individual AE
# df = pd.read_excel(os.getcwd()+'/Fault 1_Bias/Ci.xlsx',engine='openpyxl')
# # df['Ci']=df.Ci.apply(np.log)*100
# # df['C']=df.C.apply(np.log)*100
# x = df[df.columns[2:9]].to_numpy()
# scaler = preprocessing.MinMaxScaler()
# scaled_data = scaler.fit_transform(x)
# model=load_model('Test.h5')
# recon=model.predict(scaled_data) 
# recon=pd.DataFrame(recon,columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C'])



folder_path=os.getcwd()+'/Fault 1_Bias/Tsp 1.xlsx'
raw_data = pd.read_excel(folder_path, engine='openpyxl')

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


sensor_data = test_data['Qc']
sensor_features = create_features(sensor_data)
# sensor_features=pd.concat([pd.DataFrame(sensor_features),recon.Ti],axis=1)
dfs.append(pd.concat([sensor_data, sensor_features], axis=1))

dfs[0].name = 'df1'
dfs[0]['rolling_mean'].fillna(dfs[0]['rolling_mean'].mean(), inplace=True)
dfs[0]['rolling_std'].fillna(dfs[0]['rolling_std'].mean(), inplace=True)
print(dfs[0])

# # # Scale test data
scaler=MinMaxScaler()

# # Predict and calculate reconstruction error for each column
# Error_By_Sensor = pd.DataFrame()

scaled_test_data = scaler.fit_transform(dfs[0])
model = tf.keras.models.load_model('Qc.h5')
predicted_data = model.predict(dfs[0])

# # Inverse scaling
predicted_data = scaler.inverse_transform(predicted_data)
scaled_test_data = scaler.inverse_transform(scaled_test_data)



# # Convert to DataFrames
df_inverse_test_data = pd.DataFrame(scaled_test_data)
df_inverse_predicted_data = pd.DataFrame(predicted_data)

# Concatenate the DataFrames
combined_df = pd.concat([df_inverse_test_data[0], df_inverse_predicted_data[0]], axis=1)



# df_inverse_predicted_data.to_excel('Qc.xlsx')

print(combined_df.head(5))
