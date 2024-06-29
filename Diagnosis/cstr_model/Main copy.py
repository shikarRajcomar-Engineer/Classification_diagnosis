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
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
# Hide Warnings
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


def Model_development(n_features, train_data, test_data):
    # Define the encoder
    encoder = tf.keras.Sequential(name='encoder')
    encoder.add(Dense(units=20, activation='relu', input_shape=[n_features]))
    encoder.add(Dropout(0.1))
    encoder.add(Dense(units=10, activation='relu'))
    encoder.add(Dense(units=5, activation='relu'))

    # Define the decoder
    decoder = tf.keras.Sequential(name='decoder')
    decoder.add(Dense(units=10, activation='relu', input_shape=[5]))
    decoder.add(Dense(units=20, activation='relu'))
    decoder.add(Dropout(0.1))
    decoder.add(Dense(units=n_features, activation='sigmoid'))

    # Combine encoder and decoder into an autoencoder
    autoencoder = tf.keras.Sequential([encoder, decoder])

    # Compile the model
    autoencoder.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Define callbacks
    checkpoint = ModelCheckpoint('autoencoder_best.h5', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train the model
    history = autoencoder.fit(
        x=train_data, y=train_data,
        batch_size=32,
        epochs=50,
        verbose=1,
        validation_data=(test_data, test_data),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    # Plotting training history
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
    plt.show()

    # Save the best model
    best_model = tf.keras.models.load_model('autoencoder_best.h5')
    best_model.save('Ci_logtransform.h5')

    return history, autoencoder


# Preparing model data-only required if we are retraining a new model
df = pd.read_excel('Model data.xlsx',engine='openpyxl')
df['Ci']=df.Ci.apply(np.log)
df['C']=df.C.apply(np.log)
x = df[df.columns[2:9]].to_numpy()





scaler = preprocessing.MinMaxScaler()
scaled_data = scaler.fit_transform(x)
train_data, test_data = train_test_split(scaled_data, test_size=0.3)
n_features = train_data.shape[1]
Model_development(n_features,train_data, test_data)
autoencoder=load_model('Ci_logtransform.h5')



    # Load test data
raw_data = pd.read_excel('Model Data.xlsx', engine='openpyxl')
test_data = raw_data.iloc[:, 2:9]

# Scale test data
scaled_test_data = scaler.transform(test_data)
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
df2=pd.DataFrame(predictions)

# Calculate the MSLE for each column and store it in a new DataFrame called Error_By_Sensor
Error_By_Sensor = pd.DataFrame()
for col in df1.columns:
    point1 = df1[col].values.reshape(-1, 1)
    point2 = df2[col].values.reshape(-1, 1)

    # Calculate the MSLE between Y and Ypred for each column
    # tf.keras.losses.mean_squared_logarithmic_error
    # tf.keras.losses.mean_squared_error
    # tf.keras.losses.mean_absolute_error


    msle = tf.keras.losses.mean_squared_logarithmic_error(point1, point2).numpy()
    Error_By_Sensor[col] = msle

Error_By_Sensor.columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']

Error_By_Sensor.to_excel('Ci_logtransform.xlsx')













# I am importing a test file and scaling it then passing the scaled values into the model the model predictions is then inversed to reconstruct the original data(y pred)
# Overall mse is calculated and each signal mse is calculated(MSE vs MAE?)
# All datapointws are being passed but can also only pass Class =1 Anomalies







# ----
# autoencoders = ['AE_feature0.h5','AE_feature1.h5','AE_feature2.h5','AE_feature3.h5','AE_feature4.h5','AE_feature5.h5','AE_feature6.h5']
# all_results=pd.DataFrame()


# folder_path=os.getcwd()+'/Fault 1_Bias'
# for filename in os.listdir(folder_path):
#     filepath=os.path.join(folder_path,filename)
#     if os.path.isfile(filepath) and filename.endswith('.xlsx'):

#         raw_data = pd.read_excel(filepath, engine='openpyxl')
#         raw_data1=raw_data[raw_data.Class==1]
#         raw_data=raw_data.iloc[:,2:9]

#         scaler = StandardScaler()
#         scaled_data = scaler.fit_transform(raw_data)
#         scaled_data=(pd.DataFrame(scaled_data))
#         original=scaler.inverse_transform(scaled_data)
#         # Predictions dataframe works out model prediction for each sensor using the individual models and then converts them back to the original signal(ypred)
#         predictions=pd.DataFrame()
#         reconstruction_errors = []
#         predicted_originals = []
#         for i,autoencoder in enumerate(autoencoders):
#             model = load_model(autoencoder)
#             print(autoencoder)
#             print(filename)
#             predicted_data = model.predict(scaled_data[i])
#             predicted_data = pd.DataFrame(predicted_data)
#             predictions = pd.concat([predictions, predicted_data], axis=1)
#             print(predicted_data)
#         predictions=pd.DataFrame(scaler.inverse_transform(predictions))
#         df1=pd.DataFrame(scaler.inverse_transform(scaled_data))
#         df2=pd.DataFrame(predictions)

#         # Calculate the MSLE for each column and store it in a new DataFrame called Error_By_Sensor
#         Error_By_Sensor = pd.DataFrame()
#         for col in df1.columns:
#             point1 = df1[col].values.reshape(-1, 1)
#             point2 = df2[col].values.reshape(-1, 1)
#             msle = tf.keras.losses.mean_absolute_error(point1, point2).numpy()
#             Error_By_Sensor[col] = msle

#         Error_By_Sensor.columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']
#         print('(-------)')
#         print(Error_By_Sensor)
#         print('(-------)')

#         spread_df = Utils.Spread(Error_By_Sensor)
#         widest_spread = pd.DataFrame(Utils.widest_spread_sensor(spread_df),columns=['Diagnosis'])
#         results=pd.concat([spread_df,widest_spread],axis=1)
#         results['filename']=filename
#         all_results=all_results.append(results)
#         Error_By_Sensor.columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']
#         dff=pd.concat([Error_By_Sensor,raw_data1.Class],axis=1)
#         dff=dff[dff.Class==1]
#         dff=dff[['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']]

#         dff.boxplot(figsize=(10, 6))
#         plt.title('Boxplot for Each Column')
#         plt.ylabel('Values')
#         plt.xticks(rotation=45)
#         plt.show()
#         print(autoencoder)
#         print('-------------------------')
# # all_results.to_excel('all_results.xlsx')
