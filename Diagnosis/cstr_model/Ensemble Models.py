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

def Model_development(n_features, train_data, test_data, model_save_path):
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

recon=pd.read_excel('Recon.xlsx')

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
    Model_development(n_features, train_data, test_data, model_save_path)
