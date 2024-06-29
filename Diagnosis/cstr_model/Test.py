import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

def scale_data(method):
    if method == 'MinMax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method. Please choose 'MinMax'")
    return scaler

# Read data
df = pd.read_excel('Model Data.xlsx', engine='openpyxl')
rcon = pd.read_excel('Recon.xlsx', engine='openpyxl')
rcon = rcon.iloc[:37228, :] 
# Apply feature engineering to each column
dfs = []
for idx, col in enumerate(df.columns[2:9]):
    sensor_data = df[col]
    sensor_features = create_features(sensor_data)
    sensor_features=pd.concat([pd.DataFrame(sensor_features),rcon.iloc[:,idx]],axis=1)
    dfs.append(pd.concat([sensor_data, sensor_features], axis=1))

# Build and train separate autoencoders for each column
autoencoders = []

scaler = scale_data('MinMax')
for i, df_feature in enumerate(dfs):
    # Use the corresponding column from rcon.xlsx
    rcon_column = rcon.iloc[:, i]
    
    # Add the rcon column to the feature DataFrame
    df_feature_with_rcon = df_feature.copy()
    # df_feature_with_rcon['rcon'] = rcon_column.values

    # Prepare data for training
    x = df_feature_with_rcon.to_numpy()
    n_features = x.shape[1]
    scaled_data = scaler.fit_transform(x)
    train_data, test_data = train_test_split(scaled_data, test_size=0.3)

    # Build autoencoder for the current column
    input_data = keras.Input(shape=(n_features,))
    encoded = keras.layers.Dense(units=64, activation='relu')(input_data)
    encoded = keras.layers.Dense(units=32, activation='relu')(encoded)
    encoded = keras.layers.Dense(units=16, activation='relu')(encoded)
    decoded = keras.layers.Dense(units=32, activation='relu')(encoded)
    decoded = keras.layers.Dense(units=64, activation='relu')(decoded)
    decoded = keras.layers.Dense(units=n_features, activation='linear')(decoded)

    autoencoder = keras.Model(input_data, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.mean_squared_error, 'accuracy'])

    # Train autoencoder for the current column
    history = autoencoder.fit(train_data, train_data, epochs=10, batch_size=32, shuffle=True, verbose=0, validation_data=(test_data, test_data))

    autoencoders.append(autoencoder)
    autoencoder.save(f'AEmodel{i}.h5')
