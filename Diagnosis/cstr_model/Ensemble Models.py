from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.drawing.image import Image
import os

def create_features(sensor_data):
    rolling_mean = sensor_data.rolling(window=5, min_periods=1).mean()
    rolling_std = sensor_data.rolling(window=5, min_periods=1).std()
    
    features_df = pd.DataFrame({
        'original': sensor_data, 
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std
    })
    features_df.dropna(inplace=True)
    return features_df

def add_noise(data, noise_factor=0.1):
    noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    noisy_data = np.clip(noisy_data, 0., 1.)
    return noisy_data

def Model_development(n_features, train_data, test_data, model_save_path, feature_idx):
    input_data = keras.Input(shape=(n_features,))  

    encoded = keras.layers.Dense(units=64, activation='relu')(input_data)
    # encoded = keras.layers.Dropout(0.2)(encoded)
    encoded = keras.layers.Dense(units=32, activation='relu')(encoded)
    encoded = keras.layers.Dense(units=16, activation='relu')(encoded)

    decoded = keras.layers.Dense(units=32, activation='relu')(encoded)
    # decoded = keras.layers.Dropout(0.2)(decoded)
    decoded = keras.layers.Dense(units=64, activation='relu')(decoded)
    decoded = keras.layers.Dense(units=n_features, activation='linear')(decoded)

    autoencoder = keras.Model(input_data, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='mean_absolute_error', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, restore_best_weights=True)

    # Add noise to training data
    noisy_train_data = add_noise(train_data)
    noisy_test_data = add_noise(test_data)

    history = autoencoder.fit(
        x=noisy_train_data, y=train_data,
        batch_size=16,
        epochs=100,  
        verbose=0,
        validation_data=(noisy_test_data, test_data),
        callbacks=[es])

    # Plot training curves
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

    # Save plot as image
    plot_image_path = f'training_curve_feature{feature_idx}.png'
    plt.savefig(plot_image_path)
    plt.close()

    # Save model
    autoencoder.save(model_save_path)

    return history, autoencoder, plot_image_path

# Read data
df = pd.read_excel('Model Data.xlsx')
x = df[df.columns[2:9]].to_numpy()
df = pd.DataFrame(x, columns=['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C'])
df['Ci'] = df.Ci.apply(np.log) * 100
df['C'] = df.C.apply(np.log) * 100

recon = pd.read_excel('recon.xlsx')

# Apply feature engineering to each column
dfs = []
for col in df.columns:
    sensor_data = df[col]
    sensor_features = create_features(sensor_data)
    dfs.append(sensor_features)

# DataFrame to store model performance
performance_df = pd.DataFrame(columns=['Feature', 'Train Accuracy', 'Val Accuracy', 'Train Loss', 'Val Loss'])

# Prepare data and train models for each feature
for idx, df_feature in enumerate(dfs):
    # Fill NaN values with mean of respective columns
    df_feature = pd.concat([pd.DataFrame(df_feature), recon.iloc[:, idx]], axis=1)
    df_feature.fillna(df_feature.mean(), inplace=True)

    # Prepare data for training
    x = df_feature.to_numpy()
    n_features = x.shape[1]
    
    # Split data into training and testing sets
    train_data, test_data = train_test_split(x, test_size=0.25)
    
    # Initialize the scaler and fit it on the training data only
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Define path for saving model
    model_save_path = f'AE_model_feature{idx}.h5'

    # Train model and save
    history, autoencoder, plot_image_path = Model_development(n_features, train_data, test_data, model_save_path, idx)

    # Save performance metrics
    performance_df = performance_df.append({
        'Feature': f'Feature{idx}',
        'Train Accuracy': history.history['accuracy'][-1],
        'Val Accuracy': history.history['val_accuracy'][-1],
        'Train Loss': history.history['loss'][-1],
        'Val Loss': history.history['val_loss'][-1]
    }, ignore_index=True)

# Save performance DataFrame to Excel file
performance_excel_path = 'model_performance.xlsx'
with pd.ExcelWriter(performance_excel_path, engine='openpyxl') as writer:
    performance_df.to_excel(writer, index=False, sheet_name='Performance')

    # Open the workbook and get the sheet
    workbook = writer.book
    sheet = workbook['Performance']

    # Insert images
    for idx, image_path in enumerate([f'training_curve_feature{i}.png' for i in range(len(dfs))]):
        img = Image(image_path)
        img.anchor = f'H{idx + 2}'  # Adjust the cell position as needed
        sheet.add_image(img)

# Cleanup plot images
for image_path in [f'training_curve_feature{i}.png' for i in range(len(dfs))]:
    os.remove(image_path)

print(f"Model performance and training curves saved to {performance_excel_path}")