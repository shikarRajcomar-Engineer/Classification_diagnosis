
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
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')



# Prepare the dataset

# This code combines the ci fault and ti fault summary into a master file
# def Data_prep(pattern):
#     path=os.getcwd()
#     # pattern='final'

#     masterfile=pd.DataFrame()
#     files=[f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
#     for filename in files:
#         if filename.startswith(pattern):
#             df=pd.read_excel(filename)
#             masterfile=masterfile.append(df)
#     masterfile=masterfile.iloc[:,2:]
#     return masterfile


# ----------------------------------------------------------------------------------------------------------
# 1.Model Architecture
# ----------------------------------------------------------------------------------------------------------

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

    plt.figure(figsize=(12, 4))
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

    best_model = keras.models.clone_model(autoencoder)
    best_model.set_weights(autoencoder.get_weights())
    best_model.save('AE_feature.h5')

    return history,autoencoder

# ----------------------------------------------------------------------------------------------------------
# 2.Generate Evaluation Metrics
# ----------------------------------------------------------------------------------------------------------
def evaluation_mertrics(filename,error,df,method):
    """
    Calculate and visualize evaluation metrics based on classification results.

    This function calculates various evaluation metrics for a classification problem, including
    accuracy, precision, recall, F1-score, specificity, false positive rate, false negative rate,
    true negative rate, balanced accuracy, and Matthews correlation coefficient (MCC). It also
    generates a confusion matrix heatmap and exports the results to an Excel file.

    Parameters:
        filename (str): A string representing the base filename for the output Excel file.
        error (pandas.DataFrame): A DataFrame containing classification error information.
            It should have a 'Class' column representing the true class labels.
        df (pandas.DataFrame): A DataFrame containing the original data with true class labels.
        method (str): A string indicating the method used for classification.

    Returns:
        pandas.DataFrame: A DataFrame containing evaluation metrics for different class labels
        and summary statistics.

    Notes:
        - The function calculates evaluation metrics such as accuracy, precision, recall, F1-score,
          specificity, false positive rate, false negative rate, true negative rate, balanced
          accuracy, and MCC.
        - A confusion matrix heatmap is plotted and saved.
        - The results are exported to an Excel file with the format: '{filename}_Results_{method}.xlsx'.

    Example:
        # Assuming 'error' contains classification error information and 'df' contains the true labels
        filename = "evaluation"
        method = "MyMethod"
        metrics_results = evaluation_metrics(filename, error, df, method)
    """
 
    Results=[]
    method=method
    LABELS=["Normal", "Anomaly"]
    conf_matrix = confusion_matrix(df.Class, error.Class.astype(int))
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");


    # Calculate evaluation metrics
    acc = accuracy_score(df.Class, error.Class.astype(int))
    prec = precision_score(df.Class, error.Class.astype(int))
    rec = recall_score(df.Class, error.Class.astype(int))
    f1 = f1_score(df.Class, error.Class.astype(int))
    tn, fp, fn, tp = confusion_matrix(df.Class, error.Class.astype(int)).ravel()
    spec = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    tnr = tn / (tn + fp)
    bal_acc = (tp / (tp + fn) + tn / (tn + fp)) / 2
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    # Convert confusion matrix to dataframe and add evaluation metrics as columns
    df_cm = pd.DataFrame(conf_matrix, index=LABELS, columns=LABELS)
    df_cm['Accuracy'] = acc
    df_cm['Precision'] = prec
    df_cm['Recall'] = rec
    df_cm['F1'] = f1
    df_cm['Specificity'] = spec
    df_cm['False Positive Rate'] = fpr
    df_cm['False Negative Rate'] = fnr
    df_cm['True Negative Rate'] = tnr
    df_cm['Balanced Accuracy'] = bal_acc
    df_cm['MCC'] = mcc
    df_cm['file'] =filename
    Results.append(df_cm)


    Results_array = np.array(Results)

    reshaped_array = Results_array.reshape(-1, 13)
    Results=pd.DataFrame(reshaped_array)
    Results.columns=['Normal','Anomaly','Accuracy','Precision','Recall','F1','Specificity','False Positive Rate','False Negative Rate','True Negative Rate','Balanced Accuracy','MCC','file']
    print(Results)
    Results.to_excel(f'{filename}_Results_{method}.xlsx')
    return Results

# ----------------------------------------------------------------------------------------------------------
# 3.Adaptive Threshold using Mahalanobis distance
# ----------------------------------------------------------------------------------------------------------
def mahalanobis(x=None, data=None, cov=None):
    """
    Compute the Mahalanobis distance between each row of `x` and the data.

    Mahalanobis distance is a measure of the distance between a point `x` and a dataset `data`,
    taking into account the correlations between variables.

    Parameters:
        x (numpy.ndarray or None): An array representing a set of points for which to compute
            Mahalanobis distance. Each row is a separate point. If None, the function will
            only calculate the squared Mahalanobis distances for the data points.
        data (pandas.DataFrame or None): A DataFrame containing the dataset against which to
            compute Mahalanobis distance. Each row represents a separate data point, and each
            column represents a different variable.
        cov (numpy.ndarray or None): The covariance matrix to be used for computing Mahalanobis
            distance. If None, the function will estimate the covariance matrix from the data
            with ridge regularization.

    Returns:
        numpy.ndarray: An array containing the squared Mahalanobis distances between each point
        in `x` and the data points in `data`.

    Notes:
        - The Mahalanobis distance is calculated as sqrt((x - mean(data)) @ inv(cov) @ (x - mean(data)).T).
        - If `x` is None, the function will return an array of squared Mahalanobis distances for
            the data points in `data`.

    Example:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        data = pd.DataFrame([[1, 2, 3], [2, 3, 4], [5, 6, 7]])
        cov_matrix = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
        distances = mahalanobis(x, data, cov_matrix)
    """
    
    if not cov:
        # Add ridge regularization to the covariance matrix
        cov = np.cov(data.values.T) + np.eye(data.shape[1]) * 1e-6
    x_mu = x - np.mean(data.values, axis=0)
    inv_covmat = np.linalg.pinv(cov)
    left_term = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left_term, x_mu.T)
    return mahal.diagonal()

def detect_outliers_Mahalanobis(data):
    """
    Detect outliers in a DataFrame using Mahalanobis distance and percentile threshold.

    Outliers are identified by calculating the Mahalanobis distance for each row of the input
    DataFrame and comparing it to a percentile-based threshold.

    Parameters:
        data (pandas.DataFrame): A DataFrame containing the data for outlier detection.
            Each row represents a separate observation, and each column represents a different
            variable.

    Returns:
        numpy.ndarray: An array of boolean values indicating whether each corresponding row in
        the input DataFrame is an outlier (True) or not (False).

    Notes:
        - Mahalanobis distance is a measure of the distance between a point and a dataset,
          accounting for correlations between variables.
        - The function calculates the Mahalanobis distance for each row using the 'mahalanobis'
          function. It then calculates a percentile-based threshold and identifies outliers by
          comparing the Mahalanobis distances to this threshold.

    Example:
        data = pd.DataFrame({'Feature1': [1.2, 2.5, 3.1, 0.8, 2.3],
                             'Feature2': [0.9, 2.0, 2.7, 0.6, 2.1]})
        outliers = detect_outliers_Mahalanobis(data)
    """

    # Calculate Mahalanobis distance for each row
    
    dist = mahalanobis(x=data.values, data=data)

    # Calculate percentile threshold
    threshold_percentile = np.percentile(dist, 85)  # Example: using 95th percentile

    # Identify outliers based on the threshold
    outliers = dist > threshold_percentile
    return outliers