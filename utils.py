# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
plot_confusion_matrix,
plot_roc_curve,
precision_recall_curve,
average_precision_score)
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from itertools import cycle
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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

def plot_training_history(history):
    # Plot training history
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