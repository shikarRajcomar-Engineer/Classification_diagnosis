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



labels=['Actual 0', 'Actual 1', 'Actual 2']


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






def model_testing(file_name,model):

    df = pd.read_excel(file_name)
    X = df.drop(columns=['Flag'])
    y = df['Flag']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.4)
    plot_training_history(history)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)


    return accuracy, precision, recall, f1, conf_matrix


# Create models
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)

neural_network = Sequential([
    Dense(128, input_shape=([7,]), activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='relu')
])



neural_network.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001), loss=tf.keras.losses.MSE, metrics=[tf.keras.metrics.mean_squared_error,'accuracy'])





# Create a list of models
models = [neural_network]


Eval_metrics=pd.DataFrame()
confusion_mat=pd.DataFrame()



# Test each model
for model in models:
    model_name = type(model).__name__

    accuracy, precision, recall, f1, conf_matrix = model_testing('master.xlsx', model)
    data={'Accuracy':[accuracy],
        'Precision':[precision],
        'Recall':[recall],
        'F1_score':[f1],
        'Model':[model_name]
        }
    conf=pd.DataFrame(conf_matrix)
    confusion_mat=confusion_mat.append(conf)
    data = pd.DataFrame(data)
    Eval_metrics=Eval_metrics.append(data)

# Display the resulting DataFrame
print(confusion_mat)
print(Eval_metrics)







