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



# Prepare the dataset


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


# Create models
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)

# Create a list of models
models = [random_forest, logistic_regression, decision_tree]
labels=['Actual 0', 'Actual 1', 'Actual 2']

def model_testing(file_name,model):

    df = pd.read_excel(file_name)
    X = df.drop(columns=['Flag'])
    y = df['Flag']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, conf_matrix


Eval_metrics=pd.DataFrame()
confusion_mat=pd.DataFrame()
# Test each model
for model in models:
    model_name = type(model).__name__
    print(model)
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







