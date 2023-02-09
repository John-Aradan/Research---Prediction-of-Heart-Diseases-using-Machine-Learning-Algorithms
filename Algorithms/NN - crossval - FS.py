import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


# >>----

def average_confusion_matrices(confusion_matrices):
    avg_confusion_matrix = np.zeros_like(confusion_matrices[0])
    for conf_matrix in confusion_matrices:
        avg_confusion_matrix += conf_matrix
    avg_confusion_matrix = avg_confusion_matrix / len(confusion_matrices)
    return avg_confusion_matrix

def avg(float_list):
    return sum(float_list)/len(float_list)

# >>----

# 1. Import a CSV file
df = pd.read_csv("heart_cleveland_upload.csv")

# 2. Split the imported file to X and Y where X include the attributes and Y is the result.
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# >>----
confusion_matrices = []
acc_avg = []
precision_avg = []
recall_avg = []
f1_avg = []
# >>----

# 3. Perform cross-validation split into both sets.
kf = StratifiedKFold(n_splits=5, shuffle=True,)
train_size = 0.8
for train_index, test_index in kf.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    

    # Implement Fisher Score feature selection
    selector = SelectKBest(f_classif, k=5)  # take top 5 features
    selector.fit(X_train, Y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    # Fit the model on the training data
    model = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, random_state=0) # 3 layers of 100 nodes itterating 500 times
    model.fit(X_train, Y_train)
    # Make predictions on the test data
    Y_pred = model.predict(X_test)
    # Round the predictions to either 0 or 1
    Y_pred = np.round(Y_pred)

    # >>----
    acc = accuracy_score(Y_test, Y_pred)
    acc_avg.append(acc)
    precision = precision_score(Y_test, Y_pred)
    precision_avg.append(precision)
    recall = recall_score(Y_test, Y_pred)
    recall_avg.append(recall)
    f1 = f1_score(Y_test, Y_pred)
    f1_avg.append(f1)
    # ----<<
    
    # 5. Perform a confusion matrix on the prediction and actual result
    cm = confusion_matrix(Y_test, Y_pred)
    # >>----
    confusion_matrices.append(cm)
    # ----<<

avg_confusion_matrix = average_confusion_matrices(confusion_matrices)
print(avg_confusion_matrix)
print("Confusion Matrix: \n", avg_confusion_matrix)
print("Average Accuracy:", avg(acc_avg))
print("Average Precision:", avg(precision_avg))
print("Average Recall:", avg(recall_avg))
print("Average F1 Score:", avg(f1_avg))

# 6. Plot the result
classes=np.unique(Y)
plt.imshow(avg_confusion_matrix, interpolation='nearest', cmap=plt.avg_confusion_matrix.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = avg_confusion_matrix.max() / 2.
for i, j in np.ndindex(avg_confusion_matrix.shape):
    plt.text(j, i, avg_confusion_matrix[i, j],
             horizontalalignment="center",
             color="white" if avg_confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()