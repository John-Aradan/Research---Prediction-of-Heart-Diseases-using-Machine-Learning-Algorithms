import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE

# 1. Import a CSV file
df = pd.read_csv("heart_cleveland_upload.csv")

# 2. Split the imported file to X and Y where X include the attributes and Y is the result
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# 3. Perform split into both train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# 4. Implement Random Forest and give a prediction
clf = RandomForestClassifier(n_estimators=100, random_state=0)

# Use RFE to select the best features
rfe = RFE(clf)
rfe = rfe.fit(X_train, Y_train)

# Transform the train and test sets using the selected features
X_train = rfe.transform(X_train)
X_test = rfe.transform(X_test)

clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

# 5. Perform a confusion matrix on the prediction and actual result
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
acc = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
print("Accuracy: ", acc)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1)

# 6. Plot the result
classes=np.unique(Y)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("RF-RFE")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
