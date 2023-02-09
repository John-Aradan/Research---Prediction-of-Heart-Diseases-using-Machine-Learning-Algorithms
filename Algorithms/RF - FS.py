import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix

# Import the CSV file into a Pandas dataframe
df = pd.read_csv('heart_cleveland_upload.csv')

# Split the dataframe into features and target variables
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Implement Fisher Score feature selection
selector = SelectKBest(f_classif, k=5)  # take top 5 features
selector.fit(X_train, Y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

# Implement Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, Y_train)

# Predict on the test set
Y_pred = clf.predict(X_test)

# Create a confusion matrix
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

# Plot the confusion matrix
classes=np.unique(Y)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("RF-FS")
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
