import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import tree


# Import the CSV file and store it as a dataframe
df = pd.read_csv("heart_cleveland_upload.csv")

# Split the data into the attributes (X) and the result (Y)
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) # 20% of the data for testing


# Train the Decision Tree model
model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Use the trained model to make predictions on the test data
Y_pred = model.predict(X_test)

# Calculate the confusion matrix
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
classes = np.unique(Y)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("DT")
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
