from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 1. Import a CSV file
df = pd.read_csv('heart_cleveland_upload.csv')

# 2. Split the imported file to X and Y where X include the attributes and Y is the result.
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# 3. Perform split into both train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# 4. Implement Neural Network and give a prediction
# Initialize the model
model = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, random_state=0) # 3 layers of 100 nodes itterating 500 times
model.fit(X_train, Y_train)
# Make predictions on the test data
Y_pred = model.predict(X_test)
# Round the predictions to either 0 or 1
Y_pred = np.round(Y_pred)

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
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("NN")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0, 1], rotation=45)
plt.yticks(tick_marks, [0, 1])

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
