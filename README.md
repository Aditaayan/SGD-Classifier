# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import Necessary Libraries and Load Data 
2.Split Dataset into Training and Testing Sets
3.Train the Model Using Stochastic Gradient Descent (SGD)
4.Make Predictions and Evaluate Accuracy
```
## Program:

## Program to implement the prediction of iris species using SGD Classifier.
## Developed by: ADITAAYAN M
## RegisterNumber: 212223040006
*/
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrics
import matplotlib.pyplot as plt
import seaborn as sns
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```


## Output:
![368835349-817e65d2-27c8-4758-b3cf-052b473b8f3a](https://github.com/user-attachments/assets/cc0b4d85-2d79-41b2-bb74-82ddf2abe26a)

![368835394-7fea0925-030c-4aa2-b1c8-e51c639c0f83](https://github.com/user-attachments/assets/e0b48a48-e3a8-43d5-a14d-f242ce7cf316)



![368835423-7c0f9a5d-ec52-4ff9-84a5-e1095c3cc0c5](https://github.com/user-attachments/assets/bbaec29b-0c72-4b6f-8eab-553c599c7753)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
