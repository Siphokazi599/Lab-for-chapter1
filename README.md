# Lab-for-chapter1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
print("All packages imported successfully!")

     
All packages imported successfully!

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

     

# Explore the dataset
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First 5 samples:\n", X[:5])
# Explore the dataset
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First 5 samples:\n", X[:5])

     
Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Target names: ['setosa' 'versicolor' 'virginica']
First 5 samples:
 [[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Target names: ['setosa' 'versicolor' 'virginica']
First 5 samples:
 [[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]

# Simple visualization
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Iris Dataset")
plt.show()

     


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

     

iris = load_iris()
X = iris.data              # Features: sepal length, width, petal length, width
y = iris.target            # Target: 0 - Setosa, 1 - Versicolour, 2 - Virginica

# Optional: Show the first few samples
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print("Sample Data:")
print(df.head())

     
Sample Data:
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \
0                5.1               3.5                1.4               0.2   
1                4.9               3.0                1.4               0.2   
2                4.7               3.2                1.3               0.2   
3                4.6               3.1                1.5               0.2   
4                5.0               3.6                1.4               0.2   

   species  
0        0  
1        0  
2        0  
3        0  
4        0  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

     
LogisticRegression(max_iter=200)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

     
Accuracy: 1.0

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


from sklearn.cluster import KMeans

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Compare with true labels
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("K-means Clustering Results")
plt.show()

     


from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

     
Training set size: (105, 4)
Test set size: (45, 4)

from sklearn.neighbors import KNeighborsClassifier
# Create and train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

     
KNeighborsClassifier(n_neighbors=3)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# Make predictions
y_pred = model.predict(X_test)
# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

     
Model accuracy: 1.00
