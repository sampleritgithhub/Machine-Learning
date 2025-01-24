

# Step 1: Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

# Step 2: Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# Step 3: Create a DataFrame for better visualization
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target  # Add species as target
print("Dataset Head:\n", data.head())

# Step 4: Split the data into features (X) and target (y)
X = data.iloc[:, :-1].values  # Features (Sepal and Petal measurements)
y = data.iloc[:, -1].values   # Target (0: Setosa, 1: Versicolor, 2: Virginica)

# Step 5: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Standardize the features (important for distance-based algorithms)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Define the k-NN algorithm
def knn(X_train, y_train, X_test, k):
    predictions = []
    for x_test in X_test:
        # Calculate distances from the test sample to all training samples
        distances = np.sqrt(np.sum((X_train - x_test) ** 2, axis=1))
        # Find the indices of the k nearest neighbors
        neighbors_indices = np.argsort(distances)[:k]
        # Extract the labels of the k nearest neighbors
        neighbors_labels = y_train[neighbors_indices]
        # Perform majority voting
        most_common = Counter(neighbors_labels).most_common(1)
        predictions.append(most_common[0][0])
    return predictions

# Step 8: Evaluate k-NN for different values of k
k_values = range(1, 11)
accuracies = []

for k in k_values:
    y_pred = knn(X_train, y_train, X_test, k)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"\nFor k={k}:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title("Elbow Plot for k-NN")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid()
plt.show()

# Step 10: Determine the best k and display the corresponding confusion matrix
best_k = k_values[np.argmax(accuracies)]
print(f"\nOptimal k: {best_k} with Accuracy: {max(accuracies) * 100:.2f}%")

# Step 11: Final evaluation with the optimal k
y_pred_optimal = knn(X_train, y_train, X_test, best_k)
print("\nConfusion Matrix with Optimal k:\n", confusion_matrix(y_test, y_pred_optimal))
print("\nClassification Report with Optimal k:\n", classification_report(y_test, y_pred_optimal))
