# SVM Classification using Iris Dataset
# --------------------------------------

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ------------------------------------------------
# 1. Load the Iris dataset
# ------------------------------------------------

iris = datasets.load_iris()

X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)


# ------------------------------------------------
# 2. Split dataset into training and testing
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------
# 3. Standardise the features
# ------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------------
# 4. Create the SVM model
# ------------------------------------------------

svm_model = SVC(
    kernel="linear",
    C=1.0,
    random_state=42
)


# ------------------------------------------------
# 5. Train the model
# ------------------------------------------------

svm_model.fit(X_train, y_train)


# ------------------------------------------------
# 6. Make predictions on TEST dataset
# ------------------------------------------------

y_pred = svm_model.predict(X_test)


# =================================================
# 7. CONFUSION MATRIX
# =================================================

cm = confusion_matrix(y_test, y_pred)

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)


# ------------------------------------------------
# 8. Calculate Accuracy
# ------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ------------------------------------------------
# 9. Calculate Precision
# ------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nPrecision:")
print(f"{precision:.4f}")


# ------------------------------------------------
# 10. Calculate Recall
# ------------------------------------------------

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nRecall:")
print(f"{recall:.4f}")


# ------------------------------------------------
# 11. Calculate F1 Score
# ------------------------------------------------

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nF1 Score:")
print(f"{f1:.4f}")


# =================================================
# 12. Classification Report
# =================================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# =================================================
# 13. Display Confusion Matrix
# =================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title("SVM Confusion Matrix - Iris Test Dataset")
plt.show()


# =================================================
# 14. Predict a new flower
# =================================================

new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower_scaled = scaler.transform(new_flower)

prediction = svm_model.predict(new_flower_scaled)

print("\n======================================")
print("NEW FLOWER PREDICTION")
print("======================================")

print("Predicted class:", iris.target_names[prediction[0]])