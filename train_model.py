import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# Load dataset
data = pd.read_csv("diabetes.csv")


print("Dataset loaded successfully!")
print("Total records:", len(data))

print("\nFirst 5 records:")
print(data.head())


# -----------------------------------
# Handle missing values
# -----------------------------------

features = [
    "Glucose",
    "Blood Pressure",
    "Insulin",
    "BMI",
    "Age"
]

target = "Outcome"


# Replace zero values with missing values
# for features where zero is not realistic
data["Glucose"] = data["Glucose"].replace(0, pd.NA)
data["Blood Pressure"] = data["Blood Pressure"].replace(0, pd.NA)
data["Insulin"] = data["Insulin"].replace(0, pd.NA)
data["BMI"] = data["BMI"].replace(0, pd.NA)


# Fill missing values with median
for column in features:

    data[column] = data[column].fillna(
        data[column].median()
    )


# -----------------------------------
# Features and Target
# -----------------------------------

X = data[features]

y = data[target]


# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------------
# Feature Scaling
# -----------------------------------

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# -----------------------------------
# Logistic Regression
# -----------------------------------

logistic_model = LogisticRegression(
    max_iter=1000
)


logistic_model.fit(
    X_train_scaled,
    y_train
)


logistic_prediction = logistic_model.predict(
    X_test_scaled
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)


# -----------------------------------
# SVM
# -----------------------------------

svm_model = SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    probability=True
)


svm_model.fit(
    X_train_scaled,
    y_train
)


svm_prediction = svm_model.predict(
    X_test_scaled
)


svm_accuracy = accuracy_score(
    y_test,
    svm_prediction
)


# -----------------------------------
# Logistic Regression Results
# -----------------------------------

print("\n================================")
print("LOGISTIC REGRESSION")
print("================================")

print(
    "Accuracy:",
    round(logistic_accuracy * 100, 2),
    "%"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        logistic_prediction
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_prediction
    )
)


print(
    "Precision:",
    round(
        precision_score(
            y_test,
            logistic_prediction
        ),
        3
    )
)


print(
    "Recall:",
    round(
        recall_score(
            y_test,
            logistic_prediction
        ),
        3
    )
)


print(
    "F1 Score:",
    round(
        f1_score(
            y_test,
            logistic_prediction
        ),
        3
    )
)


# -----------------------------------
# SVM Results
# -----------------------------------

print("\n================================")
print("SUPPORT VECTOR MACHINE")
print("================================")

print(
    "Accuracy:",
    round(svm_accuracy * 100, 2),
    "%"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        svm_prediction
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        svm_prediction
    )
)


print(
    "Precision:",
    round(
        precision_score(
            y_test,
            svm_prediction
        ),
        3
    )
)


print(
    "Recall:",
    round(
        recall_score(
            y_test,
            svm_prediction
        ),
        3
    )
)


print(
    "F1 Score:",
    round(
        f1_score(
            y_test,
            svm_prediction
        ),
        3
    )
)


# -----------------------------------
# Compare Models
# -----------------------------------

print("\n================================")
print("MODEL COMPARISON")
print("================================")


print(
    "Logistic Regression:",
    round(
        logistic_accuracy * 100,
        2
    ),
    "%"
)


print(
    "SVM:",
    round(
        svm_accuracy * 100,
        2
    ),
    "%"
)


if logistic_accuracy > svm_accuracy:

    print(
        "\nBest Model: Logistic Regression"
    )

else:

    print(
        "\nBest Model: SVM"
    )


# -----------------------------------
# Save Models
# -----------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


with open(
    "models/logistic_model.pkl",
    "wb"
) as file:

    pickle.dump(
        logistic_model,
        file
    )


with open(
    "models/svm_model.pkl",
    "wb"
) as file:

    pickle.dump(
        svm_model,
        file
    )


with open(
    "models/scaler.pkl",
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


print("\nModels saved successfully!")