"""
Support Vector Machine on the Breast Cancer dataset with user input prediction
"""

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))

    print("\nEnter 30 feature values for prediction:")
    user_input = [float(input(f"{name}: ")) for name in cancer.feature_names]
    prediction = model.predict([user_input])
    print("Predicted Diagnosis:", "Malignant" if prediction[0] == 0 else "Benign")

if __name__ == "__main__":
    main()
