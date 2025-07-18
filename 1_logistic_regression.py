"""
Logistic Regression on the Iris dataset with user input prediction
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

    print("\nEnter values for prediction (sepal length, sepal width, petal length, petal width):")
    user_input = [float(input(f"{feature}: ")) for feature in iris.feature_names]
    prediction = model.predict([user_input])
    print("Predicted class:", iris.target_names[prediction[0]])

if __name__ == "__main__":
    main()
