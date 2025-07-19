"""
Random Forest on the Wine dataset with user input prediction
"""

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    wine = load_wine()
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

    print("\nEnter 13 feature values for prediction:")
    user_input = [float(input(f"{name}: ")) for name in wine.feature_names]
    prediction = model.predict([user_input])
    print("Predicted Wine Class:", wine.target_names[prediction[0]])

if __name__ == "__main__":
    main()
