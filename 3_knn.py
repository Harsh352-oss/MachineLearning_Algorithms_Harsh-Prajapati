"""
K-Nearest Neighbors on the Digits dataset with simplified user input
"""

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("✅ KNN Accuracy:", accuracy_score(y_test, y_pred))

    print("\n📥 Enter 64 pixel values (0–16) as comma-separated numbers (e.g. 0,0,12,7,...):")
    user_input_str = input("➡ Pixel values: ")

    try:
        user_input = [float(x.strip()) for x in user_input_str.split(',')]
        if len(user_input) != 64:
            raise ValueError("You must enter exactly 64 values.")
        prediction = model.predict([user_input])
        print("🔢 Predicted Digit:", prediction[0])
    except Exception as e:
        print("❌ Invalid input:", e)

if __name__ == "__main__":
    main()
