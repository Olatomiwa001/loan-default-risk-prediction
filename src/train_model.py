import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate sample dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open("src/loan_default_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
