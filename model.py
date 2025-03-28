import pickle

def load_model():
    with open("loan_default_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def predict(model, input_data):
    return model.predict([input_data])[0]  # Adjust based on input format

def load_model():
    with open("src/loan_default_model.pkl", "rb") as file:  # <-- Correct path
        model = pickle.load(file)
    return model