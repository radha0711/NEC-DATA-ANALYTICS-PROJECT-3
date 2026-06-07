import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
# Directory setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == "src":
    BASE_DIR = os.path.dirname(CURRENT_DIR)
else:
    BASE_DIR = CURRENT_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.pkl")
def main():
    print("--- Step 2: Model Training ---")
    
    # Check if processed data exists
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed data file '{PROCESSED_DATA_PATH}' not found.")
        print("Please run Step 1 (src/data_prep.py) before training models.")
        return
        
    # Load processed data
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data = pickle.load(f)
        
    X = data["embeddings"]
    y_names = data["names"]
    y_genders = data["genders"]
    
    print(f"Loaded {len(X)} face embeddings.")
    
    if len(X) < 5:
        print("Error: Too few samples to train models. Please gather more dataset images.")
        return
        
    # --- Name Classifier Training ---
    print("\nTraining Name Prediction Model...")
    name_encoder = LabelEncoder()
    y_names_encoded = name_encoder.fit_transform(y_names)
    
    # Split for Name validation
    # If there is only 1 class or too few samples, we might not split or handle it gracefully
    unique_names = np.unique(y_names_encoded)
    if len(unique_names) < 2:
        print("Warning: Only 1 unique person in dataset. Validation split will be bypassed.")
        X_train_n, X_test_n, y_train_n, y_test_n = X, X, y_names_encoded, y_names_encoded
    else:
        # Standard 80/20 train/test split
        X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
            X, y_names_encoded, test_size=0.2, random_state=42, stratify=y_names_encoded
        )
        
    # Linear SVM works exceptionally well for high-dimensional, normalized face encodings
    name_classifier = SVC(C=1.0, kernel='linear', probability=True, random_state=42)
    name_classifier.fit(X_train_n, y_train_n)
    
    # Evaluate Name model
    y_pred_n = name_classifier.predict(X_test_n)
    acc_n = accuracy_score(y_test_n, y_pred_n)
    print(f"Name Classifier Accuracy: {acc_n * 100:.2f}%")
    print(classification_report(y_test_n, y_pred_n, target_names=name_encoder.classes_, zero_division=0))
    
    # --- Gender Classifier Training ---
    print("\nTraining Gender Prediction Model...")
    gender_encoder = LabelEncoder()
    y_genders_encoded = gender_encoder.fit_transform(y_genders)
    
    # Split for Gender validation
    unique_genders = np.unique(y_genders_encoded)
    if len(unique_genders) < 2:
        print("Warning: Only 1 unique gender in dataset. Validation split will be bypassed.")
        X_train_g, X_test_g, y_train_g, y_test_g = X, X, y_genders_encoded, y_genders_encoded
    else:
        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
            X, y_genders_encoded, test_size=0.2, random_state=42, stratify=y_genders_encoded
        )
        
    gender_classifier = SVC(C=1.0, kernel='linear', probability=True, random_state=42)
    gender_classifier.fit(X_train_g, y_train_g)
    
    # Evaluate Gender model
    y_pred_g = gender_classifier.predict(X_test_g)
    acc_g = accuracy_score(y_test_g, y_pred_g)
    print(f"Gender Classifier Accuracy: {acc_g * 100:.2f}%")
    print(classification_report(y_test_g, y_pred_g, target_names=gender_encoder.classes_, zero_division=0))
    
    # --- Save Models and Encoders ---
    # Save Name model and encoder
    name_model_path = os.path.join(MODELS_DIR, "name_model.pkl")
    name_encoder_path = os.path.join(MODELS_DIR, "name_encoder.pkl")
    with open(name_model_path, "wb") as f:
        pickle.dump(name_classifier, f)
    with open(name_encoder_path, "wb") as f:
        pickle.dump(name_encoder, f)
        
    # Save Gender model and encoder
    gender_model_path = os.path.join(MODELS_DIR, "gender_model.pkl")
    gender_encoder_path = os.path.join(MODELS_DIR, "gender_encoder.pkl")
    with open(gender_model_path, "wb") as f:
        pickle.dump(gender_classifier, f)
    with open(gender_encoder_path, "wb") as f:
        pickle.dump(gender_encoder, f)
        
    print(f"\nModels successfully saved to '{MODELS_DIR}' directory.")
    print("Next step: Run prediction on a new image (src/predict.py) or run the camera (src/webcam_predict.py).")
if __name__ == "__main__":
    main()